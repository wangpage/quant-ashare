"""batch_scan — 247 只候选池全扫描 + 桶化推荐 + LLM 精分析 + 飞书汇总.

复用 watchlist_signal_v2 的因子引擎 (反转/打板/席位/板块),
 把股票池从硬编的 user_watchlist.yaml 扩展到 股票名称_代码.csv。

流程:
    1. 读 CSV → 清洗 6 位代码列表
    2. fetch_watchlist_kline(codes) — 自动增量补齐 cache
    3. 计算 4 类因子面板 (反转/打板/席位/板块)
    4. synthesize_composite → alpha_z 排序
    5. 桶化: 推荐(z≥1.5) / 观察(0.5≤z<1.5) / 中性 / 不推荐(z<-0.5)
    6. Top20 + Bottom10 → LLM 决策 (buy/watch/avoid + 止损止盈)
    7. 输出 CSV + Markdown 摘要
    8. 飞书 IM 推送

用法:
    python3 scripts/batch_scan.py                                 # 默认读 股票名称_代码.csv
    python3 scripts/batch_scan.py --codes-csv /path/to/other.csv  # 自定义池
    python3 scripts/batch_scan.py --dry-run                       # 不发飞书
    python3 scripts/batch_scan.py --no-llm                        # 跳过 LLM (纯量化)
    python3 scripts/batch_scan.py --top 20 --bottom 10            # 自定义 LLM 数量
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# 加载 .env
_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# 复用 watchlist_signal_v2 的因子引擎 (不复用它的 fetch, 有 parquet dtype bug)
from scripts.watchlist_signal_v2 import (
    load_lhb,
    compute_reversal_panel, synthesize_composite, latest_slice,
)
from data_adapter.em_direct import bulk_fetch_daily
from factors.alpha_limit import compute_limit_alpha
from factors.seat_network import compute_seat_alpha
from factors.sector_momentum import compute_sector_momentum, load_universe_kline


# -------- 自建稳健 fetch (绕开 v2 的 date dtype bug) --------
def fetch_kline(codes: list[str], days_back: int = 180) -> pd.DataFrame:
    """拉 codes 的日 K, 用独立 cache 文件防和 v2 冲突. 自动增量补齐."""
    today = pd.Timestamp.today().normalize()
    start = (today - pd.Timedelta(days=int(days_back * 1.5))).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    cache_path = ROOT / "cache" / f"batch_scan_kline_{end}.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = df["code"].astype(str).str.zfill(6)
        cached = set(df["code"].unique())
        missing = [c for c in codes if c not in cached]
        if not missing:
            return df[df["code"].isin(codes)].copy()
        logger.info(f"fetch_kline: cache 命中 {len(cached & set(codes))} 只, 补拉 {len(missing)} 只")
        extra = bulk_fetch_daily(missing, start, end, sleep_ms=80, progress=True)
    else:
        logger.info(f"fetch_kline: 无 cache, 全量拉 {len(codes)} 只")
        df = pd.DataFrame()
        extra = bulk_fetch_daily(codes, start, end, sleep_ms=80, progress=True)

    if not extra.empty:
        extra["date"] = pd.to_datetime(extra["date"])
        extra["code"] = extra["code"].astype(str).str.zfill(6)
        df = pd.concat([df, extra], ignore_index=True) if not df.empty else extra
        df = df.drop_duplicates(subset=["code", "date"]).sort_values(["code", "date"])
        try:
            df.to_parquet(cache_path, index=False)
            logger.info(f"fetch_kline: cache 已更新 {cache_path.name}")
        except Exception as e:
            logger.warning(f"fetch_kline: cache 写入失败 {e}, 内存使用")
    return df[df["code"].isin(codes)].copy() if not df.empty else df
from llm_layer.agents import _LLMBackend
from llm_layer import xml_parser as xp
from llm_layer.prompts_shortline import SHORTLINE_PICK_PROMPT

from notifier import feishu_client
from utils.logger import logger

OUTPUT_DIR = ROOT / "output" / "batch_scan"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CSV = Path("/Users/page/Desktop/股票/股票名称_代码.csv")
DEFAULT_USER = "ou_5be0f87dc7cec796b7ea97d0a9b5302f"


# ========== 1. 读 CSV ==========
def load_codes_csv(path: Path) -> pd.DataFrame:
    """读 股票名称_代码.csv, 清洗非法代码. 返回 DataFrame[code, name]."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)
    # 兼容列名: 代码/code, 股票名称/name
    code_col = next((c for c in ["代码", "code"] if c in df.columns), None)
    name_col = next((c for c in ["股票名称", "name"] if c in df.columns), None)
    if code_col is None:
        raise ValueError(f"CSV 缺代码列, 现有列: {list(df.columns)}")
    df["code"] = df[code_col].astype(str)
    df["name"] = df[name_col].astype(str) if name_col else df["code"]
    # 清洗: 只保留 6 位纯数字代码
    mask = df["code"].str.match(r"^\d{6}$")
    dropped = (~mask).sum()
    df = df[mask].copy()
    df["code"] = df["code"].str.zfill(6)
    df = df.drop_duplicates(subset=["code"]).reset_index(drop=True)
    logger.info(f"batch_scan: 读 {path.name} {len(df)} 只 (丢弃 {dropped} 条非法)")
    return df[["code", "name"]]


# ========== 2-4. 因子计算 ==========
def compute_all_factors(codes: list[str], name_map: dict) -> pd.DataFrame:
    """对 codes 列表算全套 v2 因子, 返回 synthesize_composite 的输出."""
    # K 线 (180 天, 自动增量补齐)
    kline = fetch_kline(codes, days_back=180)
    if kline.empty:
        logger.error("K 线拉取失败, 无法继续")
        return pd.DataFrame()
    logger.info(f"K 线覆盖 {kline['code'].nunique()} 只 (目标 {len(codes)})")

    # 反转因子
    rev_df = compute_reversal_panel(kline)
    logger.info(f"反转因子: {len(rev_df)} 只")
    # 挂上 name
    rev_df["name"] = rev_df.index.map(lambda c: name_map.get(c, c))

    # 打板因子
    try:
        limit_panel = compute_limit_alpha(kline)
        limit_slc = latest_slice(limit_panel, kline["date"].max())
    except Exception as e:
        logger.warning(f"打板因子失败: {e}")
        limit_slc = pd.DataFrame()

    # 席位因子 (compute_seat_alpha 签名: lhb_df, trading_dates)
    lhb = load_lhb()
    if lhb is not None and not lhb.empty:
        try:
            trading_dates = pd.DatetimeIndex(sorted(pd.to_datetime(kline["date"].unique())))
            seat_panel = compute_seat_alpha(lhb, trading_dates)
            seat_slc = latest_slice(seat_panel, kline["date"].max())
        except Exception as e:
            logger.warning(f"席位因子失败: {e}")
            seat_slc = pd.DataFrame()
    else:
        seat_slc = pd.DataFrame()

    # 板块动量
    try:
        universe_kline = load_universe_kline(ROOT / "cache")
        sector_df = compute_sector_momentum(
            kline, universe_kline,
            as_of=pd.Timestamp(kline["date"].max()),
            lookback=5, n_neighbors=15, corr_window=60,
        )
        # v2 里是直接拿 sector_df 当 slice, 不用 latest_slice
        sector_slc = sector_df
    except Exception as e:
        logger.warning(f"板块因子失败: {e}")
        sector_slc = pd.DataFrame()

    # 合成
    sig = synthesize_composite(
        rev_df, limit_slc, seat_slc,
        intraday_slice=None,
        sector_slice=sector_slc if not sector_slc.empty else None,
    )
    return sig


# ========== 5. 桶化 ==========
BUCKET_TIERS = [
    ("🟩 推荐",   1.5,   99.0, "strong_buy"),
    ("🟢 关注",   0.5,    1.5, "watch"),
    ("⚪ 中性",  -0.5,    0.5, "neutral"),
    ("🟡 谨慎",  -1.5,   -0.5, "caution"),
    ("🟥 不推荐", -99.0, -1.5, "avoid"),
]


def bucketize(sig: pd.DataFrame) -> pd.DataFrame:
    """给 sig 加一列 bucket."""
    def _tag(z):
        for label, lo, hi, key in BUCKET_TIERS:
            if lo <= z < hi:
                return key
        return "neutral"
    out = sig.copy()
    out["bucket"] = out["alpha_z"].map(_tag)
    return out


# ========== 6. LLM 精分析 ==========
def build_prompt(row: pd.Series, stats: dict, sent: dict) -> str:
    return SHORTLINE_PICK_PROMPT.format(
        code=row.name, name=row.get("name", row.name),
        price=row["latest_close"],
        alpha_z=row["alpha_z"],
        top_category=row.get("top_category", ""),
        rev_score=row.get("rev_score", 0),
        limit_score=row.get("limit_score", 0),
        seat_score=row.get("seat_score", 0),
        pct_5d=stats.get("pct_5d", 0),
        pct_20d=stats.get("pct_20d", 0),
        ma5=f"{stats.get('ma5', 0):.2f}",
        ma20=f"{stats.get('ma20', 0):.2f}",
        sentiment_regime=sent.get("regime", "⚪️ 平稳"),
        limit_up_count=sent.get("limit_up_count", 0),
        max_streak=sent.get("max_streak_up", 0),
        boom_rate=sent.get("boom_rate", 0),
    )


def _safe_float(s, default=None):
    try:
        return float(s) if s else default
    except (ValueError, TypeError):
        return default


def _safe_int(s, default=2):
    try:
        return int(float(s)) if s else default
    except (ValueError, TypeError):
        return default


def decide_one(row: pd.Series, stats: dict, sent: dict,
                backend: _LLMBackend) -> dict:
    code = row.name
    prompt = build_prompt(row, stats, sent)
    base = {"code": code, "name": row.get("name", code),
            "price": row["latest_close"], "alpha_z": row["alpha_z"]}
    try:
        raw = backend.chat(prompt, max_tokens=900)
        return {
            **base,
            "action": (xp.extract_tag(raw, "ACTION") or "watch").strip().lower(),
            "conviction": _safe_float(xp.extract_tag(raw, "CONVICTION"), 0.0),
            "stop_loss": _safe_float(xp.extract_tag(raw, "STOP_LOSS"), None),
            "take_profit": _safe_float(xp.extract_tag(raw, "TAKE_PROFIT"), None),
            "holding": _safe_int(xp.extract_tag(raw, "HOLDING_DAYS"), 2),
            "reason": (xp.extract_tag(raw, "EXPLANATION") or "").strip(),
            "risk": (xp.extract_tag(raw, "RISK") or "").strip(),
        }
    except Exception as e:
        return {**base, "action": "error", "reason": f"LLM 失败: {e}"}


def load_kline_stats_from(kline: pd.DataFrame, codes: list[str]) -> dict:
    """从已加载的 kline DataFrame 计算 stats, 避免重读 cache."""
    df = kline.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.sort_values(["code", "date"])
    out = {}
    for c in codes:
        sub = df[df["code"] == c].tail(21)
        if len(sub) < 21:
            continue
        close = sub["close"].values
        out[c] = {
            "pct_5d":  (close[-1] / close[-6] - 1) * 100 if len(close) >= 6 else 0,
            "pct_20d": (close[-1] / close[-21] - 1) * 100,
            "ma5":     float(close[-5:].mean()),
            "ma20":    float(close[-20:].mean()),
        }
    return out


def load_sentiment() -> dict:
    sdir = ROOT / "output" / "sentiment_cycle"
    js = sorted(sdir.glob("*.json")) if sdir.exists() else []
    if not js:
        return {"regime": "⚪️ 平稳", "limit_up_count": 0,
                "max_streak_up": 0, "boom_rate": 0.0}
    import json
    return json.loads(js[-1].read_text(encoding="utf-8"))


# backend 别名 → (实际 backend, 默认 model), 对齐 llm_shortline_pick.py
_BACKEND_MODELS: dict[str, tuple[str, str]] = {
    "qwen":        ("qwen",        "qwen-plus"),
    "dashscope":   ("dashscope",   "qwen-plus"),
    "deepseek":    ("deepseek",    "deepseek-chat"),
    "anthropic":   ("anthropic",   "claude-haiku-4-5"),
    "zhizengzeng": ("zhizengzeng", "gpt-5.4-mini"),
    "gpt5-mini":   ("zhizengzeng", "gpt-5.4-mini"),
    "gemini3":     ("zhizengzeng", "gemini-3-flash-preview"),
    "mock":        ("mock",        "mock"),
}


def run_llm_batch(targets: pd.DataFrame, kline: pd.DataFrame,
                   concurrency: int = 6) -> list[dict]:
    """并发跑 LLM. targets: DataFrame 索引=code."""
    stats_map = load_kline_stats_from(kline, list(targets.index))
    sent = load_sentiment()
    alias = os.environ.get("LLM_BACKEND", "qwen")
    backend_key, default_model = _BACKEND_MODELS.get(alias, ("qwen", "qwen-plus"))
    model = os.environ.get("LLM_MODEL", default_model)
    backend = _LLMBackend(backend_key, model)

    logger.info(f"LLM 后端: {alias} (backend={backend_key}, model={model}), 并发 {concurrency}, 目标 {len(targets)} 只")
    decisions = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(decide_one, row, stats_map.get(code, {}), sent, backend): code
                for code, row in targets.iterrows()}
        for fut in as_completed(futs):
            decisions.append(fut.result())
    # 按原顺序排序
    order = {c: i for i, c in enumerate(targets.index)}
    decisions.sort(key=lambda d: order.get(d["code"], 999))
    return decisions


# ========== 7. 产出 ==========
def write_ranking_csv(sig: pd.DataFrame, date: str) -> Path:
    path = OUTPUT_DIR / f"{date}_ranking.csv"
    cols = ["name", "bucket", "alpha_z", "latest_close", "top_category", "cat_sign",
            "rev_score", "limit_score", "seat_score"]
    if "sector_score" in sig.columns:
        cols.append("sector_score")
    out = sig[cols].copy()
    out.index.name = "code"
    out.to_csv(path, encoding="utf-8-sig")
    return path


def write_llm_csv(decisions: list[dict], date: str) -> Path:
    path = OUTPUT_DIR / f"{date}_llm.csv"
    pd.DataFrame(decisions).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def build_summary_md(sig: pd.DataFrame, decisions: list[dict], date: str,
                      total: int, top_n: int, bottom_n: int) -> str:
    counts = sig["bucket"].value_counts().to_dict()

    def _cnt(k):
        return counts.get(k, 0)

    lines = [
        f"📊 **A股批量扫描 · {date}**",
        "",
        f"**候选池** {total} 只",
        f"🟩 推荐 {_cnt('strong_buy')} · 🟢 关注 {_cnt('watch')} · "
        f"⚪ 中性 {_cnt('neutral')} · 🟡 谨慎 {_cnt('caution')} · "
        f"🟥 不推荐 {_cnt('avoid')}",
        "",
    ]

    # LLM 精分析桶: 用 decisions 拆 buy/watch/avoid
    buy = [d for d in decisions if d.get("action") == "buy"]
    avoid = [d for d in decisions if d.get("action") == "avoid"]

    # 明日推荐买 (LLM action=buy 的部分)
    if buy:
        lines.append(f"**🟩 明日推荐买 ({len(buy)} 只)**")
        for d in buy[:8]:
            conv = d.get("conviction", 0) or 0
            sl = d.get("stop_loss")
            tp = d.get("take_profit")
            stop = f" 止损¥{sl:.2f}" if sl else ""
            profit = f" 目标¥{tp:.2f}" if tp else ""
            lines.append(f"· `{d['code']} {d['name']}` ¥{d['price']:.2f} "
                         f"[z={d['alpha_z']:+.2f} 信心{conv:.0%}]{stop}{profit}")
            if d.get("reason"):
                lines.append(f"   逻辑：{d['reason'][:60]}")
        lines.append("")

    # 明日回避
    if avoid:
        lines.append(f"**🟥 明日回避 ({len(avoid)} 只)**")
        for d in avoid[:8]:
            lines.append(f"· `{d['code']} {d['name']}` ¥{d['price']:.2f} "
                         f"[z={d['alpha_z']:+.2f}]")
            if d.get("reason"):
                lines.append(f"   理由：{d['reason'][:60]}")
        lines.append("")

    # Top 10 纯量化排序 (不管 LLM)
    top10 = sig.head(10)
    lines.append("**📈 量化打分 Top 10**")
    for code, r in top10.iterrows():
        cat = f"{r.get('cat_sign', '')}{r.get('top_category', '')}"
        lines.append(f"· `{code} {r.get('name', '')}` z={r['alpha_z']:+.2f} [{cat}] ¥{r['latest_close']:.2f}")
    lines.append("")

    # Bottom 5 (量化角度最不看好)
    bot5 = sig.tail(5).iloc[::-1]
    lines.append("**📉 量化打分 Bottom 5**")
    for code, r in bot5.iterrows():
        cat = f"{r.get('cat_sign', '')}{r.get('top_category', '')}"
        lines.append(f"· `{code} {r.get('name', '')}` z={r['alpha_z']:+.2f} [{cat}]")
    lines.append("")

    lines.append(f"📎 完整 {total} 只排序表: output/batch_scan/{date}_ranking.csv")
    lines.append(f"📎 LLM 决策 ({top_n}+{bottom_n} 只): output/batch_scan/{date}_llm.csv")
    lines.append("")
    lines.append("_⚠️ 排序打分 + 风险提示，非投资建议_")
    return "\n".join(lines)


# ========== 8. 主流程 ==========
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes-csv", default=str(DEFAULT_CSV))
    ap.add_argument("--top", type=int, default=20, help="LLM 分析 Top N")
    ap.add_argument("--bottom", type=int, default=10, help="LLM 分析 Bottom N")
    ap.add_argument("--no-llm", action="store_true", help="跳过 LLM 精分析")
    ap.add_argument("--dry-run", action="store_true", help="不发飞书")
    ap.add_argument("--user-id", default=os.environ.get("LARK_USER_OPEN_ID", DEFAULT_USER))
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = Path(args.codes_csv)
    if not csv_path.exists():
        logger.error(f"找不到 {csv_path}")
        return 1

    # 1. 读 CSV
    codes_df = load_codes_csv(csv_path)
    codes = codes_df["code"].tolist()
    name_map = dict(zip(codes_df["code"], codes_df["name"]))

    # 2-4. 因子
    sig = compute_all_factors(codes, name_map)
    if sig.empty:
        logger.error("因子计算失败")
        return 1
    sig["name"] = sig.index.map(lambda c: name_map.get(c, c))

    # 5. 桶化
    sig = bucketize(sig)
    total = len(sig)
    logger.info(f"参与排序 {total} 只 (原池 {len(codes)})")

    # 写 ranking CSV
    rank_path = write_ranking_csv(sig, today)
    logger.info(f"✓ 排序表 {rank_path.name}")

    # 6. LLM
    decisions = []
    if not args.no_llm:
        kline = fetch_kline(codes, days_back=180)
        targets_top = sig.head(args.top)
        targets_bot = sig.tail(args.bottom)
        targets = pd.concat([targets_top, targets_bot])
        targets = targets[~targets.index.duplicated(keep="first")]  # 池子太小时防重
        decisions = run_llm_batch(targets, kline, concurrency=args.concurrency)
        llm_path = write_llm_csv(decisions, today)
        logger.info(f"✓ LLM 决策 {llm_path.name}")

    # 7. 摘要
    md = build_summary_md(sig, decisions, today, total,
                           args.top if not args.no_llm else 0,
                           args.bottom if not args.no_llm else 0)
    md_path = OUTPUT_DIR / f"{today}_summary.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info(f"✓ 摘要 {md_path.name}")

    # 8. 飞书
    if args.dry_run:
        print(md)
        return 0

    if not feishu_client.auth_preflight():
        logger.error("auth preflight 失败")
        return 10

    ok = feishu_client.send_im(args.user_id, md)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

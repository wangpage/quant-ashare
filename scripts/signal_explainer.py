"""V3 信号解释器 - Qwen 为今日 top-K 持仓生成人读报告.

流程:
    1. 加载缓存 kline + 龙虎榜
    2. 计算全部因子 (技术 + 反转 + 龙虎榜)
    3. 只训练 **最后一个窗口** 并预测最新日期 (省时)
    4. 取 top 25, 为每只收集上下文
    5. Qwen 并发生成解释, 组装成 markdown 报告

用法:
    python3 scripts/signal_explainer.py --pool 500 --top 25
    # 需要环境变量 DASHSCOPE_API_KEY
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# 加载 .env
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from data_adapter.sina_universe import fetch_all_ashare
from data_adapter.lhb import build_lhb_features, LHB_FACTOR_NAMES
from factors.alpha_pandas import compute_pandas_alpha
from factors.alpha_reversal import compute_advanced_alpha

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"

# 关键因子的可读名称(给 Qwen 用)
FACTOR_DESC = {
    "REV_5":       "5日反转",
    "REV_10":      "10日反转",
    "REV_20":      "20日反转",
    "MOM12_1":     "12-1月动量",
    "MOM6_1":      "6-1月动量",
    "LOW_VOL_60":  "低波动(60日)",
    "TURN_TREND":  "换手率抬升",
    "MAX_RET_20":  "近20日避免极端单日涨",
    "LHB_FLAG_10": "近10日上过龙虎榜",
    "LHB_FLAG_5":  "近5日上过龙虎榜",
    "LHB_NB_20":   "近20日龙虎榜净买入",
    "LHB_JIGOU_20": "近20日机构席位上榜",
    "LHB_COUNT_60": "近60日上榜频次",
    "AMIHUD_20":   "非流动性溢价",
}


def _fit_and_predict(X_tr, y_tr, X_te):
    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.04,
            num_leaves=31, max_depth=6, min_child_samples=80,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
        )
        m.fit(X_tr.values, y_tr.values)
    except (OSError, ImportError):
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.04,
            max_leaf_nodes=31, max_depth=6,
            min_samples_leaf=80, l2_regularization=0.1, random_state=42,
        )
        m.fit(X_tr.values, y_tr.values)
    return pd.Series(m.predict(X_te.values), index=X_te.index), m


def make_label(daily_df, horizon=30):
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    open_next = df.groupby("code")["open"].shift(-1)
    close_fwd = df.groupby("code")["close"].shift(-horizon)
    df["label"] = close_fwd / open_next - 1
    df["label"] = df.groupby("date")["label"].transform(
        lambda s: s.clip(s.quantile(0.01), s.quantile(0.99))
    )
    return df.set_index(["date", "code"])["label"]


def cs_zscore(feat_df):
    def _z(s):
        mu, sd = s.mean(), s.std()
        return (s - mu) / sd if sd > 0 else s * 0
    return feat_df.groupby(level="date").transform(_z).clip(-3, 3).fillna(0)


def load_stock_names(codes: list[str]) -> dict[str, str]:
    """从 sina 实时行情拉股票名称 (做报告用)."""
    try:
        df = fetch_all_ashare(max_pages=60)
        return df.set_index("code")["name"].to_dict()
    except Exception:
        return {c: c for c in codes}


def build_context(code: str, name: str,
                  latest_feat: pd.Series,
                  pred_score: float,
                  rank: int,
                  latest_kline: pd.DataFrame,
                  recent_lhb: pd.DataFrame) -> str:
    """为单只股票拼 Qwen 输入上下文."""
    close = latest_kline["close"].iloc[-1]
    ret5  = latest_kline["close"].iloc[-1] / latest_kline["close"].iloc[-6] - 1 if len(latest_kline) >= 6 else 0
    ret20 = latest_kline["close"].iloc[-1] / latest_kline["close"].iloc[-21] - 1 if len(latest_kline) >= 21 else 0

    # 取 z-score 最强 (绝对值最大) 的 8 个因子
    top_factors = latest_feat.abs().nlargest(8).index.tolist()
    factor_lines = []
    for f in top_factors:
        desc = FACTOR_DESC.get(f, f)
        v = latest_feat[f]
        direction = "强" if v > 0.5 else ("弱" if v < -0.5 else "中性")
        factor_lines.append(f"    - {desc}: {v:+.2f} ({direction})")

    # 龙虎榜最近 3 次
    lhb_lines = []
    if len(recent_lhb):
        for _, row in recent_lhb.head(3).iterrows():
            dt = row["TRADE_DATE"].strftime("%Y-%m-%d")
            nb = row.get("BILLBOARD_NET_AMT", 0) / 1e4  # → 万元
            expl = row.get("EXPLAIN", "")[:30]
            lhb_lines.append(f"    - {dt}: 净买入 {nb:+.0f} 万, {expl}")

    ctx = f"""股票: {code} {name}
排名: 模型打分第 {rank} 名 (score={pred_score:+.4f})
最新价: {close:.2f} 元
近 5 日: {ret5:+.2%}
近 20 日: {ret20:+.2%}

关键因子 (z-score, 截面标准化):
{chr(10).join(factor_lines)}"""
    if lhb_lines:
        ctx += "\n\n近期龙虎榜:\n" + "\n".join(lhb_lines)
    return ctx


SYSTEM_PROMPT = """你是一个资深 A股量化研究员. 输入是单只股票的模型打分与关键因子, 你要用 1 段话 (不超过 100 字) 说清楚:
1. 模型看好它的核心理由 (2-3 个关键因子组合)
2. 主要风险点 (1 个)
3. 建议仓位态度: 重仓 / 标配 / 观察 / 回避

语言:专业、简洁、不堆砌术语. 不要复述输入, 直接给结论. 不要用 markdown 格式."""


async def explain_one(backend, code: str, ctx: str) -> tuple[str, str]:
    from functools import partial
    loop = asyncio.get_event_loop()
    try:
        # _LLMBackend.chat 是同步的, 扔到线程池
        resp = await loop.run_in_executor(
            None, partial(backend.chat, f"{SYSTEM_PROMPT}\n\n---\n{ctx}\n---", 350)
        )
        return code, resp.strip()
    except Exception as e:
        return code, f"(LLM 调用失败: {e})"


async def main_async(pool: int, top_k: int, start: str, end: str):
    print(f"\n{'='*64}\n  V3 信号解释器 - {time.strftime('%Y-%m-%d %H:%M')}\n{'='*64}")

    # 1. 加载缓存
    kline_path = CACHE / f"kline_{start}_{end}_n{pool}.parquet"
    if not kline_path.exists():
        # 兜底到已有缓存
        candidates = sorted(CACHE.glob(f"kline_{start}_{end}_n*.parquet"),
                            key=lambda p: int(p.stem.split("_n")[-1]))
        if not candidates:
            print(f"❌ 无 kline 缓存. 先跑 run_real_research_v3.py"); return
        kline_path = candidates[-1]
        print(f"  使用兜底缓存 {kline_path.name}")
    daily = pd.read_parquet(kline_path)
    print(f"[1/5] kline {len(daily)} 行, {daily['code'].nunique()} 只")

    lhb_path = CACHE / f"lhb_{start}_{end}.parquet"
    lhb_df = pd.read_parquet(lhb_path) if lhb_path.exists() else pd.DataFrame()
    print(f"  龙虎榜 {len(lhb_df)} 条")

    # 2. 因子 + 对齐
    print("\n[2/5] 计算因子...")
    feat_tech = compute_pandas_alpha(daily)
    feat_rev = compute_advanced_alpha(daily)
    feat_combo = feat_tech.join(feat_rev, how="outer")
    if not lhb_df.empty:
        trading_dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
        feat_lhb = build_lhb_features(lhb_df, trading_dates)
        feat_combo = feat_combo.join(feat_lhb, how="left")
        for f in LHB_FACTOR_NAMES:
            if f in feat_combo.columns:
                feat_combo[f] = feat_combo[f].fillna(0)
    feat_z = cs_zscore(feat_combo)
    print(f"  特征 {feat_z.shape}")

    # 3. 只训最后窗口
    print("\n[3/5] 训练最近 1 年, 预测最新日期 top...")
    label = make_label(daily, horizon=30)
    aligned = feat_z.join(label.rename("label"), how="inner")
    feat_cols = [c for c in feat_z.columns if c in aligned.columns]
    X = aligned[feat_cols]
    y = aligned["label"]
    dates = X.index.get_level_values("date").unique().sort_values()
    last_date = dates[-1]
    train_start = dates[max(0, len(dates) - 252 - 30)]
    train_end = dates[-30]
    valid_tr = y.loc[train_start:train_end].notna() & X.loc[train_start:train_end].notna().all(axis=1)
    X_tr = X.loc[train_start:train_end][valid_tr]
    y_tr = y.loc[train_start:train_end][valid_tr]
    X_te = X.loc[last_date]  # 最新一天的截面
    pred_today, _ = _fit_and_predict(X_tr, y_tr, X_te)
    print(f"  预测日期: {last_date.date()}, {len(pred_today)} 只打分")

    # 4. Top K
    top = pred_today.nlargest(top_k)
    print(f"\n[4/5] Top {top_k} 股票已选出")

    # 名称
    names = load_stock_names(top.index.tolist())

    # 5. 为每只构造 context
    contexts = {}
    for rank, (code, score) in enumerate(top.items(), 1):
        name = names.get(code, code)
        try:
            # 该股的最新因子截面
            latest_feat = feat_z.xs((last_date, code))
        except KeyError:
            continue
        # 最近 21 日 kline
        k = daily[(daily["code"] == code)].sort_values("date").tail(22)
        # 龙虎榜 最近 60 日
        rec_lhb = pd.DataFrame()
        if not lhb_df.empty:
            cutoff = last_date - pd.Timedelta(days=60)
            rec_lhb = lhb_df[(lhb_df["code"] == code) &
                             (lhb_df["TRADE_DATE"] >= cutoff)].sort_values("TRADE_DATE", ascending=False)
        contexts[code] = (rank, score, name,
                          build_context(code, name, latest_feat, score, rank, k, rec_lhb))

    # 6. Qwen 并发解释
    print(f"\n[5/5] Qwen qwen-plus 并发生成解释 (并发=5)...")
    from llm_layer.agents import _LLMBackend
    backend = _LLMBackend(backend="qwen", model="qwen-plus")

    sem = asyncio.Semaphore(5)
    async def _gated(code, ctx):
        async with sem:
            return await explain_one(backend, code, ctx)
    tasks = [_gated(code, ctx) for code, (_, _, _, ctx) in contexts.items()]
    t0 = time.time()
    results = await asyncio.gather(*tasks)
    print(f"  完成 {len(results)} 条, 耗时 {time.time()-t0:.1f}s")

    explain_map = dict(results)

    # 7. 组装 markdown 报告
    lines = [
        f"# V3 今日推荐持仓 — {last_date.date()}",
        "",
        f"- 模型: LightGBM 多因子合成 (61 因子)",
        f"- 股票池: {daily['code'].nunique()} 只中小盘",
        f"- 持仓周期: 30 交易日",
        f"- 总持仓: top {len(top)} 只等权",
        "",
        "## 持仓清单",
        "",
        "| 排名 | 代码 | 名称 | 打分 | Qwen 分析 |",
        "|---|---|---|---|---|",
    ]
    for rank, (code, score) in enumerate(top.items(), 1):
        name = names.get(code, code)
        expl = explain_map.get(code, "").replace("|", "/").replace("\n", " ")
        lines.append(f"| {rank} | {code} | {name} | {score:+.4f} | {expl} |")

    lines += [
        "",
        "## 详细理由",
        "",
    ]
    for rank, (code, score) in enumerate(top.items(), 1):
        name = names.get(code, code)
        expl = explain_map.get(code, "")
        ctx = contexts.get(code, (None, None, None, ""))[3]
        lines += [
            f"### {rank}. {code} {name}",
            "",
            f"**Qwen 评价**: {expl}",
            "",
            "<details><summary>因子详情</summary>",
            "",
            "```",
            ctx,
            "```",
            "",
            "</details>",
            "",
        ]

    out = ROOT / "output" / f"signal_{last_date.strftime('%Y%m%d')}_top{top_k}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ 报告: {out}")
    # 打印 top 5 到终端
    print("\n--- Top 5 预览 ---")
    for rank, (code, score) in enumerate(top.head(5).items(), 1):
        name = names.get(code, code)
        print(f"{rank}. {code} {name}  score={score:+.4f}")
        print(f"   {explain_map.get(code, '')[:200]}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=500)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--start", default="20230101")
    ap.add_argument("--end", default="20260420")
    args = ap.parse_args()
    asyncio.run(main_async(args.pool, args.top, args.start, args.end))

"""opening_auction — 9:15-9:25 集合竞价决策.

两种模式:
    --mode live      盘中(默认): 用 akshare.stock_zh_a_spot_em 拿当日竞价价
    --mode replay    回放历史: 用 daily 缓存中的 open/close

与 watchlist_signal_v2 的 alpha_z 结合:
    alpha_z 高 + 大幅高开 + 高量比   → 🚀 追涨
    alpha_z 高 + 大幅低开 + 高量比   → ⛏️ 黄金坑
    alpha_z 高 + 平开/小幅         → 👀 观望 (等回踩)
    alpha_z 低 + 大幅高开           → ⚠️ 诱多警示 (勿接)
    alpha_z 低 + 任何走势           → 🚫 回避

建议 9:20 跑(给 akshare 预热),9:25 前把决策单推到飞书.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

env = Path(__file__).resolve().parent.parent / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
PAPER = ROOT / "output" / "paper_trade"


def load_watchlist() -> list[dict]:
    path = ROOT / "config" / "user_watchlist.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["watchlist"]


def load_latest_alpha_z() -> pd.DataFrame | None:
    """读 v2 信号 CSV (最近一天)."""
    sig_dir = PAPER / "watchlist_signals_v2"
    if not sig_dir.exists():
        return None
    csvs = sorted(sig_dir.glob("*.csv"))
    if not csvs:
        return None
    latest = csvs[-1]
    print(f"  alpha_z 来源: {latest.name}")
    df = pd.read_csv(latest, index_col=0)
    df.index = df.index.astype(str).str.zfill(6)
    return df


def fetch_spot(codes: list[str]) -> pd.DataFrame:
    """akshare 实时快照 (盘中有效, 盘前后返回前一交易日收盘)."""
    import akshare as ak
    import time
    last_err = None
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_spot_em()
            df = df.rename(columns={
                "代码": "code", "最新价": "price",
                "涨跌幅": "pct_chg", "今开": "open",
                "昨收": "prev_close", "成交量": "volume",
            })
            df["code"] = df["code"].astype(str).str.zfill(6)
            return df[df["code"].isin(codes)][
                ["code", "price", "pct_chg", "open", "prev_close", "volume"]
            ]
        except Exception as e:
            last_err = e
            print(f"  [重试 {attempt+1}/3] {e}")
            time.sleep(5)
    raise RuntimeError(f"akshare spot 失败: {last_err}")


def replay_daily(codes: list[str], date: pd.Timestamp) -> pd.DataFrame:
    """回放模式: 用 daily 缓存的某日 open/prev_close."""
    wl_cache = sorted(CACHE.glob("watchlist_kline_*.parquet"))
    if not wl_cache:
        raise FileNotFoundError("无 watchlist kline 缓存")
    df = pd.read_parquet(wl_cache[-1])
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)

    # 当日
    today = df[df["date"] == date]
    # 前一日
    all_dates = sorted(df["date"].unique())
    prev_date = next((d for d in reversed(all_dates) if d < date), None)
    if prev_date is None:
        raise ValueError(f"{date.date()} 之前无数据")
    prev = df[df["date"] == prev_date][["code", "close"]].rename(
        columns={"close": "prev_close"})

    merged = today[["code", "open", "close", "volume", "pct_chg"]].merge(
        prev, on="code", how="left")
    merged["price"] = merged["close"]  # 回放用收盘作为当日"最新价"
    return merged[merged["code"].isin(codes)]


def compute_volume_ratio(codes: list[str], today_vol: dict,
                          date: pd.Timestamp) -> dict:
    """量比: 今日开盘前 N 分钟成交量 / 近 5 日平均全日量 * 比例.

    在没有分钟级数据的前提下, 用"当日已成交量 / 近 5 日均量"近似.
    仅供参考, P2 会升级为真·开盘竞价量比.
    """
    wl_cache = sorted(CACHE.glob("watchlist_kline_*.parquet"))
    if not wl_cache:
        return {}
    df = pd.read_parquet(wl_cache[-1])
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df[df["date"] < date].sort_values("date")

    ratios = {}
    for c in codes:
        sub = df[df["code"] == c].tail(5)
        if len(sub) < 3:
            continue
        avg_vol = sub["volume"].mean()
        now_vol = today_vol.get(c, 0)
        if avg_vol > 0 and now_vol > 0:
            ratios[c] = now_vol / avg_vol
    return ratios


# ---------- 决策矩阵 ----------
def classify(alpha_z: float, gap_pct: float, vol_ratio: float) -> tuple[str, str]:
    """返回 (action_tag, rationale)."""
    hi_z = alpha_z >= 0.75
    lo_z = alpha_z <= -0.75
    big_up = gap_pct >= 3
    big_dn = gap_pct <= -3
    hi_vol = vol_ratio >= 1.5   # 放量

    if hi_z and big_up and hi_vol:
        return "🚀 追涨", f"高 alpha + 大幅高开 {gap_pct:+.1f}% + 放量 {vol_ratio:.1f}x"
    if hi_z and big_dn and hi_vol:
        return "⛏️ 黄金坑", f"高 alpha + 大幅低开 {gap_pct:+.1f}% + 放量 {vol_ratio:.1f}x, 低吸"
    if hi_z and abs(gap_pct) < 1:
        return "👀 观望", f"高 alpha 平开 {gap_pct:+.1f}%, 等回踩"
    if hi_z:
        return "🟢 适度买入", f"alpha {alpha_z:+.2f}, 开盘 {gap_pct:+.1f}%"
    if lo_z and big_up:
        return "⚠️ 诱多警示", f"低 alpha 大幅高开, 主力出货嫌疑"
    if lo_z:
        return "🚫 回避", f"alpha {alpha_z:+.2f}, 开盘 {gap_pct:+.1f}%"
    return "⚪️ 中性", f"alpha {alpha_z:+.2f}, 开盘 {gap_pct:+.1f}%"


def build_report(decisions: list[dict], date_str: str, mode: str) -> str:
    lines = [
        f"🔔 **集合竞价决策单 — {date_str} ({mode})**",
        "",
        f"Universe: {len(decisions)} 只",
        "",
    ]
    # 按 action 归类
    groups = {}
    for d in decisions:
        groups.setdefault(d["action"], []).append(d)

    order = ["🚀 追涨", "⛏️ 黄金坑", "🟢 适度买入", "👀 观望",
             "⚪️ 中性", "⚠️ 诱多警示", "🚫 回避"]
    for k in order:
        if k not in groups:
            continue
        lines.append(f"**{k} ({len(groups[k])} 只)**")
        for d in groups[k][:6]:
            lines.append(
                f"  • {d['name']} ¥{d['price']:.2f}  "
                f"开盘 {d['gap']:+.1f}%  量比 {d['vol_ratio']:.1f}x  "
                f"alpha_z={d['alpha_z']:+.2f}"
            )
        lines.append("")

    lines.append("⚠️ 量比基于开盘累计成交 / 近 5 日均量近似, 非 9:25 真·竞价量比.")
    return "\n".join(lines)


def send_lark(md: str) -> bool:
    user_id = os.environ.get("LARK_USER_OPEN_ID",
                              "ou_5be0f87dc7cec796b7ea97d0a9b5302f")
    cmd = ["lark-cli", "im", "+messages-send",
           "--as", "user", "--user-id", user_id, "--markdown", md]
    try:
        rc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if rc.returncode == 0:
            print("✓ 飞书已送达")
            return True
        print(f"❌ 飞书失败: {rc.stderr[:200]}")
    except Exception as e:
        print(f"❌ 飞书异常: {e}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="live", choices=["live", "replay"])
    ap.add_argument("--date", default=None, help="replay 模式的日期 YYYY-MM-DD")
    ap.add_argument("--dry-run-lark", action="store_true")
    args = ap.parse_args()

    print(f"\n{'='*64}\n  🔔 Opening Auction {datetime.now():%Y-%m-%d %H:%M} ({args.mode})\n{'='*64}")

    wl = load_watchlist()
    codes = [w["code"].zfill(6) for w in wl]
    name_map = {w["code"].zfill(6): w["name"] for w in wl}

    # 1. 读最新 alpha_z
    alpha_df = load_latest_alpha_z()
    if alpha_df is None:
        print("❌ 没有 v2 信号 CSV, 先跑 watchlist_signal_v2.py")
        return 1
    alpha_map = alpha_df["alpha_z"].to_dict()

    # 2. 取竞价数据
    if args.mode == "live":
        spot = fetch_spot(codes)
        spot_date = pd.Timestamp.today().normalize()
    else:
        if not args.date:
            print("❌ replay 模式必须指定 --date")
            return 1
        spot_date = pd.Timestamp(args.date)
        spot = replay_daily(codes, spot_date)
    if spot.empty:
        print("❌ 竞价数据空")
        return 1
    print(f"  竞价数据: {len(spot)} 只")

    # 3. 量比
    vol_dict = dict(zip(spot["code"], spot["volume"].fillna(0)))
    vol_ratios = compute_volume_ratio(codes, vol_dict, spot_date)

    # 4. 决策
    decisions = []
    for _, r in spot.iterrows():
        code = r["code"]
        if pd.isna(r.get("prev_close")) or r["prev_close"] == 0:
            continue
        gap = (r["open"] / r["prev_close"] - 1) * 100 if pd.notna(r.get("open")) else 0
        vr = vol_ratios.get(code, 1.0)
        az = alpha_map.get(code, 0.0)
        action, rationale = classify(az, gap, vr)
        decisions.append({
            "code": code, "name": name_map.get(code, code),
            "price": r.get("price", r.get("close", 0)),
            "gap": gap, "vol_ratio": vr, "alpha_z": az,
            "action": action, "rationale": rationale,
        })

    # 5. 展示 + 推送
    decisions.sort(key=lambda d: (-d["alpha_z"], -d["gap"]))
    md = build_report(decisions, str(spot_date.date()), args.mode)
    print(f"\n{'='*64}\n飞书消息:\n{'='*64}")
    print(md)

    # 落盘
    out_dir = PAPER / "opening_auction"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{spot_date.date()}_{args.mode}.csv"
    pd.DataFrame(decisions).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  CSV: {out}")

    if not args.dry_run_lark:
        send_lark(md)


if __name__ == "__main__":
    main()

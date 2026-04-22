"""intraday_monitor — 盘中每 5 分钟实时监控 + 形态判断 + LLM 决策 + 飞书.

设计原则:
    1. 日频因子盘中不变 → 不跑 v2, 只用昨晚已算的 alpha_z 当基准
    2. 实时信号来自: 当前价 + 今日 1 分钟 K 形态 + 距关键阈值距离
    3. LLM (qwen) 做"持/减半/全平"盘中决策, 不做多头/空头切换
    4. 低噪推送: 仅在 (a) 形态状态切换 或 (b) 触发硬止损/冲高线 或 (c) 整点 (10:30/11:00/13:30/14:30/14:55) 才推飞书
    5. 状态存 output/real_positions/intraday_state.json (上次形态), 避免刷屏

盘中形态识别 (基于今日 1 分钟 K):
    - 拉升接力:  近 30 分钟 +2% 且放量 (最近 10 根 K 平均量 > 前 20 根均量 1.5x)
    - 放量滞涨:  近 30 分钟横盘 + 放量 (上攻无力)
    - 急速下跌:  近 15 分钟 -2%
    - 缩量横盘:  近 30 分钟振幅 <1% + 缩量
    - 尾盘拉升/跳水: 14:30 后专项判断
    - 封板:      现价 ≥ 昨收 × 1.098 且 振幅 <0.3%

调用方式:
    python3 scripts/intraday_monitor.py               # 默认推飞书 (去重)
    python3 scripts/intraday_monitor.py --dry-run-lark
    python3 scripts/intraday_monitor.py --no-llm      # 关 LLM, 只规则引擎 (省时)
    python3 scripts/intraday_monitor.py --force       # 强制推, 不去重
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

env = Path(__file__).resolve().parent.parent / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

ROOT = Path(__file__).resolve().parent.parent
LARK_BIN = "/opt/homebrew/bin/lark-cli"
ACCOUNT = ROOT / "output" / "real_positions" / "account.json"
STATE = ROOT / "output" / "real_positions" / "intraday_state.json"
CACHE = ROOT / "cache"


# ---------- 数据 ----------
def _em_secid(code: str) -> str:
    code = str(code).zfill(6)
    return f"1.{code}" if code.startswith("6") else f"0.{code}"


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "*/*",
}


def _em_get(url: str, params: dict, retries: int = 3, timeout: int = 10) -> dict | None:
    """带重试的东财 GET - 先试 requests, 失败 fallback 到 curl 子进程."""
    import time as _time
    from urllib.parse import urlencode
    full_url = f"{url}?{urlencode(params)}"

    for attempt in range(retries):
        # 先试 requests
        try:
            session = requests.Session()
            session.headers.update(_HEADERS)
            r = session.get(url, params=params, timeout=timeout)
            session.close()
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass

        # fallback: curl
        try:
            out = subprocess.run(
                ["curl", "-s", "-A", _HEADERS["User-Agent"],
                 "-H", f"Referer: {_HEADERS['Referer']}",
                 "--max-time", str(timeout), full_url],
                capture_output=True, text=True, timeout=timeout + 2,
            )
            if out.returncode == 0 and out.stdout:
                return json.loads(out.stdout)
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ⚠️ curl 也失败 ({e})")
                return None
        _time.sleep(1.5 ** attempt)
    return None


def _sina_symbol(code: str) -> str:
    code = str(code).zfill(6)
    return f"sh{code}" if code.startswith("6") else f"sz{code}"


def fetch_realtime_sina(code: str) -> dict | None:
    """新浪实时行情 (盘中稳定, 无限频)."""
    sym = _sina_symbol(code)
    url = f"https://hq.sinajs.cn/list={sym}"
    try:
        out = subprocess.run(
            ["curl", "-s", "-A", _HEADERS["User-Agent"],
             "-H", "Referer: https://finance.sina.com.cn/",
             "--max-time", "10", url],
            capture_output=True, text=False, timeout=12,
        )
        if out.returncode != 0:
            return None
        raw = out.stdout.decode("gbk", errors="replace")
        # 解析: var hq_str_sh600522="名称,开,昨,现,高,低,买一,卖一,量,额,..."
        eq = raw.find("=")
        if eq < 0:
            return None
        payload = raw[eq + 1:].strip().strip(';').strip('"')
        parts = payload.split(",")
        if len(parts) < 32 or not parts[3]:
            return None
        current = float(parts[3])
        if current == 0:
            return None
        prev_close = float(parts[2])
        return {
            "price":      current,
            "open":       float(parts[1]),
            "prev_close": prev_close,
            "high":       float(parts[4]),
            "low":        float(parts[5]),
            "volume":     int(parts[8]) // 100,   # 股 → 手
            "amount":     float(parts[9]),
            "pct_chg":    (current / prev_close - 1) * 100,
            "amplitude":  (float(parts[4]) - float(parts[5])) / prev_close * 100,
            "name":       parts[0],
            "quote_time": f"{parts[30]} {parts[31]}",
            "source":     "sina",
        }
    except Exception as e:
        print(f"  ⚠️ sina 实时失败 ({e})")
        return None


def fetch_realtime_em(code: str) -> dict | None:
    """东财直连 (备用)."""
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": _em_secid(code),
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields": "f43,f44,f45,f46,f47,f48,f57,f58,f60,f170,f171",
    }
    try:
        data = _em_get(url, params)
        if not data:
            return None
        d = data.get("data", {})
        if not d or not d.get("f43"):
            return None
        return {
            "price":      d["f43"] / 100,
            "open":       d["f46"] / 100,
            "prev_close": d["f60"] / 100,
            "high":       d["f44"] / 100,
            "low":        d["f45"] / 100,
            "volume":     d["f47"],
            "amount":     d["f48"],
            "pct_chg":    d["f170"] / 100,
            "amplitude":  d["f171"] / 100,
            "name":       d.get("f58", ""),
            "source":     "em",
        }
    except Exception as e:
        print(f"  ⚠️ em 解析失败 ({e})")
        return None


def fetch_realtime(code: str) -> dict | None:
    """先 sina 后 em."""
    q = fetch_realtime_sina(code)
    if q:
        return q
    return fetch_realtime_em(code)


def fetch_today_minute(code: str, klt: int = 1) -> pd.DataFrame:
    """今日 1 分钟 K (klt=1 是 1 分钟)."""
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    today = datetime.now().strftime("%Y%m%d")
    params = {
        "secid": _em_secid(code),
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "klt": klt, "fqt": 1,
        "beg": today, "end": today, "lmt": 1000,
    }
    try:
        data = _em_get(url, params, retries=3, timeout=10)
        if not data:
            return pd.DataFrame()
        d = data.get("data", {})
        klines = d.get("klines", [])
        if not klines:
            return pd.DataFrame()
        rows = []
        for line in klines:
            parts = line.split(",")
            rows.append({
                "dt":    pd.Timestamp(parts[0]),
                "open":  float(parts[1]), "close": float(parts[2]),
                "high":  float(parts[3]), "low":   float(parts[4]),
                "volume": int(parts[5]),  "amount": float(parts[6]),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  ⚠️ 分钟 K 解析失败 ({e})")
        return pd.DataFrame()


# ---------- 形态识别 ----------
def detect_pattern_quote_only(quote: dict) -> dict:
    """无分钟 K 时, 基于 quote 做简化形态判断.

    CLV = (现价 - 日低) / (日高 - 日低), 反映日内收盘位置.
    - CLV > 0.8  收在高位, 强势
    - CLV < 0.3  收在低位, 弱势
    """
    price = quote["price"]
    open_px = quote["open"]
    high = quote["high"]
    low = quote["low"]
    prev_close = quote["prev_close"]
    pct_chg = quote["pct_chg"]

    # 封板
    if price >= round(prev_close * 1.098, 2) and high >= round(prev_close * 1.099, 2):
        return {"pattern": "🔒 封涨停", "strength": 1.0,
                "note": f"¥{price:.2f} ≈ 涨停 ¥{prev_close*1.1:.2f}"}

    hl_range = high - low
    clv = (price - low) / hl_range if hl_range > 0 else 0.5
    amp = hl_range / prev_close * 100   # 振幅 %
    vs_open = (price / open_px - 1) * 100

    # 强势组合
    if pct_chg >= 3 and clv >= 0.75:
        return {"pattern": "🚀 高开高走", "strength": 0.85,
                "note": f"涨 {pct_chg:+.1f}% 收在日内 {clv:.0%} 位置, 强势"}
    if pct_chg >= 2 and clv >= 0.6 and vs_open > 0:
        return {"pattern": "🟢 稳步走强", "strength": 0.7,
                "note": f"涨 {pct_chg:+.1f}% 开盘后续创新高"}

    # 冲高回落
    if (high / prev_close - 1) * 100 >= 3 and clv <= 0.4:
        return {"pattern": "📉 冲高回落", "strength": 0.85,
                "note": f"曾摸 ¥{high:.2f}(+{(high/prev_close-1)*100:.1f}%), 现回落到 {clv:.0%} 位"}

    # 探底反弹
    if (low / prev_close - 1) * 100 <= -2 and clv >= 0.6:
        return {"pattern": "⛏️ 探底反弹", "strength": 0.7,
                "note": f"曾探 ¥{low:.2f}(-{(1-low/prev_close)*100:.1f}%), 现回升到 {clv:.0%} 位"}

    # 下跌
    if pct_chg <= -2 and clv <= 0.4:
        return {"pattern": "📉 弱势下跌", "strength": 0.85,
                "note": f"跌 {pct_chg:+.1f}% 收在日内 {clv:.0%} 位"}

    # 窄幅
    if amp < 1.5:
        return {"pattern": "😴 窄幅震荡", "strength": 0.3,
                "note": f"振幅 {amp:.1f}%, 多空胶着"}

    # 震荡
    if 0.3 <= clv <= 0.6:
        return {"pattern": "⚪️ 日内震荡", "strength": 0.4,
                "note": f"{pct_chg:+.1f}% 振幅 {amp:.1f}% CLV {clv:.0%}"}

    # 默认分档
    if pct_chg > 0:
        return {"pattern": "🟢 小幅走强", "strength": 0.5,
                "note": f"{pct_chg:+.1f}% CLV {clv:.0%}"}
    return {"pattern": "🟡 小幅走弱", "strength": 0.5,
            "note": f"{pct_chg:+.1f}% CLV {clv:.0%}"}


def detect_pattern(minute_df: pd.DataFrame, quote: dict, cost: float) -> dict:
    """输出 {pattern, strength(0-1), note}."""
    if minute_df.empty or len(minute_df) < 5:
        return detect_pattern_quote_only(quote)

    now = datetime.now().time()
    recent_30 = minute_df.tail(30)
    recent_15 = minute_df.tail(15)
    recent_10 = minute_df.tail(10)

    price = quote["price"]
    prev_close = quote["prev_close"]
    open_px = quote["open"]

    # 量能对比
    vol_recent = recent_10["volume"].mean() if len(recent_10) >= 5 else 0
    vol_base = (minute_df.iloc[:-10]["volume"].mean()
                if len(minute_df) > 20 else vol_recent)
    vol_ratio = vol_recent / max(vol_base, 1)

    # 30 分钟涨跌 (分钟 K 闭盘价)
    if len(recent_30) >= 2:
        ret_30m = (recent_30["close"].iloc[-1] / recent_30["close"].iloc[0] - 1)
    else:
        ret_30m = 0
    if len(recent_15) >= 2:
        ret_15m = (recent_15["close"].iloc[-1] / recent_15["close"].iloc[0] - 1)
    else:
        ret_15m = 0

    amp_30 = (recent_30["high"].max() - recent_30["low"].min()) / recent_30["close"].iloc[0]

    # 封板判断
    limit_price = round(prev_close * 1.10, 2)
    is_limit = price >= round(prev_close * 1.098, 2) and quote["high"] >= limit_price - 0.01

    # 形态优先级判断
    if is_limit:
        return {"pattern": "🔒 封涨停", "strength": 1.0,
                "note": f"现价 ¥{price:.2f} ≈ 涨停 ¥{limit_price:.2f}"}

    if ret_15m <= -0.02:
        return {"pattern": "📉 急速下跌", "strength": 0.9,
                "note": f"15 分钟 {ret_15m:.1%}, 量比 {vol_ratio:.1f}x"}

    if ret_30m >= 0.02 and vol_ratio >= 1.5:
        return {"pattern": "🚀 拉升接力", "strength": 0.85,
                "note": f"30 分钟 {ret_30m:+.1%}, 放量 {vol_ratio:.1f}x"}

    if abs(ret_30m) < 0.005 and vol_ratio >= 1.5:
        return {"pattern": "⚠️ 放量滞涨", "strength": 0.7,
                "note": f"横盘 + 放量 {vol_ratio:.1f}x (上攻失力)"}

    if amp_30 < 0.01 and vol_ratio < 0.7:
        return {"pattern": "😴 缩量横盘", "strength": 0.3,
                "note": f"30 分钟振幅 {amp_30:.2%}, 缩量"}

    # 尾盘特殊判断
    if now >= time(14, 30):
        if ret_30m >= 0.01:
            return {"pattern": "🌆 尾盘拉升", "strength": 0.8,
                    "note": f"14:30 后 {ret_30m:+.1%}, 主力护盘"}
        if ret_30m <= -0.01:
            return {"pattern": "🌆 尾盘跳水", "strength": 0.85,
                    "note": f"14:30 后 {ret_30m:+.1%}, 恐慌出货"}

    # 默认
    if price > cost:
        return {"pattern": "🟢 小幅走强", "strength": 0.5,
                "note": f"{ret_30m:+.1%} 量比 {vol_ratio:.1f}x"}
    return {"pattern": "🟡 小幅走弱", "strength": 0.5,
            "note": f"{ret_30m:+.1%} 量比 {vol_ratio:.1f}x"}


# ---------- 决策引擎 (规则 + LLM) ----------
def rule_decision(pos: dict, quote: dict, pattern: dict) -> dict:
    """基于规则产出 action: hold / reduce_half / close_all / break_stop."""
    cost = pos["cost"]
    price = quote["price"]
    pnl_pct = (price - cost) / cost

    stop_pct = pos.get("stop_loss_pct", 0.05)
    target_pct = pos.get("take_profit_pct", 0.05)
    stop_price = cost * (1 - stop_pct)
    target_price = cost * (1 + target_pct)

    # 优先级 1: 硬止损
    if price <= stop_price:
        return {"action": "break_stop",
                "reason": f"🛑 破硬底 ¥{stop_price:.2f} (现 ¥{price:.2f}), T+1 开盘必卖"}

    # 优先级 2: 冲高 + 放量滞涨 = 减半
    if pnl_pct >= target_pct and pattern["pattern"] in ("⚠️ 放量滞涨", "📉 急速下跌"):
        return {"action": "reduce_half",
                "reason": f"到冲高线且 {pattern['pattern']}, 减半锁 {pnl_pct:+.1%}"}

    # 优先级 3: 尾盘跳水 + 已盈利
    now = datetime.now().time()
    if now >= time(14, 30) and pattern["pattern"] == "🌆 尾盘跳水" and pnl_pct > 0:
        return {"action": "reduce_half",
                "reason": "尾盘跳水, 减半避险 (T+1 明早再看)"}

    # 优先级 4: 急速下跌且跌破成本
    if pattern["pattern"] == "📉 急速下跌" and price < cost:
        return {"action": "watch_close",
                "reason": "急跌跌破成本, 密切观察是否破硬底"}

    # 优先级 5: 封板
    if pattern["pattern"] == "🔒 封涨停":
        return {"action": "hold",
                "reason": "封涨停, 持仓等次日高开"}

    # 默认
    if pnl_pct >= target_pct:
        return {"action": "consider_reduce",
                "reason": f"已盈利 {pnl_pct:+.1%} 达冲高线, 考虑减半锁利"}
    return {"action": "hold", "reason": f"{pattern['pattern']}, 持仓观察"}


def llm_decision(pos: dict, quote: dict, pattern: dict, minute_df: pd.DataFrame) -> dict | None:
    """Qwen 盘中决策 (只在形态切换或关键时点调用)."""
    try:
        from llm_layer.agents import _LLMBackend
        from llm_layer import xml_parser as xp
    except Exception as e:
        return None

    cost = pos["cost"]
    price = quote["price"]
    pnl_pct = (price - cost) / cost

    # 今日 K 线描述 (简)
    today_high = minute_df["high"].max() if len(minute_df) else quote["high"]
    today_low = minute_df["low"].min() if len(minute_df) else quote["low"]
    total_amount = minute_df["amount"].sum() / 1e8 if len(minute_df) else 0

    prompt = f"""你是 A 股盘中短线交易员, 持仓 T+1 中, 今天只能看盘不能卖.

持仓股: {pos['name']} ({pos.get('code', '')})
买入成本: ¥{cost:.2f}
当前现价: ¥{price:.2f} ({pnl_pct:+.2%})
今开: ¥{quote['open']:.2f}  昨收: ¥{quote['prev_close']:.2f}
今日高: ¥{today_high:.2f}  今日低: ¥{today_low:.2f}
涨跌幅: {quote['pct_chg']:+.2f}%  今累计成交: ¥{total_amount:.1f} 亿
盘中形态: {pattern['pattern']} — {pattern['note']}

硬止损: ¥{cost * 0.95:.2f}  冲高线: ¥{cost * 1.05:.2f}  退出日: {pos.get('target_exit', 'N/A')}
当前时刻: {datetime.now():%H:%M}

任务: 对**明早(T+1)开盘**给出操作建议, 今天不能动手.

<ACTION>hold_all / reduce_half / close_all</ACTION>
<NEXT_DAY_PLAN>一句话, 明早开盘怎么做</NEXT_DAY_PLAN>
<KEY_SIGNAL>今日盘面最重要的一个信号</KEY_SIGNAL>
<RISK>1 条风险</RISK>
"""
    try:
        backend = _LLMBackend("qwen", "qwen-plus")
        raw = backend.chat(prompt, max_tokens=500)
        return {
            "action": (xp.extract_tag(raw, "ACTION") or "hold_all").strip().lower(),
            "next_day": (xp.extract_tag(raw, "NEXT_DAY_PLAN") or "").strip(),
            "key_signal": (xp.extract_tag(raw, "KEY_SIGNAL") or "").strip(),
            "risk": (xp.extract_tag(raw, "RISK") or "").strip(),
        }
    except Exception as e:
        print(f"  ⚠️ LLM 失败 ({e})")
        return None


# ---------- 状态去重 ----------
def load_state() -> dict:
    if not STATE.exists():
        return {}
    return json.loads(STATE.read_text(encoding="utf-8"))


def save_state(s: dict):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2, ensure_ascii=False, default=str),
                     encoding="utf-8")


def should_push(code: str, pattern: str, action: str, price: float,
                 state: dict, force: bool = False) -> tuple[bool, str]:
    """返回 (should_push, reason). 低噪策略: 只在 (a) 形态变化 (b) action 变化
    (c) 触发止损/减半 (d) 整点 推."""
    if force:
        return True, "force"

    now = datetime.now()
    last = state.get(code, {})
    last_pattern = last.get("pattern", "")
    last_action = last.get("action", "")
    last_push_time = last.get("last_push", "")

    # 触发类必推
    if action in ("break_stop", "reduce_half"):
        return True, "trigger"

    # 形态切换必推
    if pattern != last_pattern:
        return True, f"pattern_change: {last_pattern} → {pattern}"

    # action 切换必推
    if action != last_action:
        return True, f"action_change: {last_action} → {action}"

    # 整点报告 (10:30 / 11:00 / 13:30 / 14:30 / 14:55)
    key_times = [(10, 30), (11, 0), (13, 30), (14, 30), (14, 55)]
    for h, m in key_times:
        if now.hour == h and now.minute in (m, m + 1, m + 2, m + 3, m + 4):
            # 30 分钟内只推一次
            if last_push_time:
                last_t = datetime.fromisoformat(last_push_time)
                if (now - last_t).total_seconds() < 30 * 60:
                    break
            return True, f"keytime_{h:02d}{m:02d}"

    return False, "no_change"


# ---------- 推送 ----------
def build_message(pos: dict, quote: dict, pattern: dict, rule: dict,
                   llm: dict | None) -> str:
    cost = pos["cost"]
    price = quote["price"]
    pnl_pct = (price - cost) / cost
    pnl_amt = (price - cost) * pos["shares"]
    stop_price = cost * (1 - pos.get("stop_loss_pct", 0.05))
    target_price = cost * (1 + pos.get("take_profit_pct", 0.05))

    lines = [
        f"📡 **盘中监控 {datetime.now():%H:%M}** — {pos['name']} {pos.get('code', '')}",
        "",
        f"现价 ¥{price:.2f} ({quote['pct_chg']:+.2f}%)  "
        f"盈亏 {pnl_pct:+.2%} (¥{pnl_amt:+,.0f})",
        f"今 开/高/低: ¥{quote['open']:.2f} / ¥{quote['high']:.2f} / ¥{quote['low']:.2f}",
        f"硬底 ¥{stop_price:.2f}  冲高 ¥{target_price:.2f}",
        "",
        f"**形态**: {pattern['pattern']}",
        f"  {pattern['note']}",
        "",
        f"**规则引擎**: {rule['action']}",
        f"  {rule['reason']}",
    ]
    if llm:
        lines += [
            "",
            f"**LLM 盘中决策**: {llm['action']}",
            f"  🔑 关键信号: {llm['key_signal']}",
            f"  📅 明早方案: {llm['next_day']}",
            f"  ⚠️ 风险: {llm['risk']}",
        ]
    return "\n".join(lines)


def send_lark(md: str) -> bool:
    user_id = os.environ.get("LARK_USER_OPEN_ID",
                              "ou_5be0f87dc7cec796b7ea97d0a9b5302f")
    cmd = [str(LARK_BIN), "im", "+messages-send",
           "--as", "user", "--user-id", user_id, "--markdown", md]
    try:
        env = {**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}"}
        rc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
        return rc.returncode == 0
    except Exception:
        return False


# ---------- 主 ----------
def is_trading_hours() -> bool:
    now = datetime.now().time()
    return (time(9, 30) <= now <= time(11, 30)) or (time(13, 0) <= now <= time(15, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-lark", action="store_true")
    ap.add_argument("--force", action="store_true", help="强制推, 跳过去重")
    ap.add_argument("--no-llm", action="store_true", help="关 LLM, 只规则引擎")
    args = ap.parse_args()

    if not is_trading_hours() and not args.force:
        print(f"  非交易时间 {datetime.now():%H:%M}, 退出 (--force 强跑)")
        return 0

    if not ACCOUNT.exists():
        print("  无持仓 account.json, 退出")
        return 0

    acc = json.loads(ACCOUNT.read_text(encoding="utf-8"))
    positions = acc.get("positions", {})
    if not positions:
        print("  无持仓, 退出")
        return 0

    state = load_state()
    new_state = {}

    for code, pos in positions.items():
        pos["code"] = code
        print(f"\n[{code}] {pos['name']} {datetime.now():%H:%M}")

        quote = fetch_realtime(code)
        if not quote:
            print("  ❌ 拉不到行情")
            continue

        minute_df = fetch_today_minute(code, klt=1)
        pattern = detect_pattern(minute_df, quote, pos["cost"])
        rule = rule_decision(pos, quote, pattern)

        print(f"  价 ¥{quote['price']:.2f} ({quote['pct_chg']:+.2f}%)")
        print(f"  形态 {pattern['pattern']} — {pattern['note']}")
        print(f"  规则 {rule['action']} — {rule['reason']}")

        push, push_reason = should_push(
            code, pattern["pattern"], rule["action"],
            quote["price"], state, force=args.force
        )

        llm = None
        if push and not args.no_llm:
            llm = llm_decision(pos, quote, pattern, minute_df)
            if llm:
                print(f"  LLM {llm['action']}: {llm['next_day']}")

        if push:
            print(f"  📤 推送 ({push_reason})")
            md = build_message(pos, quote, pattern, rule, llm)
            if not args.dry_run_lark:
                ok = send_lark(md)
                print(f"  {'✓' if ok else '❌'} 飞书")
            else:
                print("\n" + md)
            new_state[code] = {
                "pattern": pattern["pattern"],
                "action": rule["action"],
                "price": quote["price"],
                "last_push": datetime.now().isoformat(),
            }
        else:
            print(f"  ⏸ 跳过 ({push_reason})")
            new_state[code] = state.get(code, {})
            # 更新非推送字段
            new_state[code]["pattern"] = pattern["pattern"]
            new_state[code]["action"] = rule["action"]
            new_state[code]["price"] = quote["price"]

    save_state(new_state)


if __name__ == "__main__":
    main()

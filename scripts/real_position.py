"""real_position — 真实持仓增删查改 CLI.

子命令:
    buy   新增持仓
    sell  平仓 (自动算实现 PnL, 移入 history)
    list  列出当前持仓
    edit  修改止损/止盈参数

用法示例:

    # 加仓 (必填: code name shares cost)
    python3 scripts/real_position.py buy 600522 中天科技 1200 32.44
    # 自定义止损/止盈/持有期
    python3 scripts/real_position.py buy 600522 中天科技 1200 32.44 \\
        --stop-loss 0.05 --take-profit 0.08 --hold-days 3

    # 平仓 (自动用 akshare 实时价, 也可手动指定)
    python3 scripts/real_position.py sell 600522
    python3 scripts/real_position.py sell 600522 --price 33.80

    # 查看
    python3 scripts/real_position.py list

    # 修改止损线
    python3 scripts/real_position.py edit 600522 --stop-loss 0.03
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

env = Path(__file__).resolve().parent.parent / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

ROOT = Path(__file__).resolve().parent.parent
ACCOUNT = ROOT / "output" / "real_positions" / "account.json"


def load() -> dict:
    if not ACCOUNT.exists():
        ACCOUNT.parent.mkdir(parents=True, exist_ok=True)
        return {"owner": os.environ.get("USER", "user"),
                "note": "真实资金持仓追踪",
                "positions": {}, "history": []}
    return json.loads(ACCOUNT.read_text(encoding="utf-8"))


def save(acc: dict):
    ACCOUNT.write_text(
        json.dumps(acc, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def next_trading_day(d: datetime, days: int) -> datetime:
    """跳过周末,返回 days 个交易日后的日期."""
    d = d
    added = 0
    while added < days:
        d = d + timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d


def fetch_spot_price(code: str) -> float | None:
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        df["代码"] = df["代码"].astype(str).str.zfill(6)
        row = df[df["代码"] == code]
        if row.empty:
            return None
        return float(row.iloc[0]["最新价"])
    except Exception as e:
        print(f"  ⚠️ 实时价获取失败: {e}")
        return None


def send_lark(md: str) -> bool:
    user_id = os.environ.get("LARK_USER_OPEN_ID",
                              "ou_5be0f87dc7cec796b7ea97d0a9b5302f")
    cmd = ["lark-cli", "im", "+messages-send",
           "--as", "user", "--user-id", user_id, "--markdown", md]
    try:
        rc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return rc.returncode == 0
    except Exception:
        return False


# ---------- 子命令 ----------
def cmd_buy(args):
    acc = load()
    code = args.code.zfill(6)
    if code in acc["positions"]:
        print(f"⚠️ {code} 已有持仓, 用 edit 修改, 或先 sell 再 buy")
        return 1

    today = datetime.now()
    target_exit = next_trading_day(today, args.hold_days)
    capital = args.shares * args.cost

    acc["positions"][code] = {
        "name":             args.name,
        "shares":           args.shares,
        "cost":             args.cost,
        "buy_date":         today.strftime("%Y-%m-%d"),
        "capital":          round(capital, 2),
        "stop_loss_pct":    args.stop_loss,
        "use_ma5":          not args.no_ma5,
        "take_profit_pct":  args.take_profit,
        "holding_days":     args.hold_days,
        "target_exit":      target_exit.strftime("%Y-%m-%d"),
        "note":             args.note or "",
    }
    save(acc)

    stop_price = args.cost * (1 - args.stop_loss)
    target_price = args.cost * (1 + args.take_profit)
    md = (
        f"🟢 **新建持仓 — {args.name} ({code})**\n\n"
        f"数量: {args.shares} 股  成本: ¥{args.cost:.2f}  投入: ¥{capital:,.0f}\n"
        f"硬止损: ¥{stop_price:.2f}  冲高目标: ¥{target_price:.2f}\n"
        f"持有天数: {args.hold_days} 个交易日  退出日: {target_exit.date()}\n"
        + (f"备注: {args.note}" if args.note else "")
    )
    print(md)
    if not args.dry_run_lark:
        send_lark(md)
    print(f"\n✓ 已写入 {ACCOUNT}")
    return 0


def cmd_sell(args):
    acc = load()
    code = args.code.zfill(6)
    if code not in acc["positions"]:
        print(f"❌ 无 {code} 持仓")
        return 1

    pos = acc["positions"][code]

    # 确定卖出价
    if args.price:
        sell_price = args.price
        src = "manual"
    else:
        sell_price = fetch_spot_price(code)
        src = "akshare_spot"
        if sell_price is None:
            print("❌ 实时价拉不到, 用 --price 手动指定")
            return 1

    pnl_amt = (sell_price - pos["cost"]) * pos["shares"]
    pnl_pct = (sell_price - pos["cost"]) / pos["cost"]
    # 简化: 单边佣金 + 印花费 共 ~10bps
    cost_fee = pos["shares"] * sell_price * 10 / 10000
    net_pnl = pnl_amt - cost_fee

    record = {
        **pos, "code": code,
        "sell_date":  datetime.now().strftime("%Y-%m-%d"),
        "sell_price": sell_price,
        "pnl_amt":    round(pnl_amt, 2),
        "pnl_pct":    round(pnl_pct, 4),
        "cost_fee":   round(cost_fee, 2),
        "net_pnl":    round(net_pnl, 2),
        "price_src":  src,
        "reason":     args.reason or "manual",
    }
    acc["history"].append(record)
    del acc["positions"][code]
    save(acc)

    emoji = "🟢" if net_pnl > 0 else "🔴"
    md = (
        f"{emoji} **平仓 — {pos['name']} ({code})**\n\n"
        f"买入: ¥{pos['cost']:.2f}  卖出: ¥{sell_price:.2f} ({src})\n"
        f"数量: {pos['shares']} 股  持有 {pos['buy_date']} → {record['sell_date']}\n"
        f"毛利: ¥{pnl_amt:+,.2f} ({pnl_pct:+.2%})\n"
        f"费用: ¥{cost_fee:,.2f}\n"
        f"**净利: ¥{net_pnl:+,.2f}**\n"
        + (f"理由: {args.reason}" if args.reason else "")
    )
    print(md)
    if not args.dry_run_lark:
        send_lark(md)
    return 0


def cmd_list(args):
    acc = load()
    if not acc["positions"]:
        print("  无持仓")
        if acc.get("history"):
            print(f"  历史交易: {len(acc['history'])} 笔")
            tot_pnl = sum(h.get("net_pnl", 0) for h in acc["history"])
            print(f"  累计净利: ¥{tot_pnl:+,.2f}")
        return 0

    print(f"\n{'='*64}\n当前持仓 ({len(acc['positions'])} 只)\n{'='*64}")
    total_cost = 0
    for code, p in acc["positions"].items():
        total_cost += p["cost"] * p["shares"]
        stop = p["cost"] * (1 - p.get("stop_loss_pct", 0.05))
        target = p["cost"] * (1 + p.get("take_profit_pct", 0.05))
        print(f"{p['name']} {code}")
        print(f"  {p['shares']} 股 @ ¥{p['cost']:.2f}  成本 ¥{p['cost']*p['shares']:,.0f}")
        print(f"  硬底 ¥{stop:.2f}   冲高 ¥{target:.2f}   退出 {p.get('target_exit', 'N/A')}")
        if p.get("note"):
            print(f"  💬 {p['note']}")
        print()
    print(f"总成本: ¥{total_cost:,.0f}")

    if acc.get("history"):
        tot_pnl = sum(h.get("net_pnl", 0) for h in acc["history"])
        print(f"\n历史交易: {len(acc['history'])} 笔  累计净利: ¥{tot_pnl:+,.2f}")
    return 0


def cmd_edit(args):
    acc = load()
    code = args.code.zfill(6)
    if code not in acc["positions"]:
        print(f"❌ 无 {code} 持仓")
        return 1

    p = acc["positions"][code]
    changed = []
    if args.stop_loss is not None:
        p["stop_loss_pct"] = args.stop_loss
        changed.append(f"stop_loss_pct={args.stop_loss}")
    if args.take_profit is not None:
        p["take_profit_pct"] = args.take_profit
        changed.append(f"take_profit_pct={args.take_profit}")
    if args.hold_days is not None:
        p["holding_days"] = args.hold_days
        p["target_exit"] = next_trading_day(
            datetime.strptime(p["buy_date"], "%Y-%m-%d"), args.hold_days
        ).strftime("%Y-%m-%d")
        changed.append(f"hold_days={args.hold_days} target={p['target_exit']}")
    if args.note is not None:
        p["note"] = args.note
        changed.append(f"note={args.note[:30]}")

    if not changed:
        print("⚠️ 没有修改参数")
        return 0
    save(acc)
    print(f"✓ {p['name']} ({code}) 更新: {', '.join(changed)}")
    return 0


# ---------- 入口 ----------
def main():
    ap = argparse.ArgumentParser(description="真实资金持仓 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # buy
    p_buy = sub.add_parser("buy", help="新建持仓")
    p_buy.add_argument("code")
    p_buy.add_argument("name")
    p_buy.add_argument("shares", type=int)
    p_buy.add_argument("cost", type=float)
    p_buy.add_argument("--stop-loss", type=float, default=0.05, help="止损比例 (默认 5%)")
    p_buy.add_argument("--take-profit", type=float, default=0.05, help="止盈比例 (默认 5%)")
    p_buy.add_argument("--hold-days", type=int, default=3, help="最大持有交易日 (默认 3)")
    p_buy.add_argument("--no-ma5", action="store_true", help="不启用 MA5 止损")
    p_buy.add_argument("--note", default="")
    p_buy.add_argument("--dry-run-lark", action="store_true")
    p_buy.set_defaults(func=cmd_buy)

    # sell
    p_sell = sub.add_parser("sell", help="平仓")
    p_sell.add_argument("code")
    p_sell.add_argument("--price", type=float, help="手动指定卖出价, 否则自动拉实时价")
    p_sell.add_argument("--reason", default="", help="平仓理由(hard_stop/take_profit/expired/manual)")
    p_sell.add_argument("--dry-run-lark", action="store_true")
    p_sell.set_defaults(func=cmd_sell)

    # list
    p_list = sub.add_parser("list", help="列出持仓")
    p_list.set_defaults(func=cmd_list)

    # edit
    p_edit = sub.add_parser("edit", help="修改止损/止盈/持有期")
    p_edit.add_argument("code")
    p_edit.add_argument("--stop-loss", type=float)
    p_edit.add_argument("--take-profit", type=float)
    p_edit.add_argument("--hold-days", type=int)
    p_edit.add_argument("--note")
    p_edit.set_defaults(func=cmd_edit)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)

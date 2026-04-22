"""每日 cron 编排脚本: 数据更新 + paper trade + git push + 飞书通知.

流程:
    1. 跑 daily_data_updater (失败不阻塞, 警告后继续)
    2. 跑 paper_trade_runner
    3. 跑 watchlist_signal (专业量化信号, 内部独立推飞书)
    4. git add/commit/push paper_trade 输出
    5. 读 account.json 生成账户汇报并推送
    6. 分析师简报 (市场全景 + 明日 Top N + 昨日命中 + 风险提示)
    7. 批量扫描 (246 只候选池 全因子打分 + LLM Top20+Bottom10 推荐/回避)

用法:
    python3 scripts/cron_daily.py                     # 正常模式
    python3 scripts/cron_daily.py --skip-data         # 跳过数据更新
    python3 scripts/cron_daily.py --skip-push         # 跳过 git push
    python3 scripts/cron_daily.py --dry-run-lark      # 不发飞书
    python3 scripts/cron_daily.py --test-notify       # 只测飞书

环境:
    LARK_USER_OPEN_ID 或默认 ou_5be0f87dc7cec796b7ea97d0a9b5302f
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# cron 环境 PATH 很简陋, 硬加 homebrew/本地 bin
_extra_path = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
os.environ["PATH"] = ":".join(_extra_path + [os.environ.get("PATH", "")])

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
PAPER = ROOT / "output" / "paper_trade"

DEFAULT_LARK_USER = "ou_5be0f87dc7cec796b7ea97d0a9b5302f"


def log(msg: str, logfile=None):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile:
        logfile.write(line + "\n")
        logfile.flush()


def run_cmd(cmd: list[str], logfile=None, timeout: int = 1800) -> tuple[int, str]:
    """执行命令, 返回 (exit_code, combined_output)."""
    log(f"▶️  {' '.join(cmd)}", logfile)
    try:
        p = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True,
            timeout=timeout, check=False,
        )
        if logfile:
            logfile.write(p.stdout)
            logfile.write(p.stderr)
            logfile.flush()
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        log(f"❌ 超时 {timeout}s", logfile)
        return 124, "TIMEOUT"
    except Exception as e:
        log(f"❌ 异常 {e}", logfile)
        return 1, str(e)


def build_report(account: dict, mode: str = "daily") -> str:
    """从 account.json 构造简洁 markdown 汇报."""
    today = datetime.now().strftime("%Y-%m-%d")
    eq = account.get("equity_history", [])
    nav = eq[-1]["nav"] if eq else account.get("initial_cash", 0)
    nav0 = account.get("initial_cash", 1e6)
    total_ret = (nav - nav0) / nav0

    # 最近一日交易
    trades = account.get("trade_log", [])
    today_trades = [t for t in trades if str(t.get("date", "")).startswith(today)]

    lines = [
        f"📊 **A股 Paper Trade {today}**",
        "",
        f"• NAV: **¥{nav/1e4:.1f} 万** (初始 ¥{nav0/1e4:.0f} 万)",
        f"• 累计: **{total_ret:+.2%}**  持仓 {len(account.get('positions', {}))} 只",
    ]

    if len(eq) >= 2:
        today_pnl = eq[-1]["nav"] - eq[-2]["nav"]
        lines.append(f"• 当日 P&L: **¥{today_pnl:+.0f}**")

    if today_trades:
        buys = [t for t in today_trades if t.get("side") == "buy"]
        sells = [t for t in today_trades if t.get("side") == "sell"]
        lines.append(f"• 交易: 买 {len(buys)} / 卖 {len(sells)}")
        if sells:
            pnl_sum = sum(t.get("pnl", 0) for t in sells)
            lines.append(f"• 到期 PnL: ¥{pnl_sum:+.0f}")

    # Top 5 当前持仓
    positions = account.get("positions", {})
    if positions:
        sorted_pos = sorted(positions.items(),
                             key=lambda x: x[1].get("rank", 999))[:5]
        lines.append("")
        lines.append("**Top 5 持仓**:")
        for code, p in sorted_pos:
            lines.append(f"  #{p['rank']:<2} {code} @{p['cost']:.2f} × {p['shares']}")

    lines.append("")
    lines.append(f"📁 [GitHub](https://github.com/wangpage/quant-ashare)")
    return "\n".join(lines)


def build_error_report(stage: str, output: str) -> str:
    """失败时的通知."""
    today = datetime.now().strftime("%Y-%m-%d")
    snippet = "\n".join(output.strip().split("\n")[-10:])[:500]
    return (f"⚠️ **Paper Trade cron {today} 失败**\n\n"
            f"阶段: **{stage}**\n\n"
            f"```\n{snippet}\n```")


def build_leak_report(passed: bool, output: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    if passed:
        return f"🔒 **Leak Detector {today}**\n\n✅ 所有 4 测通过, 回测无 leak"
    snippet = "\n".join(output.strip().split("\n")[-10:])[:500]
    return (f"🚨 **Leak Detector {today} FAILED**\n\n"
            f"```\n{snippet}\n```\n\n请立即检查!")


def send_lark(markdown: str, user_id: str, dry_run: bool = False) -> bool:
    if dry_run:
        log(f"[dry-run] 飞书消息:\n{markdown}")
        return True
    cmd = ["lark-cli", "im", "+messages-send",
           "--as", "user", "--user-id", user_id,
           "--markdown", markdown]
    rc, out = run_cmd(cmd, timeout=30)
    if rc == 0:
        log(f"✓ 飞书已送达")
        return True
    log(f"❌ 飞书发送失败 (rc={rc}): {out[:200]}")
    return False


def git_commit_push(logfile=None, dry_run: bool = False) -> bool:
    """把 paper_trade 新输出提交并推送."""
    if dry_run:
        log("[dry-run] 跳过 git push", logfile)
        return True

    # 只 add paper_trade 目录 (避免误 commit 其他改动)
    rc, _ = run_cmd(
        ["git", "add", "output/paper_trade/"],
        logfile, timeout=30,
    )
    if rc != 0:
        return False

    # 检查是否有 staged
    rc, out = run_cmd(["git", "diff", "--cached", "--quiet"],
                      logfile, timeout=10)
    if rc == 0:
        log("  无新 paper_trade 变更, 跳过 commit", logfile)
        return True

    today = datetime.now().strftime("%Y-%m-%d")
    msg = f"chore(paper-trade): daily run {today}"
    rc, _ = run_cmd(["git", "commit", "-m", msg], logfile, timeout=30)
    if rc != 0:
        return False
    rc, _ = run_cmd(["git", "push", "origin", "main"], logfile, timeout=60)
    return rc == 0


def main_daily(args):
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = LOG_DIR / f"cron_{today}.log"
    with open(log_path, "a", encoding="utf-8") as logfile:
        log(f"{'='*64}", logfile)
        log(f"🌅 Cron daily @ {today}", logfile)
        log(f"{'='*64}", logfile)

        user_id = os.environ.get("LARK_USER_OPEN_ID", DEFAULT_LARK_USER)

        # 1. 数据更新
        if not args.skip_data:
            log("\n── 1/7 数据增量更新 ──", logfile)
            rc, out = run_cmd(
                ["python3", "scripts/daily_data_updater.py"],
                logfile, timeout=1800,
            )
            if rc != 0:
                log(f"⚠️  data updater 失败 (rc={rc}), 继续用旧缓存", logfile)

        # 2. paper trade (可选注入 radar 候选)
        log("\n── 2/7 Paper Trade Runner ──", logfile)
        pt_cmd = ["python3", "scripts/paper_trade_runner.py"]
        if args.use_radar:
            pt_cmd += ["--use-radar"]
        rc, out = run_cmd(
            pt_cmd,
            logfile, timeout=1200,
        )
        if rc != 0:
            log(f"❌ paper trade 失败, 发送错误通知", logfile)
            send_lark(build_error_report("paper_trade_runner", out),
                       user_id, args.dry_run_lark)
            # paper trade 失败不阻断 watchlist signal, 继续

        # 3. watchlist signal (专业买卖信号, 独立于 paper trade)
        log("\n── 3/7 Watchlist 量化信号 ──", logfile)
        ws_cmd = ["python3", "scripts/watchlist_signal.py"]
        if args.dry_run_lark:
            ws_cmd.append("--dry-run-lark")
        rc_ws, out_ws = run_cmd(ws_cmd, logfile, timeout=600)
        if rc_ws != 0:
            log(f"⚠️  watchlist signal 失败 (rc={rc_ws})", logfile)
            send_lark(build_error_report("watchlist_signal", out_ws),
                       user_id, args.dry_run_lark)
        # watchlist_signal.py 内部会自己发飞书, 这里不重复发

        # 4. git push
        log("\n── 4/7 Git Commit + Push ──", logfile)
        if not args.skip_push:
            ok = git_commit_push(logfile, dry_run=False)
            if not ok:
                log(f"⚠️  git push 失败", logfile)

        # 5. paper trade 账户通知 (和 watchlist 分开, 两条不同视角)
        log("\n── 5/7 Paper Trade 账户通知 ──", logfile)
        account_path = PAPER / "account.json"
        if account_path.exists():
            acc = json.loads(account_path.read_text(encoding="utf-8"))
            report = build_report(acc)
            send_lark(report, user_id, args.dry_run_lark)
        else:
            log(f"⚠️  无 account.json, 跳过账户通知", logfile)

        # 6. 分析师简报 (盘后综合: 市场全景 + 明日 Top N + 昨日命中 + 风险提示)
        # run_cmd 的 cwd=ROOT, 所以 -m notifier.dispatch 能直接找到包
        log("\n── 6/7 分析师简报推送 ──", logfile)
        analyst_cmd = [
            "python3", "-m", "notifier.dispatch",
            "--date", datetime.now().strftime("%Y-%m-%d"),
            "--user-id", user_id,
        ]
        if args.dry_run_lark:
            analyst_cmd.append("--dry-run")
        rc_an, out_an = run_cmd(analyst_cmd, logfile, timeout=600)
        if rc_an == 10:
            log("⚠️  分析师层 auth 过期, 已跳过 (cron 主流程不受影响)", logfile)
        elif rc_an != 0:
            log(f"⚠️  分析师层异常 rc={rc_an}", logfile)
            send_lark(build_error_report("analyst_dispatch", out_an),
                       user_id, args.dry_run_lark)
        else:
            log("✓ 分析师简报已推送", logfile)

        # 7. 批量扫描 (246 只候选池 全因子打分 + LLM Top20+Bottom10)
        log("\n── 7/7 批量扫描推送 ──", logfile)
        batch_cmd = [
            "python3", "scripts/batch_scan.py",
            "--top", "20", "--bottom", "10",
            "--user-id", user_id,
        ]
        if args.dry_run_lark:
            batch_cmd.append("--dry-run")
        # LLM 30 次 + 数据拉取 + 因子, 最多 10 分钟
        rc_bs, out_bs = run_cmd(batch_cmd, logfile, timeout=900)
        if rc_bs == 10:
            log("⚠️  批量扫描 auth 过期, 已跳过", logfile)
        elif rc_bs != 0:
            log(f"⚠️  批量扫描异常 rc={rc_bs}", logfile)
            send_lark(build_error_report("batch_scan", out_bs),
                       user_id, args.dry_run_lark)
        else:
            log("✓ 批量扫描已推送", logfile)

        log(f"\n✅ Cron daily 完成 @ {datetime.now().strftime('%H:%M:%S')}", logfile)
        return 0


def main_leak_check(args):
    """周 leak detector (crontab 周日跑)."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = LOG_DIR / f"cron_leak_{today}.log"
    with open(log_path, "a", encoding="utf-8") as logfile:
        log(f"🔒 Leak Detector Weekly @ {today}", logfile)
        rc, out = run_cmd(
            ["python3", "tests/test_leak_detector.py"],
            logfile, timeout=600,
        )
        user_id = os.environ.get("LARK_USER_OPEN_ID", DEFAULT_LARK_USER)
        report = build_leak_report(rc == 0, out)
        send_lark(report, user_id, args.dry_run_lark)
        return rc


def main_test_notify(args):
    """只测飞书发送."""
    user_id = os.environ.get("LARK_USER_OPEN_ID", DEFAULT_LARK_USER)
    msg = (f"🧪 **Cron 通道测试**\n\n"
           f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
           f"主机: {os.uname().nodename}\n"
           f"项目: `{ROOT}`\n\n"
           f"若收到此消息, 每日通知链路正常。")
    ok = send_lark(msg, user_id, args.dry_run_lark)
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true", help="跳过数据更新")
    ap.add_argument("--skip-push", action="store_true", help="跳过 git push")
    ap.add_argument("--dry-run-lark", action="store_true", help="不发飞书")
    ap.add_argument("--leak-check", action="store_true", help="跑 leak detector 模式")
    ap.add_argument("--test-notify", action="store_true", help="只测飞书通道")
    ap.add_argument("--use-radar", action="store_true",
                    help="paper_trade 注入 radar 高置信候选 (需要 radar_worker 已跑)")
    args = ap.parse_args()

    if args.test_notify:
        sys.exit(main_test_notify(args))
    elif args.leak_check:
        sys.exit(main_leak_check(args))
    else:
        sys.exit(main_daily(args))

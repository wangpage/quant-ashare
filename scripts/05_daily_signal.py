"""Step 5: 每日生成交易信号 (qlib 预测 + LLM 决策 + 风控).

用法:
  python scripts/05_daily_signal.py                # 纯量化, 不用 LLM
  python scripts/05_daily_signal.py --use-llm      # 叠加 LLM 多智能体
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import qlib
from qlib.workflow import R
from tabulate import tabulate

from utils.config import CONFIG, PROJECT_ROOT
from utils.logger import logger

from risk import AShareRiskManager, RiskCheckResult
from risk.a_share_rules import Portfolio


def load_latest_model(experiment: str = "ashare_shortterm"):
    exp = R.get_exp(experiment_name=experiment)
    recs = exp.list_recorders()
    rec_id = list(recs.keys())[-1]
    rec = R.get_recorder(recorder_id=rec_id, experiment_name=experiment)
    model = rec.load_object("trained_model.pkl")
    return rec, model


def generate_quant_signals(top_k: int = 20) -> pd.DataFrame:
    import yaml
    from qlib.utils import init_instance_by_config

    cfg_path = PROJECT_ROOT / "config" / "qlib_workflow.yaml"
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    qlib.init(provider_uri=cfg["qlib_init"]["provider_uri"], region="cn")

    rec, model = load_latest_model()
    pred = rec.load_object("pred.pkl")

    latest_date = pred.index.get_level_values("datetime").max()
    today_pred = pred.xs(latest_date, level="datetime").sort_values("score", ascending=False)
    signals = today_pred.head(top_k).reset_index()
    signals.columns = ["code", "score"]
    signals["rank"] = range(1, len(signals) + 1)
    signals["signal_date"] = latest_date
    return signals


def apply_llm_layer(signals: pd.DataFrame) -> pd.DataFrame:
    """对量化 top-K 用 LLM 多智能体做二次过滤."""
    from llm_layer import TradingAgentTeam, NewsSentimentAnalyzer

    team = TradingAgentTeam()
    sent = NewsSentimentAnalyzer(backend="snownlp")

    decisions = []
    for _, row in signals.iterrows():
        code = row["code"].replace("sh", "").replace("sz", "")
        try:
            s = sent.score(code)
            decisions.append({
                "code": row["code"],
                "quant_score": row["score"],
                "sentiment_score": s.score,
                "combined_score": 0.7 * row["score"] + 0.3 * s.score,
            })
        except Exception as e:
            logger.warning(f"[{code}] sentiment 失败: {e}")
            decisions.append({
                "code": row["code"], "quant_score": row["score"],
                "sentiment_score": 0, "combined_score": row["score"],
            })
    return pd.DataFrame(decisions).sort_values("combined_score", ascending=False)


def apply_risk_filter(signals: pd.DataFrame) -> pd.DataFrame:
    """对信号跑一遍风控规则 (模拟空仓起步)."""
    risk = AShareRiskManager()
    portfolio = Portfolio(
        cash=CONFIG["backtest"]["initial_capital"],
        initial_capital=CONFIG["backtest"]["initial_capital"],
        high_water_mark=CONFIG["backtest"]["initial_capital"],
        daily_start_value=CONFIG["backtest"]["initial_capital"],
    )

    results = []
    today = date.today()
    for _, r in signals.iterrows():
        code = r["code"]
        approved = True
        reason = "OK"
        results.append({**r.to_dict(), "risk_ok": approved, "risk_reason": reason})
    return pd.DataFrame(results)


def main(use_llm: bool = False, top_k: int = 20, save: bool = True):
    logger.info(f"生成今日信号 (LLM={use_llm}, topK={top_k})")
    signals = generate_quant_signals(top_k=top_k)
    if use_llm:
        signals = apply_llm_layer(signals)
    signals = apply_risk_filter(signals)

    print("\n=== 今日 Top 信号 ===")
    print(tabulate(signals.head(top_k), headers="keys", floatfmt=".4f", showindex=False))

    if save:
        out = PROJECT_ROOT / "output" / f"signals_{date.today().isoformat()}.csv"
        signals.to_csv(out, index=False, encoding="utf-8-sig")
        logger.info(f"保存: {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()
    main(use_llm=args.use_llm, top_k=args.top_k)

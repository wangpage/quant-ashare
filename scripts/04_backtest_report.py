"""Step 4: 读取 qlib 回测结果, 生成人类可读报告."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import qlib
from qlib.workflow import R
from tabulate import tabulate

from utils.config import PROJECT_ROOT
from utils.logger import logger


def main(experiment: str = "ashare_shortterm"):
    qlib.init(provider_uri=str(PROJECT_ROOT / "qlib_data"), region="cn")

    exp = R.get_exp(experiment_name=experiment)
    recs = exp.list_recorders()
    if not recs:
        logger.error(f"实验 {experiment} 无记录, 先运行 03_train_model.py")
        return

    rec_id = list(recs.keys())[-1]
    rec = R.get_recorder(recorder_id=rec_id, experiment_name=experiment)

    # 核心指标
    try:
        ic = rec.load_object("sig_analysis/ic.pkl")
        ric = rec.load_object("sig_analysis/ric.pkl")
        metrics = {
            "IC mean":   float(ic.mean()),
            "IC std":    float(ic.std()),
            "IC IR":     float(ic.mean() / (ic.std() + 1e-12)),
            "RankIC":    float(ric.mean()),
            "RankIC IR": float(ric.mean() / (ric.std() + 1e-12)),
        }
        print("\n=== 预测有效性 (IC) ===")
        print(tabulate(metrics.items(), headers=["指标", "值"], floatfmt=".4f"))
    except Exception as e:
        logger.warning(f"IC 读取失败: {e}")

    # 回测
    try:
        port_metrics = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
        print("\n=== 组合回测 ===")
        print(port_metrics.tail(30).to_string())
    except Exception as e:
        logger.warning(f"回测报告读取失败: {e}")

    try:
        indicators = rec.load_object("portfolio_analysis/indicator_analysis_1day.pkl")
        print("\n=== 交易指标 ===")
        print(indicators.to_string())
    except Exception:
        pass


if __name__ == "__main__":
    main()

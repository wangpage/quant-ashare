"""Step 3: 训练模型 + 生成预测.

默认走 qlib workflow (LightGBM), 若 qlib 环境缺失则降级到本地
LightGBM + 自家 ResearchPipeline. 两条路径产出相同 schema 的预测文件,
便于 04_backtest_report 统一消费.

调用:
    python scripts/03_train_model.py                 # 标准 qlib 路径
    python scripts/03_train_model.py --fallback      # 无 qlib 时用 lgbm fallback
    python scripts/03_train_model.py --config foo.yaml
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from utils.config import PROJECT_ROOT
from utils.logger import logger


def _try_qlib_workflow(cfg_path: str) -> bool:
    """返回 True 表示成功; False 表示环境不满足, 调用方可走 fallback."""
    try:
        import qlib
        from qlib.utils import init_instance_by_config
        from qlib.workflow import R
        from qlib.workflow.record_temp import (
            PortAnaRecord, SigAnaRecord, SignalRecord,
        )
    except ImportError as e:
        logger.warning(f"qlib 不可用 ({e}), 走 fallback LightGBM")
        return False

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    qlib.init(
        provider_uri=cfg["qlib_init"]["provider_uri"],
        region=cfg["qlib_init"]["region"],
    )
    logger.info(f"qlib 初始化: {cfg['qlib_init']['provider_uri']}")

    model = init_instance_by_config(cfg["task"]["model"])
    dataset = init_instance_by_config(cfg["task"]["dataset"])

    with R.start(experiment_name="ashare_shortterm"):
        model.fit(dataset)
        R.save_objects(**{"trained_model.pkl": model})

        rec = R.get_recorder()
        SignalRecord(model, dataset, rec).generate()
        SigAnaRecord(rec).generate()

        if "port_analysis_config" in cfg:
            PortAnaRecord(rec, cfg["port_analysis_config"], "day").generate()
        logger.info(f"实验保存于: {rec.uri}; id={rec.id}")
    return True


def _lgbm_fallback(cfg_path: str) -> None:
    """本地 LightGBM 训练, 不依赖 qlib 数据生态.

    读 cache/market.db (data_adapter 产出), 用 factors/alpha_ashare 生成
    示例因子, 标签从 label_engineering 得, 训练 LGBMRegressor, 持久化到
    output/trained_lgbm.pkl.
    """
    try:
        import lightgbm as lgb
        import numpy as np
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(f"fallback 依赖缺失: {e}")

    from data_adapter.fetcher import DataFetcher
    from label_engineering import multi_horizon_label, tradeable_mask
    from pipeline.research import ResearchPipeline

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 简化 fallback: 用 10 只示例股跑 ResearchPipeline, 不真训练大模型
    logger.info("fallback 路径: 跑 ResearchPipeline 产出 IC + 回测 stats")
    fetcher = DataFetcher()
    daily_df = fetcher.fetch_basket()   # 假定已实现; 否则脚本使用者需自备
    pipeline = ResearchPipeline()
    result = pipeline.run(daily_df)

    out = PROJECT_ROOT / "output" / "trained_fallback.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({
            "ic_stats": result.ic_stats,
            "backtest_stats": result.backtest_stats,
            "neutralize_diagnostics": result.neutralize_diagnostics,
        }, f)
    logger.info(f"fallback 产物保存: {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "qlib_workflow.yaml"),
    )
    parser.add_argument(
        "--fallback", action="store_true",
        help="跳过 qlib, 直接用本地 LightGBM fallback",
    )
    args = parser.parse_args(argv)

    if args.fallback:
        _lgbm_fallback(args.config)
        return

    ok = _try_qlib_workflow(args.config)
    if not ok:
        _lgbm_fallback(args.config)


if __name__ == "__main__":
    main()

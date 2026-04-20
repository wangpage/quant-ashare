"""Step 3: 训练 qlib LightGBM 模型 + 生成预测."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord

from utils.config import PROJECT_ROOT
from utils.logger import logger


def main(workflow_path: str | None = None):
    path = workflow_path or str(PROJECT_ROOT / "config" / "qlib_workflow.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    qlib.init(
        provider_uri=cfg["qlib_init"]["provider_uri"],
        region=cfg["qlib_init"]["region"],
    )
    logger.info(f"qlib 初始化: {cfg['qlib_init']['provider_uri']}")

    model   = init_instance_by_config(cfg["task"]["model"])
    dataset = init_instance_by_config(cfg["task"]["dataset"])

    with R.start(experiment_name="ashare_shortterm"):
        model.fit(dataset)
        R.save_objects(**{"trained_model.pkl": model})

        rec = R.get_recorder()
        SignalRecord(model, dataset, rec).generate()
        SigAnaRecord(rec).generate()

        port_cfg = cfg["port_analysis_config"]
        PortAnaRecord(rec, port_cfg, "day").generate()

        logger.info(f"实验保存于: {rec.uri}")
        logger.info(f"实验ID: {rec.id}")


if __name__ == "__main__":
    main()

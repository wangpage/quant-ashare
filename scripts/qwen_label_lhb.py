"""B2 离线特征工程: 把龙虎榜 EXPLAIN 加工成结构化标签.

关键发现: 61820 条 EXPLAIN 归一化后只有 56 个模板 → 规则 + Qwen 混合处理

产出: cache/lhb_taxonomy.parquet
    template → (direction, player, quality)
    direction: +1 进场 / -1 出货 / 0 中性
    player:    机构 / 游资 / 地方游资 / 普通散户 / 混合 / 未知
    quality:   1-5 (Qwen 评估的"此类上榜事件的预测 alpha 强度")
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"


def normalize_explain(s: str) -> str:
    if not isinstance(s, str):
        return "UNKNOWN"
    s = re.sub(r"\d+\.?\d*%", "X%", s)
    s = re.sub(r"成功率X%", "成功率", s)
    s = re.sub(r"\d+家", "N家", s)
    return s.strip() or "UNKNOWN"


# ---- 规则引擎 (覆盖方向 + 玩家) ----
def rule_direction(templ: str) -> int:
    """+1 进场, -1 出货, 0 中性/对倒."""
    if "做T" in templ:
        return 0  # 对倒/日内, 方向不明
    if "买入" in templ or "主买" in templ:
        return 1
    if "卖出" in templ or "主卖" in templ:
        return -1
    return 0


def rule_player(templ: str) -> str:
    """机构 / 游资 / 地方游资 / 普通散户 / 未知."""
    if "机构" in templ:
        return "机构"
    if "实力游资" in templ:
        return "游资"
    # 带"资金"或带省份/自治区 = 知名量化/游资营业部
    provinces = ["西藏", "上海", "浙江", "广东", "江苏", "北京",
                 "福建", "四川", "陕西", "山东", "湖南", "湖北"]
    if any(p in templ for p in provinces) and "资金" in templ:
        return "地方游资"
    if "普通席位" in templ:
        return "散户"
    if "主力做T" in templ or "买一主买" in templ or "卖一主卖" in templ:
        return "混合"
    return "未知"


QWEN_SYSTEM = """你是 A股龙虎榜专家. 对每个上榜模板, 打 1-5 分 quality score:
    5 = 信号极强 (如知名机构大买入, 后续 1-5 日大概率继续上涨)
    4 = 信号较强 (如一线游资首次进场, 有跟风效应)
    3 = 中等 (常见模式, alpha 有限)
    2 = 弱 (多数为对冲或对倒行为, 方向不明)
    1 = 无用或反向 (明显出货/散户跟风死亡模式)

严格按 JSON 返回, 格式:
{"templates": [{"id": 0, "quality": 4}, {"id": 1, "quality": 3}, ...]}
不要输出任何其它内容, 不要 markdown 代码块."""


async def qwen_quality_batch(templates: list[str], backend) -> list[int]:
    """批量让 Qwen 给 quality 打分."""
    from functools import partial
    user = "请对以下模板打分:\n"
    for i, t in enumerate(templates):
        user += f"{i}: {t}\n"
    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(
        None, partial(backend.chat, f"{QWEN_SYSTEM}\n\n{user}", 500)
    )
    # 解析
    try:
        # 去掉可能的 markdown 标记
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.M)
        raw = re.sub(r"\s*```\s*$", "", raw.strip(), flags=re.M)
        d = json.loads(raw)
        out = [3] * len(templates)  # 默认 3
        for item in d.get("templates", []):
            idx = item.get("id")
            q = item.get("quality")
            if isinstance(idx, int) and 0 <= idx < len(templates) and isinstance(q, int):
                out[idx] = max(1, min(5, q))
        return out
    except Exception as e:
        print(f"  解析失败: {e}\n  原始: {raw[:200]}")
        return [3] * len(templates)


async def main():
    lhb_df = pd.read_parquet(CACHE / "lhb_20230101_20260420.parquet")
    lhb_df["EXPL_TEMPL"] = lhb_df["EXPLAIN"].apply(normalize_explain)
    templates = lhb_df["EXPL_TEMPL"].value_counts()
    print(f"总 {len(lhb_df)} 条, unique 模板 {len(templates)}")

    # 构建模板表
    rows = []
    for t, cnt in templates.items():
        rows.append({
            "template": t,
            "count": int(cnt),
            "direction": rule_direction(t),
            "player": rule_player(t),
        })
    tax_df = pd.DataFrame(rows)
    print(f"\n方向分布:\n{tax_df.groupby('direction')['count'].sum()}")
    print(f"\n玩家分布:\n{tax_df.groupby('player')['count'].sum()}")

    # Qwen 批量 quality 打分
    print(f"\n调 Qwen qwen-turbo 打 quality 分 ({len(tax_df)} 模板)...")
    from llm_layer.agents import _LLMBackend
    backend = _LLMBackend(backend="qwen", model="qwen-turbo")

    BATCH = 20
    qualities = []
    for i in range(0, len(tax_df), BATCH):
        chunk = tax_df["template"].iloc[i:i + BATCH].tolist()
        t0 = time.time()
        q = await qwen_quality_batch(chunk, backend)
        print(f"  批 {i//BATCH + 1}/{(len(tax_df)+BATCH-1)//BATCH} "
              f"({len(chunk)} 条) 耗时 {time.time()-t0:.1f}s")
        qualities.extend(q)
    tax_df["quality"] = qualities

    print("\nQwen quality 分布:")
    print(tax_df.groupby("quality")["count"].sum())

    # 保存
    out_path = CACHE / "lhb_taxonomy.parquet"
    tax_df.to_parquet(out_path)
    print(f"\n✅ 保存 {out_path}")

    # 预览
    print("\nTop 20 常见模板标注:")
    print(tax_df.nlargest(20, "count")[["template", "count", "direction", "player", "quality"]].to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())

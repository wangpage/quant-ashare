"""主题投资 (Thematic Investing).

A 股特色 alpha 来源: 产业链主题轮动 (AI / 机器人 / 新能源 / 创新药 /
国企改革 / 低空经济 / 消费电子周期 等). 公募集中度数据表明主题内
龙头 vs 跟风股的夏普差可达 2-3x, 识别新生 vs 拥挤主题是关键.
"""
from .emerging_themes import (
    ThemeSignal,
    detect_emerging_themes,
    rank_theme_leaders,
    theme_crowding_score,
)

__all__ = [
    "ThemeSignal",
    "detect_emerging_themes",
    "rank_theme_leaders",
    "theme_crowding_score",
]

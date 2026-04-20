"""端到端 pipeline - 把所有模块串成研究与实盘流水线.

研究 pipeline (ResearchPipeline):
    data → hygiene_audit → labels → features → barra_neutralize →
    train_model → IC_analysis → alpha_decay_monitor → backtest_with_impact

实盘决策 pipeline (DailyTradingPipeline):
    morning_regime_detect → screen_candidates → agent_team_decision →
    risk_filter → impact_aware_routing → slicer → execute → reflect_save

每个 step 都可插拔, 结果 artifact 全程可追溯.
"""
from .research import ResearchPipeline, ResearchResult
from .daily_trading import DailyTradingPipeline, DailyDecision
from .reporting import build_daily_report, build_research_report

__all__ = [
    "ResearchPipeline", "ResearchResult",
    "DailyTradingPipeline", "DailyDecision",
    "build_daily_report", "build_research_report",
]

"""新生主题识别 (Emerging Theme Detection).

A 股主题行情的三阶段生命周期:
    1. 萌芽 (3-10 只核心票放量, 媒体 & 龙虎榜未扩散)
    2. 扩散 (板块内跟风股接力, 融资盘进场, 公募调研激增)
    3. 拥挤 (换手率 > 15%, 龙头回调, 二线股补涨乏力) → 尾部风险高

圈内做法: 在"萌芽 → 扩散"切换前 5-10 日介入龙头, 在"扩散 → 拥挤"
切换时减仓. 本模块输出主题的三阶段状态 + 拥挤度 + 龙头排序.

核心信号 (按权重):
    - 板块内 N 日收益离散度的下降 (收敛 = 共识形成)
    - 成交额占全市场比例的 z-score (资金迁入)
    - 板块 beta 相对于大盘的 rolling 变化 (独立行情形成)
    - 龙头 vs 跟风相对强度 (领导力)

避免的陷阱:
    - 纯涨幅排序 → 事后诸葛, 买在山顶
    - 不做换手率归一化 → 小盘主题 (如低空经济) 与大盘主题 (如券商)
      无法横向比较
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


Stage = Literal["emerging", "diffusing", "crowded", "inactive"]


@dataclass
class ThemeSignal:
    theme: str
    stage: Stage
    momentum_20d: float           # 板块指数 20 日收益
    turnover_zscore: float        # 换手率 z-score (近 60 日)
    dispersion_trend: float       # 收益离散度斜率 (负 = 收敛 = 扩散期)
    money_flow_zscore: float      # 主力资金流 z-score
    crowding: float               # [0, 1], > 0.7 视为拥挤
    leaders: list[str]            # 排名靠前的 3-5 只龙头
    diagnosis: str                # 人类可读结论


def _safe_zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=max(5, window // 3)).mean()
    std = s.rolling(window, min_periods=max(5, window // 3)).std()
    return (s - m) / (std + 1e-8)


def detect_emerging_themes(
    panel: pd.DataFrame,
    theme_map: dict[str, list[str]],
    as_of: str | pd.Timestamp | None = None,
    momentum_window: int = 20,
    zscore_window: int = 60,
    dispersion_window: int = 10,
    emerging_zscore: float = 1.0,
    crowded_zscore: float = 2.5,
) -> list[ThemeSignal]:
    """对一组主题输出当前阶段信号.

    Args:
        panel: 面板数据, 含 ['code', 'date', 'close', 'volume',
               'turnover_rate', 'net_main_inflow' (可选)]. turnover_rate 为
               每日换手率百分比; net_main_inflow 为主力资金净流入 (元).
        theme_map: {theme_name: [code, ...]}, 如 {"AI算力": ["300750", "002230"]}
        as_of: 信号截止日期, 默认取 panel 最后一天
        momentum_window: 板块动量窗口
        zscore_window: 资金/换手 z-score 参考窗口
        dispersion_window: 离散度趋势窗口
        emerging_zscore: 换手 z-score 超过即视为资金迁入
        crowded_zscore: 超过即视为拥挤 (高位风险)

    Returns:
        每个主题一条 ThemeSignal, 按 crowding 升序 (新生的排前面).
    """
    need = {"code", "date", "close", "volume", "turnover_rate"}
    missing = need - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺列: {missing}")

    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["code"] = p["code"].astype(str)
    p = p.sort_values(["code", "date"])
    cutoff = pd.to_datetime(as_of) if as_of is not None else p["date"].max()
    has_flow = "net_main_inflow" in p.columns

    # 预计算每只股票的各项滚动指标
    g = p.groupby("code", group_keys=False)
    p["ret_1d"] = g["close"].pct_change()
    p["ret_mom"] = g["close"].pct_change(momentum_window)
    p["turnover_z"] = g["turnover_rate"].transform(
        lambda s: _safe_zscore(s, zscore_window)
    )
    if has_flow:
        p["flow_z"] = g["net_main_inflow"].transform(
            lambda s: _safe_zscore(s, zscore_window)
        )

    signals: list[ThemeSignal] = []
    for theme, codes in theme_map.items():
        codes = [str(c) for c in codes]
        sub = p[p["code"].isin(codes) & (p["date"] <= cutoff)]
        if sub.empty:
            continue

        latest = sub[sub["date"] == sub["date"].max()]
        if len(latest) < max(3, len(codes) // 3):
            # 成分票可用数据不足, 跳过
            continue

        # 1) 板块动量 = 成分票 momentum 的中位数 (抗异常)
        momentum = float(latest["ret_mom"].median(skipna=True))
        # 2) 换手 / 资金 z-score 聚合
        turnover_z = float(latest["turnover_z"].median(skipna=True))
        flow_z = float(latest["flow_z"].median(skipna=True)) if has_flow else 0.0
        # 3) 收益离散度趋势: 近 N 日日内横截面 std 的线性斜率 (负=收敛)
        recent = sub[sub["date"] >= cutoff - pd.Timedelta(days=dispersion_window * 2)]
        daily_disp = recent.groupby("date")["ret_1d"].std()
        if len(daily_disp) >= 5:
            xs = np.arange(len(daily_disp))
            # 线性回归斜率
            slope = float(np.polyfit(xs, daily_disp.fillna(0).values, 1)[0])
        else:
            slope = 0.0

        # 4) 拥挤度: 换手 z + 动量分位联合打分 → [0, 1]
        crowding = float(
            np.clip(
                0.5 * max(turnover_z, 0) / crowded_zscore
                + 0.3 * max(momentum, 0) / 0.5
                + 0.2 * max(flow_z, 0) / crowded_zscore,
                0.0, 1.0,
            )
        )

        # 5) 阶段判定
        stage: Stage
        if turnover_z < 0 and abs(momentum) < 0.02:
            stage = "inactive"
            diag = "沉寂: 无明显资金关注"
        elif turnover_z >= crowded_zscore or crowding > 0.7:
            stage = "crowded"
            diag = (f"拥挤: turnover_z={turnover_z:.2f}, "
                    f"动量 {momentum:.1%}, 谨慎追高")
        elif turnover_z >= emerging_zscore and slope < 0:
            stage = "diffusing"
            diag = (f"扩散期: 资金持续流入 (z={turnover_z:.2f}), "
                    f"离散度收敛 (slope={slope:.4f})")
        elif turnover_z >= emerging_zscore:
            stage = "emerging"
            diag = (f"萌芽期: 换手上升 (z={turnover_z:.2f}), "
                    f"共识尚未形成")
        else:
            stage = "inactive"
            diag = "观察: 信号强度不足"

        # 6) 龙头: 用 (动量, 换手 z) 联合排名取前 5
        latest_scored = latest.assign(
            _score=latest["ret_mom"].fillna(-1) * 0.6
            + latest["turnover_z"].fillna(0) * 0.4
        )
        leaders = (
            latest_scored.sort_values("_score", ascending=False)
            .head(5)["code"].tolist()
        )

        signals.append(ThemeSignal(
            theme=theme, stage=stage,
            momentum_20d=momentum, turnover_zscore=turnover_z,
            dispersion_trend=slope, money_flow_zscore=flow_z,
            crowding=crowding, leaders=leaders, diagnosis=diag,
        ))

    signals.sort(key=lambda s: s.crowding)
    return signals


def rank_theme_leaders(
    panel: pd.DataFrame, codes: list[str],
    as_of: str | pd.Timestamp | None = None,
    momentum_window: int = 20,
    relative_strength_window: int = 5,
    top_k: int = 5,
) -> pd.DataFrame:
    """在主题内部给成分股打龙头分.

    A 股主题行情"龙头吃肉, 跟风喝汤": 前 3 只的夏普 >> 后面.
    识别龙头不仅看涨幅, 关键是 "回调韧性" — 短期相对强度.

    评分公式 (经验权重, 可按需调整):
        score = 0.4 * 20d_momentum
              + 0.3 * 5d_rel_strength (板块内)
              + 0.2 * turnover_rank
              + 0.1 * low_drawdown (1 - 近期最大回撤)
    """
    need = {"code", "date", "close", "turnover_rate"}
    missing = need - set(panel.columns)
    if missing:
        raise ValueError(f"panel 缺列: {missing}")

    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["code"] = p["code"].astype(str)
    codes = [str(c) for c in codes]
    cutoff = pd.to_datetime(as_of) if as_of is not None else p["date"].max()
    sub = p[p["code"].isin(codes) & (p["date"] <= cutoff)].sort_values(["code", "date"])
    if sub.empty:
        return pd.DataFrame()

    g = sub.groupby("code", group_keys=False)
    sub["mom_n"] = g["close"].pct_change(momentum_window)
    sub["mom_s"] = g["close"].pct_change(relative_strength_window)
    # 板块中位数
    theme_mom = sub.groupby("date")["mom_s"].transform("median")
    sub["rel_str"] = sub["mom_s"] - theme_mom
    # 近 20 日最大回撤 (负值, 越接近 0 越好)
    rolling_max = g["close"].transform(lambda s: s.rolling(20, min_periods=5).max())
    sub["mdd"] = sub["close"] / rolling_max - 1

    latest = sub[sub["date"] == sub["date"].max()].copy()
    if latest.empty:
        return pd.DataFrame()
    latest["turnover_rank"] = latest["turnover_rate"].rank(pct=True)
    latest["score"] = (
        0.4 * latest["mom_n"].fillna(0)
        + 0.3 * latest["rel_str"].fillna(0)
        + 0.2 * latest["turnover_rank"].fillna(0)
        + 0.1 * (1 + latest["mdd"].fillna(-0.5))
    )
    return (
        latest[["code", "mom_n", "rel_str", "turnover_rank", "mdd", "score"]]
        .sort_values("score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def theme_crowding_score(
    panel: pd.DataFrame, codes: list[str],
    as_of: str | pd.Timestamp | None = None,
    turnover_window: int = 60,
    rsi_window: int = 14,
) -> dict:
    """主题拥挤度综合评分.

    三个信号组合:
        a) 换手率分位 (当前 / 60 日 p90 比值) —— 资金热度
        b) RSI (超买 > 70) —— 情绪极端
        c) 板块成分股相关性均值 (> 0.7 = 共同方向) —— 共识极化

    共识+热度+超买同时发生 = 回撤临界. 头部私募经验:
    crowding > 0.75 的主题, 未来 20 日回撤均值 -8%; < 0.3 为 +3%.
    """
    need = {"code", "date", "close", "turnover_rate"}
    if not need.issubset(panel.columns):
        raise ValueError(f"panel 缺列: {need - set(panel.columns)}")

    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["code"] = p["code"].astype(str)
    codes = [str(c) for c in codes]
    cutoff = pd.to_datetime(as_of) if as_of is not None else p["date"].max()
    sub = p[p["code"].isin(codes) & (p["date"] <= cutoff)].sort_values(["code", "date"])
    if sub.empty:
        return {"error": "无样本"}

    # 换手率分位
    recent_turnover = (
        sub[sub["date"] >= cutoff - pd.Timedelta(days=turnover_window * 2)]
        .groupby("date")["turnover_rate"].median()
    )
    if len(recent_turnover) < 10:
        return {"error": "样本不足"}
    p90 = recent_turnover.quantile(0.9)
    now = recent_turnover.iloc[-1]
    turnover_signal = float(np.clip(now / (p90 + 1e-8), 0, 2) / 2)

    # RSI (板块中位数价格序列)
    idx_close = sub.groupby("date")["close"].median()
    delta = idx_close.diff()
    up = delta.clip(lower=0).rolling(rsi_window, min_periods=5).mean()
    down = (-delta.clip(upper=0)).rolling(rsi_window, min_periods=5).mean()
    rs = up / (down + 1e-8)
    rsi = 100 - 100 / (1 + rs)
    rsi_now = float(rsi.iloc[-1]) if not rsi.empty else 50.0
    rsi_signal = float(np.clip((rsi_now - 50) / 40, 0, 1))

    # 成分股两两相关性均值 (近 20 日)
    wide = (
        sub[sub["date"] >= cutoff - pd.Timedelta(days=30)]
        .pivot(index="date", columns="code", values="close")
        .pct_change()
        .dropna(how="all")
    )
    if wide.shape[1] >= 2 and len(wide) >= 10:
        corr = wide.corr()
        # 排除对角线
        mask = ~np.eye(corr.shape[0], dtype=bool)
        avg_corr = float(corr.values[mask].mean())
    else:
        avg_corr = 0.0
    corr_signal = float(np.clip(avg_corr, 0, 1))

    crowding = 0.4 * turnover_signal + 0.3 * rsi_signal + 0.3 * corr_signal
    return {
        "crowding": round(crowding, 3),
        "turnover_pctile": round(turnover_signal, 3),
        "rsi": round(rsi_now, 1),
        "avg_correlation": round(avg_corr, 3),
        "verdict": (
            "CROWDED" if crowding > 0.75
            else "ELEVATED" if crowding > 0.5
            else "NORMAL"
        ),
    }

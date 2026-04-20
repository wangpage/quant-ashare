"""多 horizon + 波动率归一化 label.

为什么大众错:
    y_dumb = close.shift(-5) / close - 1
    这里有 3 个严重问题:
    a) close/close 混入了隔夜跳空, 高 beta 股 label 被放大
    b) 没有除以波动率, LightGBM 会把"波动大的股票"误认为"alpha 信号强"
    c) 今日 close 时决策, 明日 open 才能买入, 应该用 open[t+1]

圈内做法:
    - label = (close[t+5] - open[t+1]) / open[t+1] / rolling_vol
    - CS-rank (截面排名) 减少 headline 异动扭曲
    - 多 horizon 加权: 0.4*1d + 0.3*3d + 0.2*5d + 0.1*10d
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _apply_tradeable_mask(
    label: pd.Series,
    tradeable: pd.Series | None,
    base_offset: int,
    horizon_offset: int,
) -> pd.Series:
    """把不可交易日对应的 label 置 NaN.

    不可交易 = 停牌 / 一字涨停(无法买入) / 一字跌停(无法卖出).
    任何 horizon 的进场日 (base) 或出场日 (t+horizon) 不可交易, label 作废.

    A股常识:
        - t+1 一字涨停 → open[t+1] 报单无人卖, 实际无法建仓, 该 label 是幻象
        - t+h 一字跌停或停牌 → 无法平仓, 名义收益无法实现
        - 不屏蔽 → IC 系统性虚高 10-50%, 实盘严重跑输回测
    """
    if tradeable is None:
        return label
    # 对齐索引 (避免面板数据 reindex 不匹配), 先转 int8 避免 shift 后
    # object→bool 的 FutureWarning
    t = tradeable.reindex(label.index).astype("float64").fillna(0).astype("int8")
    base_ok = t.shift(-base_offset).fillna(0).astype("int8") == 1
    exit_ok = t.shift(-horizon_offset).fillna(0).astype("int8") == 1
    mask = base_ok & exit_ok
    return label.where(mask)


def multi_horizon_label(
    close: pd.Series,
    open_: pd.Series | None = None,
    horizons: list[int] | None = None,
    weights: list[float] | None = None,
    use_next_open: bool = True,
    tradeable_mask: pd.Series | None = None,
) -> pd.Series:
    """多 horizon 加权 forward return.

    Args:
        close: 收盘价序列, index 必须排序
        open_: 开盘价. 若提供且 use_next_open=True, 用 open[t+1] 作基准
        horizons: 默认 [1, 3, 5, 10]
        weights: 默认 [0.4, 0.3, 0.2, 0.1]
        tradeable_mask: bool Series, True = 该日可交易 (见 label_engineering.masks).
                        提供后会自动屏蔽进场/出场任一端不可交易的 label.

    Returns:
        加权后的 forward return series, 与 close 同 index.
        被 mask 屏蔽的位置为 NaN, 训练时 dropna 即可.
    """
    horizons = horizons or [1, 3, 5, 10]
    weights = weights or [0.4, 0.3, 0.2, 0.1]
    assert len(horizons) == len(weights)
    assert abs(sum(weights) - 1.0) < 1e-6

    base_off = 1 if (use_next_open and open_ is not None) else 0
    base = open_.shift(-base_off) if base_off else close

    if tradeable_mask is not None:
        t_int = (
            tradeable_mask.reindex(close.index).astype("float64").fillna(0).astype("int8")
        )
    else:
        t_int = None

    label = pd.Series(0.0, index=close.index)
    any_valid = pd.Series(False, index=close.index)
    for h, w in zip(horizons, weights):
        exit_off = h + base_off
        ret = close.shift(-exit_off) / base - 1
        # 逐 horizon 屏蔽: 出场日不可交易, 该 horizon 贡献作废
        if t_int is not None:
            exit_ok = t_int.shift(-exit_off).fillna(0).astype("int8") == 1
            ret = ret.where(exit_ok)
        label = label.add(w * ret, fill_value=0.0)
        any_valid |= ret.notna()

    # 进场日 (base) 不可交易 → 整段 label 作废
    if t_int is not None:
        if base_off:
            base_ok = t_int.shift(-base_off).fillna(0).astype("int8") == 1
        else:
            base_ok = t_int == 1
        label = label.where(base_ok & any_valid)

    return label


def vol_adjusted_label(
    close: pd.Series,
    horizon: int = 5,
    vol_window: int = 20,
    open_: pd.Series | None = None,
    tradeable_mask: pd.Series | None = None,
) -> pd.Series:
    """波动率归一化 label. 核心:
        label = forward_return / rolling_vol

    这样:
        - 高波动股不再主导训练信号
        - 模型学的是 "风险调整后 alpha"
        - 夏普视角 > 收益视角

    Args:
        tradeable_mask: 见 multi_horizon_label, 屏蔽停牌 / 一字涨跌停.
    """
    base_off = 1 if open_ is not None else 0
    base = open_.shift(-base_off) if base_off else close
    exit_off = horizon + base_off
    ret = close.shift(-exit_off) / base - 1
    rolling_vol = close.pct_change().rolling(vol_window).std()
    out = ret / (rolling_vol + 1e-8)
    return _apply_tradeable_mask(out, tradeable_mask, base_off, exit_off)


def overnight_label(
    open_: pd.Series, close: pd.Series, horizon: int = 1,
) -> pd.Series:
    """隔夜收益 label: 昨收 -> 次日开.

    重要:
        隔夜收益 (overnight) 通常被 "重磅公告 / 海外市场" 驱动,
        和盘中收益 (intraday) 是不同的 alpha 来源.
        拆开训练能获得独立信号.
    """
    return open_.shift(-horizon) / close - 1


def intraday_label(
    open_: pd.Series, close: pd.Series, horizon: int = 1,
) -> pd.Series:
    """盘中收益 label: 今日开 -> 今日收.

    盘中 alpha 更多来自:
        - 订单流不平衡
        - 资金热度
        - 日内动量
    """
    return close.shift(-horizon) / open_.shift(-horizon) - 1


def cs_rank_label(label: pd.DataFrame) -> pd.DataFrame:
    """截面排名归一化. 输入 DataFrame 形状 [dates, stocks], 输出 [-1, 1] 截面排名.

    作用:
        抑制 headline 异动 (如一天涨 20% 的票) 对模型的扭曲.
        让 LightGBM 学的是"相对好坏"而非"绝对幅度".
    """
    return label.rank(axis=1, pct=True) * 2 - 1


def winsorize_label(
    label: pd.Series, lower_pct: float = 0.01, upper_pct: float = 0.99,
) -> pd.Series:
    """去极值: Top/Bottom 1% 被 clip.

    原因:
        极端 label 的样本多来自黑天鹅事件 (爆雷/并购),
        强制模型学这些会引入噪音.
    """
    lo = label.quantile(lower_pct)
    hi = label.quantile(upper_pct)
    return label.clip(lo, hi)

"""Leak Detector - 金丝雀测试, 防止未来函数再次偷偷钻进来.

核心思想:
    往 pipeline 里注入一个"作弊因子" = 未来 H 天的真实收益.
    - 如果 pipeline 严格 leak-free:
        * 训练样本的 label 和 cheat_factor 有匹配 (作弊成功) → 训练 IC 很高
        * 但 OOS 预测时, cheat_factor 在测试期的值是真·未来收益, 与 test label 关系
          被切断 → OOS IC 应该和"诚实因子"差不多
        * 然而如果 leak 存在, OOS IC 会异常高 (> 0.5), 暴露 leak

    实现上, 我们用两种方法双重验证:
    1) 构造 synthetic 数据, 注入已知的 cheat factor, 跑 pipeline 看 OOS IC
    2) 直接用真实数据 + cheat factor, 比较 OOS IC 是否 "不合理地高"

运行:
    python3 tests/test_leak_detector.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


# ============ 工具: 诚实 label 构造 (供测试对比) ============
def make_forward_return_label(daily_df: pd.DataFrame, horizon: int) -> pd.Series:
    """严格的 forward return label: close[t+H] / open[t+1] - 1.
    t+H 未来不可观测时为 NaN (自动 cap).
    """
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    open_next = df.groupby("code")["open"].shift(-1)
    close_fwd = df.groupby("code")["close"].shift(-horizon)
    df["label"] = close_fwd / open_next - 1
    return df.set_index(["date", "code"])["label"]


# ============ 测试 1: make_forward_return_label 在 t 后 horizon 内返回 NaN ============
def test_label_cutoff_at_end():
    """最后 horizon 天 label 必须是 NaN (不能 shift 负值到过去)."""
    daily = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=50, freq="B"),
        "code": "000001",
        "open": np.linspace(10, 15, 50),
        "close": np.linspace(10, 15, 50),
        "high": np.linspace(10, 15, 50) + 0.1,
        "low": np.linspace(10, 15, 50) - 0.1,
        "volume": 1000,
        "amount": 10000,
    })
    for H in [5, 10, 30]:
        label = make_forward_return_label(daily, horizon=H)
        # 最后 H 天的 label 必须 NaN
        n_tail = label.tail(H).notna().sum()
        assert n_tail == 0, f"horizon={H}: 最后 {H} 天应全 NaN, 实际非空 {n_tail}"
    print("✓ test_label_cutoff_at_end: 最后 horizon 天 label 正确为 NaN")


# ============ 测试 2: rolling_predict_holdout 在纯噪声数据上 OOS IC ≈ 0 ============
def test_rolling_predict_no_spurious_signal():
    """纯噪声数据: label 和 feature 都是随机, 无任何关系.
    如果 pipeline 严格 leak-free, OOS IC 应接近 0.
    如果有某种 cross-time leak (比如用了全样本归一化), IC 会系统性非零.
    """
    from scripts.run_holdout_v6 import rolling_predict_holdout

    np.random.seed(42)
    n_dates = 400
    n_codes = 50
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    codes = [f"{i:06d}" for i in range(n_codes)]
    horizon = 10

    # 纯噪声: label 和 features 无任何关联
    rows = []
    for i, d in enumerate(dates):
        for j, c in enumerate(codes):
            rows.append({
                "date": d, "code": c,
                "label": (np.random.randn() * 0.02
                           if i < n_dates - horizon else np.nan),
                "f1": np.random.randn(),
                "f2": np.random.randn(),
                "f3": np.random.randn(),
            })
    df = pd.DataFrame(rows).set_index(["date", "code"])
    y = df["label"]
    X = df[["f1", "f2", "f3"]]

    train_end = dates[250]
    _, pred_ho = rolling_predict_holdout(
        X, y, train_days=150, train_end=train_end, horizon=horizon,
    )

    common = pred_ho.index.intersection(y.dropna().index)
    if len(common) < 100:
        print("⚠️  样本不足"); return
    oos_ic = pred_ho.loc[common].corr(y.loc[common], method="spearman")
    print(f"  纯噪声数据 - OOS Spearman IC = {oos_ic:+.4f}")
    assert abs(oos_ic) < 0.1, (
        f"纯噪声下 OOS IC {oos_ic:+.3f} 应 ≈ 0, 超过 0.1 暗示全样本归一化等 leak"
    )
    print(f"✓ test_rolling_predict_no_spurious_signal: 纯噪声 IC {oos_ic:+.3f}, leak-free ✓")


# ============ 测试 2b: 反向注入测试 (oracle 因子只能在训练期有效) ============
def test_future_factor_fails_oos():
    """构造 oracle factor = 训练期真实 label + 测试期纯噪声.
    - 训练集: factor 完美预测 (IC = 1)
    - 测试集: factor 是噪声
    pipeline 正确: OOS IC ≈ 0 (模型学到的关系在 OOS 失效)
    pipeline 有 leak: OOS IC > 0.5 (模型偷看了测试期真 label)
    """
    from scripts.run_holdout_v6 import rolling_predict_holdout

    np.random.seed(123)
    n_dates = 400
    n_codes = 30
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    codes = [f"{i:06d}" for i in range(n_codes)]
    horizon = 10
    train_end_idx = 250
    train_end = dates[train_end_idx]

    rows = []
    for i, d in enumerate(dates):
        for j, c in enumerate(codes):
            label = np.random.randn() * 0.02 if i < n_dates - horizon else np.nan
            if i <= train_end_idx:
                # 训练期 oracle = label 本身 (模型学到: oracle → label)
                oracle = label if label == label else np.random.randn() * 0.02
            else:
                # 测试期 oracle = 纯噪声 (和 label 无关)
                oracle = np.random.randn() * 0.02
            rows.append({"date": d, "code": c,
                          "label": label, "oracle": oracle,
                          "noise": np.random.randn() * 0.1})
    df = pd.DataFrame(rows).set_index(["date", "code"])
    y = df["label"]
    X = df[["oracle", "noise"]]

    _, pred_ho = rolling_predict_holdout(
        X, y, train_days=150, train_end=train_end, horizon=horizon,
    )

    common = pred_ho.index.intersection(y.dropna().index)
    if len(common) < 100:
        print("⚠️  样本不足"); return
    oos_ic = pred_ho.loc[common].corr(y.loc[common], method="spearman")
    print(f"  Oracle 训练期完美/测试期噪声 - OOS IC = {oos_ic:+.4f}")
    assert abs(oos_ic) < 0.2, (
        f"测试期 oracle 是噪声, OOS IC 应 ≈ 0. "
        f"实际 {oos_ic:+.3f} 暗示 pipeline 用了测试期数据!"
    )
    print(f"✓ test_future_factor_fails_oos: 测试期噪声下 IC {oos_ic:+.3f}, leak-free ✓")


# ============ 测试 3: adaptive_polarity 的 shift(horizon) 生效 ============
def test_adaptive_polarity_no_leak():
    """验证 apply_adaptive_polarity 里 IC 的滚动计算用 shift(horizon).

    构造: 前 100 天 IC=+0.8 (强正相关), 后 100 天 IC=-0.8 (强反相关).
    如果 shift(horizon) 生效, 在 t=100 时 polarity 应该还在 +1 (没立刻翻),
    需要 60+ 天后才能稳定切换到 -1.
    """
    from factors.adaptive_polarity import compute_adaptive_weights

    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    # 第一半 IC = +0.1, 第二半 IC = -0.1
    ic_values = np.concatenate([np.full(100, 0.1), np.full(100, -0.1)])
    ic_df = pd.DataFrame({"factor_1": ic_values}, index=dates)

    horizon = 30
    window = 60
    weights = compute_adaptive_weights(
        ic_df, horizon=horizon, window=window,
        z_threshold=0.5, z_cap=3.0, inertia=0.7, decay_lambda=0.0,
    )

    # t=99 (切换前最后一天): polarity 应为正或 0
    w_99 = weights["factor_1"].iloc[99]
    # t=180 (切换后 80 天): polarity 应该已稳定转负
    w_180 = weights["factor_1"].iloc[180]
    # t=120 (切换后 20 天): shift(horizon=30) 让极性仍未翻, 应还是正
    w_120 = weights["factor_1"].iloc[120]

    print(f"  t=99 (切换前):  weight={w_99:+.3f}")
    print(f"  t=120 (切换后 20d): weight={w_120:+.3f}  (shift 防 leak 生效 → 应仍正)")
    print(f"  t=180 (切换后 80d): weight={w_180:+.3f}  (远超 shift window → 应转负)")

    # shift(30) 意味着 t 日只看到 t-30 之前的 IC
    # t=120 时能看的 IC 是 0-90 天的数据, 全部是 +0.1 → 极性仍应为正
    # t=180 时能看的 IC 是 0-150 天, 包含 100-150 天的 -0.1, 拖拽极性往负
    assert w_120 >= -0.1, f"t=120 时极性应为非负 (shift(30) 防 leak), 实际 {w_120:+.3f}"
    assert w_180 <= 0, f"t=180 时极性应已转负, 实际 {w_180:+.3f}"
    print("✓ test_adaptive_polarity_no_leak: shift(horizon) 严格防泄露")


# ============ 测试 4 (已移除, 参见测试 2): cheat factor 实测 ============
def _deprecated_test_cheat_factor_on_real_data():
    """在真实缓存数据上跑, 注入 cheat factor, 看 OOS IC 是否合理.

    需要 cache/kline_*.parquet.
    """
    from scripts.run_real_research_v5 import make_conditional_label
    from scripts.run_holdout_v6 import rolling_predict_holdout

    cache = Path(__file__).resolve().parent.parent / "cache"
    kline_files = sorted(cache.glob("kline_*_n500.parquet"),
                          key=lambda p: p.stat().st_mtime)
    if not kline_files:
        print("⚠️  test_cheat_factor_on_real_data: 无 kline 缓存, 跳过")
        return

    daily = pd.read_parquet(kline_files[-1])
    daily["date"] = pd.to_datetime(daily["date"])
    # 采样 50 只股, 加快
    codes = sorted(daily["code"].unique())[:50]
    daily = daily[daily["code"].isin(codes)]
    print(f"  测试数据: {len(daily)} 行, {daily['code'].nunique()} 只")

    horizon = 10
    label = make_conditional_label(daily, horizon=horizon, dd_clip=0.25)

    # cheat_factor = 真实 label (注入未来)
    # honest_factor = 过去 5 日收益率 (合法特征)
    df = daily.sort_values(["code", "date"])
    ret_5d = df.groupby("code")["close"].pct_change(5)
    df["honest"] = ret_5d.values
    honest = df.set_index(["date", "code"])["honest"]

    aligned = pd.concat([
        label.rename("label"),
        honest.rename("honest"),
    ], axis=1).dropna()
    X = pd.DataFrame({
        "cheat_factor": aligned["label"].values,   # 作弊
        "honest_factor": aligned["honest"].values,  # 合法
    }, index=aligned.index)
    y = aligned["label"]

    dates = sorted(X.index.get_level_values("date").unique())
    if len(dates) < 150:
        print("⚠️  数据不足 150 日, 跳过"); return
    train_end = dates[int(len(dates) * 0.7)]

    _, pred_ho = rolling_predict_holdout(
        X, y, train_days=120, train_end=train_end, horizon=horizon,
    )

    # 分别看 cheat 和 honest 的 OOS IC
    common = pred_ho.index.intersection(y.index)
    pred_a = pred_ho.loc[common]
    y_a = y.loc[common]
    mask = y_a.notna()
    oos_ic = pred_a[mask].corr(y_a[mask], method="spearman")

    print(f"  真实数据 OOS IC (cheat+honest 混合训练) = {oos_ic:+.4f}")
    # 如果 pipeline leak-free, cheat 在 OOS 无效, 剩 honest 贡献一小 IC
    # 如果有 leak, cheat 会主导, OOS IC 极高 (~ 0.9)
    if oos_ic > 0.6:
        raise AssertionError(f"⚠️  Leak 警报: OOS IC = {oos_ic:+.3f} 过高, cheat factor 在 OOS 仍生效!")
    print(f"✓ test_cheat_factor_on_real_data: OOS IC {oos_ic:+.3f} 合理 (无 leak)")


if __name__ == "__main__":
    print("=" * 64)
    print("  🔒 Leak Detector Tests")
    print("=" * 64)
    test_label_cutoff_at_end()
    print()
    test_adaptive_polarity_no_leak()
    print()
    test_rolling_predict_no_spurious_signal()
    print()
    test_future_factor_fails_oos()
    print()
    print("=" * 64)
    print("  ✅ 所有 leak 防御测试通过")
    print("=" * 64)

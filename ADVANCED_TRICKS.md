# 量化暗门大全 - 大众量化师不知道但头部私募在用的 15 个技巧

> 本文档汇总 A股头部量化机构 (幻方、九坤、明汯、灵均、诚奇等) 在实盘中
> 使用但很少公开讨论的技术细节. 所有内容基于公开文献 / 行业访谈 /
> 从业者经验, 可验证、可落地.

---

## 📚 已实现模块对应表

| 暗门 | 模块 | 核心文件 |
|---|---|---|
| #1 标签工程 | `label_engineering/` | `horizons.py`, `masks.py` |
| #2 市场微结构 | `market_microstructure/` | `impact.py`, `order_flow.py`, `spread_factors.py` |
| #3 因子衰减监控 | `alpha_decay/` | `monitor.py`, `crowding.py` |
| #4 事件屏蔽 | `corporate_actions/` | `event_mask.py`, `event_factors.py` |
| #5 Barra 中性化 | `barra_neutralize/` | `style_factors.py`, `neutralize.py` |
| #6 组合优化 | `portfolio_opt/` | `risk_parity.py`, `vol_targeting.py`, `mvo.py` |

---

## 🔍 15 个圈内 tricks 详解

### ① 标签工程 比 因子工程 重要 10 倍

**大众**: `y = close[t+5]/close[t+1] - 1`

**真相**:
- `close/close` 包含隔夜跳空噪声, 高 beta 股 label 被放大
- 今收决策, 次开入场 → 应该用 `close[t+5]/open[t+1]`
- 原始收益 label 会让模型把 "波动大" 误认为 "alpha 强"
- 需要 ATR / vol 归一化, 夏普视角 > 收益视角
- 多 horizon 加权 (1d × 0.4 + 3d × 0.3 + 5d × 0.2 + 10d × 0.1) 减少单点噪声
- 截面排名 CSRank 抑制 headline 异动

**实现**: `label_engineering/horizons.py`

---

### ② 涨跌停 / 停牌 / 公告 样本 必须屏蔽

**大众**: 全量训练数据.

**真相**:
回测 IC 会系统性虚高 20-30% (幻方内部估计), 实盘暴雷. 原因:
- 涨停当天买不到 (无卖单), 但回测按成交价计入
- 停牌日价格冻结, 恢复后跳空被当成"信号"
- 财报披露前 2 天常有信息泄露, 模型学到的是泄露
- 解禁日前 10 天已经被内部人提前反应, 模型看到是"均线破位"

**实现**: `label_engineering/masks.py`, `corporate_actions/event_mask.py`

---

### ③ Level2 真正的 alpha 是微结构, 不是"更快的 K 线"

**大众**: Level2 = 看得更快.

**真相**: 公开 tick-level alpha:
- **OIR (Order Imbalance Ratio)**: 短期价格预测王牌
- **VPIN**: 诺贝尔奖级预警工具, 闪崩前 2 周准确 > 90%
- **撤单率 > 0.7**: 疑似幌骗 (spoofing), 该股异常
- **Kyle's Lambda**: 单位订单的价格冲击, 流动性深度代理
- **Micro-price**: 按挂单量加权的中间价, 比 (bid+ask)/2 精确

**实现**: `market_microstructure/order_flow.py`, `spread_factors.py`

---

### ④ 冲击成本 不是 "固定 2 bps"

**大众**: `cost = 0.0002 × notional`

**真相**: Almgren-Chriss + sqrt 律:
```
cost_bps = k × σ × sqrt(Q / V) × 10000
```
- 茅台 1 亿 (日成交 50 亿): ~5 bps
- 小盘 1 亿 (日成交 3 亿): ~50 bps
- 固定滑点**低估小盘 10 倍**, 策略实盘立崩

**实现**: `market_microstructure/impact.py`

---

### ⑤ Alpha 都会衰减, 区别只是快慢

**大众**: 训完模型就不管.

**真相**: 头部私募每日跟踪:
- `rolling_IC / all_time_IC 比值`, < 0.5 立即下线
- 半衰期估计 (指数拟合), A股 alpha 平均半衰期 6-9 个月
- 公开论文 / 研报的因子, 18 月内一定失效 (Kakushadze 2019)

**实现**: `alpha_decay/monitor.py`

---

### ⑥ 因子拥挤度 - 为什么你的 alpha 突然不赚钱了

**大众**: 以为自己的 alpha 独特.

**真相**: A股量化规模 2019 → 2024 增长 10 倍 (2000亿 → 2 万亿).
- Turnover 上升 + IC 下降 = 铁证拥挤
- 融券余额变化 = 反向资金规模
- 公开策略 6 个月内被稀释
- 实盘 IC 比回测 IC 普遍低 30-40%

**监控指标**:
```python
crowding = (short_turnover / long_turnover) / |(short_ret / long_ret)|
# > 1.5: 警报
# > 2.0: 立刻降仓
```

**实现**: `alpha_decay/crowding.py`

---

### ⑦ Barra 风格因子 中性化 (大众做假中性化)

**大众**: 减去市值 + 行业 dummy (2 维).

**真相**:
```
alpha_raw = β_1×Size + β_2×Beta + β_3×Momentum
          + β_4×ResVol + β_5×Liquidity + β_6×NonLinSize
          + Σ β_ind×Industry + ε

alpha_clean = ε  ← 只有残差才是真 alpha
```
不做透 Barra 中性化, 你赚的是 **风格轮动**, 不是 alpha.

**实现**: `barra_neutralize/style_factors.py`, `neutralize.py`

---

### ⑧ 多 Alpha 组合分散, 单因子生命周期 6-18 月

**大众**: 找一个神 alpha 长期打工.

**真相**: 所有 alpha 都会死, 但相关性 < 0.3 的多个 alpha 组合:
- 组合半衰期无穷大
- 分散后夏普 √N 倍增长
- 头部: 一个 pod 5 个正交因子, 两两 corr < 0.3

**实现**: `alpha_decay/crowding.py::alpha_portfolio_correlation`

---

### ⑨ 风险平价 vs 等权重

**大众**: Top-20 等权.

**真相**:
- 等权让高波动股票主导组合波动 (常被忽视)
- 风险平价: 每只票对总风险贡献相等
- Bridgewater 全天候策略 10 年夏普 1.3
- A股实证: 风险平价 vs 等权, 夏普 +0.3, 最大回撤 -5%

**实现**: `portfolio_opt/risk_parity.py`

---

### ⑩ 波动率目标 (Vol Targeting)

**大众**: 固定仓位.

**真相**: 组合年化波动率锚定 15%:
- 波动上升自动减仓, 避免危机放大
- 波动下降自动加仓, 吃到稳定 alpha
- 夏普普遍提升 20-30% (Moreira & Muir 2017, JFE)

**实现**: `portfolio_opt/vol_targeting.py`

---

### ⑪ Black-Litterman 贝叶斯先验融合

**大众**: 直接用 alpha 信号.

**真相**: 你的 alpha 信号总是有噪声, 和市场先验 (CAPM 隐含收益) 贝叶斯融合:
```
posterior_μ = f(market_equilibrium_prior, your_views, your_confidence)
```
- 稳定性暴增 (不会因单点噪声暴仓)
- 高盛/桥水都在用
- 特别适合 LLM agent 输出的 view

**实现**: `portfolio_opt/mvo.py::black_litterman_posterior`

---

### ⑫ 解禁 / 大宗折价 / 高管减持 是明牌空头

**大众**: 忽视这些.

**真相**:
- 解禁前 10 天价格已经反应, 解禁后 3 天集中抛压
- 大宗折价 > 3% = 内部人套现, 之后 60 天负收益概率 65%+
- 高管 3 人集中减持 (同月) = 预警信号, 未来 60 天跑输大盘 15%

**实现**: `corporate_actions/event_factors.py`

---

### ⑬ 开盘 10 分钟 + 收盘 15 分钟 不交易

**大众**: 全天 TWAP.

**真相**:
- 开盘前 10 分钟波动极大 (开盘 auction 释放)
- 收盘前 15 分钟流动性差 + 机构对倒, 冲击成本翻倍
- 最优交易时段: **10:00-11:30 + 13:30-14:30**
- 高频机构 (jump/renaissance) 都避开极端时段

**建议**: 加入 `execution/time_windows.py` (未来扩展)

---

### ⑭ 数据清洗 的 17 个陷阱 (节选)

1. **幸存偏差**: Wind/聚源历史股票池只包含现存公司, 退市的消失 → IC 虚高
2. **前视偏差**: 财报"报告期"作为可用时间 → 错误, 应用"披露日"
3. **复权因子**: 后复权历史价 vs 实盘前复权, 分布偏移严重
4. **代码变更**: 000022 改为 001872 等, 不处理导致断档
5. **送股配股**: 不做除权调整, 错误触发止损
6. **港股 ADR**: 汇率对齐 / 节假日差异
7. **指数调仓**: 纳入/剔除时点影响策略评估
8. **停牌恢复**: 首日不应作为样本 (定价失真)
9. **Level2 NATS 协议**: 开盘延迟 10-20s, 别用头 5 分钟数据
10. **数据源时钟**: akshare / tushare 服务器时间 vs 交易所时间对齐

**建议**: 新建 `data_hygiene/` 模块, 逐条扫描

---

### ⑮ 资金流的 "前瞻" vs "滞后"

**大众**: 盯北向资金.

**真相**:
- **北向是盘后数据**: 收盘后公布, 次日已经 price-in, 开盘就没用
- **真正前瞻**:
  - 两融余额**变化率** (而非余额水平)
  - 大宗交易**笔数趋势** (机构出货节奏)
  - ETF **份额变化** (基金净申购代理)
  - 期权 **Put/Call Ratio 的变化率**
  - IV skew (OTM put IV - OTM call IV) 上升 → 恐慌前瞻
- **同业持仓**: Wind 能看到公募季报, 但 "同一拥挤" 提前 30 天要靠资金流

---

## 📊 上线 checklist

实盘前必须过完:

- [ ] 回测是否屏蔽 涨跌停 / 停牌 / 财报窗口 / 新股 / ST ?
- [ ] 冲击成本用 sqrt 律, 不是固定 bps ?
- [ ] 风格中性化用 Barra 10 因子, 不是 Size + Industry ?
- [ ] 样本外回测至少 6 个月 ?
- [ ] rolling_IC / all_time_IC > 0.7 (最近 60 天) ?
- [ ] 组合日换手率 < 60% (避免手续费吞噬) ?
- [ ] 单票最大仓位 < 5%, 单行业 < 25% ?
- [ ] 波动率目标 + 回撤熔断 (15% limit) ?
- [ ] 因子与公开因子 (FF5) 相关性 < 0.4 ?
- [ ] 盘后资金流因子**不能**在盘中用 ?
- [ ] 解禁日 / 财报日 前后 2 天不新开仓 ?
- [ ] Level2 订单流因子仅用于 T+1 以下频率 ?

---

## 🔬 进一步学习

**论文**:
- Almgren, Chriss (2000) - Optimal Execution of Portfolio Transactions
- Easley, López de Prado, O'Hara (2012) - Flow Toxicity and Liquidity
- Moreira, Muir (2017) - Volatility-Managed Portfolios
- Kakushadze (2016) - 101 Formulaic Alphas (WorldQuant)
- Barra (2012) - CNE5 China Equity Risk Model

**书籍**:
- López de Prado - *Advances in Financial Machine Learning* (2018)
- Gappy - *Quantitative Portfolio Management* (2020)
- Grinold & Kahn - *Active Portfolio Management* (2000)

**代码学习**:
- WorldQuant BRAIN 平台公开的 alpha 101
- QuantConnect Lean 引擎 (冲击成本建模优秀)
- Zipline (但已停更)

---

## ⚠️ 免责声明

所有技巧基于历史数据与公开研究. 量化投资有风险, 实盘前必须:
1. 至少 6 个月样本外验证
2. 3 个月模拟盘
3. 小资金起步 (< 预计规模 10%)
4. 严守风控规则

本文档作为教育资源, 不构成投资建议.

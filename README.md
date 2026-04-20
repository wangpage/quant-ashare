# quant-ashare

> **机构级 A股量化交易系统** - 融合 Hermes Agent 多智能体决策、Barra 风格因子中性化、Level2 微结构 alpha、Almgren-Chriss 冲击成本建模的开源实现.

[![Tests](https://img.shields.io/badge/numerical%20tests-65%2F65-brightgreen)](tests/test_numerical_correctness.py) [![Smoke](https://img.shields.io/badge/smoke%20tests-~120-blue)](tests/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

不是又一个抄 Alpha158 的项目. 本项目致力于实现 **头部私募在用但很少公开的圈内技巧**, 覆盖从数据清洗到执行层的完整量化链路.

**当前状态**: 已实装 9 个核心模块, 规划 15 个. 实装清单见 [功能矩阵](#-已实装-vs-规划功能矩阵), 详解见 [ADVANCED_TRICKS.md](ADVANCED_TRICKS.md).

---

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────┐
│  LLM 决策层 (Hermes XML + 多轮辩论)                       │
│  [基本面] [技术] [情绪] → [Bull vs Bear] → [交易] → [风控] │
├─────────────────────────────────────────────────────────┤
│  qlib 核心 + Alpha158 + 35 A股特化因子                   │
├─────────────────────────────────────────────────────────┤
│  【圈内暗门 9 大模块 - 已实装】                             │
│  标签工程 / 微结构因子 / 因子衰减监控 / 事件屏蔽              │
│  Barra 中性化 / 组合优化 / 数据清洗 / 主题投资 / 执行层       │
├─────────────────────────────────────────────────────────┤
│  数据源: akshare + 东财/新浪直连 + Level2 NATS            │
├─────────────────────────────────────────────────────────┤
│  风控: 涨跌停 / T+1 / 凯利 / 回撤熔断 / Regime 乘数        │
└─────────────────────────────────────────────────────────┘
```

## ✨ 核心特性

### 🧠 多智能体 LLM 决策
- Hermes-3 风格 **XML 结构化推理** (`<THINKING>`, `<REASONING>`, `<REFLECTION>`)
- **Bull vs Bear 多轮辩论**, 由 Judge agent 最终裁决
- 三分析师并行 (asyncio), 延迟降 3x
- 支持 **Claude / GPT / DeepSeek / Kimi / Qwen / GLM** 统一后端

### 📊 已实装 vs 规划功能矩阵

✅ = 已有代码 + 测试覆盖; 🚧 = 规划 / 骨架; ❌ = 仅文档提及

| # | 技巧 | 状态 | 大众盲点 → 本项目实现 |
|---|---|:---:|---|
| 1 | 标签工程 | ✅ | `close/close` → 多 horizon + vol 归一化 + CSRank + **停牌/一字板屏蔽** |
| 2 | 涨跌停 / 停牌屏蔽 | ✅ | 全量训练 → 自动屏蔽 (回测 IC 虚高 20-30%) |
| 3 | Level2 微结构 alpha | ✅ | OIR / **VPIN** / Cancel Ratio / Kyle's λ |
| 4 | 冲击成本建模 | ✅ | 固定 2bps → **Almgren-Chriss + sqrt 律** |
| 5 | Barra 风格中性化 | ✅ | Size+Industry → **CNE5 六风格因子** |
| 6 | 因子衰减监控 | ✅ | 训完不管 → rolling IC 自动下线 |
| 7 | 组合优化 | ✅ | Top-K 等权 → **风险平价 + Black-Litterman** |
| 8 | 事件屏蔽 | ✅ | 无 → 解禁/财报/大宗自动屏蔽 |
| 9 | 数据清洗 | ✅ | 幸存者 / 前视 / 复权 (含**小幅分红/累计漂移**) |
| 10 | 主题投资识别 | ✅ | 纯涨幅 → 萌芽/扩散/拥挤三阶段 + 龙头排序 |
| 11 | Regime 自适应 | 🚧 | 8 种市场状态 + 仓位乘数 (骨架已有) |
| 12 | Level2 时序保护 | ✅ | 指数退避 / 时钟漂移监控 / 乱序检测 |
| 13 | 资金流前瞻 | 🚧 | 主力资金 + 北向资金领先信号 (规划中) |
| 14 | 涨停连板因子 | ❌ | 连板高度 + 炸板概率 (规划中) |
| 15 | 龙虎榜游资特征 | ❌ | 知名营业部联动图 (规划中) |

详解见 [ADVANCED_TRICKS.md](ADVANCED_TRICKS.md). 进度会随每次 release 更新.

### 🧬 记忆与自进化
- **Memory Curator**: 每笔交易后 LLM 提炼反思存 SQLite + FTS5
- **Skill Factory**: 自动从成功交易聚类生成复用规则
- **Weekly Nudge**: 每周合并去重记忆

### 🚀 端到端 Pipeline
- **ResearchPipeline**: 数据体检 → 标签 → 屏蔽 → Barra → IC → 带冲击回测
- **DailyTradingPipeline**: Regime → 筛选 → Agent → 风控 → 冲击路由

---

## 📈 真实回测 (最简基线)

**2024-01-01 ~ 2026-04-20**, 10 只蓝筹股 (宁德/茅台/五粮液/平安/招行/比亚迪...) + 仅 3 个示例因子 + Top-10 等权:

| 指标 | 值 |
|---|---|
| 年化收益 | **11.03%** |
| 最大回撤 | **-8.55%** |
| 夏普比率 | **0.79** |
| 换手率 | 0.45% |

> 这是**最弱基线**. 上 Alpha158 + Barra 残差 + 风险平价组合预期夏普 1.5-2.0.

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/wangpage/quant-ashare.git
cd quant-ashare
pip install -r requirements.txt
```

### 🌐 启动 Web UI (推荐)

最快看到项目效果的方式. 自带 4 个交互页面:
- **📡 今日信号仪表盘** - 大盘 regime + 主题评分 + top-K 信号 + 资金分配
- **🔍 个股详情** - K 线 / 因子画像 / 主题归属
- **🧠 Agent 辩论记录** - 三分析师 + Bull vs Bear + 交易员 + 风控
- **📈 回测与策略曲线** - 净值 / 回撤 / IC 衰减 / Barra 分解

```bash
# 默认 Mock 模式, 无需任何配置即可体验
bash scripts/run_webapp.sh
# 浏览器打开 http://localhost:8501

# 接真实数据 (需要配置 akshare / LLM API)
QUANT_WEB_MODE=real bash scripts/run_webapp.sh
```

![webapp 架构](./README.md)

### 🧪 测试诚实说明

测试分两层, 不要混淆:

| 类型 | 文件 | 数量 | 说明 |
|---|---|---|---|
| **数值正确性 (单元)** | `tests/test_numerical_correctness.py` | **65** | 真数值断言: IC/ICIR 精确匹配, 最大回撤闭式公式, ATR 止损 3 种情况, Barra 残差正交性 (corr<0.05), AC 冲击 sqrt 律单调性, VPIN/OIR 边界, URL 占位符识别, Regime 崩盘/狂热识别, alpha 衰减监控单调性 |
| **冒烟/契约 (集成)** | 其他 `tests/test_*.py` | ~120 | 主要验证"非空 / 字段存在 / 函数可跑通", 不做数值校对 |

```bash
# 数值正确性 (最该看的)
python3 tests/test_numerical_correctness.py           # 65/65

# 冒烟测试
python3 tests/test_parser.py                          # CSV 解析 60/60
python3 tests/test_phase1_xml.py                      # Hermes XML 34/34
python3 tests/test_phase2_memory_regime.py            # Memory+Regime 35/35
python3 tests/test_advanced_tricks.py                 # 6 暗门 55/55
```

**不再声称 "219/219 passing"** — 之前 README 的数字把冒烟测试的断言累加 (40+ 个 `_a` 断言/文件) 算成通过率, 现改为按**测试函数**口径: 共 46 个 `def test_*`, 其中 10 个含**真数值断言** (本次新增).

### 跑真实数据 Research Pipeline

```bash
# 拉 10 只蓝筹股 + 2 年数据 + 跑完整 pipeline (含 Barra)
python3 scripts/run_real_research.py --n 10 --start 20240101 --end 20260420

# 输出: annual_return, sharpe, max_drawdown + 完整报告
# 保存: output/research_YYYYMMDD_HHMM.md
```

### LLM Agent 决策 (以 DeepSeek 为例, 便宜)

```bash
# 1. 注册 https://platform.deepseek.com, 充 5 元
# 2. 设环境变量
export DEEPSEEK_API_KEY="sk-xxxxx"

# 3. 跑决策
python3 -c "
import asyncio
from llm_layer import TradingAgentTeam
team = TradingAgentTeam(
    backend='deepseek',
    analyst_model='deepseek-chat',
    researcher_model='deepseek-reasoner',
    trader_model='deepseek-chat',
    risk_model='deepseek-chat',
)
# ... 详见 llm_layer/agents.py
"
```

### Level2 实时行情接入

对接 base32.cn (测试账号公开):

```bash
# 盘中 (09:30-15:00) 运行
python3 tests/test_realdata_intraday.py
```

详细计划见 [LEVEL2_LIVE_TEST_PLAN.md](LEVEL2_LIVE_TEST_PLAN.md).

---

## 📁 项目结构

```
quant-ashare/
├── config/                    # YAML 配置
├── data_adapter/              # akshare + 东财/新浪直连
├── data_hygiene/          ⭐ 数据清洗暗门 (幸存/前视/复权/停牌)
├── factors/                   # A股特化因子 35 个
├── label_engineering/     ⭐ 标签工程 (多 horizon + vol 归一化)
├── market_microstructure/ ⭐ OIR / VPIN / Almgren-Chriss
├── corporate_actions/     ⭐ 解禁/财报/大宗事件
├── barra_neutralize/      ⭐ CNE5 六风格因子
├── alpha_decay/           ⭐ IC 衰减 + 拥挤度监控
├── portfolio_opt/         ⭐ 风险平价 + MVO + Black-Litterman
├── execution/             ⭐ TWAP/VWAP + 冲击感知 + 时段避让
├── pipeline/              ⭐ 研究 + 实盘端到端
├── market_regime/            # 8 种市场状态分类器
├── memory/                   # 交易记忆 + Skill Factory
├── llm_layer/                # Hermes XML + 多智能体
├── risk/                     # A股风控
├── level2/                   # NATS Level2 接入
├── scripts/                  # 一键脚本
├── tests/                    # 219 测试用例
└── ADVANCED_TRICKS.md    ⭐ 15 个圈内 tricks 详解
```

---

## 🔬 核心技术文档

| 文档 | 内容 |
|---|---|
| [ADVANCED_TRICKS.md](ADVANCED_TRICKS.md) | 15 个头部私募在用的 tricks + 上线 checklist |
| [LEVEL2_LIVE_TEST_PLAN.md](LEVEL2_LIVE_TEST_PLAN.md) | Level2 NATS 盘中接入作战手册 |
| [tests/LEVEL2_GUIDE.md](tests/LEVEL2_GUIDE.md) | 本地 NATS + 生产环境接入指南 |

---

## 🧪 测试成绩 (诚实口径)

```
test_numerical_correctness (真数值断言):     65/65 ✓  ← 看这个
test_parser (CSV 字段契约):                  60/60 ✓
test_advanced_tricks (暗门模块接口):         55/55 ✓
test_phase1_xml (XML 解析):                  34/34 ✓
test_phase2_memory_regime (记忆+regime):    35/35 ✓
──────────────────────────────────────────────
test_* 函数总数: 46 个
  - 真数值断言: 10 个 (IC/回撤/Barra残差/ATR/AC冲击/URL validator 等)
  - 冒烟/契约: 36 个
```

**更新记录**:
- v0.2 (2026-04): 接入真实 `RegimeDetector` (data_providers); NATS URL fail-fast 校验;
  新增 10 个真数值断言测试; README 测试数字诚实化
- v0.1 初版: 早期 README 声称 "219/219" 是把冒烟测试中的小断言 `_a(...)`
  累加而得, 单元测试函数实际只 36 个, 且几乎无数值正确性校验. 已修正.

---

## 🛡️ 免责声明

**本项目仅供学习与研究**, 不构成任何投资建议.

量化投资存在**实质性风险**. 任何 "95% 胜率"、"夏普 5+" 的宣传都是**过拟合或话术**.
真实量化 alpha 的核心是 (胜率-50%) × 盈亏比 × 高频次, 不是单次准度.

实盘前必须:
1. ✅ 至少 6 个月样本外回测
2. ✅ 3 个月模拟盘验证
3. ✅ 从小资金起步 (< 预计规模 10%)
4. ✅ 严守风控规则 (最大回撤 / 熔断 / 止损)

---

## 🙏 致谢的开源项目

本项目学习了以下优秀项目的思想:
- [microsoft/qlib](https://github.com/microsoft/qlib) - AI 量化平台基石
- [akfamily/akshare](https://github.com/akfamily/akshare) - A股免费数据源
- [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) - XML 结构化推理
- [HKUDS/AI-Trader](https://github.com/HKUDS/AI-Trader) - 多智能体辩论
- [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL) - 强化学习金融
- [hsliuping/TradingAgents-CN](https://github.com/hsliuping/TradingAgents-CN) - 中文金融 agent

---

## 📮 贡献

欢迎 Issue / PR. 特别欢迎:
- 新的 A股特化因子实现
- 头部私募已公开的 tricks 补充
- 性能优化 (向量化 / Cython)
- 更多后端 LLM 支持 (本地 llama / vLLM)

## 📄 License

[MIT](LICENSE) - 详见 LICENSE 文件, 含金融软件专属免责条款.

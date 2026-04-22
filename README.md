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

## 📈 真实回测 — 含严格 OOS 修正记录

> ⚠️ **重要诚实声明**: 早期版本 (V3-V5) 的"夏普 1.86 / IR 1.21"包含了 **训练集 label 泄露**(label horizon=30, 但训练截止 = train_end, 后 30 天 label 在真实时点不可观测). **修复 leak 后** OOS IR 跌至 **-0.13 ~ +0.2** 之间. **本项目当前定位为研究框架, 不是已验证的实盘策略.**

### 演化记录(全部 walk-forward 严格 OOS)

| 版本 | 因子构成 | Hold-out IR(干净) | 关键发现 |
|---|---|---|---|
| V1 baseline | 10 大蓝筹 + 3 玩具因子 | 0.79(其实是纯 beta) | 蓝筹无 alpha,Sharpe 来自市场 |
| V3 | 500 中小盘 + 龙虎榜 5 因子 | — | 含 leak,数字不再呈报 |
| V4 | + IC 聚类去共线 + 动态滑点 | — | 框架修正 |
| V5 | + 涨停/连板 + 高管增减持 | — | 含 leak |
| V6 (B+ 自适应极性) | + IC 时序 z-score 极性切换 | **+0.96 → 修复后 -0.13** | leak 暴露 |
| V7 (+ regime 因子) | + 大盘动量/广度 | -1.38 | 加 regime 反加剧过拟合 |
| V8 (+ 主力资金流) | + 超大单/大单/散户净流入 | **-0.75** | 牛市 regime 中无效 |

### 为什么"漂亮数字"全部作废

**走完一遍后的真相**:
1. **30 日 horizon 的策略,模型必然滞后 1-2 个月** — 真实场景下 train_end 时只能看到 train_end - 30 之前的 label
2. 之前的 V3-V6 都用了 `t ≤ train_end` 的全部 label,**含未来未观测信息**
3. 修干净后,**任何公开因子组合在 2025-2026 中小盘牛市的 OOS IR ≈ 0**
4. 同期 universe 等权 benchmark Sharpe **2.5-3.0**(极端 FOMO 牛市),**多头选股策略天生劣势**

### 真 alpha 在哪里(不再粉饰)

公开因子 + LightGBM + 月频选股,**真实 OOS 夏普天花板 0.5-1.0**. 想突破需要:
- ✅ Wind/Choice 付费数据(¥5-20 万/年)
- ✅ Level2 tick 数据(券商开户送) + 分钟级回测框架
- ✅ 自监督模型(Kronos 等)取代手工因子
- ✅ 等待 regime 切换(单边牛市里多头选股本就是 worst case)
- ❌ 继续堆公开因子 — 已被 4000+ 量化机构挖干净

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
python3 scripts/run_real_research_v5.py --n 10 --start 20240101 --end 20260420

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
├── analyst/              ⭐ 分析师推送层 (市场全景 + 因子聚合 → 飞书简报)
├── notifier/             ⭐ 飞书推送统一出口 (subprocess 封 lark-cli)
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

## 🔔 每日自动化 (Cron + 飞书通知)

**7 步流水线**(每日 15:30 自动跑,约 5-8 分钟完成, 见 [scripts/cron_daily.py](scripts/cron_daily.py)):

```
1/7 数据增量更新     (daily_data_updater.py)
2/7 Paper Trade     (paper_trade_runner.py)
3/7 Watchlist 信号   (watchlist_signal.py  ← 自选池 7 因子)
4/7 Git Commit/Push (同步 paper trade 产物到 GitHub)
5/7 账户通知         → 飞书 IM (NAV + 持仓 + 当日 P&L)
6/7 分析师简报       → 飞书 IM (市场全景 + 明日 Top3 + 昨日命中 + 风险)
7/7 批量扫描         → 飞书 IM (246 只候选池 + LLM Top20/Bottom10 推荐/回避)
```

任一环节失败会发错误告警到飞书,不阻塞后续步骤.

### 推送内容差异

| 步骤 | 目标 | 内容 |
|---|---|---|
| 5/7 账户通知 | 自我复盘 | NAV / 累计收益 / 当日 P&L / Top 5 持仓 |
| 6/7 分析师简报 | 明日决策(自选池) | 上证/深证/创业板涨跌 · 板块 Top5 · 昨日 IC · 自选池精选 Top3 + 止损止盈 + 风险点 |
| 7/7 批量扫描 | 全池排序(246 只) | 🟩推荐/🟢关注/⚪中性/🟡谨慎/🟥不推荐 桶化 · 量化 Top10+Bottom5 · LLM 精分析 Top20+Bottom10 |

### 核心脚本

- **[scripts/cron_daily.py](scripts/cron_daily.py)**: 7 步编排器, 支持 `--dry-run-lark` / `--skip-data` / `--skip-push` / `--test-notify` / `--leak-check` 各种调试模式
- **[scripts/watchlist_signal_v2.py](scripts/watchlist_signal_v2.py)**: 自选池 18 因子信号(反转 + 打板 + 席位 + 板块)
- **[scripts/batch_scan.py](scripts/batch_scan.py)**: 任意股票池扩池扫描
  ```bash
  python3 scripts/batch_scan.py                              # 默认 股票名称_代码.csv
  python3 scripts/batch_scan.py --codes-csv /path/to/pool.csv
  python3 scripts/batch_scan.py --top 20 --bottom 10         # LLM 精分析数量
  python3 scripts/batch_scan.py --no-llm                     # 纯量化模式
  ```
- **[notifier/dispatch.py](notifier/dispatch.py)**: 分析师简报推送入口
  ```bash
  python3 -m notifier.dispatch --date 2026-04-22             # 今日简报
  python3 -m notifier.dispatch --date 2026-04-22 --dry-run   # 不发飞书, 打印 Markdown
  ```

### 安装 cron

```bash
bash scripts/install_cron.sh              # 安装
bash scripts/install_cron.sh --dry-run    # 预览
bash scripts/install_cron.sh --uninstall  # 卸载
```

默认配置: 交易日 15:30 跑全流水线; 周日 20:00 跑 leak detector 自检.

### 飞书推送 - 身份配置

**重要**: 飞书不允许 `user` 身份给自己的 open_id 发 P2P 消息(API 成功但客户端不显示). 必须用 `bot` 身份(即自建应用机器人, 如本项目配的 `cccli`).

配置步骤:
1. 飞书开放平台 [open.feishu.cn](https://open.feishu.cn) 建自建应用(例如 `cccli`), 开通 `im:message` 权限
2. 在飞书客户端添加这个机器人为好友
3. 本地装 [lark-cli](https://github.com/larksuite/lark-cli)
   ```bash
   lark-cli config init       # 填 app_id / app_secret
   lark-cli auth login        # user 身份登录(交互式)
   lark-cli auth status       # 确认 valid
   ```
4. 设置推送目标(环境变量覆盖默认值):
   ```bash
   export LARK_USER_OPEN_ID=ou_xxxxx
   ```

**为什么用 `--as bot`**: lark-cli 在 user 身份下给自己的 open_id 发消息, 飞书客户端不展示; 改 `--as bot` 后消息以"机器人推送"形式出现在正常 IM 列表里. [notifier/feishu_client.py](notifier/feishu_client.py) 默认 `as_user=False`, 遇到 auth 过期返回 exit 10 让 cron 打标但不阻塞.

---

## 🔬 研究档案 (V6-V8 严格 OOS 实验)

**[scripts/run_holdout_v6.py](scripts/run_holdout_v6.py)**: B+ 自适应极性 (IC z-score + 显著性过滤 + 惯性 + 横截面归一化), 暴露了早期版本的 label leak.

**[scripts/run_holdout_v8.py](scripts/run_holdout_v8.py)**: V8 主力资金流驱动 - 集成超大单/大单/中单/小单 12 个资金流因子. 在 2025-2026 单边牛市里 IR 仍负 (-0.75), 证明: **多头选股策略在 FOMO 牛市天生劣势, 不是技术问题**.

**[factors/adaptive_polarity.py](factors/adaptive_polarity.py)**: 4+1 层防护的因子极性自适应:
1. IC 时序 z-score (置信度, 不是绝对值)
2. 显著性过滤 (|z|<0.8 weight=0)
3. 惯性 EMA (防 regime 抖动)
4. 横截面归一化 (Σ|w|=1)
5. IC 衰减加权 (近期权重大)

**[factors/alpha_regime.py](factors/alpha_regime.py)**: 大盘 regime 因子 (动量/广度/波动率), 时序 z-score, 不走截面 z.

**[factors/alpha_limit.py](factors/alpha_limit.py)**: 涨停/连板/炸板/一字板/启动期压缩 等 16 个 A股独有因子.

**[data_adapter/insider.py](data_adapter/insider.py)**: 高管/大股东增减持 36000+ 条事件.

**[data_adapter/fundflow.py](data_adapter/fundflow.py)**: 主力资金流 (akshare 个股近 120 日明细).

**[scripts/paper_trade_runner.py](scripts/paper_trade_runner.py)**: 每日自动化 paper trade - 维护真账户 + T+1 买卖 + 硬止损(亏 5%/破 MA5)+ 持仓档案.

---

## 🛡️ 免责声明

**本项目仅供学习与研究**, 不构成任何投资建议.

量化投资存在**实质性风险**. 任何 "95% 胜率"、"夏普 5+" 的宣传都是**过拟合或话术**.
真实量化 alpha 的核心是 (胜率-50%) × 盈亏比 × 高频次, 不是单次准度.

**本项目自身的诚实记录**: V3-V5 早期 README 写的"夏普 1.86 / IR 1.21"含 label leak,
修复后 OOS IR ≈ 0. 这恰好印证了上面那段话 — 任何"漂亮回测"都要严格 OOS 验证.

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

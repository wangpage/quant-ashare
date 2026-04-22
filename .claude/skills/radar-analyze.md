---
name: radar-analyze
description: Radar 事件深度复盘 - 从本地 memory.db 拉指定时间窗/代码/题材的 radar_event 和已写入的 triage+deep analysis, 结合用户 watchlist 和当前市场上下文, 做对话式量化复盘. 不是自动化流程, 是让用户"对某条/某批新闻做一次完整的人工推演"的交互式技能. 触发: "分析 radar 事件" / "深挖这条新闻" / "对今天下午的消息做复盘" / "看看 XX 题材最近的 radar" / "那批事件怎么看".
---

# Radar 事件深度复盘

## 适用场景

自动 daemon (`scripts/radar_worker.py`) 已经在后台为每条 radar_event 写入 `triage` + (对高价值事件的) `deep` analysis. 这个 skill 专门用于用户想**对某批或某条事件做进一步推演**的交互式工作流, 例如:

- "今天下午 14:00 之后的 radar 事件帮我复盘一下"
- "算力租赁这批消息我该怎么看"
- "600519 相关的 radar 最近都推了什么"
- "这条消息之前 daemon 怎么分析的, 和我想的不一样"

## 数据源

所有数据来自 `cache/memory.db` 的 `memories` 表 (`kind='radar_event'`), metadata 字段包含:
- `source` / `radar_id` / `score` / `tags` / `url` (原始数据)
- `analysis.triage.{event_type, entities, tradability, novelty, oneline}` (Haiku 产出)
- `analysis.deep.{thesis, targets, already_priced_in, evidence, supply_chain, risk_flags}` (Opus 产出, 仅部分事件)

## 工作流

### 第 1 步: 理解用户意图

明确以下一个或多个过滤条件, 不清楚就问:
- **时间窗**: 最近 N 小时 / 某日某时段 / 今天
- **标的**: 特定 code / 特定 watchlist 股票
- **来源**: cls / 同花顺 / 特定 source
- **题材**: 从 triage.entities 或 tags 过滤
- **事件类型**: single_name_catalyst / theme_momentum / policy_* 等

### 第 2 步: 从 DB 拉事件

用 Python 读 memory.db (不要直接用 sqlite3 CLI):

```python
import sys; sys.path.insert(0, '/Users/page/Desktop/股票/quant_ashare')
from memory.storage import MemoryStore
import time, json

store = MemoryStore()
# 时间窗示例: 最近 2 小时
since = int(time.time()) - 2 * 3600
events = store.query_radar_events(since_ts=since, limit=50)

# 按需过滤
relevant = []
for e in events:
    a = e.metadata.get('analysis', {})
    tr = a.get('triage', {})
    if tr.get('event_type') == 'noise':
        continue
    # 更多过滤: code, tradability, source ...
    relevant.append(e)
```

### 第 3 步: 拉当前市场上下文

对用户 watchlist 和 triage 抓到的 entities 都拉一下最新市场上下文:

```python
from llm_layer.market_context import build_context_text
ctx_by_code = {code: build_context_text(code) for code in codes_of_interest}
```

Watchlist 路径: `股票自选.csv` (一行一个 6 位 code) 或 `config/user_watchlist.yaml`.

### 第 4 步: 生成复盘

在回答用户的过程中, **你就是量化分析师**. 输出需要覆盖:

1. **事件分布扫描** (2-4 行): 总数 / 按 event_type 分档 / 高价值事件数
2. **关键事件列表** (Top 3-5):
   - 时间 / 来源 / 核心内容 / daemon 给的 triage + deep
   - 你对 daemon 分析的**认可或补充**: 是否漏掉了重要标的? 传导链有没有盲区?
3. **跨事件联动**: 多条事件指向同一题材/供应链时显式点出
4. **可操作候选** (最重要):
   - 带具体 code / 方向 / 进场逻辑 / 风险
   - 对照 watchlist 说明"这条新闻对你持仓的 XX 有什么影响"
   - **显式区分**: "已 price-in 的(避免追高)" vs "仍有空间的(可布局)" vs "风险事件(减仓)"
5. **盲区标注**: 没有 market_context 缓存的 code 要说明, 不要假装技术面判断

### 第 5 步: 迭代

用户会追问, 你继续深挖. 典型追问:
- "XX 这个标的为什么不在清单里 / 应该在清单里"
- "再看看二跳标的"
- "把这批事件 5 日之后的表现回测一下" (这个已超出 skill 能力, 转到 backtest 工具)
- "daemon 这条分析我不同意" → 让用户说为什么, 然后你独立用现有数据再推一遍

## 重要约束

- **不要调用 radar_worker**: 它是 daemon, skill 只读 DB
- **不要伪造数据**: 如果 analysis 字段为空 (daemon 还没处理到), 直接说"daemon 未分析"
- **不要编股票代码**: supply_chain 推理遵循常识, 但要验证 code 存在再列出
- **不要只复述 daemon 输出**: skill 的价值在于结合 watchlist 和用户意图做**用户专属的二次分析**, 否则用户直接看 daemon 输出就够了
- **输出长度**: 默认简洁(3 段以内), 用户要详细再展开

## 工作目录

默认在 `/Users/page/Desktop/股票/quant_ashare` 下操作. 如果 cwd 不对, 先 `cd` 或用绝对路径.

## 典型输入示例

- "/radar-analyze 今天下午"
- "/radar-analyze 算力租赁"
- "/radar-analyze 600519"
- "/radar-analyze 最近 6 小时里 tradability=high 的"

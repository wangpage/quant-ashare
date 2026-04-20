# Level2 真实数据盘中测试 - 明早 9:30 Checklist

> 目标: 盘中 (2026-04-21 周二 09:30-10:00) 接入 base32.cn 真实 Level2 NATS,
> 采集 300750/600519 真实逐笔, 验证延迟 p50/p95/p99 和整条框架.

## ⏰ 时间安排

| 时间 | 动作 | 耗时 |
|---|---|---|
| 09:20 | 开始准备, 打开终端, cd 项目目录 | 1 min |
| 09:25 | 确认网络+账号, 运行健康检查 | 2 min |
| 09:30 | **开盘! 立即启动 T6 测试脚本** | 0.5 min |
| 09:30-09:35 | 开盘 5 分钟高延迟时段, **观察** (文档说可能 10-20s 延迟) | 5 min |
| 09:35-10:00 | **正式采集** 2-3 分钟稳定期数据 | 25 min |
| 10:00 | 停止, 查看报告 | 1 min |

## 🎯 一键命令

```bash
cd /Users/page/Desktop/股票/quant_ashare

# 环境健康检查 (09:25 运行一次)
python3 tests/test_parser.py                              # 必须 60/60
python3 tests/test_smoke_real.py                          # 三节点连通性

# 盘中采集 (09:30 准时运行, 采集 2 分钟)
python3 tests/test_realdata_intraday.py
```

## ✅ 预检清单 (09:25 前完成)

```bash
# 1. 确认 Python 依赖就位
python3 -c "import nats, numpy, pandas, tabulate; print('OK')"

# 2. 确认浏览器测试账号信息
#    账号: level2_test
#    密码: level2@test
#    标的: 300750 (宁德时代), 600519 (贵州茅台)

# 3. 确认网络能连到 base32.cn (任一通即可)
curl -sI --max-time 5 https://db.base32.cn | head -1
python3 -c "
import socket
for host in ['db.base32.cn', '43.143.73.95', '43.138.245.99']:
    try:
        socket.create_connection((host, 31886), timeout=3)
        print(f'✓ {host}:31886')
    except Exception as e:
        print(f'✗ {host}: {e}')
"

# 4. 确认配置正确
cat config/level2.yaml | grep -A 3 "connection:"
```

## 📊 预期结果

### 正常情况 (深圳数据)

| 指标 | 预期 | 说明 |
|---|---|---|
| trans 消息/分 | 50-200 条 | 300750 日线活跃度 |
| order 消息/分 | 200-500 条 | 逐笔委托远多于成交 |
| simple 消息/分 | ~6000 条 | 每 10ms 刷新 |
| p50 延迟 | **< 50 ms** | 深圳正常延迟 |
| p95 延迟 | < 200 ms | |
| p99 延迟 | < 500 ms | |
| 解析错误率 | **< 0.01%** | CSV 格式稳定 |

### 正常情况 (上海数据)

| 指标 | 预期 |
|---|---|
| p50 延迟 | **< 200 ms** (文档说 ~150ms) |
| p99 延迟 | < 1000 ms |

### 开盘前 5 分钟 (异常但预期)

文档明说: **开盘时段延时可能 10-20 秒, 不可控**
- **不要惊慌**, 这是交易所本身的问题
- 策略上要避免开盘 10 分钟交易 (已在 `execution/time_windows.py` 实现)

## 🔧 可能遇到的问题

### 问题 1: 连接失败
```
Connection failed: TimeoutError
```
**排查**:
```bash
# 检查防火墙 / 公司内网
ping 43.143.73.95

# 检查端口 31886
nc -zv 43.143.73.95 31886
```

### 问题 2: 认证失败
```
Authorization Violation
```
**排查**:
- 确认 `level2_test` / `level2@test` 未被改
- 确认账号没被封 (联系微信 lky1811512)

### 问题 3: 连上但无数据
```
订阅 10 个 topic, 等待 30s...
收到消息数: 0
```
**排查**:
- 当前时间是否 09:15-15:00
- 测试账号是否只能订阅 300750 / 600519
- 是否开了"合成订单簿" (`rapid`) 权限 → test 账号可能没这个, 改 `enable_types`:
  ```yaml
  enable_types:
    trans: true
    order: true
    rapid: false        # 关掉
    simple: true
  ```

### 问题 4: 延迟异常高 (> 5 秒)
**可能原因**:
- 你不在华南/华东机房, 物理距离远
- 开盘前 5 分钟 (正常)
- 中间有代理 / VPN

### 问题 5: Python 3.9 兼容问题
已修: 所有文件都有 `from __future__ import annotations`

## 📁 数据采集后

测试脚本会自动:
1. 打印屏幕报告 (p50/p95/p99, 按类型/按股票分组)
2. 保存 JSON 报告到 `output/realdata_<timestamp>.json`
3. 样本消息 5 条展示 (原始 CSV)

查看历史报告:
```bash
ls output/realdata_*.json
python3 -c "
import json, glob
for f in sorted(glob.glob('output/realdata_*.json'))[-3:]:
    d = json.load(open(f))
    print(f)
    print(f\"  QPS: {d['qps']:.1f}, trans: {d['by_type']['trans']}, lat p99: -\")
"
```

## 🚀 成功后下一步

1. **跑更长时间**: 把 `DURATION_SECONDS` 从 120 改到 1800 (30 分钟)
2. **接入分钟K聚合**: 运行 `tests/test_integration_mock.py` 但用真 NATS 数据
3. **叠加因子**: 把 trans/order/book 数据喂给 `market_microstructure/order_flow.py`
   计算 OIR / VPIN / Cancel Ratio
4. **接入 LLM 决策**: 分钟级因子 + regime → agent team → 风控 → 模拟下单

## ⚠️ 不要做的事

- ❌ 不要一直连 (文档说单 IP 连接数 > 50 会被封)
- ❌ 不要订阅超过 100 个 topic (测试账号 2 只票 × 4 topic = 8 个, OK)
- ❌ 不要盘中下载历史数据库 (会挤占实时行情)
- ❌ 不要把 test 账号的数据用于实盘 (仅支持 300750/600519)

## 📞 遇到问题联系

- **微信**: lky1811512 (文档里的数据咨询)
- **文档**: https://docs.qq.com/doc/DQ0ZOalhtV3BFdVFV
- **错误码**: https://github.com/nats-io/nats-server/blob/main/server/errors.json

---

**明早 9:30 见, 准时开跑!**

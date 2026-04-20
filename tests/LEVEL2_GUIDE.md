# Level2 测试指南 - 从本地 Mock 到真实环境

## 一、测试矩阵

| 测试 | 文件 | 需要环境 | 运行时长 | 现状 |
|---|---|---|---|---|
| T2 解析正确性 | `test_parser.py` | 无 | 1秒 | ✅ 通过 |
| T3-mock 延迟压测 | `test_latency_mock.py` | 无 | 30秒 | ✅ 通过 |
| T4-mock 集成 | `test_integration_mock.py` | 无 | 10秒 | ✅ 通过 |
| T1 联通性 | `test_connectivity.py` | 真 NATS | 30秒 | ⏸ 待凭证 |
| T3 真实压测 | `test_latency.py` | 真 NATS | 60秒 | ⏸ 待凭证 |
| T4 真实集成 | `test_integration.py` | 真 NATS | 60秒 | ⏸ 待凭证 |

## 二、现在就能跑（无需任何安装）

```bash
cd /Users/page/Desktop/股票/quant_ashare

# 解析正确性 (23/23 通过)
python3 tests/test_parser.py

# Mock 压测 - 验证框架性能
python3 tests/test_latency_mock.py

# Mock 集成 - 验证 tick→分钟K→风控 链路
python3 tests/test_integration_mock.py
```

## 三、本地完整 NATS Broker 方案（模拟生产）

### 3.1 安装 nats-server

```bash
# 用 brew 安装 (你系统已有 brew)
brew install nats-server
nats-server --version   # 验证
```

### 3.2 启动本地 broker

```bash
# 前台启动, 默认 4222 端口
nats-server --auth level2_test --auth_token level2@test

# 或后台启动 + 写日志
nats-server --auth level2_test --auth_token level2@test \
  --log /tmp/nats.log --pid /tmp/nats.pid &
```

### 3.3 配置指向本地

编辑 `config/level2.yaml`:

```yaml
connection:
  servers:
    shanghai:
      host: "nats://127.0.0.1:4222"
  active: "shanghai"
  auth:
    user: "level2_test"
    password: "level2@test"
```

### 3.4 启动一个 publisher (模拟数据源)

另开一个终端:
```bash
python3 -c "
import asyncio, nats, json, time, random
async def pub():
    nc = await nats.connect('nats://127.0.0.1:4222',
                            user='level2_test', password='level2@test')
    codes = ['300750', '600519']
    price = {'300750': 245.88, '600519': 1688.0}
    while True:
        c = random.choice(codes)
        price[c] *= (1 + random.gauss(0, 0.002))
        msg = json.dumps({
            'code': c, 'ts': time.time_ns(),
            'price': round(price[c], 2),
            'vol': random.choice([100, 500, 1000]),
            'amt': price[c] * 500,
            'bs_flag': random.choice(['B', 'S']),
            'tid': int(time.time_ns())
        }).encode()
        await nc.publish(f'l2.trade.{c}', msg)
        await asyncio.sleep(0.001)   # 1000 msg/s
asyncio.run(pub())
"
```

### 3.5 跑真实 NATS 版本的测试

```bash
pip install nats-py tabulate numpy pandas
python3 tests/test_connectivity.py
python3 tests/test_latency.py
python3 tests/test_integration.py
```

## 四、接入生产 Level2 的前置工作

拿到真实凭证后, 只需修改 `config/level2.yaml` 的 4 处:

```yaml
connection:
  servers:
    shanghai:
      host: "nats://真实IP:真实端口"    # 文档里的 IP:PORT
      backup: "nats://备用IP:端口"
  auth:
    user: "你的生产账号"
    password: "你的生产密码"

topics:
  trade:     "券商的真实topic格式.{code}"    # 不同券商不同
  order:     "..."
  orderbook: "..."

message_format: "json"        # 或 "protobuf", 看文档
```

字段映射如果和默认不一样, 改 `level2/parser.py` 里的 `FIELD_MAP`.

## 五、常见真实生产问题

| 问题 | 原因 | 排查 |
|---|---|---|
| 连上但收不到数据 | topic 格式错 | `nats sub "l2.>"` 看有哪些 topic |
| 延迟 > 500ms | 物理网络差 | 换就近机房 / 用专线 |
| 数据解析错误率高 | 字段名不符 | 抓一条原始消息, 对照 `FIELD_MAP` |
| 消息丢包 | 处理太慢 | 加大 `RingBuffer` / 多 worker 消费 |
| 盘中收不到 | 未订阅盘前初始化 | 7:00 前上线, 并重放 |

## 六、商业 Level2 采购路径汇总

| 渠道 | 成本 | 适合 |
|---|---|---|
| 国金/华鑫/东财 券商 | 50万门槛 + 200元/月 | 个人散户 |
| 掘金 GM Terminal | 2000+/月 | 量化个人 |
| 聚宽 JoinQuant | 1万+/年 | 量化研究 |
| Wind L2 / 恒生聚源 | 5-50万/年 | 机构 |
| AK/Tushare（不含tick） | 免费 | Level1 日线分钟级 |

## 七、为什么用 NATS 协议?

NATS 优势:
- 微秒级延迟 (比 Kafka 低一个数量级)
- 支持百万级 QPS
- 订阅模式灵活 (wildcard: `l2.trade.>`)
- 无需消息持久化 (行情是实时流, 历史走另外接口)

如果券商提供 TCP/WebSocket 而非 NATS, 只需替换 `level2/nats_client.py` 为对应协议客户端, 其他层不变.

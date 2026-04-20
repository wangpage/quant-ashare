"""T5: 连 base32.cn 公网 ClickHouse 拿真实盘后数据, 做离线解析+质量审计.

公网高速地址: db.base32.cn:9000 (TCP) / 8123 (HTTP)
账号: 未知 (文档未给免费账号, 需客户经理开通)

本测试做两件事:
  A. 尝试匿名或测试账号连 ClickHouse
  B. 如连得上, 拉一段 trans/order 数据, 喂给 parser 验证字段兼容性
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabulate import tabulate
from level2.parser import Level2Parser


TRIAL_CREDS = [
    ("default", ""),
    ("level2_test", "level2@test"),
    ("reader", "reader"),
    ("guest", "guest"),
]


def try_ch():
    from clickhouse_driver import Client
    from clickhouse_driver.errors import Error as ChError

    for user, pw in TRIAL_CREDS:
        try:
            c = Client(host="db.base32.cn", port=9000,
                       user=user, password=pw,
                       compression="zstd", connect_timeout=5)
            res = c.execute("SHOW DATABASES")
            print(f"✓ 连通: user={user}, 可见数据库: {[r[0] for r in res][:10]}")
            return c, user
        except ChError as e:
            print(f"  ✗ {user}: {str(e)[:120]}")
        except Exception as e:
            print(f"  ✗ {user}: {type(e).__name__}: {str(e)[:120]}")
    return None, None


def audit_trans(c):
    print("\n-- 抽样深圳逐笔成交 (share.trans) --")
    try:
        rows = c.execute("""
            SELECT * FROM share.trans
            WHERE SecurityID = '300750' AND TradeDate >= yesterday()
            LIMIT 5
        """)
        if not rows:
            print("  (空, 可能昨日无数据)")
            return
        cols = c.execute("DESCRIBE share.trans")
        print("  字段:", [r[0] for r in cols])
        for r in rows:
            print(" ", r)

        # 把第一行重构成 CSV 喂给 parser
        p = Level2Parser("csv")
        csv_line = ",".join(str(x) for x in rows[0])
        import time
        t = p.parse_trans(csv_line.encode(), time.time_ns())
        print(f"\n  解析结果: code={t.code if t else None}, "
              f"price={t.price if t else None}, volume={t.volume if t else None}")
    except Exception as e:
        print(f"  失败: {e}")


def main():
    print("="*70)
    print("  T5 ClickHouse 公网盘后数据测试")
    print("  db.base32.cn:9000 (TCP)")
    print("="*70)

    client, user = try_ch()
    if not client:
        print("\n⚠ 所有试用账号均不可用")
        print("  文档说明: ClickHouse 账号需 '联系客户经理开通' / '私聊获取'")
        print("  微信: lky1811512")
        return 1

    try:
        db = client.execute("SHOW TABLES FROM share")
        print(f"\n-- share 库的表: {[r[0] for r in db]} --")
    except Exception as e:
        print(f"SHOW TABLES 失败: {e}")

    audit_trans(client)
    return 0


if __name__ == "__main__":
    sys.exit(main())

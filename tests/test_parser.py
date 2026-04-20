"""T2: 数据解析正确性 - 用文档示例 CSV 校验.

所有用例直接来自《沪深Level2高级行情接口文档》0x02, 字段值已核对.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabulate import tabulate
from level2.parser import Level2Parser


def _a(cond, msg): return bool(cond), msg


# ============ T2.1 逐笔成交 ============
def test_trans_sh():
    """文档 0x02-2.1 上海案例."""
    p = Level2Parser("csv")
    msg = b"688111,2024-12-19,100851000,1734574131116,T,310.960,100,1,7130505,4576430,4573387"
    t = p.parse_trans(msg, time.time_ns())
    return [
        _a(t is not None, "trans上海 非空"),
        _a(t and t.code == "688111", "code=688111"),
        _a(t and t.trade_date == "2024-12-19", "日期"),
        _a(t and t.exchange_time == 100851000, "交易所时间"),
        _a(t and t.server_time_ms == 1734574131116, "服务器时间"),
        _a(t and t.tick_type == "T", "上海TickType=T"),
        _a(t and t.price == 310.960, "价格"),
        _a(t and t.volume == 100, "成交量"),
        _a(t and t.buy_no == 4576430, "买方序号"),
        _a(t and t.sell_no == 4573387, "卖方序号"),
    ]


def test_trans_sz():
    """文档 0x02-2.1 深圳案例 (TickType=1)."""
    p = Level2Parser("csv")
    msg = b"000002,2024-12-19,101346670,1734574426689,1,7.900,1300,2013,19973342,19912753,19973341"
    t = p.parse_trans(msg, time.time_ns())
    return [
        _a(t is not None, "trans深圳 非空"),
        _a(t and t.code == "000002", "code=000002"),
        _a(t and t.tick_type == "1", "深圳TickType=1"),
        _a(t and t.volume == 1300, "成交量1300"),
    ]


# ============ T2.2 逐笔委托 ============
def test_order():
    """文档 0x02-2.2 案例."""
    p = Level2Parser("csv")
    msg = b"688111,2024-12-19,101346510,1734574426617,A,1,308.550,200,1,7660642,4898191"
    o = p.parse_order(msg, time.time_ns())
    return [
        _a(o is not None, "order 非空"),
        _a(o and o.code == "688111", "code"),
        _a(o and o.tick_type == "A", "上海新增委托A"),
        _a(o and o.side == 1, "买方 side=1"),
        _a(o and o.price == 308.550, "价格"),
        _a(o and o.volume == 200, "数量"),
        _a(o and o.order_no == 4898191, "订单号"),
    ]


# ============ T2.5 合成订单簿 5档 ============
def test_rapid():
    """文档 0x02-2.5 案例."""
    p = Level2Parser("csv")
    msg = (b"300750.XSHE,2026-02-11,93835400,1770773915402,"
           b"364.970,366.900,1727855,632959354.11,366.290,0.0000,"
           b"368.000,364.500,437.960,291.980,8041,"
           b"366.300,366.280,100,800,"
           b"366.310,366.250,100,600,"
           b"366.590,366.240,200,200,"
           b"366.600,366.230,3400,200,"
           b"366.610,366.210,200,5700,"
           b"1,1")
    r = p.parse_rapid(msg, time.time_ns())
    return [
        _a(r is not None, "rapid 非空"),
        _a(r and r.code_exchange == "300750.XSHE", "代码.交易所"),
        _a(r and r.code == "300750", "code 提取"),
        _a(r and r.pre_close == 364.970, "前收盘"),
        _a(r and r.last_price == 366.290, "最新价"),
        _a(r and r.num_trades == 8041, "成交笔数"),
        _a(r and len(r.ask_prices) == 5 and len(r.bid_prices) == 5, "5档"),
        _a(r and r.ask_prices[0] == 366.300, "卖1价"),
        _a(r and r.bid_prices[0] == 366.280, "买1价"),
        _a(r and r.ask_volumes[0] == 100, "卖1量"),
        _a(r and r.bid_volumes[0] == 800, "买1量"),
        _a(r and r.spread > 0, "spread>0"),
        _a(r and r.bid_count1 == 1, "买1笔数"),
        _a(r and -1 <= r.imbalance <= 1, "imbalance 范围"),
    ]


# ============ T2.6 简化订单簿 ============
def test_simple():
    """文档 0x02-2.6 案例."""
    p = Level2Parser("csv")
    # 盘前数据 (09:15)
    msg = b"300750.XSHE,91500020,1774919700026,413.000,0,0.00,0.000,0.0000,0,413.990,0.000,100,0"
    s = p.parse_simple(msg, time.time_ns())
    return [
        _a(s is not None, "simple 非空"),
        _a(s and s.code == "300750", "code"),
        _a(s and s.pre_close == 413.000, "前收"),
        _a(s and s.ask_price1 == 413.990, "卖1价"),
        _a(s and s.ask_volume1 == 100, "卖1量"),
    ]


def test_simple_intraday():
    """盘中有交易的 simple."""
    p = Level2Parser("csv")
    msg = (b"600519.XSHG,93228350,1774920777887,"
           b"1420.000,1016992,1496494713.65,1476.500,0.0000,6427,"
           b"1476.800,1476.550,100,100")
    s = p.parse_simple(msg, time.time_ns())
    return [
        _a(s is not None, "simple盘中 非空"),
        _a(s and s.last_price == 1476.500, "最新价"),
        _a(s and s.total_volume == 1016992, "总成交量"),
        _a(s and abs(s.spread - 0.250) < 1e-6, f"spread=0.25 实际{s and s.spread}"),
        _a(s and abs(s.mid_price - 1476.675) < 1e-6, "mid"),
    ]


# ============ T2.4 十档订单簿 ============
def test_depth():
    """文档 0x02-2.4 案例 (61字段)."""
    p = Level2Parser("csv")
    # 注: 文档案例是 2.4 的完整10档
    msg = (b"300750,2,2026-02-24,130124000,2,1771909285375,"
           b"365.340,372.640,15633188,5751363780.58,"
           b"564400,355.20,3068361,382.18,362.990,0.0000,"
           b"373.500,362.500,438.410,292.270,77995,"
           b"363.000,362.990,800,9200,"
           b"363.100,362.980,700,300,"
           b"363.150,362.910,2800,400,"
           b"363.160,362.900,100,200,"
           b"363.200,362.890,300,600,"
           b"363.230,362.880,900,400,"
           b"363.240,362.840,100,400,"
           b"363.250,362.830,2500,600,"
           b"363.310,362.750,100,100,"
           b"363.330,362.690,200,600")
    d = p.parse_depth(msg, time.time_ns())
    return [
        _a(d is not None, "depth 非空"),
        _a(d and d.code == "300750", "code"),
        _a(d and len(d.ask_prices) == 10, "10档卖"),
        _a(d and len(d.bid_prices) == 10, "10档买"),
        _a(d and d.last_price == 362.990, "最新价"),
        _a(d and d.ask_prices[0] == 363.000, "卖1价"),
        _a(d and d.bid_prices[0] == 362.990, "买1价"),
        _a(d and d.num_trades == 77995, "成交笔数"),
    ]


# ============ T2.7 资金流 ============
def test_flow():
    """文档 0x02-2.7 案例."""
    p = Level2Parser("csv")
    msg = (b"301291,2024-12-06,11:29:58,640,1733455803129,"
           b"41075200.26,955339,3440,"
           b"43471904.67,1015460,4489,"
           b"31378883.78,732374,404,"
           b"32413443.21,754833,421,"
           b"17346646.16,401900,44,"
           b"14846031.35,344291,44,"
           b"7292010.00,170700,3,"
           b"7175987.00,166600,5")
    fl = p.parse_flow(msg, time.time_ns())
    return [
        _a(fl is not None, "flow 非空"),
        _a(fl and fl.code == "301291", "code"),
        _a(fl and fl.retail_buy_amount == 41075200.26, "散户买入额"),
        _a(fl and fl.inst_buy_count == 3, "机构买入笔数"),
        _a(fl and -1 <= fl.big_money_bias <= 1, "大单方向范围"),
    ]


def test_malformed():
    p = Level2Parser("csv")
    assert p.parse_trans(b"not,enough,fields", 0) is None
    assert p.parse_rapid(b"bad", 0) is None
    return [
        _a(p.stats["errors"] >= 2, "错误计数"),
        _a("trans" in p.stats["errors_by_type"], "错误按类型分"),
    ]


def main():
    tests = [
        ("trans_sh", test_trans_sh), ("trans_sz", test_trans_sz),
        ("order", test_order), ("rapid", test_rapid),
        ("simple_preopen", test_simple), ("simple_intraday", test_simple_intraday),
        ("depth", test_depth), ("flow", test_flow),
        ("malformed", test_malformed),
    ]
    rows = []
    total_pass = total = 0
    for name, fn in tests:
        for ok, desc in fn():
            rows.append([name, desc, "✓" if ok else "✗"])
            total_pass += ok
            total += 1

    print("\n==== T2 CSV 解析测试 (依据官方文档案例) ====")
    print(tabulate(rows, headers=["模块", "用例", "结果"]))
    print(f"\n通过率: {total_pass}/{total}")
    sys.exit(0 if total_pass == total else 1)


if __name__ == "__main__":
    main()

"""按 code 拉基本面文本, 供 TradingAgentTeam.analyze_fundamental 消费.

多源 fallback (按优先级):
  1. cache/fundamentals_cache.json  (24h 内命中直接返回)
  2. 雪球 stock_individual_basic_info_xq  (稳定, 39 字段, 含行业/员工/法人/业务)
  3. 同花顺 stock_financial_abstract_ths  (补充财务: 净利润/ROE/营收)
  4. 东财 stock_individual_info_em  (最全但近期限速常失败)
  5. cache/market.db.stock_info  (最终降级)
  6. 兜底: "{code} {name} (基本面数据暂不可得)"

每源独立重试 + 总时长限制, 不让任何单源拖垮整体.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from utils.config import PROJECT_ROOT
from utils.logger import logger

CACHE_PATH = PROJECT_ROOT / "cache" / "fundamentals_cache.json"
CACHE_TTL_HOURS = 24


def _read_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_cache(cache: dict[str, dict]) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=0),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"fundamentals cache 写失败: {e}")


def _cache_fresh(entry: dict) -> bool:
    ts = entry.get("_fetched_at", 0)
    return (time.time() - ts) < CACHE_TTL_HOURS * 3600


def _yi_fmt(v) -> str:
    """把 元 格式化成 亿/万亿/亿."""
    try:
        v = float(v)
    except Exception:
        return "--"
    if v >= 1e12:
        return f"{v/1e12:.2f}万亿"
    if v >= 1e8:
        return f"{v/1e8:.1f}亿"
    if v >= 1e4:
        return f"{v/1e4:.0f}万"
    return f"{v:.0f}"


def _fetch_em(code: str, retries: int = 2) -> dict | None:
    """东财. 字段: 总市值/流通市值/总股本/行业/上市时间/股票简称."""
    try:
        import akshare as ak
    except ImportError:
        return None
    for a in range(retries):
        try:
            df = ak.stock_individual_info_em(symbol=code)
            if df is None or df.empty:
                return None
            out = dict(zip(df["item"], df["value"]))
            out["_source"] = "em"
            return out
        except Exception:
            if a < retries - 1:
                time.sleep(1 * (2 ** a))
    return None


def _fetch_xq(code: str, retries: int = 2) -> dict | None:
    """雪球. 字段最全: 公司简称/全名/业务/行业/员工数/法人/上市日期/注册资本."""
    try:
        import akshare as ak
    except ImportError:
        return None
    # 雪球 API 需要 SH/SZ 前缀
    prefix = "SH" if code.startswith(("6", "9")) else "SZ"
    sym = prefix + code
    for a in range(retries):
        try:
            df = ak.stock_individual_basic_info_xq(symbol=sym)
            if df is None or df.empty:
                return None
            out = dict(zip(df["item"], df["value"]))
            out["_source"] = "xq"
            return out
        except Exception:
            if a < retries - 1:
                time.sleep(1 * (2 ** a))
    return None


def _fetch_ths_financial(code: str, retries: int = 2) -> dict | None:
    """同花顺最近一期财务: 净利润/ROE/营收/EPS/资产负债率等."""
    try:
        import akshare as ak
    except ImportError:
        return None
    for a in range(retries):
        try:
            df = ak.stock_financial_abstract_ths(
                symbol=code, indicator="按报告期"
            )
            if df is None or df.empty:
                return None
            # 取最新一期
            last = df.iloc[-1].to_dict()
            last["_source"] = "ths"
            return last
        except Exception:
            if a < retries - 1:
                time.sleep(1 * (2 ** a))
    return None


def _ms_to_date(v) -> str:
    """1528646400000 (ms) → '2018-06-11'."""
    try:
        from datetime import datetime as _dt
        ms = int(float(v))
        return _dt.fromtimestamp(ms / 1000).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _fetch_live(code: str) -> dict | None:
    """多源 fallback. 先试雪球(最稳), 再东财, 最后 DB."""
    # 源 1: 雪球
    xq = _fetch_xq(code)
    if xq:
        out = {
            "股票简称": xq.get("org_short_name_cn", ""),
            "公司全称": xq.get("org_name_cn", ""),
            "主营业务": xq.get("main_operation_business", ""),
            "行业": (xq.get("affiliate_industry") or {}).get("ind_name", ""),
            "法人": xq.get("legal_representative", ""),
            "董事长": xq.get("chairman", ""),
            "员工数": xq.get("staff_num", ""),
            "上市日期": _ms_to_date(xq.get("listed_date")),
            "成立日期": _ms_to_date(xq.get("established_date")),
            "实际控制人": xq.get("actual_controller", ""),
            "注册资本": xq.get("reg_asset", ""),
            "_fetched_at": time.time(),
            "_source": "xq",
        }
        # 补财务
        fin = _fetch_ths_financial(code)
        if fin:
            for k in ("报告期", "净利润", "净利润同比增长率",
                      "营业总收入", "营业总收入同比增长率",
                      "销售毛利率", "净资产收益率", "资产负债率",
                      "基本每股收益"):
                if k in fin:
                    out[f"fin_{k}"] = fin[k]
        return out

    # 源 2: 东财
    em = _fetch_em(code)
    if em:
        em["_fetched_at"] = time.time()
        return em

    logger.warning(f"[{code}] 雪球/东财 都失败")
    return None


def _fetch_db_fallback(code: str) -> dict | None:
    """从 cache/market.db 读 stock_info (若有)."""
    try:
        db = PROJECT_ROOT / "cache" / "market.db"
        if not db.exists():
            return None
        with sqlite3.connect(db) as c:
            row = c.execute(
                "SELECT name, industry, market_cap FROM stock_info WHERE code = ?",
                [code],
            ).fetchone()
        if not row:
            return None
        return {
            "股票简称": row[0] or "",
            "行业": row[1] or "",
            "总市值": row[2] or 0,
            "_source": "db",
            "_fetched_at": time.time(),
        }
    except Exception as e:
        logger.debug(f"db fallback 失败 {code}: {e}")
        return None


def get_fundamentals_dict(code: str, force_refresh: bool = False) -> dict:
    """内部字典形式, 供其它模块 (如 candidate_data_builder) 按需取字段."""
    cache = _read_cache()
    if not force_refresh and code in cache and _cache_fresh(cache[code]):
        return cache[code]

    data = _fetch_live(code)
    if data is None:
        data = _fetch_db_fallback(code)
    if data is None:
        data = {"_fetched_at": time.time(), "_source": "none"}

    cache[code] = data
    _write_cache(cache)
    return data


def fetch_fundamentals_text(code: str, name_hint: str = "",
                             force_refresh: bool = False) -> str:
    """给 agent 的文本. name_hint 是外部调用方已知的名字(如从 watchlist csv)."""
    d = get_fundamentals_dict(code, force_refresh=force_refresh)
    src = d.get("_source", "unknown")

    if src in ("none", "unknown"):
        return f"{code} {name_hint} (基本面数据不可得)"

    # 雪球源 (主) + 同花顺财务 (附加)
    if src == "xq":
        parts = [f"{code} {d.get('股票简称') or name_hint or '--'}"]
        if d.get("行业"):
            parts.append(f"行业:{d['行业']}")
        if d.get("主营业务"):
            biz = d["主营业务"][:50]
            parts.append(f"主营:{biz}")
        if d.get("员工数"):
            parts.append(f"员工 {d['员工数']}人")
        if d.get("注册资本"):
            try:
                rc = float(d["注册资本"])
                parts.append(f"注册资本 {_yi_fmt(rc)}")
            except Exception:
                pass
        if d.get("上市日期"):
            parts.append(f"上市 {d['上市日期']}")
        if d.get("实际控制人"):
            parts.append(f"实控人 {d['实际控制人']}")

        # 财务(若同花顺能拉到)
        fin_parts = []
        if d.get("fin_报告期"):
            fin_parts.append(f"最近报告 {d['fin_报告期']}")
        if d.get("fin_净利润"):
            gr = d.get("fin_净利润同比增长率")
            tag = f"(同比{gr})" if gr and str(gr) != "False" else ""
            fin_parts.append(f"净利润 {d['fin_净利润']}{tag}")
        if d.get("fin_营业总收入"):
            gr = d.get("fin_营业总收入同比增长率")
            tag = f"(同比{gr})" if gr and str(gr) != "False" else ""
            fin_parts.append(f"营收 {d['fin_营业总收入']}{tag}")
        if d.get("fin_净资产收益率"):
            fin_parts.append(f"ROE {d['fin_净资产收益率']}")
        if d.get("fin_销售毛利率"):
            fin_parts.append(f"毛利率 {d['fin_销售毛利率']}")
        if d.get("fin_资产负债率"):
            fin_parts.append(f"资产负债率 {d['fin_资产负债率']}")

        s = " | ".join(parts)
        if fin_parts:
            s += "  ||  财务: " + " / ".join(fin_parts)
        return s

    # 东财源(备)
    if src == "em":
        name = d.get("股票简称") or name_hint or "--"
        parts = [f"{code} {name}"]
        if d.get("行业"):
            parts.append(f"行业:{d['行业']}")
        if d.get("总市值"):
            parts.append(f"总市值 {_yi_fmt(d['总市值'])}")
        if d.get("流通市值"):
            parts.append(f"流通市值 {_yi_fmt(d['流通市值'])}")
        if d.get("上市时间"):
            ls = str(d["上市时间"])
            if len(ls) == 8 and ls.isdigit():
                ls = f"{ls[:4]}-{ls[4:6]}-{ls[6:]}"
            parts.append(f"上市 {ls}")
        return " | ".join(parts) + " _(em)_"

    # DB fallback
    if src == "db":
        name = d.get("股票简称") or name_hint or "--"
        parts = [f"{code} {name}"]
        if d.get("行业"):
            parts.append(f"行业:{d['行业']}")
        if d.get("总市值"):
            parts.append(f"总市值 {_yi_fmt(d['总市值'])}")
        return " | ".join(parts) + " _(db)_"

    return f"{code} {name_hint} (未知基本面源)"


def fetch_fundamentals_batch(codes: list[str],
                              force_refresh: bool = False) -> dict[str, str]:
    """批量, 顺序执行(akshare 速率限制), 用 cache 能省很多次 call."""
    out = {}
    for c in codes:
        out[c] = fetch_fundamentals_text(c, force_refresh=force_refresh)
    return out


if __name__ == "__main__":
    # 调试
    import sys
    codes = sys.argv[1:] or ["300750", "603163", "300034"]
    for c in codes:
        print(fetch_fundamentals_text(c))

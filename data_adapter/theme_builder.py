"""构造 theme_map: {theme_name: [codes]}, 供 detect_emerging_themes 消费.

策略 (两档):
  - quick: 只拉当日涨幅 >threshold 的板块 (~30 个), ~2 min
  - full: 拉全部 ~800 个概念板块 (~10 min, 慎用)

缓存到 cache/theme_map_{mode}_{YYYYMMDD}.json, 24h 过期.

失败时返回空 dict, 调用方应处理这种情况.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from utils.config import PROJECT_ROOT
from utils.logger import logger

CACHE_DIR = PROJECT_ROOT / "cache"


def _cache_path(mode: str) -> Path:
    today = datetime.now().strftime("%Y%m%d")
    return CACHE_DIR / f"theme_map_{mode}_{today}.json"


def _read_cache(mode: str) -> dict | None:
    p = _cache_path(mode)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d
    except Exception:
        return None


def _write_cache(mode: str, data: dict) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(mode).write_text(
            json.dumps(data, ensure_ascii=False, indent=0),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"theme_map cache 写失败: {e}")


def _safe_ak_call(fn, *args, retries: int = 3, backoff: float = 2.0, **kwargs):
    last_err = None
    for a in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if a < retries - 1:
                time.sleep(backoff * (2 ** a))
    raise last_err


def build_theme_map(mode: str = "quick",
                     min_pct_chg: float = 3.0,
                     force_refresh: bool = False) -> dict[str, list[str]]:
    """构造 {theme_name: [6位 code]} 映射.

    Args:
        mode: "quick" 只拉当日涨幅 >=min_pct_chg% 的板块; "full" 拉全部
        min_pct_chg: quick 模式下的涨幅阈值(%)
        force_refresh: 跳过缓存

    Returns:
        dict. 空表示拉取彻底失败.
    """
    if not force_refresh:
        cached = _read_cache(mode)
        if cached:
            logger.info(f"theme_map 命中缓存 ({mode}), {len(cached)} 个题材")
            return cached

    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare 未安装, theme_map 无法构造")
        return {}

    # 1) 拉所有概念板块的 当日涨跌 快照
    try:
        board_df = _safe_ak_call(ak.stock_board_concept_name_em)
    except Exception as e:
        logger.error(f"stock_board_concept_name_em 失败: {e}")
        return {}

    logger.info(f"概念板块总数: {len(board_df)} 个")

    # 找涨幅列 (可能叫 '涨跌幅' 或类似)
    pct_col = next(
        (c for c in board_df.columns if "涨跌幅" in c or "pct" in c.lower()),
        None,
    )
    if mode == "quick":
        if pct_col is None:
            logger.warning("找不到涨跌幅列, 改用 full 模式")
            mode = "full"
        else:
            boards = board_df[board_df[pct_col] >= min_pct_chg].copy()
            # 按涨幅降序, 上限 50 个板块避免太慢
            boards = boards.sort_values(pct_col, ascending=False).head(50)
            logger.info(
                f"quick 模式筛出 {len(boards)} 个强势板块 (涨幅>={min_pct_chg}%)"
            )
    else:
        boards = board_df
        logger.info(f"full 模式拉全部 {len(boards)} 个板块")

    name_col = next(
        (c for c in boards.columns if c in ("板块名称", "name", "概念名称")),
        "板块名称",
    )

    theme_map: dict[str, list[str]] = {}
    n_ok = n_fail = 0
    t0 = time.time()
    for i, r in enumerate(boards.itertuples(), 1):
        name = getattr(r, name_col, None)
        if not name:
            continue
        try:
            cons = _safe_ak_call(
                ak.stock_board_concept_cons_em, symbol=name,
                retries=2, backoff=1.5,
            )
            code_col = next(
                (c for c in cons.columns if c in ("代码", "code", "股票代码")),
                "代码",
            )
            codes = cons[code_col].astype(str).str.zfill(6).tolist()
            theme_map[name] = codes
            n_ok += 1
        except Exception as e:
            n_fail += 1
            logger.debug(f"[{name}] cons 拉取失败: {e}")
        if i % 10 == 0:
            logger.info(
                f"  进度 {i}/{len(boards)}  ok={n_ok} fail={n_fail} "
                f"耗时 {time.time()-t0:.0f}s"
            )
        time.sleep(0.3)  # 避免被限速

    logger.info(
        f"theme_map 构造完成: {len(theme_map)} 个题材, "
        f"ok={n_ok} fail={n_fail}, 耗时 {time.time()-t0:.1f}s"
    )
    if theme_map:
        _write_cache(mode, theme_map)
    return theme_map


def code_to_themes(theme_map: dict[str, list[str]]) -> dict[str, list[str]]:
    """反向索引: {code: [theme_names]}."""
    out: dict[str, list[str]] = {}
    for theme, codes in theme_map.items():
        for c in codes:
            out.setdefault(c, []).append(theme)
    return out


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    m = build_theme_map(mode=mode)
    print(f"题材数: {len(m)}")
    for k, v in list(m.items())[:5]:
        print(f"  {k} ({len(v)} 只): {v[:5]}")

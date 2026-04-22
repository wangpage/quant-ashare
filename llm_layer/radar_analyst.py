"""Radar 事件 LLM 分析器: Haiku triage + Opus deep analyze.

两段式设计:
  1. triage(event)       Haiku, 每条必跑, <1s, 分类 + 判定是否需要深挖
  2. deep_analyze(...)   Opus,  仅候选跑, ~3s, 产出可交易候选清单

输出统一用 Hermes XML 风格 (与 llm_layer/agents.py 保持一致),
通过 xml_parser.extract_tag 抽取. 不用严格 JSON.

Anthropic key 缺失时自动降级到 DashScope/Qwen (环境变量 DASHSCOPE_API_KEY),
避免本地没配 Anthropic 就无法调试. 模型可通过环境变量覆盖:
  RADAR_TRIAGE_BACKEND=anthropic  RADAR_TRIAGE_MODEL=claude-haiku-4-5
  RADAR_DEEP_BACKEND=anthropic    RADAR_DEEP_MODEL=claude-opus-4-7
"""
from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from typing import Any

from utils.logger import logger

from . import xml_parser as xp
from .agents import _LLMBackend


# ==================== code<->name 权威校验 ====================
# LLM 经常把 code 和 name 张冠李戴 (e.g. "300035 钢研高纳" 实为中科电气).
# 优先走本地 JSON 缓存 (24h 有效), 失效才去 akshare 刷新;
# akshare 挂了但缓存还在 → 用过期缓存 (日内 name 变化极少, 比 fail-open 安全);
# 彻底拉不到才 fail-open, 不让校验层打垮主流程.
_NAME_MAP_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cache", "stock_name_map.json",
)
_NAME_MAP_TTL_S = 24 * 3600


def _load_name_map_cache() -> dict[str, str]:
    import json
    try:
        with open(_NAME_MAP_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_name_map_cache(m: dict[str, str]) -> None:
    import json
    try:
        os.makedirs(os.path.dirname(_NAME_MAP_CACHE), exist_ok=True)
        with open(_NAME_MAP_CACHE, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[name-check] 写缓存失败: {e}")


def _fetch_name_map_from_akshare() -> dict[str, str]:
    """主接口走交易所官网(stock_info_a_code_name), 失败退到东财 spot."""
    import akshare as ak
    try:
        df = ak.stock_info_a_code_name()
        return dict(zip(df["code"].astype(str).str.zfill(6), df["name"]))
    except Exception as e_primary:
        logger.info(f"[name-check] stock_info_a_code_name 失败 ({e_primary}), 退 spot_em")
        df = ak.stock_zh_a_spot_em()
        return dict(zip(df["代码"].astype(str).str.zfill(6), df["名称"]))


@lru_cache(maxsize=1)
def _code_name_map() -> dict[str, str]:
    """A 股 code→name 权威表. 24h 本地 JSON 缓存, 过期刷 akshare."""
    fresh = (
        os.path.exists(_NAME_MAP_CACHE)
        and (time.time() - os.path.getmtime(_NAME_MAP_CACHE) < _NAME_MAP_TTL_S)
    )
    if fresh:
        m = _load_name_map_cache()
        if m:
            logger.info(f"[name-check] 用本地缓存 {len(m)} 条")
            return m

    try:
        m = _fetch_name_map_from_akshare()
        _save_name_map_cache(m)
        logger.info(f"[name-check] 刷新 akshare {len(m)} 条, 已写缓存")
        return m
    except Exception as e:
        stale = _load_name_map_cache()
        if stale:
            logger.warning(
                f"[name-check] akshare 失败 ({e}), 回退过期缓存 {len(stale)} 条"
            )
            return stale
        logger.warning(f"[name-check] 无缓存且 akshare 失败, 降级不校验: {e}")
        return {}


def _normalize_name(s: str) -> str:
    """宽松匹配: 去掉空格/横杠和 ST/退/U/XD/DR 前后缀."""
    if not s:
        return ""
    s = re.sub(r"[\s\-]", "", s)
    s = re.sub(r"^(\*?ST|XD|DR)", "", s)
    s = re.sub(r"(\*?ST|XD|DR|退|U)$", "", s)
    return s


def _validate_code_name(code: str, name: str) -> tuple[str, str] | None:
    """校验 (code, name). 返回 (code, name_corrected) 或 None (应丢弃).

    规则:
      - code 格式非法 (非 6 位数字) → None
      - 映射表不可用 → 原样放行 (fail-open)
      - code 不在 A 股列表 → None (典型 LLM 瞎编代码)
      - name 为空或前后缀差异 → 填/覆盖为真 name
      - name 与真 name 明显不符 → 以真 name 为准, logger.warning
    """
    code = (code or "").strip()
    if not code.isdigit() or len(code) != 6:
        return None
    m = _code_name_map()
    if not m:
        return (code, (name or "").strip())
    true_name = m.get(code)
    if not true_name:
        logger.warning(f"[name-check] 丢弃: code={code} 不在 A 股列表 (LLM name='{name}')")
        return None
    if not name:
        return (code, true_name)
    if _normalize_name(name) == _normalize_name(true_name):
        return (code, true_name)
    logger.warning(
        f"[name-check] 纠正: code={code} LLM='{name}' → 真='{true_name}'"
    )
    return (code, true_name)


# ==================== 事件分类法 ====================
EVENT_TYPES = [
    "policy_macro",          # 利率/汇率/财政
    "policy_sector",         # 行业补贴/监管/准入
    "single_name_catalyst",  # 个股业绩/中标/调研/合作
    "theme_momentum",        # 题材异动,伴随明显价格行为
    "flow_data",             # 资金流/龙虎榜/大单
    "fundamental_data",      # 宏观/行业数据发布
    "geopolitical",          # 地缘/军事/外交
    "noise",                 # 无交易价值
]

TRADABILITY = ["high", "medium", "low", "none"]


# ==================== 后端选择 ====================
def _pick_backend(env_backend: str, env_model: str,
                  default_backend: str, default_model: str,
                  fallback_backend: str = "qwen",
                  fallback_model: str = "qwen-turbo") -> _LLMBackend:
    """按环境变量选 backend/model, Anthropic key 缺失时回落到国产."""
    backend = os.getenv(env_backend, default_backend)
    model = os.getenv(env_model, default_model)
    if backend == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        logger.warning(
            f"ANTHROPIC_API_KEY 未设, 回落到 {fallback_backend}/{fallback_model}. "
            f"想用 Claude 请在 .env 加 ANTHROPIC_API_KEY."
        )
        backend, model = fallback_backend, fallback_model
    return _LLMBackend(backend=backend, model=model)


def _triage_backend() -> _LLMBackend:
    return _pick_backend(
        "RADAR_TRIAGE_BACKEND", "RADAR_TRIAGE_MODEL",
        "anthropic", "claude-haiku-4-5",
        "qwen", "qwen-turbo",
    )


def _deep_backend() -> _LLMBackend:
    return _pick_backend(
        "RADAR_DEEP_BACKEND", "RADAR_DEEP_MODEL",
        "anthropic", "claude-opus-4-7",
        "qwen", "qwen-plus",
    )


# ==================== Triage Prompt ====================
_TRIAGE_PROMPT = """你是一个 A 股量化新闻事件分类器. 输入一条 radar 事件, 输出结构化标签.
严格按 XML 标签格式输出, 不要多余的叙述.

=== 分类法 ===
<EVENT_TYPE> 必为以下之一:
  policy_macro          宏观政策(利率/汇率/财政/外储)
  policy_sector         行业政策(补贴/监管/准入/反垄断)
  single_name_catalyst  个股催化(业绩/中标/调研/合作/高管变动)
  theme_momentum        题材异动(伴随明显价格行为, 如"XX 概念午后拉升")
  flow_data             资金流(龙虎榜/大单/北向/主力净流入)
  fundamental_data      宏观或行业数据发布(PMI/CPI/社融/销量)
  geopolitical          地缘/军事/外交
  noise                 无实际交易价值(会议致辞/常规外事/一般报道)

=== 实体格式 ===
<ENTITIES>: 分号分隔. 每个实体 "代码:名称:角色".
  代码缺失则 "--:XX公司:角色". 角色 ∈ {{受益, 受损, 竞品受益, 供应链受益, 无关}}.
  没有识别到任何 A 股相关实体就写 "NONE".

=== 其他字段 ===
<TRADABILITY> ∈ {{high, medium, low, none}}:
  high    明确个股/题材 + 盘中可交易
  medium  方向明确但标的较多, 需进一步筛选
  low     模糊利好或间接传导
  none    纯宏观/纯噪音, 无法直接落到标的
<NOVELTY> 0.0-1.0 浮点. 重复报道/复读类接近 0.2, 首发利好接近 0.8+
<NEEDS_DEEP> true/false. 以下情况设 true:
  - event_type 为 theme_momentum 或 single_name_catalyst
  - tradability 为 high
  - 供应链传导不明显但可能有二跳机会
<ONELINE> 不超过 60 字的事件定性总结.

=== 输出样例 ===
<EVENT_TYPE>theme_momentum</EVENT_TYPE>
<ENTITIES>300857:协创数据:受益;002261:拓维信息:受益</ENTITIES>
<TRADABILITY>high</TRADABILITY>
<NOVELTY>0.7</NOVELTY>
<NEEDS_DEEP>true</NEEDS_DEEP>
<ONELINE>算力租赁板块午后持续走强, 龙头协创数据涨 15% 创新高</ONELINE>

=== 待分类事件 ===
source: {source}
code(扩展猜测): {code}
score(扩展关键词分): {score}
tags: {tags}
title: {title}
content: {content}

开始输出(只输出 XML 标签):"""


def _fmt_triage_prompt(event: dict) -> str:
    return _TRIAGE_PROMPT.format(
        source=event.get("source", "?"),
        code=event.get("code") or "--",
        score=event.get("score", 0),
        tags=",".join(event.get("tags") or []) or "--",
        title=(event.get("title") or event.get("content") or "")[:200],
        content=(event.get("content") or event.get("title") or "")[:600],
    )


def _parse_entities(raw: str) -> list[dict]:
    """'300857:协创数据:受益;002261:拓维:受益' → [{code, name, role}, ...].

    对带 code 的实体做 code↔name 权威校验; 无 code 的实体(如 '--:XX:受益')原样保留,
    因为 triage 阶段允许模糊实体.
    """
    if not raw or raw.strip().upper() == "NONE":
        return []
    out = []
    for seg in raw.split(";"):
        parts = [p.strip() for p in seg.strip().split(":")]
        if len(parts) < 2:
            continue
        raw_code = parts[0] if parts[0] and parts[0] != "--" else ""
        raw_name = parts[1]
        role = parts[2] if len(parts) > 2 else ""
        if raw_code:
            validated = _validate_code_name(raw_code, raw_name)
            if validated is None:
                continue
            code, name = validated
        else:
            code, name = "", raw_name
        out.append({"code": code, "name": name, "role": role})
    return out


def _parse_bool(raw: str, default: bool = False) -> bool:
    if not raw:
        return default
    s = raw.strip().lower()
    return s in ("true", "yes", "1", "是")


def _parse_float(raw: str, default: float = 0.0) -> float:
    if not raw:
        return default
    try:
        import re as _re
        m = _re.search(r"-?\d+(?:\.\d+)?", raw)
        return float(m.group(0)) if m else default
    except Exception:
        return default


# ==================== 对外入口 ====================
def triage(event: dict) -> dict:
    """对一条 radar 事件做分类 + 初步判定.

    Args:
        event: 至少含 title/source, 可含 content/code/score/tags

    Returns:
        {
          event_type: str,          # EVENT_TYPES 之一
          entities: list[dict],
          tradability: str,         # TRADABILITY 之一
          novelty: float,           # 0.0-1.0
          needs_deep: bool,
          oneline: str,
          _raw: str,                # 原始 LLM 输出, 调试用
          _latency_s: float,
          _backend: str,
          _model: str,
        }
    """
    prompt = _fmt_triage_prompt(event)
    llm = _triage_backend()
    t0 = time.time()
    try:
        raw = llm.chat(prompt, max_tokens=500)
    except Exception as e:
        logger.error(f"triage LLM 调用失败: {e}")
        return {
            "event_type": "noise",
            "entities": [],
            "tradability": "none",
            "novelty": 0.0,
            "needs_deep": False,
            "oneline": f"[triage error] {e}",
            "_raw": "",
            "_latency_s": time.time() - t0,
            "_backend": llm.backend,
            "_model": llm.model,
            "_error": str(e),
        }

    event_type = xp.extract_tag(raw, "EVENT_TYPE").lower()
    if event_type not in EVENT_TYPES:
        event_type = "noise"
    tradability = xp.extract_tag(raw, "TRADABILITY").lower()
    if tradability not in TRADABILITY:
        tradability = "low"
    return {
        "event_type": event_type,
        "entities": _parse_entities(xp.extract_tag(raw, "ENTITIES")),
        "tradability": tradability,
        "novelty": max(0.0, min(1.0, _parse_float(xp.extract_tag(raw, "NOVELTY")))),
        "needs_deep": _parse_bool(xp.extract_tag(raw, "NEEDS_DEEP")),
        "oneline": xp.extract_tag(raw, "ONELINE")[:120],
        "_raw": raw,
        "_latency_s": round(time.time() - t0, 2),
        "_backend": llm.backend,
        "_model": llm.model,
    }


def should_deep_analyze(triage_result: dict) -> bool:
    """daemon 端的硬规则兜底, 避免过度信任模型 NEEDS_DEEP."""
    et = triage_result.get("event_type")
    tr = triage_result.get("tradability")
    if et == "noise" or tr == "none":
        return False
    if tr == "high":
        return True  # 高可交易性强制深挖
    return bool(triage_result.get("needs_deep"))


# ==================== Deep Analyze Prompt ====================
_DEEP_PROMPT = """你是一个 A 股量化分析师. 对一条已经过 triage 的新闻事件做决策级分析,
回答: 这条消息能不能做、做谁、怎么做、期限多长、是否已 price-in.

=== 推理框架 ===
1) 核心逻辑: 用 1-2 句点出新闻→市场影响的关键链路.
2) 传导链: 一跳受益/受损 (直接相关) → 二跳受益/受损 (供应链 / 替代品 / 竞争对手).
   注意区分:"事件标的"本身 vs "事件提到的标的" vs "因供应链传导应该关注的标的".
3) 是否已 price-in: 读每个标的的技术面文本块, 判断:
   - 20 日涨幅 > 同期 HS300 15%+ → partially priced in
   - 位于 250 日高点 > 90% → 大概率 fully/partially priced in, 追高风险大
   - 量比急升 + 位置低 → 刚启动, 可能还有空间
4) 给可交易候选: 代码 + 方向(long/short/avoid) + 进场参考 + 信心 + 半衰期(小时).
5) 风险标: 连板数/拥挤度/停牌风险/消息真伪.

=== 输出格式 ===
严格按 XML 标签. TARGETS 内部用 <TARGET>...</TARGET> 分块. 数字不要带单位中的空格.

<THESIS>核心影响逻辑, 不超过 60 字</THESIS>
<TARGETS>
<TARGET>
<CODE>300857</CODE>
<NAME>协创数据</NAME>
<DIRECTION>avoid</DIRECTION>
<THESIS>已5日+60%高位, 追高无安全边际</THESIS>
<ENTRY>--</ENTRY>
<CONF>0.75</CONF>
<HALF_LIFE>24</HALF_LIFE>
</TARGET>
<TARGET>
<CODE>002261</CODE>
<NAME>拓维信息</NAME>
<DIRECTION>long</DIRECTION>
<THESIS>同题材补涨股, 20日跑输龙头25个点</THESIS>
<ENTRY>回踩 22-23 区间</ENTRY>
<CONF>0.55</CONF>
<HALF_LIFE>48</HALF_LIFE>
</TARGET>
</TARGETS>
<ALREADY_PRICED_IN>partial</ALREADY_PRICED_IN>
<EVIDENCE>龙头250日区间97%已接近满, 跟风股仍在60日区间40%</EVIDENCE>
<SUPPLY_CHAIN>一跳:算力租赁直接受益企业;二跳:光模块/IDC代建商/液冷供应链</SUPPLY_CHAIN>
<RISK_FLAGS>题材5日拥挤度高;龙头连板风险;若大盘回调率先杀跌</RISK_FLAGS>

=== 事件信息 ===
source: {source}
title: {title}
content: {content}
triage_oneline: {oneline}
triage_event_type: {event_type}
triage_entities: {entities_fmt}

=== 标的技术面(从本地缓存读出的文本块, 无缓存的标的会显式标注) ===
{market_contexts}

=== 补充说明 ===
- 如果某个标的的技术面是 "无本地历史数据缓存", 你仍可以给出基于新闻本身的方向判断,
  但 CONF 应相应降低 (-0.1 到 -0.2), 并在 RISK_FLAGS 里标"无技术面验证".
- 如果 triage 判定 tradability=low, 宁可输出少量高质量候选 (TARGETS 可为空)
  也不要凑数.
- 半衰期 (HALF_LIFE) 用小时为单位. 常见值: 单日题材 8-24, 持续题材 48-120,
  基本面催化 120-240.
- 供应链分析优先使用常识推理, 不要编造不存在的公司代码.

开始输出(只输出 XML 标签):"""


def _format_entities(entities: list[dict]) -> str:
    if not entities:
        return "(triage 未识别出具体标的)"
    return "; ".join(
        f"{e.get('code') or '--'}:{e.get('name','')}:{e.get('role','')}"
        for e in entities
    )


def _inline_extract(body: str, tag: str) -> str:
    """容错的子标签抽取: 优先匹配闭合标签, 否则回退到下一个任意 <tag> 边界.

    专治 LLM 偶发性输出错误闭合 (如 <NAME>xxx</CODE>), 避免吃到后续所有文本.
    不依赖 xml_parser.KNOWN_TAGS, 因为 TARGET 子结构 (CODE/NAME/DIRECTION/...)
    不在那个白名单里.
    """
    import re as _re
    m = _re.search(
        rf"<{tag}>(.*?)</{tag}>",
        body, _re.DOTALL | _re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    # 回退: 到下一个任意 <xxx> 或 </xxx> 或字符串末尾
    m = _re.search(
        rf"<{tag}>(.*?)(?:<\/?\w+[^>]*>|$)",
        body, _re.DOTALL | _re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return ""


def _extract_targets(raw: str) -> list[dict]:
    """从 <TARGETS>...</TARGETS> 块中解析多个 <TARGET> 子结构."""
    from . import xml_parser as xp
    import re as _re

    targets_block = xp.extract_tag(raw, "TARGETS")
    if not targets_block:
        return []
    out = []
    for m in _re.finditer(r"<TARGET>(.*?)</TARGET>", targets_block, _re.DOTALL):
        body = m.group(1)
        code_raw = _inline_extract(body, "CODE")
        name_raw = _inline_extract(body, "NAME")
        validated = _validate_code_name(code_raw, name_raw)
        if validated is None:
            # code 非法或不在 A 股列表, 放弃该 target 防止幻觉流入下游
            continue
        code, name = validated
        t = {
            "code": code,
            "name": name,
            "direction": _inline_extract(body, "DIRECTION").lower(),
            "thesis": _inline_extract(body, "THESIS"),
            "entry": _inline_extract(body, "ENTRY"),
            "conf": _parse_float(_inline_extract(body, "CONF"), 0.0),
            "half_life_hours": int(_parse_float(_inline_extract(body, "HALF_LIFE"), 0)),
        }
        if t["direction"] not in ("long", "short", "avoid"):
            t["direction"] = "avoid"
        out.append(t)
    return out


def deep_analyze(event: dict, triage_result: dict,
                 market_contexts: dict[str, str] | None = None) -> dict:
    """Opus 深度分析, 产出可交易候选清单.

    Args:
        event: 原始 radar event dict
        triage_result: triage() 返回值
        market_contexts: {code: context_text}, 每个 entity 的技术面文本块.
                         None 时 deep_analyze 会自己懒加载 (避免强耦合).

    Returns:
        {
          thesis: str,
          targets: list[dict],       # [{code, name, direction, thesis, entry, conf, half_life_hours}]
          already_priced_in: str,    # full|partial|no
          evidence: str,
          supply_chain: str,
          risk_flags: str,
          _raw: str,
          _latency_s: float,
          _backend: str,
          _model: str,
        }
    """
    # 懒加载 market context
    if market_contexts is None:
        from . import market_context as mc
        market_contexts = {}
        codes_to_check = []
        for e in triage_result.get("entities", []):
            if e.get("code"):
                codes_to_check.append(e["code"])
        if event.get("code") and event["code"] not in codes_to_check:
            codes_to_check.append(event["code"])
        for code in codes_to_check[:6]:  # 上限防止 prompt 爆炸
            market_contexts[code] = mc.build_context_text(code)

    if market_contexts:
        mc_text = "\n".join(f"- {v}" for v in market_contexts.values())
    else:
        mc_text = "(没有关联个股代码, 仅凭新闻本身推理)"

    prompt = _DEEP_PROMPT.format(
        source=event.get("source", "?"),
        title=(event.get("title") or "")[:200],
        content=(event.get("content") or event.get("title") or "")[:800],
        oneline=triage_result.get("oneline", ""),
        event_type=triage_result.get("event_type", ""),
        entities_fmt=_format_entities(triage_result.get("entities", [])),
        market_contexts=mc_text,
    )

    llm = _deep_backend()
    t0 = time.time()
    try:
        raw = llm.chat(prompt, max_tokens=1500)
    except Exception as e:
        logger.error(f"deep_analyze LLM 调用失败: {e}")
        return {
            "thesis": f"[deep_analyze error] {e}",
            "targets": [],
            "already_priced_in": "",
            "evidence": "",
            "supply_chain": "",
            "risk_flags": "",
            "_raw": "",
            "_latency_s": time.time() - t0,
            "_backend": llm.backend,
            "_model": llm.model,
            "_error": str(e),
        }

    return {
        "thesis": xp.extract_tag(raw, "THESIS")[:200],
        "targets": _extract_targets(raw),
        "already_priced_in": xp.extract_tag(raw, "ALREADY_PRICED_IN").strip().lower() or "no",
        "evidence": xp.extract_tag(raw, "EVIDENCE")[:400],
        "supply_chain": xp.extract_tag(raw, "SUPPLY_CHAIN")[:400],
        "risk_flags": xp.extract_tag(raw, "RISK_FLAGS")[:400],
        "_raw": raw,
        "_latency_s": round(time.time() - t0, 2),
        "_backend": llm.backend,
        "_model": llm.model,
    }

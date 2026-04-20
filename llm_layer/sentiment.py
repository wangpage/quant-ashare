"""新闻/公告/股吧 情绪分析, 输出 [-1, 1] 的情绪得分作为量价外的附加特征.

数据源:
  - akshare: stock_news_em (东财个股新闻)
  - akshare: news_cctv (央视新闻 - 宏观)
  - 股吧: 东财股吧帖子 (爬虫, 此处省略)

情绪打分方式:
  - 默认: SnowNLP (零成本 baseline)
  - 可选: LLM (OpenAI/Anthropic) - 更准但要 API key
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from utils.logger import logger


@dataclass
class SentimentScore:
    code: str
    date: str
    score: float               # [-1, +1]
    confidence: float          # [0, 1]
    sample_size: int
    backend: str


class NewsSentimentAnalyzer:
    def __init__(
        self,
        backend: Literal["snownlp", "openai", "anthropic"] = "snownlp",
        model: str = "claude-haiku-4-5",
    ):
        self.backend = backend
        self.model = model
        self._llm_client = None

    # ---------- 数据抓取 ----------
    def fetch_news(self, code: str, limit: int = 30) -> pd.DataFrame:
        import akshare as ak
        try:
            df = ak.stock_news_em(symbol=code)
        except Exception as e:
            logger.warning(f"[{code}] 新闻抓取失败: {e}")
            return pd.DataFrame()
        if df.empty:
            return df
        # 按日期倒序, 取最近 limit 条
        return df.head(limit)

    # ---------- SnowNLP baseline ----------
    @staticmethod
    def _snownlp_score(texts: list[str]) -> float:
        from snownlp import SnowNLP
        if not texts:
            return 0.0
        scores = []
        for t in texts:
            try:
                s = SnowNLP(t).sentiments  # [0, 1]
                scores.append(s * 2 - 1)   # 转 [-1, 1]
            except Exception:
                pass
        return float(sum(scores) / len(scores)) if scores else 0.0

    # ---------- LLM scoring ----------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _llm_score(self, texts: list[str]) -> tuple[float, float]:
        prompt = (
            "你是A股分析师。给定一组新闻标题+摘要, 综合判断对该股价短期(1-5日)走势的影响。"
            "输出严格的JSON: {\"score\": -1到1的小数, \"confidence\": 0到1, \"reason\": \"简要\"}。"
            "score>0 看涨, <0 看跌, 0 中性。\n\n新闻:\n" +
            "\n".join(f"- {t[:200]}" for t in texts[:20])
        )
        if self.backend == "anthropic":
            import anthropic
            if self._llm_client is None:
                self._llm_client = anthropic.Anthropic()
            resp = self._llm_client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
        elif self.backend == "openai":
            import openai
            if self._llm_client is None:
                self._llm_client = openai.OpenAI()
            resp = self._llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            text = resp.choices[0].message.content
        else:
            raise ValueError(self.backend)
        import json, re
        m = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if not m:
            return 0.0, 0.0
        try:
            j = json.loads(m.group(0))
            return float(j.get("score", 0)), float(j.get("confidence", 0))
        except Exception:
            return 0.0, 0.0

    # ---------- 对外接口 ----------
    def score(self, code: str) -> SentimentScore:
        df = self.fetch_news(code, limit=30)
        today = time.strftime("%Y-%m-%d")
        if df.empty:
            return SentimentScore(code, today, 0.0, 0.0, 0, self.backend)

        # 兼容 akshare 不同版本字段
        title_col = "新闻标题" if "新闻标题" in df.columns else df.columns[0]
        content_col = "新闻内容" if "新闻内容" in df.columns else title_col
        texts = [f"{r[title_col]} {r.get(content_col, '')}" for _, r in df.iterrows()]

        if self.backend == "snownlp":
            s = self._snownlp_score(texts)
            return SentimentScore(code, today, s, 0.5, len(texts), "snownlp")
        else:
            s, conf = self._llm_score(texts)
            return SentimentScore(code, today, s, conf, len(texts), self.backend)

    def score_batch(self, codes: list[str]) -> pd.DataFrame:
        out = []
        for c in codes:
            try:
                out.append(self.score(c).__dict__)
            except Exception as e:
                logger.warning(f"[{c}] sentiment 失败: {e}")
            time.sleep(0.2)
        return pd.DataFrame(out)


if __name__ == "__main__":
    a = NewsSentimentAnalyzer(backend="snownlp")
    print(a.score("600519"))

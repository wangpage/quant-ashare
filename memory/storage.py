"""记忆存储层: SQLite + FTS5 全文检索.

为什么不用 chromadb / faiss?
  - chromadb 依赖重, 且对中文分词支持不佳
  - FTS5 自带 unicode61 tokenizer + trigram 近似
  - 我们先跑起来, 后期可切换成向量检索

表设计:
  memories          主存储: id, kind, ts, code, content, outcome_pnl, metadata_json
  memories_fts      FTS5 索引: content, code, kind
  skills            自动生成的交易 skill: id, pattern_name, conditions,
                    actions, success_rate, sample_count, created_at
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

from utils.config import PROJECT_ROOT
from utils.logger import logger


MemoryKind = Literal["reflection", "rule", "risk_lesson",
                     "fundamental", "technical", "sentiment",
                     "trade", "market_event"]


@dataclass
class MemoryRecord:
    id: int | None
    kind: str
    ts: int                        # Unix 秒
    code: str                      # 关联股票, 无则 ""
    content: str                   # 反思/规则 文本
    outcome_pnl: float | None      # 交易结果百分比, 规则类则 None
    metadata: dict                 # 自由字段


class MemoryStore:
    """线程安全的记忆存储."""

    def __init__(self, db_path: str | Path | None = None):
        db_path = Path(db_path) if db_path else PROJECT_ROOT / "cache" / "memory.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def _init_schema(self):
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    ts INTEGER NOT NULL,
                    code TEXT DEFAULT '',
                    content TEXT NOT NULL,
                    outcome_pnl REAL,
                    metadata_json TEXT DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
                CREATE INDEX IF NOT EXISTS idx_memories_code ON memories(code);
                CREATE INDEX IF NOT EXISTS idx_memories_ts ON memories(ts);

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content, code, kind,
                    tokenize = 'trigram'
                );

                CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content, code, kind)
                    VALUES (new.id, new.content, new.code, new.kind);
                END;
                CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memories BEGIN
                    DELETE FROM memories_fts WHERE rowid = old.id;
                END;
                CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE ON memories BEGIN
                    UPDATE memories_fts SET content = new.content,
                        code = new.code, kind = new.kind
                    WHERE rowid = new.id;
                END;

                CREATE TABLE IF NOT EXISTS skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_name TEXT UNIQUE NOT NULL,
                    conditions TEXT NOT NULL,
                    actions TEXT NOT NULL,
                    success_rate REAL DEFAULT 0.0,
                    avg_pnl REAL DEFAULT 0.0,
                    sample_count INTEGER DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    last_updated INTEGER NOT NULL,
                    active INTEGER DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_skills_active ON skills(active);
            """)

    # ---------- 写 ----------
    def add(
        self, kind: str, content: str,
        code: str = "", outcome_pnl: float | None = None,
        metadata: dict | None = None, ts: int | None = None,
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO memories(kind, ts, code, content, outcome_pnl, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (kind, ts or int(time.time()), code, content, outcome_pnl,
                 json.dumps(metadata or {}, ensure_ascii=False)),
            )
            return cur.lastrowid

    def add_many(self, records: list[dict]) -> int:
        n = 0
        for r in records:
            self.add(**r)
            n += 1
        return n

    # ---------- 读 ----------
    def recent(self, kind: str | None = None, code: str = "",
               days: int = 30, limit: int = 50) -> list[MemoryRecord]:
        cutoff = int(time.time()) - days * 86400
        sql = "SELECT * FROM memories WHERE ts >= ?"
        params: list[Any] = [cutoff]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        if code:
            sql += " AND code = ?"
            params.append(code)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def search(self, query: str, kind: str | None = None,
               limit: int = 10) -> list[MemoryRecord]:
        """全文检索: trigram FTS5 优先, 短查询(<3字)回退 LIKE."""
        if not query or not query.strip():
            return []

        use_fts = len(query.strip()) >= 3

        if use_fts:
            sql = ("SELECT m.* FROM memories_fts f "
                   "JOIN memories m ON f.rowid = m.id "
                   "WHERE memories_fts MATCH ?")
            params: list[Any] = [query]
            if kind:
                sql += " AND m.kind = ?"
                params.append(kind)
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)
            try:
                with self._conn() as c:
                    rows = c.execute(sql, params).fetchall()
                if rows:
                    return [self._row_to_record(r) for r in rows]
            except sqlite3.OperationalError:
                pass

        # 回退: LIKE 扫描 (短查询 or FTS 无果)
        sql = "SELECT * FROM memories WHERE content LIKE ?"
        params = [f"%{query}%"]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def delete(self, id: int):
        with self._conn() as c:
            c.execute("DELETE FROM memories WHERE id = ?", (id,))

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"], kind=row["kind"], ts=row["ts"],
            code=row["code"], content=row["content"],
            outcome_pnl=row["outcome_pnl"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    # ---------- skill 相关 ----------
    def upsert_skill(self, pattern_name: str, conditions: str, actions: str,
                     success_rate: float, avg_pnl: float,
                     sample_count: int) -> int:
        now = int(time.time())
        with self._conn() as c:
            existing = c.execute(
                "SELECT id FROM skills WHERE pattern_name = ?", (pattern_name,)
            ).fetchone()
            if existing:
                c.execute("""
                    UPDATE skills SET conditions=?, actions=?,
                        success_rate=?, avg_pnl=?, sample_count=?,
                        last_updated=?
                    WHERE id=?
                """, (conditions, actions, success_rate, avg_pnl,
                      sample_count, now, existing["id"]))
                return existing["id"]
            cur = c.execute("""
                INSERT INTO skills(pattern_name, conditions, actions,
                    success_rate, avg_pnl, sample_count,
                    created_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (pattern_name, conditions, actions, success_rate,
                  avg_pnl, sample_count, now, now))
            return cur.lastrowid

    def active_skills(self, min_samples: int = 5,
                      min_success: float = 0.55) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("""
                SELECT * FROM skills
                WHERE active=1 AND sample_count >= ? AND success_rate >= ?
                ORDER BY success_rate DESC
            """, (min_samples, min_success)).fetchall()
        return [dict(r) for r in rows]

    def deactivate_skill(self, id: int):
        with self._conn() as c:
            c.execute("UPDATE skills SET active=0 WHERE id=?", (id,))

    def stats(self) -> dict:
        with self._conn() as c:
            n_mem = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            n_by_kind = dict(c.execute(
                "SELECT kind, COUNT(*) FROM memories GROUP BY kind"
            ).fetchall())
            n_skill = c.execute(
                "SELECT COUNT(*) FROM skills WHERE active=1"
            ).fetchone()[0]
        return {"memories_total": n_mem, "by_kind": n_by_kind,
                "active_skills": n_skill}


# ==================== 便捷函数 ====================
_default_store: MemoryStore | None = None


def default_store() -> MemoryStore:
    global _default_store
    if _default_store is None:
        _default_store = MemoryStore()
    return _default_store


def recall_for_agent(code: str, kind: str,
                     query: str = "", limit: int = 3) -> str:
    """给 agent 用的格式化召回: 返回 3-5 条相关经验的纯文本."""
    store = default_store()
    records = []
    if query:
        records = store.search(query, kind=kind, limit=limit)
    if not records and code:
        records = store.recent(kind=kind, code=code, days=180, limit=limit)
    if not records:
        records = store.recent(kind=kind, days=90, limit=limit)
    if not records:
        return "无"
    lines = []
    for i, r in enumerate(records, 1):
        pnl = f" [实盘{r.outcome_pnl:+.1%}]" if r.outcome_pnl is not None else ""
        lines.append(f"{i}. [{r.kind}]{pnl} {r.content[:200]}")
    return "\n".join(lines)

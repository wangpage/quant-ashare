"""Stock Radar 扩展配套 API 服务 (独立 FastAPI, 不动 Streamlit webapp).

Week 1 范围:
  - POST /radar/events  接收插件上行事件流, 写入 cache/memory.db
  - GET  /radar/stats   自检: 今日事件数 / 分源 / 高分数
  - GET  /healthz       存活检查

启动:
  pip install fastapi uvicorn
  uvicorn api_server:app --port 9876
  (Chrome 扩展默认 push 到 http://localhost:9876)

后续 (Week 2+):
  GET /radar/signals  —— 包装 watchlist_signal.py 输出
  GET /radar/config   —— 下发席位/题材/黑白名单字典
"""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from memory.storage import MemoryStore

log = logging.getLogger("radar_api")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="quant_ashare Radar API", version="0.1.0")

# Chrome 扩展跨域, 宽松放开 (localhost 本机使用, 非公开服务)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

store = MemoryStore()


@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": int(time.time())}


@app.post("/radar/events")
async def ingest_radar_events(req: Request) -> dict[str, Any]:
    payload = await req.json()
    events = payload.get("events") or []
    if not isinstance(events, list):
        raise HTTPException(400, "events must be a list")
    if len(events) > 500:
        raise HTTPException(413, "batch too large, max 500")

    n = store.add_radar_events(events)
    log.info("radar ingest: received=%d inserted=%d (source=%s)",
             len(events), n,
             ",".join({e.get("source", "?") for e in events[:20]}))
    return {
        "received": len(events),
        "inserted": n,
        "duplicates": len(events) - n,
        "ts": int(time.time()),
    }


@app.get("/radar/stats")
def radar_stats(hours: int = 24) -> dict[str, Any]:
    since = int(time.time()) - hours * 3600
    s = store.radar_stats(since)
    return {"window_hours": hours, **s}


@app.get("/radar/recent")
def radar_recent(source: str | None = None, min_score: int = 0,
                 limit: int = 50, hours: int = 24) -> dict[str, Any]:
    since = int(time.time()) - hours * 3600
    events = store.query_radar_events(
        since_ts=since, source=source, min_score=min_score, limit=limit)
    return {
        "count": len(events),
        "events": [
            {
                "ts": e.ts, "code": e.code, "content": e.content,
                "source": e.metadata.get("source"),
                "score": e.metadata.get("score"),
                "tags": e.metadata.get("tags"),
                "url": e.metadata.get("url"),
            }
            for e in events
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="127.0.0.1", port=9876, reload=False)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend_service import memory_store, process_query
from law_query_categories import normalize_category
from standards import load_standards


class QueryRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    category: str = "auto"


app = FastAPI(title="Wastewater Law Query API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def sse_event(event: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {data}\n\n"


def run_graph_query(query: str) -> dict[str, Any]:
    from graph import run_query

    return run_query(query)


def query_stream(request: QueryRequest) -> Iterable[str]:
    if not request.query.strip():
        yield sse_event("error", {"message": "Query is required."})
        return

    category = normalize_category(request.category)

    try:
        yield sse_event("status", {"status": "memory", "label": "整理對話記憶"})
        yield sse_event("status", {"status": "retrieving", "label": "依查詢面向檢索法規"})
        yield sse_event("status", {"status": "judging", "label": "建立結構化判斷"})
        payload = process_query(
            query=request.query,
            conversation_id=request.conversation_id,
            category=category,
            run_graph=run_graph_query,
        )
        yield sse_event("final", payload)
        yield sse_event("status", {"status": "done", "label": "已完成"})
    except ValueError as exc:
        yield sse_event("error", {"message": str(exc)})
    except Exception as exc:
        yield sse_event("error", {"message": f"Backend query failed: {exc}"})


@app.get("/api/health")
def health() -> dict[str, Any]:
    standards = load_standards()
    chroma_path = Path("chroma_db")
    return {
        "status": "ok",
        "standards_count": len(standards),
        "conversation_count": memory_store.count(),
        "rag_db_exists": chroma_path.exists(),
        "rag_db_path": str(chroma_path),
        "categories": [
            "auto",
            "effluent_standard",
            "industry_scope",
            "sewer_system",
            "facility_special",
            "permit_plan",
            "monitoring_reporting",
            "penalty",
        ],
    }


@app.post("/api/query")
def query(request: QueryRequest) -> StreamingResponse:
    return StreamingResponse(
        query_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

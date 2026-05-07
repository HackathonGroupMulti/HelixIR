"""
FastAPI server that drives the HelixIR dashboard.

Endpoints
---------
GET  /api/report          Latest profile report (JSON)
POST /api/profile         Submit a new analysis result
GET  /api/benchmarks      Latest benchmark results
POST /api/benchmarks      Append a benchmark result
WS   /ws                  Live event stream (broadcasts on every POST)

The server is intentionally stateless between restarts — it holds the last
report and benchmark list in memory.  Run with:

    helix serve            # via CLI
    uvicorn helix.server:app --reload --port 8765
"""
from __future__ import annotations
import asyncio
import json
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="HelixIR", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# In-memory state
# --------------------------------------------------------------------------

_latest_report: dict | None = None
_benchmark_results: list[dict] = []
_ws_clients: set[WebSocket] = set()


# --------------------------------------------------------------------------
# WebSocket broadcast helper
# --------------------------------------------------------------------------

async def _broadcast(event: str, payload: Any) -> None:
    dead: set[WebSocket] = set()
    msg = json.dumps({"event": event, "data": payload})
    for ws in list(_ws_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


# --------------------------------------------------------------------------
# REST endpoints
# --------------------------------------------------------------------------

@app.get("/api/report")
async def get_report() -> JSONResponse:
    if _latest_report is None:
        return JSONResponse({"error": "No report available yet"}, status_code=404)
    return JSONResponse(_latest_report)


class ProfilePayload(BaseModel):
    graph: dict
    roofline: dict
    passes: list[dict]
    num_ops: int
    total_flops: int
    total_bytes: int


@app.post("/api/profile")
async def post_profile(payload: ProfilePayload) -> dict:
    global _latest_report
    _latest_report = payload.model_dump()
    asyncio.create_task(_broadcast("report", _latest_report))
    return {"status": "ok"}


@app.get("/api/benchmarks")
async def get_benchmarks() -> list[dict]:
    return _benchmark_results


class BenchmarkPayload(BaseModel):
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    iterations: int
    flops: int = 0
    achieved_tflops: float = 0.0
    peak_tflops: float = 0.0
    efficiency_pct: float = 0.0


@app.post("/api/benchmarks")
async def post_benchmark(payload: BenchmarkPayload) -> dict:
    entry = payload.model_dump()
    _benchmark_results.append(entry)
    asyncio.create_task(_broadcast("benchmark", entry))
    return {"status": "ok", "total": len(_benchmark_results)}


@app.delete("/api/benchmarks")
async def clear_benchmarks() -> dict:
    _benchmark_results.clear()
    asyncio.create_task(_broadcast("benchmarks_cleared", {}))
    return {"status": "ok"}


# --------------------------------------------------------------------------
# WebSocket endpoint
# --------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _ws_clients.add(ws)
    try:
        # Send current state immediately on connect
        if _latest_report:
            await ws.send_text(json.dumps({"event": "report", "data": _latest_report}))
        if _benchmark_results:
            await ws.send_text(json.dumps({"event": "benchmarks_init", "data": _benchmark_results}))

        while True:
            await ws.receive_text()   # keep-alive; client messages ignored
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


# --------------------------------------------------------------------------
# Utility: push a helix.analyze() result from Python
# --------------------------------------------------------------------------

def push_report(report: dict, server_url: str = "http://localhost:8765") -> None:
    """
    Convenience helper — call from a training script to push the HelixIR
    analysis report to the running dashboard server.

        report = helix.analyze(my_fn, *args)
        helix.server.push_report(report)
    """
    import urllib.request

    def _serialise(obj: Any) -> Any:
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, (list, tuple)):
            return [_serialise(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        return obj

    payload = _serialise(report)
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{server_url}/api/profile",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())

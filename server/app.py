# server/app.py
import asyncio
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.middleware.cors import CORSMiddleware

# ---- import your DB models/helpers ----
from event_log import Session as DBSession, Event, Person, Vehicle  # adjust import path if needed

app = FastAPI(title="Surveillance Dashboard", version="0.1")

# If you open client from another origin/port, enable CORS here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the standalone client
app.mount("/static", StaticFiles(directory="server/web"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("server/web/index.html")


# ---------- DB Session Dependency ----------
def get_db():
    db = DBSession()
    try:
        yield db
    finally:
        db.close()


# ---------- WebSocket Manager ----------
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: Dict[str, Any]):
        if not self.active:
            return
        data = json.dumps(message, default=str)
        # broadcast to all; remove dead connections
        stale = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)

manager = ConnectionManager()

# ---------- In-memory metrics (live only) ----------
metrics = {
    "workers_active": 0,
    "lpr_queue_size": 0,
    "active_tracks": 0,
    "last_update": None
}

# ---------- Pydantic models for ingest ----------
class TelemetryIn(BaseModel):
    workers_active: int
    lpr_queue_size: int
    active_tracks: int

class LiveDetectionIn(BaseModel):
    track_id: str
    label: str
    confidence: float
    license_plate: Optional[str] = None


# ---------- WebSocket endpoint ----------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            # This server is push-only; if you want client->server messages,
            # you can await ws.receive_text() here.
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ---------- Telemetry ingest (from detection pipeline) ----------
@app.post("/api/ingest/telemetry")
async def ingest_telemetry(t: TelemetryIn):
    metrics["workers_active"] = t.workers_active
    metrics["lpr_queue_size"] = t.lpr_queue_size
    metrics["active_tracks"] = t.active_tracks
    metrics["last_update"] = datetime.utcnow()

    await manager.broadcast({"type": "metrics", "data": metrics})
    return {"status": "ok"}


# ---------- Live detection ingest (from detection pipeline) ----------
@app.post("/api/ingest/live")
async def ingest_live(d: LiveDetectionIn):
    payload = {
        "track_id": d.track_id,
        "label": d.label,
        "confidence": d.confidence,
        "license_plate": d.license_plate
    }
    await manager.broadcast({"type": "live", "data": payload})
    return {"status": "ok"}


# ---------- DB endpoints: recent events ----------
@app.get("/api/events/recent")
def get_recent_events(limit: int = 100, db: Session = Depends(get_db)):
    """
    Returns recent events joined with subclass tables when present.
    """
    q = (
        db.query(Event, Vehicle, Person)
        .outerjoin(Vehicle, Vehicle.event_id == Event.event_id)
        .outerjoin(Person, Person.event_id == Event.event_id)
        .order_by(Event.last_seen.desc())
        .limit(limit)
        .all()
    )

    rows = []
    for ev, veh, per in q:
        rows.append({
            "event_id": ev.event_id,
            "object_id": ev.object_id,
            "class_type": ev.class_type,
            "time_first_detected": ev.time_first_detected,
            "last_seen": ev.last_seen,
            "vehicle": {
                "vehicle_type": getattr(veh, "vehicle_type", None),
                "color": getattr(veh, "color", None),
                "license_plate": getattr(veh, "license_plate", None),
            } if veh else None,
            "person": {
                "appearance": getattr(per, "appearance", None),
                "behavior": getattr(per, "human_behavior", None).name if per and per.human_behavior else None
            } if per else None
        })
    return rows


# ---------- Simple metrics GET (for initial load) ----------
@app.get("/api/metrics")
def get_metrics():
    return metrics

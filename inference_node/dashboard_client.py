import requests
import time
from config.secret import server_base


def post_telemetry(_last_metrics_post, workers_active: int, lpr_queue_size: int, active_tracks: int, camera_id="cam_01"):
    """Live health and performance stats"""
    now = time.time()
    if now - _last_metrics_post < 0.8:
        return
    _last_metrics_post = now
    try:
        requests.post(f"{server_base}/api/ingest/telemetry", json={
            "camera_id": camera_id,
            "workers_active": workers_active,
            "lpr_queue_size": lpr_queue_size,
            "active_tracks": active_tracks
        }, timeout=0.4)
    except Exception:
        pass


def post_live(_last_live_sent, live_throttle_sec, track_id: str, label: str, confidence: float, license_plate: str | None = None, camera_id="cam_01"):
    """Live feed from camera"""
    now = time.time()
    last = _last_live_sent.get(track_id, 0)
    if now - last < live_throttle_sec and license_plate is None:
        return
    _last_live_sent[track_id] = now
    try:
        requests.post(f"{server_base}/api/ingest/live", json={
            "camera_id": camera_id,
            "track_id": str(track_id),
            "label": label,
            "confidence": float(confidence),
            "license_plate": license_plate
        }, timeout=0.4)
    except Exception:
        pass
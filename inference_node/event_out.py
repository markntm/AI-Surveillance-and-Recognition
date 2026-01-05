import time
import requests

SERVER_BASE = "http://127.0.0.1:8000"

def emit(event: dict):
    event.setdefault("timestamp", time.time())
    try:
        requests.post(
            f"{SERVER_BASE}/api/ingest/event",
            json=event,
            timeout=0.3
        )
    except Exception:
        pass

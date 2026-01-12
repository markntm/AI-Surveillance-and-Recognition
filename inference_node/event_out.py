import time
import requests
from config.secret import server_base


def emit(event: dict):
    event.setdefault("timestamp", time.time())
    try:
        requests.post(
            f"{server_base}/api/ingest/event",
            json=event,
            timeout=0.3
        )
    except Exception:
        pass

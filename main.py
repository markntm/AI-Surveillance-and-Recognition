# main.py
# Orchestrates capture -> COCO detect -> tracking -> enqueue for LPR -> overlay results.
import time
import cv2
import numpy as np
import queue
import threading

from obj_detector import ObjectDetector
from obj_tracker import Tracker
from lpr_worker import LPRWorker
from utilities import crop_bbox
import mss  # if you want screen capture

# ---------------- CONFIG ----------------
YOLO_COCO_PATH = "yolo11n.pt"
YOLO_LPR_PATH = "ultralytics/runs/detect/train2/weights/best.pt"
TESSEACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows example
NUM_LPR_WORKERS = 1
LPR_QUEUE_MAXSIZE = 8
LPR_COOLDOWN_SECONDS = 3.0   # wait this long before re-processing same track
PLATE_RETENTION_SECONDS = 60  # keep plate text in cache for this long

# Vehicle class ids (COCO): 2=car,3=motorbike,5=bus,7=truck (verify via model.names)
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

# ---------------- Setup ----------------


detector = ObjectDetector(YOLO_COCO_PATH, conf=0.35)
tracker = Tracker()

lpr_task_q = queue.Queue(maxsize=LPR_QUEUE_MAXSIZE)
lpr_result_q = queue.Queue()

# Cache & bookkeeping
last_lpr_request = {}   # track_id -> timestamp when we last enqueued LPR for this track
plate_cache = {}        # track_id -> {'text':..., 'conf':..., 'ts':...}

# Video source (webcam)
cap = cv2.VideoCapture(0)

# Start LPR worker threads
workers = []


def worker_setup():
    for i in range(NUM_LPR_WORKERS):
        w = LPRWorker(lpr_task_q, lpr_result_q, YOLO_LPR_PATH, tesseract_cmd=TESSEACT_CMD, conf=0.2, name=f"LPR-{i}")
        w.start()
        workers.append(w)


def YOLO_programme():
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # 1) Detect objects (COCO)
            detections = detector.detect(frame)  # list of dicts

            # 2) Track (DeepSORT)
            tracks = tracker.update(detections, frame)

            # 3) Display & queue LPR tasks for vehicles
            for tr in tracks:
                tid = tr["track_id"]
                x1,y1,x2,y2 = tr["bbox"]
                label = tr["label"] if tr["label"] is not None else "obj"

                # Draw box + id
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"ID {tid} | {label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # If this track is a vehicle, consider LPR
                if tr.get("label") in [detector.model.names[i] for i in VEHICLE_CLASS_IDS]:
                    # cooldown check
                    now = time.time()
                    last = last_lpr_request.get(tid, 0)
                    if now - last > LPR_COOLDOWN_SECONDS:
                        # crop vehicle region
                        vehicle_crop = crop_bbox(frame, tr["bbox"], pad=0.05)  # slight padding
                        if vehicle_crop is not None and not lpr_task_q.full():
                            try:
                                lpr_task_q.put_nowait((tid, vehicle_crop, {"frame_ts": now}))
                                last_lpr_request[tid] = now
                            except queue.Full:
                                pass

            # 4) Handle LPR results that workers produced
            while not lpr_result_q.empty():
                res = lpr_result_q.get_nowait()
                tid = res["track_id"]
                plate_cache[tid] = {
                    "text": res["plate_text"],
                    "conf": res["plate_conf"],
                    "ts": res["ts"]
                }

            # 5) Overlay plate_cache on frame (for tracks still visible)
            for tid, data in list(plate_cache.items()):
                # Remove stale cache entries
                if time.time() - data["ts"] > PLATE_RETENTION_SECONDS:
                    del plate_cache[tid]
                    continue
                # Find track bbox for overlay
                track_bbox = next((tr["bbox"] for tr in tracks if tr["track_id"] == tid), None)
                if track_bbox:
                    x1,y1,x2,y2 = track_bbox
                    txt = f"{data['text']} ({data['conf']:.2f})"
                    cv2.putText(frame, txt, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            # 6) FPS and display
            fps = 1.0 / max(1e-6, time.time() - t0)
            cv2.putText(frame, f"FPS: {fps:.2f}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            cv2.imshow("Vehicle + LPR (parallel)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        # Stop workers
        for w in workers:
            w.stop()
        # Wait for them to finish
        for w in workers:
            w.join(timeout=2)


if __name__ == "__main__":
    worker_setup()
    YOLO_programme()

import time
import cv2
import queue

from obj_detector import ObjectDetector
from obj_tracker import Tracker
from lpr_worker import LPRWorker
from utilities import crop_bbox, extract_dominant_color, infer_human_behavior
from event_log import log_event, log_person, log_vehicle, HumanBehavior, VehicleBehavior
from const import constants

# ---------------- Setup ----------------
detector = ObjectDetector(constants["YOLO_COCO_PATH"], conf=0.35)
tracker = Tracker()

lpr_task_q = queue.Queue(maxsize=constants["LPR_QUEUE_MAXSIZE"])
lpr_result_q = queue.Queue()

# Cache & bookkeeping
last_lpr_request = {}   # track_id -> timestamp when we last enqueued LPR for this track
plate_cache = {}        # track_id -> {'text':..., 'conf':..., 'ts':...}

# Video source (webcam)
cap = cv2.VideoCapture(0)

# Start LPR worker threads
workers = []


def worker_setup():
    for i in range(constants["NUM_LPR_WORKERS"]):
        w = LPRWorker(lpr_task_q, lpr_result_q, constants["YOLO_LPR_PATH"], tesseract_cmd=constants["TESSERACT_CMD"], conf=0.2, name=f"LPR-{i}")
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
                tid = str(tr["track_id"])
                x1,y1,x2,y2 = tr["bbox"]
                label = tr["label"] if tr["label"] is not None else "obj"

                # --- DB: log/update Event ---
                event = log_event(tid, label)

                # Draw box + id
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"ID {tid} | {label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # If this track is a vehicle, consider LPR
                if tr.get("label") in [detector.model.names[i] for i in constants["VEHICLE_CLASS_IDS"]]:
                    # cooldown check
                    now = time.time()
                    last = last_lpr_request.get(tid, 0)

                    # --- DB: vehicle info without plate ---
                    vehicle_type = label
                    vehicle_color = extract_dominant_color(frame, tr["bbox"])
                    log_vehicle(event, vehicle_type, vehicle_color, None, VehicleBehavior.unknown)

                    if now - last > constants["LPR_COOLDOWN_SECONDS"]:
                        # crop vehicle region
                        vehicle_crop = crop_bbox(frame, tr["bbox"], pad=0.05)  # slight padding
                        if vehicle_crop is not None and not lpr_task_q.full():
                            try:
                                lpr_task_q.put_nowait((tid, vehicle_crop, {"frame_ts": now}))
                                last_lpr_request[tid] = now
                            except queue.Full:
                                pass

                elif tr.get("label") == "person":
                    appearance = "unknown"
                    behavior = infer_human_behavior(tr) or HumanBehavior.unknown
                    log_person(event, appearance, behavior)

            # 4) Handle LPR results that workers produced
            while not lpr_result_q.empty():
                res = lpr_result_q.get_nowait()
                tid = res["track_id"]
                plate_cache[tid] = {
                    "text": res["plate_text"],
                    "conf": res["plate_conf"],
                    "ts": res["ts"]
                }

                # --- DB: update vehicle row with plate ---
                event = log_event(tid, "vehicle")
                existing = plate_cache[tid]
                log_vehicle(event, None, None, existing["text"], VehicleBehavior.unknown)


            # 5) Overlay plate_cache on frame (for tracks still visible)
            for tid, data in list(plate_cache.items()):
                # Remove stale cache entries
                if time.time() - data["ts"] > constants["PLATE_RETENTION_SECONDS"]:
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

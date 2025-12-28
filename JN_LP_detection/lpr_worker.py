import threading
import time
import numpy as np
import cv2
import pytesseract
from ultralytics import YOLO

class LPRWorker(threading.Thread):
    def __init__(self, task_queue, result_queue, lpr_model_path, tesseract_cmd=None, conf=0.2, name=None):
        super().__init__(daemon=True, name=name)
        self.task_queue = task_queue        # receives tuples (track_id, vehicle_crop, meta)
        self.result_queue = result_queue    # sends back dict {track_id, plate_text, plate_conf, ts}
        self.lpr_model_path = lpr_model_path
        self.conf = conf
        self.stop_flag = threading.Event()
        self.tesseract_cmd = tesseract_cmd

    def run(self):
        # Load an LPR model inside the thread to avoid sharing model across threads
        lpr_model = YOLO(self.lpr_model_path)

        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

        while not self.stop_flag.is_set():
            try:
                task = self.task_queue.get(timeout=0.5)  # (track_id, vehicle_crop, meta)
            except Exception:
                continue

            track_id, vehicle_crop, meta = task
            ts = time.time()

            # Basic safety checks
            if vehicle_crop is None or vehicle_crop.size == 0:
                self.task_queue.task_done()
                continue

            # Run the LPR model on the cropped vehicle image
            try:
                results = lpr_model.predict(source=vehicle_crop, conf=self.conf)
            except Exception as e:
                # send empty result on error
                self.result_queue.put({"track_id": track_id, "plate_text": "", "plate_conf": 0.0, "ts": ts})
                self.task_queue.task_done()
                continue

            # For each plate detection (in vehicle_crop coordinates)
            plate_texts = []
            plates_conf = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box_xyxy, conf in zip(boxes.xyxy, boxes.conf):
                    px1, py1, px2, py2 = map(int, box_xyxy.tolist())
                    # ensure bounds
                    px1 = max(0, px1); py1 = max(0, py1)
                    px2 = min(vehicle_crop.shape[1], px2); py2 = min(vehicle_crop.shape[0], py2)
                    if px2 <= px1 or py2 <= py1:
                        continue
                    plate_crop = vehicle_crop[py1:py2, px1:px2]

                    # Preprocess for OCR (you can factor this into a helper)
                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                    gray = cv2.fastNlMeansDenoising(gray, h=30)
                    gray = cv2.equalizeHist(gray)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # OCR config: uppercase + digits whitelist
                    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    text = pytesseract.image_to_string(thresh, config=custom_config).strip().replace(" ", "")

                    plate_texts.append(text)
                    plates_conf.append(float(conf.item() if hasattr(conf, "item") else conf))

            # Choose best plate read (e.g., highest detection confidence)
            best_text = ""
            best_conf = 0.0
            if plate_texts:
                idx = int(np.argmax(plates_conf)) if len(plates_conf) > 1 else 0
                best_text = plate_texts[idx]
                best_conf = plates_conf[idx]

            # Send result back
            self.result_queue.put({
                "track_id": track_id,
                "plate_text": best_text,
                "plate_conf": best_conf,
                "ts": ts
            })

            self.task_queue.task_done()

    def stop(self):
        self.stop_flag.set()

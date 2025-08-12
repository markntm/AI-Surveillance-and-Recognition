from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import time
import pytesseract
import mss


# CONSTANTS

YOLO_COCO_PATH = "yolo11n.pt"
YOLO_LPR_PATH = "ultralytics/runs/detect/train2/weights/best.pt"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
VEHICLE_CLASSES = [2,3,5,7]


def extract_text(results, frame):
    extracted_texts = []

    for box in results[0].boxes:
        # Get coordinates as integers
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop the detected license plate
        crop = frame[y1:y2, x1:x2]

        # --- PREPROCESSING STEPS ---
        # 1. Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 2. Resize (helps OCR on small plates)
        scale_factor = 2.0
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # 3. Denoise
        gray = cv2.fastNlMeansDenoising(gray, h=30)

        # 4. Increase contrast
        gray = cv2.equalizeHist(gray)

        # 5. Thresholding (binarization)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # --- OCR ---
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(thresh, config=custom_config)

        # Clean text
        text = text.strip().replace(" ", "")

        extracted_texts.append((text, float(box.conf[0]), (x1, y1, x2, y2)))

        print(extracted_texts)

    return str(extracted_texts)


"""
def extract_text(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, h=30)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    return text.strip().replace(" ", "")
"""

"""
def run_yolo(source_type="video", model_type="coco", screen_region=None, confidence_thresh=0.40):
    # Selects YOLO model
    if model_type=="coco":
        model_path = YOLO_COCO_PATH
    else:
        model_path = YOLO_LPR_PATH

    model = YOLO(model_path)

    # Initialize DeepSORT tracker
    tracker = DeepSort(max_age=30, n_init=3)

    # Sets source
    if source_type == "video":
        cap = cv2.VideoCapture(0)
    elif source_type == "screen":
        sct = mss.mss()

    # Main program
    while True:
        start_time = time.time()

        # Grabbing frame
        if source_type == "video":
            ret, frame = cap.read()
            if not ret:
                break
        elif source_type == "screen":
            screenshot = np.array(sct.grab(screen_region))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        # Run YOLO detection
        results = model.predict(source=frame, conf=confidence_thresh)
        detections = results[0].boxes

        # Convert detections to DeepSORT input format
        dets_for_deepsort = []
        if detections is not None and detections.xyxy is not None:
            for box, cls, conf in zip(detections.xyxy, detections.cls, detections.conf):
                x1, y1, x2, y2 = box.tolist()
                dets_for_deepsort.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), int(cls.item())))

        # Run DeepSORT tracker
        tracks = tracker.update_tracks(dets_for_deepsort, frame=frame)

        # Draw detections + OCR (if LPR mode)
        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            class_id = track.get_det_class()
            confidence = track.get_det_conf()
            class_name = model.names[int(class_id)] if class_id is not None else "Unknown"
            conf_str = f"{confidence:.2f}" if confidence is not None else "?.??"

            label = f"ID {track.track_id} | {class_name} ({conf_str})"

            # Draw box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            # If running License Plate Recognition model, run OCR
            if model_path == YOLO_LPR_PATH:
                plate_text = extract_text(results, frame)
                cv2.putText(frame, plate_text.strip(), (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Display FPS
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Show the frame
        cv2.imshow(f"YOLOv8 + DeepSORT ({model_type.upper()} - {source_type})", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    if source_type == "video":
        cap.release()
    cv2.destroyAllWindows()
"""

def run_yolo(source_type="video", screen_region=None, conf_thresh_coco=0.4, conf_thresh_lpr=0.2):
    # Load both models once
    coco_model = YOLO(YOLO_COCO_PATH)
    lpr_model = YOLO(YOLO_LPR_PATH)

    tracker = DeepSort(max_age=30, n_init=3)

    if source_type == "video":
        cap = cv2.VideoCapture(0)
    elif source_type == "screen":
        sct = mss.mss()

    while True:
        start_time = time.time()

        # Get frame
        if source_type == "video":
            ret, frame = cap.read()
            if not ret:
                break
        else:
            screenshot = np.array(sct.grab(screen_region))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        # -------- Step 1: Detect vehicles with COCO model --------
        coco_results = coco_model.predict(source=frame, conf=conf_thresh_coco)
        detections = coco_results[0].boxes

        dets_for_deepsort = []
        for box, cls, conf in zip(detections.xyxy, detections.cls, detections.conf):
            if int(cls) in VEHICLE_CLASSES:  # Only vehicles
                x1, y1, x2, y2 = map(int, box.tolist())
                dets_for_deepsort.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), int(cls.item())))

        # Track vehicles
        tracks = tracker.update_tracks(dets_for_deepsort, frame=frame)

        # -------- Step 2: For each tracked vehicle, run LPR --------
        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            vehicle_crop = frame[y1:y2, x1:x2]

            # Run LPR model inside vehicle crop
            lpr_results = lpr_model.predict(source=vehicle_crop, conf=conf_thresh_lpr)

            for lp_box in lpr_results[0].boxes:
                px1, py1, px2, py2 = map(int, lp_box.xyxy[0])
                plate_crop = vehicle_crop[py1:py2, px1:px2]

                # Run OCR on plate
                plate_text = extract_text(plate_crop)

                # Draw plate box (in vehicle coordinates)
                cv2.rectangle(vehicle_crop, (px1, py1), (px2, py2), (255, 0, 0), 2)
                cv2.putText(vehicle_crop, plate_text, (px1, py2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # FPS display
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("Vehicle + LPR Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if source_type == "video":
        cap.release()
    cv2.destroyAllWindows()



monitor = {"top": 0, "left": 0, "width": 1280, "height": 720}

source = input("Select source type: ")

run_yolo(source_type=source, screen_region=monitor)
    # source type = video / screen
    # model type = coco / lpr

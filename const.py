from central_server.secret import recognized_license_plates, unrecognized_license_plates

constants = {
    "recLP": recognized_license_plates,
    "unrecLP": unrecognized_license_plates,
    "YOLO_COCO_PATH": "yolo11n.pt",
    "YOLO_LPR_PATH": "ultralytics/runs/detect/train2/weights/best.pt",
    "TESSERACT_CMD": r"C:\Program Files\Tesseract-OCR\tesseract.exe",  # Windows example
    "NUM_LPR_WORKERS": 1,
    "LPR_QUEUE_MAXSIZE": 8,
    "LPR_COOLDOWN_SECONDS": 3.0,  # wait this long before re-processing same track
    "PLATE_RETENTION_SECONDS": 60, # keep plate text in cache for this long

    # Vehicle class ids (COCO): 2=car,3=motorbike,5=bus,7=truck (verify via model.names)
    "VEHICLE_CLASS_IDS": [2, 3, 5, 7]
}

import cv2


def crop_bbox(frame, bbox, pad=0):
    """Crop box with optional padding, keep it inside frame bounds."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    if pad:
        pw = int((x2 - x1) * pad)
        ph = int((y2 - y1) * pad)
        x1 -= pw; x2 += pw; y1 -= ph; y2 += ph
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


class VideoStream:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame


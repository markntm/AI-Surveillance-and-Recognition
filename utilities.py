import cv2
import numpy as np
from collections import Counter

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


def extract_dominant_color(image, k=3):
    """
    Extract the dominant color from an image using k-means clustering.

    Args:
        image (np.ndarray): The cropped vehicle image in BGR format.
        k (int): Number of clusters for k-means.

    Returns:
        tuple: (B, G, R) dominant color values.
    """
    if image is None or image.size == 0:
        return None

    # Convert image to a 2D array of pixels
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Define criteria for k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Apply k-means
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Find most common cluster
    label_counts = Counter(labels.flatten())
    dominant_index = label_counts.most_common(1)[0][0]
    dominant_color = centers[dominant_index]

    return tuple(map(int, dominant_color))  # Convert to int BGR


def infer_human_behavior(track_history):
    """
    Analyze human movement patterns to infer behavior.

    Args:
        track_history (list): List of (x, y) positions over time.

    Returns:
        str: Behavior classification (placeholder).
    """
    # Placeholder logic — can be expanded to detect:
    # - Loitering
    # - Walking speed
    # - Following another person
    # - Running
    if len(track_history) < 2:
        return "Unknown"

    # Calculate simple speed estimation
    dx = track_history[-1][0] - track_history[0][0]
    dy = track_history[-1][1] - track_history[0][1]
    distance = (dx**2 + dy**2) ** 0.5

    if distance < 10:
        return "Loitering"
    elif distance < 50:
        return "Walking"
    else:
        return "Running"


class VideoStream:
    """
    Iterator class for reading frames from a video source.
    Can be used in a for-loop directly.
    """
    def __init__(self, source):
        """
        Args:
            source (str or int): Path to video file or camera index.
        """
        self.cap = cv2.VideoCapture(source)

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame

    def release(self):
        """Manually release the video capture."""
        if self.cap.isOpened():
            self.cap.release()

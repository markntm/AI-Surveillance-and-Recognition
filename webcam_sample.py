from webcam_detect import yoloScript
import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

yoloScript(cap)

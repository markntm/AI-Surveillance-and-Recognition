from webcam_detect import yoloScriptB
import mss
import time

# for screen capture
sct = mss.mss()

# Define monitor (entire screen or a region -- 1 = main screen)
monitor = sct.monitors[1]

yoloScriptB(sct, monitor)

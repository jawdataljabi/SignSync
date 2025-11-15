import pyvirtualcam
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

with pyvirtualcam.Camera(width=width, height=height, fps=fps) as cam:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cam.sleep_until_next_frame()

        # ?? sending to another python process?
        # process_frame(frame)

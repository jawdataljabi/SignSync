import cv2
import pyvirtualcam
import zmq

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5555")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)

with pyvirtualcam.Camera(width=width, height=height, fps=fps) as cam:
    print("Virtual camera:", cam.device)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cam.sleep_until_next_frame()

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        socket.send(encoded.tobytes())

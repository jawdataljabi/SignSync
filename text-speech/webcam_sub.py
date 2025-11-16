import zmq
import numpy as np
import cv2
import mediapipe as mp

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:5555")
socket.setsockopt(zmq.SUBSCRIBE, b"")

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic()

while True:
    jpg_bytes = socket.recv()

    # ---- DECODE JPEG ----
    jpg_np = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(jpg_np, cv2.IMREAD_COLOR)

    # safety copy (MediaPipe modifies)
    frame = frame.copy()

    # MediaPipe processing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    cv2.imshow("Processed Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
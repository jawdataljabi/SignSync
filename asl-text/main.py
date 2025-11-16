import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# 1. Initialize MediaPipe Holistic and OpenCV VideoCapture
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# 2. Load your trained LSTM model
model = tf.keras.models.load_model("my_model.keras")

# Map class indices to labels (adjust to your training)
CLASS_LABELS = {
    0: "Gesture_0",  # treat as "send"
    1: "Gesture_1",  # treat as "word"
    # add more if needed
}

# 3. Sequence buffer for frames
sequence_length = 30
sequence = []

# 4. Inference and smoothing parameters
PREDICTION_STRIDE = 2       # run model every 2 frames (~15 Hz at 30 fps)
SMOOTHING_WINDOW = 10       # number of recent predictions to average
CONFIDENCE_THRESHOLD = 0.7  # minimum probability to show a gesture

predictions_buffer = []     # last SMOOTHING_WINDOW prob vectors
stable_label = None         # label we display
frame_index = 0             # frame counter

# 5. Sentence buffer and state for edge detection
sentence_buffer = []        # list of tokens (you decide what the token means)
last_stable_label = None    # previous stable label

def handle_sentence(tokens):
    """
    Replace this with your external function.
    For now it just prints the tokens with "sentence:" prefix for GUI recognition.
    """
    sentence_text = "".join(tokens)
    print(f"sentence:{sentence_text}")

# 6. Preprocessing function - must match training
def extract_keypoints(results):
    # Pose: 33 landmarks, each (x, y, z, visibility)
    pose = np.zeros(33 * 4, dtype=np.float32)
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility]
             for lm in results.pose_landmarks.landmark],
            dtype=np.float32
        ).flatten()

    # Face: 468 landmarks, each (x, y, z)
    face = np.zeros(468 * 3, dtype=np.float32)
    if results.face_landmarks:
        face = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.face_landmarks.landmark],
            dtype=np.float32
        ).flatten()

    # Left hand: 21 landmarks, each (x, y, z)
    lh = np.zeros(21 * 3, dtype=np.float32)
    if results.left_hand_landmarks:
        lh = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32
        ).flatten()

    # Right hand: 21 landmarks, each (x, y, z)
    rh = np.zeros(21 * 3, dtype=np.float32)
    if results.right_hand_landmarks:
        rh = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32
        ).flatten()

    # Final per-frame feature vector length = 1662
    return np.concatenate([pose, face, lh, rh])

# 7. Live loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw landmarks (optional)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )

    # Extract keypoints and update sequence buffer
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    if len(sequence) > sequence_length:
        sequence.pop(0)

    frame_index += 1

    # Only run prediction when we have a full window and match the stride
    if len(sequence) == sequence_length and frame_index % PREDICTION_STRIDE == 0:
        sequence_input = np.array(sequence, dtype=np.float32).reshape(
            1, sequence_length, 1662
        )

        raw_probs = model.predict(sequence_input, verbose=0)[0]  # (num_classes,)

        # Update buffer of recent probability vectors
        predictions_buffer.append(raw_probs)
        if len(predictions_buffer) > SMOOTHING_WINDOW:
            predictions_buffer.pop(0)

        # Compute smoothed probabilities
        smoothed_probs = np.mean(predictions_buffer, axis=0)
        best_class = int(np.argmax(smoothed_probs))
        best_conf = float(smoothed_probs[best_class])

        # Either show a gesture or "no gesture" based on confidence
        if best_conf >= CONFIDENCE_THRESHOLD:
            stable_label = best_class
        else:
            stable_label = None

        # 8. Sentence logic: edge detection on stable_label
        if stable_label != last_stable_label:
            # Rising edge on Gesture_1: add a token to buffer
            if stable_label == 1:
                # You can push whatever token you want: class index, label string, etc.
                sentence_buffer.append("six-seven")

            # Rising edge on Gesture_0: send and clear buffer
            elif stable_label == 0:
                sentence_buffer.append("seven-eight-nine")
            elif not stable_label:
                if sentence_buffer:
                    handle_sentence(sentence_buffer)
                    sentence_buffer.clear()

            # Update last_stable_label after handling edges
            last_stable_label = stable_label

    # Display the current stable label or "no gesture"
    if stable_label is None:
        label_str = "..."
    else:
        label_str = CLASS_LABELS.get(stable_label, str(stable_label))

    label_text = f"Prediction: {label_str}"
    cv2.putText(
        frame,
        label_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Show the buffer contents for debugging
    buffer_text = "Buffer: " + " ".join(str(t) for t in sentence_buffer)
    cv2.putText(
        frame,
        buffer_text,
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.imshow("ASL Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

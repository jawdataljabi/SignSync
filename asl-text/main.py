import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyvirtualcam
import sys
import os
import threading

# 1. Initialize MediaPipe Holistic and OpenCV VideoCapture
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

cap = cv2.VideoCapture(0)

# Get video properties for virtual camera
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)

# Check command-line argument for showing camera (default state)
SHOW_CAMERA = "--show-camera" in sys.argv or os.getenv("SHOW_CAMERA", "0") == "1"
# Thread-safe flag for camera display
camera_lock = threading.Lock()
# Track previous state to detect transitions
prev_show_camera = SHOW_CAMERA

# 2. Load your trained 1D CNN model
model = tf.keras.models.load_model("best_cnn_asl_model.keras")

# Map class indices to labels (custom mappings)
CLASS_LABELS = {
    # Custom word mappings
    0: "Reset",      # A
    1: "Hello",      # B
    4: "You",        # E
    6: "Class",      # G
    7: "In",         # H
    8: "Good",       # I
    9: "How",        # J
    11: "No",        # L
    14: "Yes",       # O
    17: "Love",      # R
    18: "EOS",       # S
    20: "Thank You", # U
    23: "Me",        # X
    24: "Goodbye",   # Y
    # Unmapped classes will show as "Class_X"
}

# 3. No sequence buffer needed - this is a single-frame 1D CNN model

# Preprocessing options - try different combinations if model doesn't work well
# ASL Alphabet models often use raw MediaPipe coordinates (normalized 0-1) without centering
PREPROCESSING_MODE = "centered_scaled"  # Options: "raw", "centered", "centered_scaled"
# "raw" = Use MediaPipe coordinates as-is (normalized 0-1)
# "centered" = Center relative to wrist (current)
# "centered_scaled" = Center and normalize by hand size

# 4. Inference and smoothing parameters
PREDICTION_STRIDE = 1        # run model every frame for faster response (~30 Hz at 30 fps)
SMOOTHING_WINDOW = 5         # smaller window for quicker updates (reduced from 10)
CONFIDENCE_THRESHOLD = 0.6   # minimum probability to show a gesture

predictions_buffer = []      # last SMOOTHING_WINDOW prob vectors
stable_label = None          # label we display
frame_index = 0              # frame counter

# 5. Sentence buffer and state for edge detection
sentence_buffer = []         # list of tokens (you decide what the token means)
last_stable_label = None     # previous stable label


def handle_sentence(tokens):
    """
    Replace this with your external function.
    For now it just prints the tokens.
    """
    print("sentence:" + " ".join(tokens))


def read_stdin_commands():
    """Read commands from stdin in a separate thread."""
    global SHOW_CAMERA
    while True:
        try:
            # Read from stdin (blocking read works for subprocess stdin)
            line = sys.stdin.readline()
            
            if not line:
                # EOF reached
                break
            
            line = line.strip().lower()
            if line == "show_camera":
                with camera_lock:
                    SHOW_CAMERA = True
            elif line == "hide_camera":
                with camera_lock:
                    SHOW_CAMERA = False
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            # Silently ignore errors to keep the thread running
            pass



# 6. Preprocessing function - must match training
# Model expects 63 features (21 hand landmarks * 3 coordinates)
def extract_keypoints(results):
    # Extract hand keypoints (prefer right hand, fallback to left hand)
    hand_keypoints = np.zeros(21 * 3, dtype=np.float32)  # 63 features
    hand_landmarks = None
    
    # Try right hand first
    if results.right_hand_landmarks:
        hand_landmarks = results.right_hand_landmarks
    # Fallback to left hand if right hand not detected
    elif results.left_hand_landmarks:
        hand_landmarks = results.left_hand_landmarks
    
    if hand_landmarks:
        # Extract raw coordinates from MediaPipe (already normalized 0-1)
        raw_keypoints = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )
        
        if PREPROCESSING_MODE == "raw":
            # Use raw MediaPipe coordinates (normalized 0-1) - most common for ASL models
            hand_keypoints = raw_keypoints.flatten()
        elif PREPROCESSING_MODE == "centered":
            # Center coordinates relative to wrist (landmark 0)
            wrist = raw_keypoints[0]  # Wrist is landmark 0
            centered = raw_keypoints - wrist
            hand_keypoints = centered.flatten()
        elif PREPROCESSING_MODE == "centered_scaled":
            # Center and normalize by hand size
            wrist = raw_keypoints[0]  # Wrist is landmark 0
            centered = raw_keypoints - wrist
            # Normalize by maximum distance from wrist (hand size)
            distances = np.linalg.norm(centered, axis=1)
            max_dist = np.max(distances)
            if max_dist > 0:
                centered = centered / max_dist
            hand_keypoints = centered.flatten()
        else:
            # Default to raw
            hand_keypoints = raw_keypoints.flatten()
    
    return hand_keypoints  # Returns 63 features (format depends on PREPROCESSING_MODE)


# 7. Start stdin reader thread
stdin_thread = threading.Thread(target=read_stdin_commands, daemon=True)
stdin_thread.start()

# 8. Live loop
try:
    with pyvirtualcam.Camera(width=width, height=height, fps=fps) as cam:
        print("Virtual camera:", cam.device)
        print("Press Ctrl+C to exit")
        
        # Use local variable to track previous state in loop
        local_prev_show_camera = prev_show_camera
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB for virtual camera and send it
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam.send(frame_rgb)
            cam.sleep_until_next_frame()

            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks (optional)
            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                )
            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                )
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                )

            # Extract keypoints (63 features for one hand)
            keypoints = extract_keypoints(results)
            
            # Check if we have hand keypoints
            hand_detected = np.any(keypoints != 0)

            frame_index += 1

            # Initialize stable_label for this frame (use last value if no new prediction)
            stable_label = last_stable_label
            
            # Run prediction based on stride and if hand is detected
            if hand_detected and frame_index % PREDICTION_STRIDE == 0:
                # Reshape to (1, 63, 1) for 1D CNN model
                model_input = keypoints.reshape(1, 63, 1).astype(np.float32)

                raw_probs = model.predict(model_input, verbose=0)[0]  # (num_classes,)

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
                    # Rising edge: add a letter/token to buffer (only if not already in buffer)
                    if stable_label is not None:
                        # Check if class is in CLASS_LABELS - skip if unknown
                        if stable_label not in CLASS_LABELS:
                            print(f"Skipped unknown class: {stable_label}")
                        else:
                            letter = CLASS_LABELS[stable_label]
                            
                            # Special handling for "Reset" - clear the buffer
                            if letter == "Reset":
                                if sentence_buffer:
                                    print(f"Buffer cleared (reset detected): {sentence_buffer}")
                                    sentence_buffer.clear()
                                else:
                                    print("Reset detected (buffer already empty)")
                            # Special handling for "EOS" - send buffer to stdout and clear
                            elif letter == "EOS":
                                if sentence_buffer:
                                    handle_sentence(sentence_buffer)
                                    sentence_buffer.clear()
                                    print("Buffer sent to stdout and cleared (EOS detected)")
                                else:
                                    print("EOS detected (buffer already empty)")
                            else:
                                # Only append if word doesn't already exist in buffer
                                if letter not in sentence_buffer:
                                    sentence_buffer.append(letter)
                                    print(f"Buffer updated: {sentence_buffer}")
                                else:
                                    print(f"Skipped duplicate: {letter} (already in buffer)")
                    
                    # Update last_stable_label after handling edges
                    last_stable_label = stable_label

            # Display the current stable label or "no gesture"
            if stable_label is None:
                label_str = "No hand / Low confidence"
                color = (0, 0, 255)  # Red
                conf_str = ""
            else:
                label_str = CLASS_LABELS.get(stable_label, f"Class_{stable_label}")
                color = (0, 255, 0)  # Green
                if len(predictions_buffer) > 0:
                    smoothed_probs = np.mean(predictions_buffer, axis=0)
                    conf_str = f" ({smoothed_probs[stable_label]:.2f})"
                else:
                    conf_str = ""

            label_text = f"Prediction: {label_str}{conf_str}"
            cv2.putText(
                frame,
                label_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

            # Show the buffer contents
            buffer_text = "Buffer: " + " ".join(str(t) for t in sentence_buffer)
            cv2.putText(
                frame,
                buffer_text,
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # Display the frame in a window (if enabled) - check with lock
            with camera_lock:
                show_camera = SHOW_CAMERA
            
            # Check if state changed (camera was just turned off)
            if local_prev_show_camera and not show_camera:
                # Camera was just turned off - close the window once
                try:
                    cv2.destroyWindow("ASL Gesture Recognition")
                except:
                    pass
            local_prev_show_camera = show_camera
            
            if show_camera:
                cv2.imshow("ASL Gesture Recognition", frame)
                # Check for 'q' key to exit (only if camera window is shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    cap.release()
    cv2.destroyAllWindows()

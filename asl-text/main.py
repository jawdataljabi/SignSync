import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Initialize MediaPipe Hands and OpenCV VideoCapture
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(1)

# Load your trained LSTM model (from .keras file)
model = tf.keras.models.load_model('my_model.keras')

# Prepare a buffer for hand landmarks (e.g., 30 frames for the sequence)
sequence_length = 30
sequence = []

# Preprocessing function (to normalize and format data for your model)
def preprocess_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    return np.array(landmarks).flatten()  # Flatten to 63 features (21 landmarks * 3 coords)

# Loop for live feed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    # If hands are detected, extract the landmarks
    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            # Extract and preprocess the landmarks
            hand_landmarks = preprocess_landmarks(landmarks)

            # Append the current frame's hand landmarks to the sequence
            sequence.append(hand_landmarks)

            # Keep the sequence at the desired length
            if len(sequence) > sequence_length:
                sequence.pop(0)  # Keep only the last `sequence_length` frames

            # If we have a complete sequence, make a prediction
            if len(sequence) == sequence_length:
                # Convert the sequence into a numpy array and reshape it for the model
                sequence_input = np.array(sequence).reshape(1, sequence_length, 63)  # Reshape for LSTM (batch_size, seq_len, features)
                prediction = model.predict(sequence_input)

                # Output the prediction (can be a gesture, label, etc.)
                predicted_label = np.argmax(prediction, axis=1)[0]  # Get the predicted label index
                cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with prediction label
    cv2.imshow("ASL Gesture Recognition", frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

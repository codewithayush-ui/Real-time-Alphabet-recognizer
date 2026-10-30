import cv2
import json
import numpy as np
from collections import deque, Counter
import mediapipe as mp
import joblib
from utils import extract_hand_features

# Load trained model and class labels
model = joblib.load('models/alphabet_model.pkl')
with open('models/classes.json', 'r') as f:
    classes = json.load(f)

vote_buffer = deque(maxlen=7)
word_formed = ""

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        detected_letter = None
        confidence_val = 0.0

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                features = extract_hand_features(hand_lms.landmark)
                features = np.array(features).reshape(1, -1)

                probs = model.predict_proba(features)[0]
                idx = np.argmax(probs)
                confidence_val = probs[idx]

                if confidence_val >= 0.60:
                    pred_class = model.classes_[idx]
                    vote_buffer.append(pred_class)
                else:
                    vote_buffer.append("?")

                if len(vote_buffer) > 0:
                    most_voted, count = Counter(vote_buffer).most_common(1)[0]
                    if most_voted != "?":
                        detected_letter = most_voted

        # Display UI
        cv2.rectangle(frame, (15, 15), (440, 140), (25, 25, 25), -1)
        cv2.rectangle(frame, (15, 15), (440, 140), (0, 200, 0), 2)

        if detected_letter:
            cv2.putText(frame, f"Letter: {detected_letter}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Conf: {confidence_val * 100:.1f}%", (240, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
        else:
            cv2.putText(frame, "Show sign...", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)

        cv2.putText(frame, f"Word: {word_formed}", (30, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "[Space] Add   [C] Clear   [Q] Quit", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("ASL Alphabet Recognizer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and detected_letter:
            word_formed += detected_letter
        elif key == ord('c') or key == ord('C'):
            word_formed = ""

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import csv
import os
import time
from utils import extract_hand_features

# Choose the letters you want to train
TARGET_LETTERS = ['A', 'B', 'C', 'D', 'L', 'V', 'W', 'Y']
SAMPLES_PER_LETTER = 150

os.makedirs('data', exist_ok=True)
csv_path = os.path.join('data', 'alphabet_data.csv')

# Initialize CSV with 63 feature columns
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'feat_{i}' for i in range(63)]
        writer.writerow(header)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    for letter in TARGET_LETTERS:
        # Preparation countdown screen
        print(f"\nNext letter: '{letter}'. Get your hand sign ready!")
        start_time = time.time()
        
        while time.time() - start_time < 3:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            remaining = int(4 - (time.time() - start_time))
            cv2.putText(frame, f"Pose for Letter: {letter}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            cv2.putText(frame, f"Starting in: {remaining}s", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
            cv2.imshow("Alphabet Data Collector", frame)
            cv2.waitKey(1)

        # Active recording phase
        collected = 0
        while collected < SAMPLES_PER_LETTER:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                    
                    features = extract_hand_features(hand_lms.landmark)
                    with open(csv_path, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([letter] + features)

                    collected += 1

            cv2.putText(frame, f"Recording Letter '{letter}': {collected}/{SAMPLES_PER_LETTER}", 
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "Vary angle and distance slightly", 
                        (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Alphabet Data Collector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

cap.release()
cv2.destroyAllWindows()
print(f"\nFinished collection. Saved to {csv_path}")
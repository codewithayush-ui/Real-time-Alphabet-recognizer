import os
import cv2
import csv
import mediapipe as mp
from utils import extract_hand_features

# The exact path matching your folder structure
KAGGLE_DIR = os.path.join('kaggle_data', 'asl_alphabet_train')
CSV_OUTPUT = os.path.join('data', 'alphabet_data.csv')

# Start with these 8 distinct letters
TARGET_LETTERS = ['A', 'B', 'C', 'D', 'L', 'V', 'W', 'Y']
SAMPLES_PER_LETTER = 300  # 300 images per letter gives plenty of data

os.makedirs('data', exist_ok=True)

# Write header row for 63 coordinates
with open(CSV_OUTPUT, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['label'] + [f'feat_{i}' for i in range(63)])

mp_hands = mp.solutions.hands

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
) as hands:
    for letter in TARGET_LETTERS:
        folder_path = os.path.join(KAGGLE_DIR, letter)
        if not os.path.exists(folder_path):
            print(f"[!] Warning: Folder '{folder_path}' not found, skipping...")
            continue

        all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        selected_images = all_images[:SAMPLES_PER_LETTER]
        print(f"Processing letter '{letter}': {len(selected_images)} images...")

        valid_count = 0
        for img_name in selected_images:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_img)

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    features = extract_hand_features(hand_lms.landmark)
                    with open(CSV_OUTPUT, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([letter] + features)
                    valid_count += 1

        print(f"  -> Successfully extracted {valid_count} hand samples for '{letter}'")

print(f"\n[DONE] Coordinates successfully saved to {CSV_OUTPUT}")
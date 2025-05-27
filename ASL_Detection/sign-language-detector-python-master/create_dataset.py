import os
import pickle

import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

print("Processing images and extracting hand landmarks...")
class_count = 0
total_images = 0
processed_images = 0

# Count total number of classes and images
for dir_ in os.listdir(DATA_DIR):
    if os.path.isdir(os.path.join(DATA_DIR, dir_)) and not dir_.startswith('.'):
        class_count += 1
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            if img_path.endswith('.jpg') or img_path.endswith('.png'):
                total_images += 1

print(f"Found {class_count} classes with {total_images} total images")

for dir_ in sorted(os.listdir(DATA_DIR)):
    if not os.path.isdir(os.path.join(DATA_DIR, dir_)) or dir_.startswith('.'):
        continue
        
    print(f"Processing class: {dir_}")
    
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        if not (img_path.endswith('.jpg') or img_path.endswith('.png')):
            continue
            
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            print(f"Warning: Could not read image {os.path.join(DATA_DIR, dir_, img_path)}")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)
            processed_images += 1
        else:
            print(f"Warning: No hand landmarks detected in {os.path.join(DATA_DIR, dir_, img_path)}")

print(f"Processed {processed_images} out of {total_images} images successfully")
print(f"Created dataset with {len(data)} samples across {len(set(labels))} classes")

print("Saving dataset to data.pickle...")
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print("Dataset creation complete!")

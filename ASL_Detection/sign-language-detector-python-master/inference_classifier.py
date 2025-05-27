import pickle
import os
import cv2
import mediapipe as mp
import numpy as np

print("Loading model...")
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
le = model_dict.get('label_encoder', None)  # Get the label encoder if available

# Read class mapping from file
class_mapping = {}
if os.path.exists('./data/class_mapping.txt'):
    with open('./data/class_mapping.txt', 'r') as f:
        for line in f:
            idx, label = line.strip().split(':', 1)
            class_mapping[int(idx)] = label.strip()
    print(f"Loaded {len(class_mapping)} classes from mapping file")
else:
    print("Warning: No class mapping file found")
    # Fallback default mapping
    class_mapping = {0: 'A', 1: 'B', 2: 'C'}

# Use webcam index 0 instead of 2 (adjust if needed)
print("Initializing camera...")
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Display the available classes
print(f"Available classes: {len(class_mapping)} signs")
for idx, label in class_mapping.items():
    print(f"  {idx}: {label}")

print("Press 'q' to quit")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera. Check camera index.")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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

        # Find the maximum feature length from our trained model
        if hasattr(model, 'n_features_in_'):
            max_length = model.n_features_in_
        else:
            # Estimate based on common hand landmark counts in MediaPipe
            max_length = 21 * 2 * 2  # 21 landmarks, x and y, normalized coordinates

        # Pad data if necessary
        if len(data_aux) < max_length:
            data_aux = data_aux + [0.0] * (max_length - len(data_aux))
        elif len(data_aux) > max_length:
            data_aux = data_aux[:max_length]

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        prediction_index = int(prediction[0])
        
        # If we have a label encoder, convert the prediction back to the original label
        if le is not None:
            # First convert model's numeric prediction back to the original class label
            original_label = le.inverse_transform([prediction_index])[0]
            # This is likely a string representation of the original folder number
            asl_character = class_mapping.get(int(original_label), f"Unknown ({original_label})")
        else:
            # Direct mapping without label encoder (less likely)
            asl_character = class_mapping.get(prediction_index, f"Unknown ({prediction_index})")

        # Draw the prediction on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, asl_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Display instructions on the frame
    cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('ASL Sign Language Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_FILE = "gesture_model.h5"
LABEL_MAP_FILE = "label_map.json"
IMG_SIZE = (128, 128)
PAD_MARGIN = 30  # same margin as training

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def load_model_and_label_map():
    model = keras.models.load_model(MODEL_FILE)
    with open(LABEL_MAP_FILE, "r") as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    return model, inv_label_map

def detect_and_crop_hand(img, results):
    if not results.multi_hand_landmarks:
        return None
    h, w, _ = img.shape
    hand_landmarks = results.multi_hand_landmarks[0]
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    min_x, max_x = int(min(xs) * w), int(max(xs) * w)
    min_y, max_y = int(min(ys) * h), int(max(ys) * h)

    pad = PAD_MARGIN
    min_x = max(min_x - pad, 0)
    min_y = max(min_y - pad, 0)
    max_x = min(max_x + pad, w)
    max_y = min(max_y + pad, h)

    cropped = img[min_y:max_y, min_x:max_x]
    if cropped.size == 0:
        return None
    resized = cv2.resize(cropped, IMG_SIZE)
    return resized

def main():
    model, inv_label_map = load_model_and_label_map()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        recognized_gesture = "None"
        confidence = 0.0

        if results.multi_hand_landmarks:
            # Draw the first detected hand
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            cropped = detect_and_crop_hand(frame, results)
            if cropped is not None:
                # Preprocess for MobileNet
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                x_input = np.array([cropped_rgb], dtype="float32")
                x_input = preprocess_input(x_input)

                preds = model.predict(x_input)[0]
                class_id = np.argmax(preds)
                confidence = float(np.max(preds))
                recognized_gesture = inv_label_map.get(class_id, "Unknown")

        if confidence > 0.5:
            text = f"{recognized_gesture} ({confidence:.2f})"
        else:
            text = "No confident prediction"

        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)
        cv2.imshow("Gesture Prediction", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

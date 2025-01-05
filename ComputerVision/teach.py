import os
import json
import cv2
import time
import shutil
import numpy as np

import mediapipe as mp
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

GESTURE_DB_FILE = "gestures.json"
SAVED_GESTURES_DIR = "saved_gestures"
MODEL_FILE = "gesture_model.h5"
LABEL_MAP_FILE = "label_map.json"

IMG_SIZE = (128, 128)
NUM_IMAGES_REQUIRED = 20  # Number of images required per gesture
EPOCHS = 50               # Train for 50 epochs now
PAD_MARGIN = 30           # Increase padding around bounding box

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def load_gesture_db():
    if not os.path.exists(GESTURE_DB_FILE):
        return {}
    with open(GESTURE_DB_FILE, "r") as f:
        return json.load(f)

def save_gesture_db(db):
    with open(GESTURE_DB_FILE, "w") as f:
        json.dump(db, f, indent=4)

def save_temp_shot(frame, shot_num):
    temp_dir = "temp_shots"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    filename = os.path.join(temp_dir, f"shot_{int(time.time())}_{shot_num}.jpg")
    cv2.imwrite(filename, frame)
    return filename

def detect_and_crop_hand(img):
    """
    Use MediaPipe to detect bounding box of the first hand in img.
    Crop that region with a larger margin (PAD_MARGIN).
    Return the resized 128x128 image.
    Return None if no hand is found.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None

    h, w, _ = img.shape
    hand_landmarks = results.multi_hand_landmarks[0]
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    min_x, max_x = int(min(xs) * w), int(max(xs) * w)
    min_y, max_y = int(min(ys) * h), int(max(ys) * h)

    # Increase margin around the bounding box
    pad = PAD_MARGIN
    min_x = max(min_x - pad, 0)
    min_y = max(min_y - pad, 0)
    max_x = min(max_x + pad, w)
    max_y = min(max_y + pad, h)

    cropped = img[min_y:max_y, min_x:max_x]
    if cropped.size == 0:
        return None
    cropped_resized = cv2.resize(cropped, IMG_SIZE)
    return cropped_resized

def save_gesture(gesture_name, shot_files):
    """
    1) For each shot in shot_files, detect/crop the hand with margin
       and save the cropped image to the permanent folder.
    2) Update gestures.json.
    3) Retrain if all gestures have enough images.
    """
    if not os.path.exists(SAVED_GESTURES_DIR):
        os.makedirs(SAVED_GESTURES_DIR)

    gesture_folder = os.path.join(SAVED_GESTURES_DIR, gesture_name)
    if not os.path.exists(gesture_folder):
        os.makedirs(gesture_folder)

    new_image_paths = []
    for sf in shot_files:
        raw_img = cv2.imread(sf)
        if raw_img is None:
            continue
        cropped = detect_and_crop_hand(raw_img)
        if cropped is None:
            continue
        base_name = os.path.basename(sf)
        new_path = os.path.join(gesture_folder, base_name)
        cv2.imwrite(new_path, cropped)
        new_image_paths.append(new_path)

    # Remove original temp shots
    for sf in shot_files:
        if os.path.exists(sf):
            os.remove(sf)

    # Update gesture DB
    db = load_gesture_db()
    if gesture_name not in db:
        db[gesture_name] = []
    db[gesture_name].extend(new_image_paths)
    save_gesture_db(db)

    # Check if gesture has enough images
    if len(db[gesture_name]) < NUM_IMAGES_REQUIRED:
        print(f"Gesture '{gesture_name}' now has {len(db[gesture_name])} images. Need {NUM_IMAGES_REQUIRED} before training.")
        return
    else:
        # If all gestures have enough images, train the model
        if all(len(db[g]) >= NUM_IMAGES_REQUIRED for g in db.keys()):
            train_model()
        else:
            print("Not all gestures have enough images. Skipping training for now.")

def train_model():
    """
    Transfer learning with MobileNetV2:
    - Load each gesture's cropped images.
    - Data augment + pre-process.
    - Train for 50 epochs.
    - Save model + label map.
    """
    gesture_folders = os.listdir(SAVED_GESTURES_DIR)
    images = []
    labels = []
    label_map = {}
    current_label_id = 0

    for gesture_name in gesture_folders:
        gesture_path = os.path.join(SAVED_GESTURES_DIR, gesture_name)
        if not os.path.isdir(gesture_path):
            continue
        label_map[gesture_name] = current_label_id
        current_label_id += 1

        for img_file in os.listdir(gesture_path):
            if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(gesture_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                images.append(img)
                labels.append(label_map[gesture_name])

    if not images:
        print("No images found. Skipping training.")
        return

    X = np.array(images, dtype="float32")
    y = np.array(labels, dtype="int")

    print(f"Collected {len(X)} images from {len(label_map)} gestures: {list(label_map.keys())}")

    # Data augmentation
    data_gen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_input
    )

    X = X.reshape((-1, IMG_SIZE[1], IMG_SIZE[0], 3))

    batch_size = 16

    train_iter = data_gen.flow(X, y, batch_size=batch_size)

    # Use MobileNetV2 for transfer learning
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[1], IMG_SIZE[0], 3))
    for layer in base_model.layers:
        layer.trainable = False  # Freeze base layers

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(len(label_map), activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    steps_per_epoch = len(X) // batch_size
    model.fit(train_iter, epochs=EPOCHS, steps_per_epoch=steps_per_epoch)

    # Save model + label map
    model.save(MODEL_FILE)
    with open(LABEL_MAP_FILE, "w") as f:
        json.dump(label_map, f)

    print(f"Training complete. Model saved to {MODEL_FILE}, label map saved to {LABEL_MAP_FILE}.")

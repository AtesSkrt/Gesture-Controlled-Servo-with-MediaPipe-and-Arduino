# Gesture-Controlled Servo with MediaPipe and Arduino

This project uses Python (with MediaPipe, OpenCV, TensorFlow/Keras) to detect hand gestures from a webcam, classify them using a machine learning model, and send corresponding servo angle commands to an Arduino over serial. The Arduino then moves the servo to the specified angle in real time.

---

## Project Video Demo

[![Project Video](https://img.youtube.com/vi/ju75onOlbgQ/0.jpg)](https://www.youtube.com/watch?v=ju75onOlbgQ)

Click the image above or watch the video below:

[https://www.youtube.com/watch?v=ju75onOlbgQ](https://www.youtube.com/watch?v=ju75onOlbgQ)

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Hardware Requirements](#hardware-requirements)
4. [Software Requirements](#software-requirements)
5. [Project Structure](#project-structure)
6. [Setup](#setup)
7. [Usage](#usage)
8. [Demo](#demo)
9. [Troubleshooting](#troubleshooting)
10. [License](#license)

---

## Overview

**Goal:** Control a servo motor via hand gestures detected from a webcam.

### Key Components

- **MediaPipe** for hand-tracking and bounding box detection.
- **TensorFlow/Keras** for training and predicting custom gestures.
- **OpenCV** for camera input and real-time display.
- **Arduino code** that receives serial commands (angles) from Python and moves the servo.
- A simple GUI for teaching new gestures and an optional prediction mode for real-time classification.

---

## Features

- **Real-Time Gesture Tracking**:
  - Uses MediaPipe Hands to detect bounding boxes on the user’s hand.

- **Custom Gesture Training**:
  - Collect 20 images for a new gesture.
  - Automatically crops and augments images.
  - Trains a CNN (or a transfer learning model using MobileNetV2).

- **Gesture Prediction**:
  - Classifies gestures in real time.
  - Displays recognized gesture name on the video feed (with confidence).

- **Servo Control**:
  - Sends the servo angle over serial to an Arduino.
  - Arduino code interprets the angle and moves the servo accordingly.

- **Double Fist Exit (Optional)**:
  - If the user forms fists on both hands, the camera feed automatically closes.

---

## Hardware Requirements

- Arduino (e.g., Arduino Nano, UNO)
- Servo Motor (e.g., SG90)
- USB Cable for communication/power
- Wires to connect the servo:
  - Signal → Pin 9
  - VCC → 5V
  - GND → GND

---

## Software Requirements

- Python 3.7+
- OpenCV
  ```bash
  pip install opencv-python
  ```
- MediaPipe
  ```bash
  pip install mediapipe
  ```
- TensorFlow/Keras:
  - For Apple Silicon:
    ```bash
    pip install tensorflow-macos tensorflow-metal
    ```
  - For other systems:
    ```bash
    pip install tensorflow
    ```
- PyQt5 or PySide6 (for GUI, if included)
- Arduino IDE

---

## Project Structure

```
Gesture-Servo-Project/
├── ui.py               # Main settings GUI (if applicable)
├── teach_ui.py         # UI for capturing images and teaching gestures
├── teach.py            # Logic: saving images, training model (transfer learning)
├── predict.py          # Real-time gesture prediction
├── mode_selector.py    # Optional: window with "Servo | Teach | Predict" modes
├── main.py             # Original servo logic (if separate)
├── ArduinoCode.ino     # Arduino code for reading angles & controlling servo
├── gestures.json       # Stores saved gestures (path references)
├── label_map.json      # Mapping between gesture names & class IDs
├── gesture_model.h5    # Trained model file (created after training)
├── saved_gestures/     # Folder with subfolders for each gesture’s images
└── README.md           # Explanation of the project
```

---

## Setup

### Clone the Repository
```bash
git clone https://github.com/YourUsername/Gesture-Servo-Project.git
cd Gesture-Servo-Project
```

### Install Python Dependencies
```bash
pip install opencv-python mediapipe tensorflow-macos tensorflow-metal pyqt5
```

### Upload Arduino Code
1. Open `ArduinoCode.ino` in the Arduino IDE.
2. Select your board (e.g., Arduino Nano) and port.
3. Upload.

### Servo Wiring
- Servo signal → Arduino pin 9
- Servo VCC → 5V
- Servo GND → GND

---

## Usage

### Teach a Gesture
1. Run the Teach UI to capture images and train a new gesture:
   ```bash
   python teach_ui.py
   ```
2. Enter a gesture name and take 20 shots.
3. Images are automatically cropped and saved.
4. Once every gesture has 20+ images, the model trains for 50 epochs and saves `gesture_model.h5`.

### Real-Time Prediction
1. Run the prediction script:
   ```bash
   python predict.py
   ```
2. Displays a webcam feed with recognized gestures.
3. Sends servo angles over serial if integrated with servo logic.

### Servo Control
The Python scripts (e.g., `predict.py` or your main script) must have serial code like:
```python
import serial
ser = serial.Serial('COM3', 9600)  # or '/dev/ttyUSB0', etc.
angle = 90
ser.write((str(angle) + "\n").encode())
```

---

## Demo

- **Teach a new gesture** (like “Thumbs Up”):
  - Capture 20 images from various angles.
  - Train a model automatically.

- **Predict**:
  - Show the recognized gesture label over the video feed.

- **Servo**:
  - Arduino receives angles (e.g., 0° for a fist, 90° for open palm, etc.) and moves the servo accordingly.
  - In distance mode, servo motor moves according to the distance between 2 fingers of the left hand.
  - If you show one finger, it goes to finger mode; if you show two fingers, it goes to distance mode.

---

## Troubleshooting

- **Partial Hand Cropped**: Increase the `PAD_MARGIN` in `teach.py` and `predict.py`.
- **Low Accuracy**: Gather more diverse images, increase the number of shots per gesture, or augment data further.
- **No Serial Connection**: Check the correct port in Python (`/dev/cu.usbserial-xxxx` on Mac, `COM3` on Windows).
- **Servo Not Moving**: Verify servo wiring, 5V supply, and that the Arduino receives angles in a valid `[0–180]` range.

---

## License

You are free to modify and redistribute it. Credits are appreciated but not required.

Feel free to open an issue or pull request if you have suggestions or encounter problems. Happy coding!

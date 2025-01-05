## Project Video Demo

[![Project Video](https://img.youtube.com/vi/ju75onOlbgQ/0.jpg)](https://www.youtube.com/watch?v=ju75onOlbgQ)

Click the image above or watch the video below:

https://www.youtube.com/watch?v=ju75onOlbgQ


# Gesture-Controlled-Servo-with-MediaPipe-and-Arduino
This project uses Python (with MediaPipe, OpenCV, TensorFlow/Keras) to detect hand gestures from a webcam, classify them using a machine learning model, and send corresponding servo angle commands to an Arduino over serial. The Arduino then moves the servo to the specified angle in real time.

Table of Contents

Overview

Features

Hardware Requirements

Software Requirements

Project Structure

Setup

Usage

Demo

Troubleshooting

License


Overview

Goal: Control a servo motor via hand gestures detected from a webcam.

Key Components:

-MediaPipe for hand-tracking and bounding box detection.
-TensorFlow / Keras for training and predicting custom gestures.
-OpenCV for camera input and real-time display.
-Arduino code that receives serial commands (angles) from Python and moves the servo.
-The project also includes a simple GUI for “teaching” new gestures and an optional “predict” mode to classify gestures in real time.

Features

Real-Time Gesture Tracking:
Uses MediaPipe Hands to detect bounding boxes on the user’s hand.
Custom Gesture Training:
Collect 20 images for a new gesture.
Automatically crops and augments images.
Trains a CNN (or a transfer learning model using MobileNetV2).
Gesture Prediction:
Classifies gestures in real time.
Displays recognized gesture name on the video feed (with confidence).
Servo Control:
Sends the servo angle over serial to an Arduino.
Arduino code interprets the angle and moves the servo accordingly.
Double Fist Exit (Optional):
If the user forms fists on both hands, the camera feed automatically closes.
Hardware Requirements

Arduino (e.g., Arduino Nano, UNO)
Servo Motor (e.g., SG90)
USB Cable for communication/power
Wires to connect the servo (Signal -> Pin 9, VCC -> 5V, GND -> GND)


Software Requirements

Python 3.7+
OpenCV (pip install opencv-python)
MediaPipe (pip install mediapipe)
TensorFlow/Keras:
For Apple Silicon:
pip install tensorflow-macos tensorflow-metal
For other systems:
pip install tensorflow
PyQt5 or PySide6 (for GUI, if included)
Arduino IDE (to upload code to Arduino)
Project Structure

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
Setup

Clone this repository:
git clone https://github.com/YourUsername/Gesture-Servo-Project.git
cd Gesture-Servo-Project
Install Python dependencies:
pip install opencv-python mediapipe tensorflow-macos tensorflow-metal pyqt5
Upload Arduino Code:
Open ArduinoCode.ino in the Arduino IDE.
Select your board (e.g., Arduino Nano) and port.
Upload.
Servo Wiring:
Servo signal -> Arduino pin 9
Servo VCC -> 5V
Servo GND -> GND
Usage

Run the Teach UI (for capturing images and training a new gesture):
python teach_ui.py
Enter a gesture name and take 20 shots.
Images are automatically cropped and saved.
Once every gesture has 20+ images, the model trains for 50 epochs and saves gesture_model.h5.
Prediction (Real-Time):
python predict.py
Displays a webcam feed with recognized gestures.
Sends servo angles over serial if integrated with servo logic.
Servo Control:
The Python scripts (e.g., predict.py or your main script) must have serial code like:
import serial
ser = serial.Serial('COM3', 9600)  # or '/dev/ttyUSB0', etc.
angle = 90
ser.write((str(angle) + "\n").encode())
Demo

Teach a new gesture (like “Thumbs Up”):
Capture 20 images from various angles.
Train a model automatically.
Predict:
Show the recognized gesture label over the video feed.
Arduino receives angles (e.g., 0° for a fist, 90° for open palm, etc.) and moves the servo accordingly.
Troubleshooting

Partial Hand Cropped: Increase the PAD_MARGIN in teach.py and predict.py.
Low Accuracy: Gather more diverse images, increase the number of shots per gesture, or augment data further.
No Serial Connection: Check the correct port in Python (/dev/cu.usbserial-xxxx on Mac, COM3 on Windows).
Servo Not Moving: Verify servo wiring, 5V supply, and that the Arduino receives angles in a valid [0–180] range.
License

You are free to modify and redistribute it. Credits are appreciated but not required.

Feel free to open an issue or pull request if you have suggestions or encounter problems. Happy coding!

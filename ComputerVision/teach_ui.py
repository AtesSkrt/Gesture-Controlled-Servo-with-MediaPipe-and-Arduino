from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QLineEdit, QListWidget, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage
import cv2
import sys
import os
import teach  # Import the teach logic from teach.py
import time


class TeachUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Teach Gestures")
        self.setGeometry(200, 100, 1280, 720)

        # Initialize attributes for teaching gestures
        self.num_shots = 0
        self.shots_required = 20  # Updated to require 20 images
        self.current_gesture_name = ""
        self.captured_images = []

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Camera feed
        self.lbl_camera = QLabel()
        self.lbl_camera.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_camera)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_teach = QPushButton("Teach Gesture")
        self.btn_teach.clicked.connect(self.on_teach_gesture)
        btn_layout.addWidget(self.btn_teach)

        self.btn_take_shot = QPushButton("Take Shot")
        self.btn_take_shot.clicked.connect(self.on_take_shot)
        self.btn_take_shot.setEnabled(False)
        btn_layout.addWidget(self.btn_take_shot)

        layout.addLayout(btn_layout)

        # Gesture Name Input
        self.input_gesture_name = QLineEdit()
        self.input_gesture_name.setPlaceholderText("Gesture Name")
        self.input_gesture_name.setEnabled(False)
        layout.addWidget(self.input_gesture_name)

        # Saved Gestures List
        lbl_saved = QLabel("Saved Gestures:")
        layout.addWidget(lbl_saved)
        self.list_saved_gestures = QListWidget()
        layout.addWidget(self.list_saved_gestures)

        # Load existing gestures
        self.load_saved_gestures()

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            sys.exit(1)

        # Timer to update the camera feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(50)  # ~20 FPS

    def load_saved_gestures(self):
        """Load saved gestures into the list widget."""
        gestures = teach.load_gesture_db()
        for g in gestures.keys():
            self.list_saved_gestures.addItem(g)

    def on_teach_gesture(self):
        """Start teaching a new gesture."""
        self.num_shots = 0
        self.captured_images.clear()
        self.input_gesture_name.setEnabled(True)
        self.btn_take_shot.setEnabled(True)

    def on_take_shot(self):
        """Capture a frame and store it as a shot."""
        ret, frame = self.cap.read()
        if not ret:
            return
        # Save the frame as a temporary shot
        shot_filename = teach.save_temp_shot(frame, self.num_shots)
        self.captured_images.append(shot_filename)
        self.num_shots += 1
        print(f"Shot {self.num_shots} taken: {shot_filename}")

        # Check if required shots are completed
        if self.num_shots >= self.shots_required:
            self.btn_take_shot.setEnabled(False)
            gesture_name = self.input_gesture_name.text().strip()
            if not gesture_name:
                gesture_name = f"Gesture_{int(time.time())}"  # Fallback name
            teach.save_gesture(gesture_name, self.captured_images)
            self.list_saved_gestures.addItem(gesture_name)
            self.input_gesture_name.setText("")
            self.input_gesture_name.setEnabled(False)
            print(f"Gesture '{gesture_name}' saved with {self.num_shots} images.")

    def update_camera(self):
        """Capture and display the camera feed."""
        ret, frame = self.cap.read()
        if ret:
            # Resize the frame to fit the QLabel dimensions
            h, w, ch = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bytes_per_line = ch * w
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.lbl_camera.setPixmap(pixmap.scaled(
                self.lbl_camera.width(), self.lbl_camera.height(),
                Qt.KeepAspectRatio
            ))

    def closeEvent(self, event):
        """Clean up resources when the window is closed."""
        self.cap.release()
        event.accept()

def main():
    app = QApplication([])
    window = TeachUI()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()

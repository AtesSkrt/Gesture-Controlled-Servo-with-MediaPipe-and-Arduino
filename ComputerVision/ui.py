from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QSlider, QCheckBox, QPushButton, QSpacerItem, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt
import json
import os
import subprocess

CONFIG_FILE = "config.json"
DEFAULT_SETTINGS = {
    "LIMIT_SERIAL_WRITES": True,
    "USE_MULTITHREADING": True,
    "RESIZE_WIDTH": 1280,
    "RESIZE_HEIGHT": 720,
    "PROCESS_INTERVAL": 0.05,
    "MAX_DISTANCE_ANGLE": 180,
    "MAX_DETECTABLE_DIST": 200
}

class MainUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = self.load_settings()
        self.init_ui()

    def load_settings(self):
        """Load settings from config.json or return defaults."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except:
                pass
        return DEFAULT_SETTINGS.copy()

    def save_settings(self):
        """Save current settings to config.json."""
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.settings, f, indent=4)

    def init_ui(self):
        self.setWindowTitle("Gesture Control Settings")
        self.setGeometry(100, 100, 700, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout()

        # Limit Serial Writes
        frame_limit_serial = QFrame()
        limit_serial_layout = QVBoxLayout(frame_limit_serial)
        lbl_limit_serial_desc = QLabel("Limit Serial Writes:")
        lbl_limit_serial_desc.setStyleSheet("font-weight: bold;")
        lbl_limit_serial_explanation = QLabel("Limits Arduino updates to changes in angles only, reducing unnecessary serial communication.")
        lbl_limit_serial_explanation.setWordWrap(True)
        self.checkbox_limit_serial = QCheckBox("Enabled")
        self.checkbox_limit_serial.setChecked(self.settings["LIMIT_SERIAL_WRITES"])
        limit_serial_layout.addWidget(lbl_limit_serial_desc)
        limit_serial_layout.addWidget(lbl_limit_serial_explanation)
        limit_serial_layout.addWidget(self.checkbox_limit_serial)
        main_layout.addWidget(frame_limit_serial)

        # Use Multithreading
        frame_multithread = QFrame()
        multithread_layout = QVBoxLayout(frame_multithread)
        lbl_multithread_desc = QLabel("Use Multithreading:")
        lbl_multithread_desc.setStyleSheet("font-weight: bold;")
        lbl_multithread_explanation = QLabel("Enables multithreading to handle Arduino communication separately from the main processing loop, improving performance.")
        lbl_multithread_explanation.setWordWrap(True)
        self.checkbox_multithread = QCheckBox("Enabled")
        self.checkbox_multithread.setChecked(self.settings["USE_MULTITHREADING"])
        multithread_layout.addWidget(lbl_multithread_desc)
        multithread_layout.addWidget(lbl_multithread_explanation)
        multithread_layout.addWidget(self.checkbox_multithread)
        main_layout.addWidget(frame_multithread)

        # Resize Width
        frame_width = QFrame()
        width_layout = QVBoxLayout(frame_width)
        lbl_width_desc = QLabel("Resize Width:")
        lbl_width_desc.setStyleSheet("font-weight: bold;")
        lbl_width_explanation = QLabel("Defines the width of the camera frame. Lower values improve performance but reduce video quality.")
        lbl_width_explanation.setWordWrap(True)
        self.slider_width = QSlider(Qt.Horizontal)
        self.slider_width.setMinimum(320)
        self.slider_width.setMaximum(1920)
        self.slider_width.setValue(self.settings["RESIZE_WIDTH"])
        self.slider_width.setTickInterval(100)
        self.slider_width.setTickPosition(QSlider.TicksBelow)
        lbl_width_value = QLabel(str(self.slider_width.value()))
        self.slider_width.valueChanged.connect(lambda v: lbl_width_value.setText(str(v)))
        width_layout.addWidget(lbl_width_desc)
        width_layout.addWidget(lbl_width_explanation)
        width_layout.addWidget(self.slider_width)
        width_layout.addWidget(lbl_width_value)
        main_layout.addWidget(frame_width)

        # Resize Height
        frame_height = QFrame()
        height_layout = QVBoxLayout(frame_height)
        lbl_height_desc = QLabel("Resize Height:")
        lbl_height_desc.setStyleSheet("font-weight: bold;")
        lbl_height_explanation = QLabel("Defines the height of the camera frame. Lower values improve performance but reduce video quality.")
        lbl_height_explanation.setWordWrap(True)
        self.slider_height = QSlider(Qt.Horizontal)
        self.slider_height.setMinimum(240)
        self.slider_height.setMaximum(1080)
        self.slider_height.setValue(self.settings["RESIZE_HEIGHT"])
        self.slider_height.setTickInterval(100)
        self.slider_height.setTickPosition(QSlider.TicksBelow)
        lbl_height_value = QLabel(str(self.slider_height.value()))
        self.slider_height.valueChanged.connect(lambda v: lbl_height_value.setText(str(v)))
        height_layout.addWidget(lbl_height_desc)
        height_layout.addWidget(lbl_height_explanation)
        height_layout.addWidget(self.slider_height)
        height_layout.addWidget(lbl_height_value)
        main_layout.addWidget(frame_height)

        # Process Interval
        frame_interval = QFrame()
        interval_layout = QVBoxLayout(frame_interval)
        lbl_interval_desc = QLabel("Process Interval (s):")
        lbl_interval_desc.setStyleSheet("font-weight: bold;")
        lbl_interval_explanation = QLabel("Controls how often the program processes gestures. Lower values improve responsiveness but use more resources.")
        lbl_interval_explanation.setWordWrap(True)
        self.slider_interval = QSlider(Qt.Horizontal)
        self.slider_interval.setMinimum(1)
        self.slider_interval.setMaximum(50)
        self.slider_interval.setValue(int(self.settings["PROCESS_INTERVAL"] * 100))
        self.slider_interval.setTickInterval(5)
        self.slider_interval.setTickPosition(QSlider.TicksBelow)
        lbl_interval_value = QLabel(f"{self.slider_interval.value() / 100:.2f}")
        self.slider_interval.valueChanged.connect(lambda v: lbl_interval_value.setText(f"{v / 100:.2f}"))
        interval_layout.addWidget(lbl_interval_desc)
        interval_layout.addWidget(lbl_interval_explanation)
        interval_layout.addWidget(self.slider_interval)
        interval_layout.addWidget(lbl_interval_value)
        main_layout.addWidget(frame_interval)

        # Max Distance Angle
        frame_max_angle = QFrame()
        max_angle_layout = QVBoxLayout(frame_max_angle)
        lbl_max_angle_desc = QLabel("Max Distance Angle:")
        lbl_max_angle_desc.setStyleSheet("font-weight: bold;")
        lbl_max_angle_explanation = QLabel("Defines the maximum servo angle (in degrees) when using distance-based control.")
        lbl_max_angle_explanation.setWordWrap(True)
        self.slider_max_angle = QSlider(Qt.Horizontal)
        self.slider_max_angle.setMinimum(0)
        self.slider_max_angle.setMaximum(180)
        self.slider_max_angle.setValue(self.settings["MAX_DISTANCE_ANGLE"])
        self.slider_max_angle.setTickInterval(10)
        self.slider_max_angle.setTickPosition(QSlider.TicksBelow)
        lbl_max_angle_value = QLabel(str(self.slider_max_angle.value()))
        self.slider_max_angle.valueChanged.connect(lambda v: lbl_max_angle_value.setText(str(v)))
        max_angle_layout.addWidget(lbl_max_angle_desc)
        max_angle_layout.addWidget(lbl_max_angle_explanation)
        max_angle_layout.addWidget(self.slider_max_angle)
        max_angle_layout.addWidget(lbl_max_angle_value)
        main_layout.addWidget(frame_max_angle)

        # Max Detectable Distance
        frame_max_dist = QFrame()
        max_dist_layout = QVBoxLayout(frame_max_dist)
        lbl_max_dist_desc = QLabel("Max Detectable Distance (px):")
        lbl_max_dist_desc.setStyleSheet("font-weight: bold;")
        lbl_max_dist_explanation = QLabel("Specifies the maximum distance (in pixels) that the program maps to servo angles.")
        lbl_max_dist_explanation.setWordWrap(True)
        self.slider_max_dist = QSlider(Qt.Horizontal)
        self.slider_max_dist.setMinimum(50)
        self.slider_max_dist.setMaximum(500)
        self.slider_max_dist.setValue(self.settings["MAX_DETECTABLE_DIST"])
        self.slider_max_dist.setTickInterval(50)
        self.slider_max_dist.setTickPosition(QSlider.TicksBelow)
        lbl_max_dist_value = QLabel(str(self.slider_max_dist.value()))
        self.slider_max_dist.valueChanged.connect(lambda v: lbl_max_dist_value.setText(str(v)))
        max_dist_layout.addWidget(lbl_max_dist_desc)
        max_dist_layout.addWidget(lbl_max_dist_explanation)
        max_dist_layout.addWidget(self.slider_max_dist)
        max_dist_layout.addWidget(lbl_max_dist_value)
        main_layout.addWidget(frame_max_dist)

        # Save & Run Button
        btn_save_run = QPushButton("Save & Run Camera")
        btn_save_run.clicked.connect(self.on_save_and_run)
        main_layout.addWidget(btn_save_run)

        # Spacer
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        main_layout.addSpacerItem(spacer)

        main_widget.setLayout(main_layout)

    def on_save_and_run(self):
        """Save settings, then open the mode selector."""
        # Save your config.json as before
        # Instead of running main.py directly, let's call mode_selector.py
        subprocess.Popen(["python3", "mode_selector.py"])
        # close this window or leave it open, up to you
        # self.close()

def main():
    app = QApplication([])
    window = MainUI()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()

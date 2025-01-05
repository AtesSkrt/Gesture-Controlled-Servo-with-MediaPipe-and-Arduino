# mode_selector.py
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
import subprocess

class ModeSelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Mode")
        self.setGeometry(200, 200, 400, 200)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        lbl_info = QLabel("Choose a mode:")
        layout.addWidget(lbl_info)

        btn_servo = QPushButton("Servo")
        btn_servo.clicked.connect(self.on_servo)
        layout.addWidget(btn_servo)

        btn_teach = QPushButton("Teach")
        btn_teach.clicked.connect(self.on_teach)
        layout.addWidget(btn_teach)

        btn_predict = QPushButton("Predict")
        btn_predict.clicked.connect(self.on_predict)
        layout.addWidget(btn_predict)

    def on_servo(self):
        # This runs your existing servo-based script (e.g., "main.py")
        subprocess.Popen(["python3", "main.py"])

    def on_teach(self):
        # Launch the teach UI + camera logic
        subprocess.Popen(["python3", "teach_ui.py"])

    def on_predict(self):
        # Launch a script that does predictions in real-time
        subprocess.Popen(["python3", "predict.py"])

def main():
    app = QApplication([])
    window = ModeSelector()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()

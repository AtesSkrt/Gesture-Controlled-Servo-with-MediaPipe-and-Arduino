import cv2
import mediapipe as mp
import serial
import time
import math

# -----------------------------
# SETTINGS
# -----------------------------
ARDUINO_PORT = '/dev/cu.usbserial-21220'  # Replace with your port
BAUD_RATE = 9600
MAX_SERVO_ANGLE = 180  # Maximum angle for the servo
PROCESS_INTERVAL = 0.1  # Process gestures every 100ms
RESIZE_WIDTH = 640  # Downscale frame width for faster processing
RESIZE_HEIGHT = 480  # Downscale frame height for faster processing

# -----------------------------
# SERIAL SETUP
# -----------------------------
try:
    ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for Arduino to reset
    print(f"Connected to Arduino on port {ARDUINO_PORT}")
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    ser = None

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def map_range(value, in_min, in_max, out_min, out_max):
    """Linear mapping from one range to another."""
    # Clamp the value within in_min and in_max
    value = max(min(value, in_max), in_min)
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def distance_2d(x1, y1, x2, y2):
    """Euclidean distance between two 2D points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def is_right_hand_closed(landmarks, image_width, image_height):
    """
    Determine if the right hand is fully closed (no fingers extended).
    We'll define 'closed' if all finger tips are below their PIP joints.
    """
    # Finger TIP/PIP IDs (MediaPipe) for index, middle, ring, pinky
    finger_tip_pip = [(8, 6), (12, 10), (16, 14), (20, 18)]

    for (tip_id, pip_id) in finger_tip_pip:
        tip_y = landmarks[tip_id].y * image_height
        pip_y = landmarks[pip_id].y * image_height
        if tip_y < pip_y:  # Means finger is extended
            return False

    # Check thumb (simplistic approach, comparing TIP and IP)
    thumb_tip_y = landmarks[4].y * image_height
    thumb_ip_y = landmarks[3].y * image_height

    if thumb_tip_y < thumb_ip_y:  # If thumb tip is higher than IP => extended
        return False

    return True


def measure_thumb_index_distance(landmarks, image_width, image_height):
    """Measure the distance between the thumb tip (4) and index tip (8)."""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    x1, y1 = thumb_tip.x * image_width, thumb_tip.y * image_height
    x2, y2 = index_tip.x * image_width, index_tip.y * image_height

    return distance_2d(x1, y1, x2, y2)


# -----------------------------
# MAIN SCRIPT
# -----------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    last_process_time = time.time()
    in_distance_mode = False  # Flag to track if we're in "distance" mode
    last_angle_sent = None  # Track the last angle sent to Arduino

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Downscale the input resolution for faster processing
        if RESIZE_WIDTH and RESIZE_HEIGHT:
            frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

        # Flip horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        current_time = time.time()
        # We only process gestures at intervals
        if (current_time - last_process_time) > PROCESS_INTERVAL:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            last_process_time = current_time

            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw the landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    # Check left or right
                    hand_label = results.multi_handedness[idx].classification[0].label

                    # We're only interested in the right hand for this mode
                    if hand_label == "Right":
                        # If not in distance mode, check if right hand is closed
                        if not in_distance_mode:
                            if is_right_hand_closed(hand_landmarks.landmark, w, h):
                                in_distance_mode = True
                                print("Switched to DISTANCE MODE.")
                        else:
                            # If in distance mode, measure distance between thumb & index
                            dist = measure_thumb_index_distance(hand_landmarks.landmark, w, h)

                            # Convert distance to angle: from 0 distance => 0°, large distance => 180°
                            # We'll define a max distance for 180° (e.g., ~200 pixels or tune to your preference)
                            max_detectable_dist = 200
                            angle = map_range(dist, 0, max_detectable_dist, 0, MAX_SERVO_ANGLE)
                            angle = int(angle)  # Round to nearest integer

                            # Send angle if changed
                            if ser and angle != last_angle_sent:
                                ser.write((str(angle) + "\n").encode())
                                print(f"Distance = {dist:.2f}, Angle = {angle}")
                                last_angle_sent = angle
            else:
                # No hands detected => optionally reset state or remain in distance mode
                pass

        # Text feedback
        mode_text = "DISTANCE MODE" if in_distance_mode else "FIST to ENTER DISTANCE MODE"
        cv2.putText(
            frame,
            mode_text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )

        cv2.imshow("Distance Mode", frame)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()


if __name__ == "__main__":
    main()

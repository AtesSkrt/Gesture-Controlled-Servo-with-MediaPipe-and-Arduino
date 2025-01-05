import cv2
import mediapipe as mp
import serial
import time
import threading
import math
import json
import os

# -------------------------------------
# LOAD SETTINGS FROM config.json
# -------------------------------------
CONFIG_FILE = "config.json"
DEFAULT_SETTINGS = {
    "LIMIT_SERIAL_WRITES": True,
    "RESIZE_WIDTH": 1280,
    "RESIZE_HEIGHT": 720,
    "PROCESS_INTERVAL": 0.05,
    "USE_MULTITHREADING": True,
    "MAX_DISTANCE_ANGLE": 180,
    "MAX_DETECTABLE_DIST": 200
}

def load_settings():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                user_settings = json.load(f)
            return {**DEFAULT_SETTINGS, **user_settings}
        except:
            pass
    return DEFAULT_SETTINGS.copy()

settings = load_settings()

LIMIT_SERIAL_WRITES = settings["LIMIT_SERIAL_WRITES"]
RESIZE_WIDTH = settings["RESIZE_WIDTH"]
RESIZE_HEIGHT = settings["RESIZE_HEIGHT"]
PROCESS_INTERVAL = settings["PROCESS_INTERVAL"]
USE_MULTITHREADING = settings["USE_MULTITHREADING"]
MAX_DISTANCE_ANGLE = settings["MAX_DISTANCE_ANGLE"]
MAX_DETECTABLE_DIST = settings["MAX_DETECTABLE_DIST"]

# -------------------------------------
# ARDUINO SERIAL SETUP
# -------------------------------------
arduino_port = '/dev/cu.usbserial-21220'  # Update if needed
baud_rate = 9600

try:
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    time.sleep(2)
    print(f"Connected to Arduino on port {arduino_port}")
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    ser = None

# -------------------------------------
# MEDIAPIPE HANDS SETUP
# -------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------------------
# HELPER FUNCTIONS
# -------------------------------------
def map_range(value, in_min, in_max, out_min, out_max):
    value = max(min(value, in_max), in_min)
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def distance_2d(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def is_finger_extended(landmark_list, tip_id, pip_id, w, h):
    tip_y = landmark_list[tip_id].y * h
    pip_y = landmark_list[pip_id].y * h
    return tip_y < pip_y

def is_thumb_raised(landmark_list, w, h):
    thumb_tip = landmark_list[4]
    thumb_ip  = landmark_list[3]
    thumb_mcp = landmark_list[2]
    wrist     = landmark_list[0]

    thumb_tip_x  = thumb_tip.x  * w
    thumb_tip_y  = thumb_tip.y  * h
    thumb_ip_y   = thumb_ip.y   * h
    thumb_mcp_y  = thumb_mcp.y  * h
    wrist_x      = wrist.x      * w
    thumb_mcp_x  = thumb_mcp.x  * w

    is_far = abs(thumb_tip_x - wrist_x) > abs(thumb_mcp_x - wrist_x)
    is_above = (thumb_tip_y < thumb_ip_y) and (thumb_tip_y < thumb_mcp_y)
    return is_far and is_above

def is_fist(landmark_list, w, h):
    """Return True if *no fingers* are extended (fist)."""
    # Index, Middle, Ring, Pinky
    fingers = [(8,6), (12,10), (16,14), (20,18)]
    for tip_id, pip_id in fingers:
        tip_y = landmark_list[tip_id].y * h
        pip_y = landmark_list[pip_id].y * h
        if tip_y < pip_y:  # finger is extended
            return False

    # Check thumb
    thumb_tip_y = landmark_list[4].y * h
    thumb_ip_y  = landmark_list[3].y * h
    if thumb_tip_y < thumb_ip_y:
        return False

    return True

def count_raised_fingers(landmark_list, w, h):
    """
    Count how many fingers (including thumb) are extended.
    This is used for the right hand to decide which mode to use.
    """
    thumb_up   = is_thumb_raised(landmark_list, w, h)
    index_up   = is_finger_extended(landmark_list, 8, 6, w, h)
    middle_up  = is_finger_extended(landmark_list, 12, 10, w, h)
    ring_up    = is_finger_extended(landmark_list, 16, 14, w, h)
    pinky_up   = is_finger_extended(landmark_list, 20, 18, w, h)

    return sum([thumb_up, index_up, middle_up, ring_up, pinky_up])

def classify_left_hand_gesture(landmark_list, w, h):
    """
    Count how many fingers are extended on the LEFT hand to map to angles 0-100.
    """
    return count_raised_fingers(landmark_list, w, h)

def measure_left_thumb_index_distance(landmark_list, w, h):
    thumb_tip = landmark_list[4]
    index_tip = landmark_list[8]

    x1, y1 = thumb_tip.x * w, thumb_tip.y * h
    x2, y2 = index_tip.x * w, index_tip.y * h
    return distance_2d(x1, y1, x2, y2)

def left_hand_angle(count_raised):
    angle_map = {
        0: 0,
        1: 20,
        2: 40,
        3: 60,
        4: 80,
        5: 100
    }
    return angle_map.get(count_raised, 0)

# -------------------------------------
# ARDUINO COMMUNICATION
# -------------------------------------
last_angle_sent = None

def send_to_arduino(angle):
    global last_angle_sent
    if ser:
        ser.write((str(angle) + "\n").encode())
        print(f"Moving servo to {angle} degrees.")
        last_angle_sent = angle

# -------------------------------------
# MAIN LOOP
# -------------------------------------
def main():
    global last_angle_sent

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Start in "fingers" mode
    current_mode = "fingers"
    last_process_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        if RESIZE_WIDTH and RESIZE_HEIGHT:
            frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        current_time = time.time()
        do_process = (current_time - last_process_time) > PROCESS_INTERVAL

        left_fist = False
        right_fist = False

        left_finger_count = 0
        left_distance = 0

        if do_process:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)
            last_process_time = current_time

            if result.multi_hand_landmarks and result.multi_handedness:
                # Track how many fingers are up on the RIGHT hand
                right_finger_count = 0

                for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    label = result.multi_handedness[idx].classification[0].label

                    # Check fists for possible exit
                    if is_fist(hand_landmarks.landmark, w, h):
                        if label == "Left":
                            left_fist = True
                        else:
                            right_fist = True

                    if label == "Right":
                        # Count how many fingers are up on the right hand
                        right_finger_count = count_raised_fingers(hand_landmarks.landmark, w, h)

                    else:  # Left hand
                        if current_mode == "fingers":
                            # Count # of fingers up on left hand
                            left_finger_count = classify_left_hand_gesture(hand_landmarks.landmark, w, h)
                        else:
                            # distance mode
                            left_distance = measure_left_thumb_index_distance(hand_landmarks.landmark, w, h)

                # If both fists => exit camera feed
                if left_fist and right_fist:
                    print("Both hands are fists. Exiting camera feed.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                # Decide mode based on RIGHT hand finger count:
                #   1 finger => finger count mode
                #   2 fingers => distance mode
                if right_finger_count == 1:
                    current_mode = "fingers"
                elif right_finger_count == 2:
                    current_mode = "distance"
                # If 0, 3, 4, 5, do nothing special. (Customize if needed.)

            # Determine servo angle based on current mode
            if current_mode == "fingers":
                angle_to_move = left_hand_angle(left_finger_count)
            else:  # distance mode
                angle_to_move = int(map_range(left_distance, 0, MAX_DETECTABLE_DIST, 0, MAX_DISTANCE_ANGLE))

            # Send angle to Arduino if changed
            if (not LIMIT_SERIAL_WRITES) or (LIMIT_SERIAL_WRITES and angle_to_move != last_angle_sent):
                if USE_MULTITHREADING:
                    threading.Thread(target=send_to_arduino, args=(angle_to_move,)).start()
                else:
                    send_to_arduino(angle_to_move)

        # UI overlay
        cv2.putText(frame, f"MODE: {current_mode.upper()}", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if current_mode == "fingers":
            cv2.putText(frame, f"Left Fingers: {left_finger_count}", (10,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, f"Distance: {int(left_distance)} px", (10,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Show
        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

    if ser:
        ser.close()

if __name__ == "__main__":
    main()

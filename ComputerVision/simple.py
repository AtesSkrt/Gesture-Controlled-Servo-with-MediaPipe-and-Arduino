import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Camera not accessible.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame not captured.")
        break

    cv2.imshow("Test Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

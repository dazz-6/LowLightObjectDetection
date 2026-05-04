import cv2
import time
from ultralytics import YOLO
print("Script started")
# ==============================
# LOAD YOUR TRAINED MODEL
# ==============================
model = YOLO("best.pt")   # IMPORTANT: your trained weights

# ==============================
# OPTIONAL: CLAHE ENHANCEMENT
# ==============================
def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# ==============================
# WEBCAM SETUP
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not working")
    exit()
else:
    print("✅ Camera opened")

prev_time = 0
print("Press 'q' to quit")
# ==============================
# REAL-TIME LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    frame = cv2.resize(frame, (640, 480))

    # OPTIONAL: enable if needed
    # frame = apply_clahe(frame)

    # ==============================
    # YOLO DETECTION
    # ==============================
    results = model(frame, conf=0.4)

    annotated = results[0].plot()

    # ==============================
    # FPS CALCULATION
    # ==============================
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(
        annotated,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # ==============================
    # DISPLAY OUTPUT
    # ==============================
    cv2.imshow("Occlusion-Robust Real-Time Detection", annotated)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()
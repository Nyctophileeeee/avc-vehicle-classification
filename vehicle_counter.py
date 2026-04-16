import cv2
import csv
import time
from ultralytics import YOLO
from collections import defaultdict

# === CONFIG ===
MODEL_PATH = r'C:\Users\ayush\Documents\GitHub\avc-vehicle-classification\yolo11n.pt'
OUTPUT_CSV = r'C:\Users\ayush\Documents\GitHub\avc-vehicle-classification\vehicle_log.csv'
COUNT_LINE_Y = 300
VIDEO_PATH = r'C:\Users\ayush\Documents\GitHub\avc-vehicle-classification\traffic.mp4'

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)
CLASS_NAMES = {2: 'car', 5: 'bus', 7: 'truck'}
VEHICLE_CLASSES = [2, 5, 7]

# === STATE ===
counted_ids = set()
counts = defaultdict(int)
track_history = defaultdict(list)

# === CSV SETUP ===
csv_file = open(OUTPUT_CSV, 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['timestamp', 'track_id', 'class', 'confidence', 'x', 'y', 'weather'])
WEATHER = "Normal"

print("="*50)
print("AVC&C — Vehicle Counter")
print(f"Model: {MODEL_PATH}")
print(f"Source: {VIDEO_PATH}")
print("Press Q to quit | Press W to cycle weather")
print("="*50)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Cannot open video!")
    print("traffic.mp4 not found — switching to webcam")
    cap = cv2.VideoCapture(0)

print(f"Opened! FPS: {cap.get(cv2.CAP_PROP_FPS)}")

WEATHER_OPTIONS = ["Normal", "Rain", "Fog", "Snow", "Sand"]
weather_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video ended or no frame")
        break

    results = model.track(frame, persist=True, conf=0.15, iou=0.3,
                          classes=VEHICLE_CLASSES, verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes     = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes   = results[0].boxes.cls.cpu().numpy().astype(int)
        confs     = results[0].boxes.conf.cpu().numpy()

        for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            class_name = CLASS_NAMES.get(int(cls), 'unknown')
            color = (0,255,170) if cls==2 else (0,165,255) if cls==5 else (255,100,0)

            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
            cv2.putText(frame, f'{class_name} #{track_id} {conf:.2f}',
                       (int(x1), int(y1)-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if track_id not in counted_ids:
                track_history[track_id].append(cy)
                if len(track_history[track_id]) >= 2:
                    prev_y = track_history[track_id][-2]
                    if prev_y < COUNT_LINE_Y <= cy or cy < COUNT_LINE_Y <= prev_y:
                        counted_ids.add(track_id)
                        counts[class_name] += 1
                        timestamp = time.strftime('%H:%M:%S')
                        writer.writerow([timestamp, track_id, class_name,
                                        f'{conf:.2f}', cx, cy, WEATHER])
                        csv_file.flush()
                        print(f"Counted: {class_name} #{track_id} | Total: {dict(counts)}")

    h, w = frame.shape[:2]
    cv2.line(frame, (0, COUNT_LINE_Y), (w, COUNT_LINE_Y), (0,255,255), 2)
    cv2.putText(frame, 'COUNT LINE', (10, COUNT_LINE_Y-8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    cv2.rectangle(frame, (0,0), (210,135), (0,0,0), -1)
    cv2.putText(frame, f"Cars:   {counts['car']}",        (10,30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,170), 2)
    cv2.putText(frame, f"Buses:  {counts['bus']}",        (10,60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
    cv2.putText(frame, f"Trucks: {counts['truck']}",      (10,90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,0), 2)
    cv2.putText(frame, f"Total:  {sum(counts.values())}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Weather: {WEATHER}", (w-200, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,100), 2)

    cv2.imshow('AVC&C - Live Vehicle Counter | Press Q to quit', frame)

    key = cv2.waitKey(10) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        break
    elif cv2.getWindowProperty('AVC&C - Live Vehicle Counter | Press Q to quit',
                               cv2.WND_PROP_VISIBLE) < 1:
        break
    elif key == ord('w'):
        weather_idx = (weather_idx + 1) % len(WEATHER_OPTIONS)
        WEATHER = WEATHER_OPTIONS[weather_idx]
        print(f"Weather: {WEATHER}")

cap.release()
csv_file.close()
cv2.destroyAllWindows()
print(f"\nFinal counts: {dict(counts)}")
print(f"CSV saved to: {OUTPUT_CSV}")
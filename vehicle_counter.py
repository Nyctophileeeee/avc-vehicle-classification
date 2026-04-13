import cv2
import csv
import time
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

# === CONFIG ===
MODEL_PATH = r'C:\Users\ayush\Documents\GitHub\avc-vehicle-classification\runs\detect\runs\train\avc_model_v15\weights\best.pt'
VIDEO_PATH = r'C:\Users\ayush\Documents\GitHub\avc-vehicle-classification\traffic.mp4'
OUTPUT_CSV = r'C:\Users\ayush\Documents\GitHub\avc-vehicle-classification\vehicle_log.csv'
COUNT_LINE_Y = 400  # horizontal line position for counting

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)
CLASS_NAMES = {0: 'car', 1: 'bus', 2: 'truck'}

# === TRACKING STATE ===
counted_ids = set()
counts = defaultdict(int)
track_history = defaultdict(list)

# === CSV SETUP ===
csv_file = open(OUTPUT_CSV, 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['timestamp', 'track_id', 'class', 'confidence', 'x', 'y'])

# === VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")
print(f"Model loaded: {MODEL_PATH}")
print("Processing... Press Q to quit")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model.track(frame, persist=True, conf=0.3, iou=0.5, verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            class_name = CLASS_NAMES.get(cls, 'unknown')

            # Draw box
            color = (0, 255, 170) if cls == 0 else (0, 165, 255) if cls == 1 else (255, 100, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f'{class_name} #{track_id} {conf:.2f}',
                       (int(x1), int(y1)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Count when crossing line
            if track_id not in counted_ids:
                track_history[track_id].append(cy)
                if len(track_history[track_id]) >= 2:
                    prev_y = track_history[track_id][-2]
                    if prev_y < COUNT_LINE_Y <= cy or cy < COUNT_LINE_Y <= prev_y:
                        counted_ids.add(track_id)
                        counts[class_name] += 1
                        timestamp = time.strftime('%H:%M:%S')
                        writer.writerow([timestamp, track_id, class_name, f'{conf:.2f}', cx, cy])
                        csv_file.flush()

    # Draw count line
    cv2.line(frame, (0, COUNT_LINE_Y), (frame.shape[1], COUNT_LINE_Y), (0, 255, 255), 2)
    cv2.putText(frame, 'COUNT LINE', (10, COUNT_LINE_Y - 8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw counts
    y_pos = 30
    cv2.putText(frame, f"Cars: {counts['car']}", (10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 170), 2)
    cv2.putText(frame, f"Buses: {counts['bus']}", (10, y_pos+30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    cv2.putText(frame, f"Trucks: {counts['truck']}", (10, y_pos+60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
    cv2.putText(frame, f"Total: {sum(counts.values())}", (10, y_pos+90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('AVC&C - Vehicle Counter', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()

print(f"\nFinal counts:")
for k, v in counts.items():
    print(f"  {k}: {v}")
print(f"CSV saved to: {OUTPUT_CSV}")
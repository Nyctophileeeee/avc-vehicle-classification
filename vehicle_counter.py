import cv2
from ultralytics import YOLO
import csv
from datetime import datetime

# Load model
model = YOLO("yolo11n.pt")

# Load plate detector
plate_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
)

cap = cv2.VideoCapture("src/traffic.mp4")

counts = {"car": 0, "bus": 0, "truck": 0}
counted_ids = set()
class_names = {2: "car", 5: "bus", 7: "truck"}
class_colours = {
    "car":   (0, 255, 100),
    "bus":   (0, 180, 255),
    "truck": (255, 80, 80),
}
LINE_Y = 300

with open("vehicle_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "vehicle_id", "class",
                     "total_cars", "total_buses", "total_trucks"])

def blur_plates(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 20))
    for (x, y, w, h) in plates:
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], (51, 51), 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, "PLATE BLURRED", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return frame, len(plates)

print("Starting AVC System with YOLO11... Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video finished!")
        break

    frame, plates_blurred = blur_plates(frame)

    results = model.track(
        frame, persist=True,
        classes=[2, 5, 7], conf=0.4, verbose=False
    )

    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 255, 255), 2)
    cv2.putText(frame, "COUNTING LINE", (10, LINE_Y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes
        for box, cls, track_id in zip(boxes.xyxy, boxes.cls, boxes.id):
            x1, y1, x2, y2 = map(int, box)
            obj_id = int(track_id)
            class_name = class_names.get(int(cls), None)
            if class_name is None:
                continue
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            colour = class_colours[class_name]
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.circle(frame, (cx, cy), 4, colour, -1)
            cv2.putText(frame, f"{class_name} #{obj_id}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
            if cy > LINE_Y and obj_id not in counted_ids:
                counts[class_name] += 1
                counted_ids.add(obj_id)
                print(f"✅ {class_name.upper()} counted! "
                      f"Cars:{counts['car']} Buses:{counts['bus']} Trucks:{counts['truck']}")
                with open("vehicle_log.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.now().strftime("%H:%M:%S"),
                                     obj_id, class_name,
                                     counts["car"], counts["bus"], counts["truck"]])

    cv2.putText(frame, "AVC System - Bradford Council | YOLO11", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Privacy Active | Plates blurred: {plates_blurred}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    y_pos = 90
    for name, count in counts.items():
        cv2.putText(frame, f"{name.upper()}: {count}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, class_colours[name], 2)
        y_pos += 35

    cv2.imshow("AVC System - Bradford Council", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\n" + "="*40)
print(f"Cars: {counts['car']} | Buses: {counts['bus']} | Trucks: {counts['truck']}")
print(f"TOTAL: {sum(counts.values())}")
print("YOLO11 + Privacy blur complete!")
print("="*40)
from ultralytics import YOLO

# Load pre-trained model — downloads automatically
model = YOLO("yolo11n.pt")

# Run detection on webcam
results = model(
    source="src/traffic.mp4",
    show=True,           
    classes=[2, 5, 7],   
    conf=0.4             
)

print("Detection complete!")
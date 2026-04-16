from ultralytics import YOLO

model = YOLO('yolo11n.pt')

model.train(
    data=r'C:\Users\ayush\Documents\GitHub\avc-vehicle-classification\src\data\train data\data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    name='avc_model_v2',
    project='runs/train',
    patience=10,
    save=True,
    device='cpu'
)

print("Training complete!")
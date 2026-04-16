import os
import shutil
import random
from pathlib import Path

random.seed(42)

BASE = Path("C:/Users/ayush/Documents/GitHub/avc-vehicle-classification/src/data/train data")

OUT = BASE
for split in ["train", "val", "test"]:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

# === ALL DATA SOURCES ===
SOURCES = [
    # Original weather data
    BASE / "Fog" / "Fog",
    BASE / "Rain" / "Rain",
    BASE / "Sand (1)" / "Sand",
    BASE / "Snow (1)" / "Snow",
    BASE / "normal condition day plus night",
    # New datasets
    BASE / "Night time Detection.v1i.yolov11" / "train",
    BASE / "Night time Detection.v1i.yolov11" / "valid",
    BASE / "rainy.v1i.yolov11" / "train",
    BASE / "rainy.v1i.yolov11" / "valid",
    BASE / "Traffic Lights 3.v2i.yolov11" / "train",
    BASE / "Traffic Lights 3.v2i.yolov11" / "valid",
]

all_pairs = []

for source in SOURCES:
    if not source.exists():
        print(f"SKIPPING (not found): {source}")
        continue

    img_dir = source / "images" if (source / "images").exists() else source
    lbl_dir = source / "labels" if (source / "labels").exists() else None

    # Also check for YOLO_darknet subfolder
    for sub in source.iterdir() if source.is_dir() else []:
        if "YOLO_darknet" in sub.name:
            lbl_dir = sub
            break

    count = 0
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        for img in img_dir.glob(ext):
            if lbl_dir:
                label = lbl_dir / (img.stem + ".txt")
            else:
                label = img.with_suffix(".txt")
            all_pairs.append((img, label))
            count += 1

    print(f"Found {count} images in: {source.name}")

print(f"\nTotal images: {len(all_pairs)}")

random.shuffle(all_pairs)
n = len(all_pairs)
train_end = int(n * 0.7)
val_end   = int(n * 0.9)

splits = {
    "train": all_pairs[:train_end],
    "val":   all_pairs[train_end:val_end],
    "test":  all_pairs[val_end:]
}

for split, pairs in splits.items():
    for img_path, label_path in pairs:
        shutil.copy2(img_path, OUT / "images" / split / img_path.name)
        if label_path.exists():
            shutil.copy2(label_path, OUT / "labels" / split / label_path.name)
    print(f"{split}: {len(pairs)} images")

print("\nDone! Ready to train.")
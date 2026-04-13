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

all_pairs = []

for dataset_folder in BASE.iterdir():
    if not dataset_folder.is_dir():
        continue
    if dataset_folder.name in ["images", "labels"]:
        continue

    # Try YOLO_darknet structure first
    yolo_dir = None
    for sub in dataset_folder.iterdir():
        if "YOLO_darknet" in sub.name:
            yolo_dir = sub
            break

    if yolo_dir:
        imgs = list((yolo_dir).glob("*.jpg")) + list((yolo_dir).glob("*.png"))
        for img in imgs:
            lbl = yolo_dir / (img.stem + ".txt")
            all_pairs.append((img, lbl))
        print(f"{dataset_folder.name}: {len(imgs)} images (YOLO darknet)")

    else:
        # Try train/valid/test structure
        for split_name in ["train", "valid", "val"]:
            split_dir = dataset_folder / split_name / "images"
            label_dir = dataset_folder / split_name / "labels"
            if split_dir.exists():
                imgs = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
                for img in imgs:
                    lbl = label_dir / (img.stem + ".txt")
                    all_pairs.append((img, lbl))
                print(f"{dataset_folder.name}/{split_name}: {len(imgs)} images")

print(f"\nTotal images found: {len(all_pairs)}")

random.shuffle(all_pairs)
train_end = int(0.7 * len(all_pairs))
val_end = int(0.85 * len(all_pairs))

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
    print(f"{split}: {len(pairs)} images copied")

print("\nDone! Dataset ready for training.")
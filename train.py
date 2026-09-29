import os
import argparse
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


def parse_annotation_line(line: str) -> Tuple[float, float, float, float, int]:
    line = line.strip()
    if not line:
        raise ValueError("Empty annotation line")

    # Accept format: xmin,ymin,xmax,ymax,class_id
    if "," in line and "(" not in line:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 5:
            raise ValueError(f"Invalid annotation format: {line}")
        x_min, y_min, x_max, y_max, class_id = parts[:5]
        return float(x_min), float(y_min), float(x_max), float(y_max), int(class_id)

    # Accept format: (xmin,ymin),(xmax,ymax),class_id
    cleaned = line.replace("(", "").replace(")", "")
    parts = [p.strip() for p in cleaned.split(',')]
    if len(parts) < 5:
        raise ValueError(f"Invalid annotation format: {line}")

    x_min, y_min = float(parts[0]), float(parts[1])
    x_max, y_max = float(parts[2]), float(parts[3])
    class_id = int(parts[4])
    return x_min, y_min, x_max, y_max, class_id


class DetectionDataset(Dataset):
    def __init__(self, image_dir: str, annotation_dir: str, image_size: Tuple[int, int] = (480, 480)):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_size = image_size
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)
        annotation_path = os.path.join(self.annotation_dir, os.path.splitext(image_name)[0] + ".txt")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        original_h, original_w = image.shape[:2]
        resized = cv2.resize(image_rgb, self.image_size, interpolation=cv2.INTER_AREA)

        boxes = []
        labels = []

        if os.path.exists(annotation_path):
            with open(annotation_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        x_min, y_min, x_max, y_max, class_id = parse_annotation_line(line)
                    except ValueError:
                        continue

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(original_w, x_max)
                    y_max = min(original_h, y_max)

                    if x_max <= x_min or y_max <= y_min:
                        continue

                    scale_x = self.image_size[0] / original_w
                    scale_y = self.image_size[1] / original_h

                    box = [
                        x_min * scale_x,
                        y_min * scale_y,
                        x_max * scale_x,
                        y_max * scale_y,
                    ]
                    boxes.append(box)
                    labels.append(int(class_id))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }

        return torch.tensor(resized).permute(2, 0, 1), target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train an object detection model with Faster R-CNN")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing images and annotations")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--image-size", type=int, nargs=2, default=(480, 480))
    args = parser.parse_args()

    image_dir = os.path.join(args.data_dir, "images")
    annotation_dir = os.path.join(args.data_dir, "annotations")

    if not os.path.exists(image_dir) or not os.path.exists(annotation_dir):
        raise FileNotFoundError(
            "The dataset folders were not found. Expected: data/images and data/annotations"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DetectionDataset(image_dir=image_dir, annotation_dir=annotation_dir, image_size=tuple(args.image_size))

    if len(dataset) == 0:
        raise ValueError("No images were found in the provided dataset directory.")

    indices = torch.randperm(len(dataset)).tolist()
    split = max(1, int(0.2 * len(dataset)))
    train_indices = indices[:-split]
    val_indices = indices[-split:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    num_classes = 2
    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(args.epochs):
        model.train()
        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()
        print(f"Epoch {epoch + 1}/{args.epochs} completed.")

    torch.save(model.state_dict(), "model_fasterrcnn.pth")
    print("Training complete. Model saved to model_fasterrcnn.pth")


if __name__ == "__main__":
    main()

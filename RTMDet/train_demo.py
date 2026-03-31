"""Minimal RTMDet training demo with synthetic data.

This script demonstrates the full training pipeline:
  1. Generate synthetic images with random colored rectangles
  2. Train the RTMDet model with dynamic label assignment
  3. Run inference and decode predictions

Usage:
    python -m RTMDet.train_demo
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch
import torch.optim as optim
from torch import Tensor

from .model import RTMDet
from .loss import RTMDetLoss


# =============================================================================
# Configuration
# =============================================================================
NUM_CLASSES = 5         # Small number for demo
IMAGE_SIZE = 320        # Input image size (RTMDet typically uses 640, we use 320 for speed)
NUM_IMAGES = 32         # Synthetic dataset size
BATCH_SIZE = 4          # Training batch size
NUM_EPOCHS = 50         # Training epochs
LR = 1e-3               # Learning rate
STRIDES = (8, 16, 32)   # Feature map strides

CLASS_NAMES = ["circle", "square", "triangle", "star", "diamond"]


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_sample(
    image_size: int = 320,
    num_classes: int = 5,
    max_objects: int = 3,
) -> Tuple[Tensor, List[dict]]:
    """Generate a synthetic training image with random colored rectangles.

    Each image contains 1 to max_objects randomly placed rectangles.
    Unlike YOLO which encodes targets into a grid, RTMDet expects
    raw box annotations — the SimOTA assigner handles assignment.

    Args:
        image_size: Width and height of the square image.
        num_classes: Number of possible object classes.
        max_objects: Maximum number of objects per image.

    Returns:
        image: Tensor [3, image_size, image_size] with values in [0, 1].
        annotations: List of dicts with "box" [x1,y1,x2,y2] and "class_id".
    """
    image = torch.rand(3, image_size, image_size) * 0.3  # dark background

    num_objects = random.randint(1, max_objects)
    annotations = []

    for _ in range(num_objects):
        min_size = image_size // 10
        max_size = image_size // 3

        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        x1 = random.randint(0, image_size - w)
        y1 = random.randint(0, image_size - h)
        x2 = x1 + w
        y2 = y1 + h

        color = torch.rand(3)
        image[:, y1:y2, x1:x2] = color.view(3, 1, 1)

        class_id = random.randint(0, num_classes - 1)
        annotations.append({
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "class_id": class_id,
        })

    return image, annotations


def create_synthetic_dataset(
    num_images: int = 32,
) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
    """Create a batch of synthetic images and their annotations.

    Unlike YOLOv1 which pre-encodes targets into a grid tensor,
    RTMDet keeps annotations as variable-length lists of boxes
    because SimOTA handles the assignment dynamically during training.

    Returns:
        images: Tensor [num_images, 3, IMAGE_SIZE, IMAGE_SIZE].
        gt_boxes_list: List of [M_i, 4] tensors (xyxy).
        gt_labels_list: List of [M_i] tensors (class IDs).
    """
    images = []
    gt_boxes_list = []
    gt_labels_list = []

    for _ in range(num_images):
        img, anns = generate_synthetic_sample(IMAGE_SIZE, NUM_CLASSES)
        images.append(img)

        boxes = torch.tensor([a["box"] for a in anns], dtype=torch.float32)
        labels = torch.tensor([a["class_id"] for a in anns], dtype=torch.long)
        gt_boxes_list.append(boxes)
        gt_labels_list.append(labels)

    return torch.stack(images), gt_boxes_list, gt_labels_list


# =============================================================================
# Training Loop
# =============================================================================

def train():
    """Run the full training demo."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Image size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"Classes: {NUM_CLASSES}, Strides: {STRIDES}")

    # Feature map sizes
    for s in STRIDES:
        h = IMAGE_SIZE // s
        print(f"  Stride {s:2d}: {h}×{h} = {h*h} points")
    total_points = sum((IMAGE_SIZE // s) ** 2 for s in STRIDES)
    print(f"  Total: {total_points} prediction points per image\n")

    # ── Create model and loss ──
    model = RTMDet(
        num_classes=NUM_CLASSES,
        input_size=IMAGE_SIZE,
        strides=STRIDES,
    ).to(device)

    criterion = RTMDetLoss(
        num_classes=NUM_CLASSES,
        strides=STRIDES,
    )

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Generate synthetic dataset ──
    print("Generating synthetic dataset...")
    images, gt_boxes_list, gt_labels_list = create_synthetic_dataset(NUM_IMAGES)
    images = images.to(device)
    gt_boxes_list = [b.to(device) for b in gt_boxes_list]
    gt_labels_list = [l.to(device) for l in gt_labels_list]
    print(f"Dataset: {images.shape[0]} images")
    print()

    # ── Training loop ──
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_losses = {"loss_cls": 0.0, "loss_reg": 0.0, "loss_total": 0.0}
        num_batches = 0

        for i in range(0, NUM_IMAGES, BATCH_SIZE):
            batch_images = images[i : i + BATCH_SIZE]
            batch_gt_boxes = gt_boxes_list[i : i + BATCH_SIZE]
            batch_gt_labels = gt_labels_list[i : i + BATCH_SIZE]

            # Forward pass
            cls_scores, bbox_preds = model(batch_images)

            # Compute loss
            losses = criterion(cls_scores, bbox_preds, batch_gt_boxes, batch_gt_labels)

            # Backward pass
            optimizer.zero_grad()
            losses["loss_total"].backward()

            # Gradient clipping (common in modern detectors for stable training)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)

            optimizer.step()

            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            num_batches += 1

        # Print epoch summary
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg = {k: v / num_batches for k, v in epoch_losses.items()}
            print(
                f"Epoch [{epoch + 1:3d}/{NUM_EPOCHS}] "
                f"Total: {avg['loss_total']:.4f}  "
                f"Cls: {avg['loss_cls']:.4f}  "
                f"Reg: {avg['loss_reg']:.4f}"
            )

    # ── Inference demo ──
    print("\n" + "=" * 60)
    print("INFERENCE DEMO")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        test_image = images[:1]
        cls_scores, bbox_preds = model(test_image)

        detections = model.decode_predictions(
            cls_scores, bbox_preds,
            score_threshold=0.3,
            nms_iou_threshold=0.5,
        )

        print(f"\nDetections for test image (conf > 0.3):")
        if detections[0]:
            for det in detections[0]:
                box = det["box"]
                print(
                    f"  Class: {CLASS_NAMES[det['class_id']]:10s}  "
                    f"Conf: {det['confidence']:.3f}  "
                    f"Box: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]"
                )
        else:
            print("  No detections above threshold.")

    print("\nDone!")


if __name__ == "__main__":
    train()

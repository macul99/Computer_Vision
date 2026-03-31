"""Minimal YOLOv1 training demo with synthetic data.

This script demonstrates the full training pipeline:
  1. Generate synthetic images with random colored rectangles
  2. Encode ground truth boxes into the YOLO grid format
  3. Train the YOLOv1 model
  4. Run inference and decode predictions

This is intentionally simple — meant for understanding, not production use.

Usage:
    python -m yolo.train_demo
"""

from __future__ import annotations

import random
from typing import Tuple

import torch
import torch.optim as optim
from torch import Tensor

from .model import YOLOv1
from .loss import YOLOv1Loss


# =============================================================================
# Configuration
# =============================================================================
S = 7              # Grid size (7×7)
B = 2              # Boxes per cell
C = 5              # Number of classes (small number for demo)
IMAGE_SIZE = 448   # Input image size
NUM_IMAGES = 32    # Synthetic dataset size
BATCH_SIZE = 4     # Training batch size
NUM_EPOCHS = 50    # Training epochs
LR = 1e-4          # Learning rate

CLASS_NAMES = ["circle", "square", "triangle", "star", "diamond"]


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_sample(
    image_size: int = 448,
    num_classes: int = 5,
    max_objects: int = 3,
) -> Tuple[Tensor, list[dict]]:
    """Generate a synthetic training image with random colored rectangles.

    Each image contains 1 to max_objects randomly placed rectangles.
    Each rectangle represents an "object" with a random class label.

    This replaces a real dataset loader for demonstration purposes.

    Args:
        image_size: Width and height of the square image.
        num_classes: Number of possible object classes.
        max_objects: Maximum number of objects per image.

    Returns:
        image: Tensor [3, image_size, image_size] with values in [0, 1].
        annotations: List of dicts, each with:
            "box": [x1, y1, x2, y2] in absolute pixel coordinates
            "class_id": int in [0, num_classes-1]
    """
    # Start with a random background
    image = torch.rand(3, image_size, image_size) * 0.3  # dark background

    num_objects = random.randint(1, max_objects)
    annotations = []

    for _ in range(num_objects):
        # Random box position and size (ensure minimum size)
        min_size = image_size // 10  # at least ~45 pixels
        max_size = image_size // 3   # at most ~150 pixels

        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        x1 = random.randint(0, image_size - w)
        y1 = random.randint(0, image_size - h)
        x2 = x1 + w
        y2 = y1 + h

        # Random color for this object
        color = torch.rand(3)

        # Draw the rectangle on the image
        image[:, y1:y2, x1:x2] = color.view(3, 1, 1)

        # Random class
        class_id = random.randint(0, num_classes - 1)

        annotations.append({
            "box": [x1, y1, x2, y2],
            "class_id": class_id,
        })

    return image, annotations


def encode_targets(
    annotations: list[dict],
    S: int = 7,
    B: int = 2,
    C: int = 5,
    image_size: int = 448,
) -> Tensor:
    """Encode ground truth annotations into the YOLOv1 target grid format.

    =======================================================================
    TARGET ENCODING PROCESS
    =======================================================================

    For each ground truth box:
    1. Find which grid cell contains the box center
    2. Compute cell-relative coordinates (x, y) and image-relative (w, h)
    3. Set the objectness flag = 1 for that cell
    4. Set the one-hot class vector

    Important: If two objects fall in the same cell, only the last one
    is kept (YOLOv1 limitation: one object per cell).

    Target tensor layout per cell: [x, y, w, h, obj, class_one_hot]
      - x, y: center offset within the cell (0 to 1)
      - w, h: width/height relative to image (0 to 1)
      - obj: 1.0 if object present, 0.0 otherwise
      - class_one_hot: one-hot vector of length C

    Args:
        annotations: List of {"box": [x1,y1,x2,y2], "class_id": int}.
        S, B, C: YOLO grid parameters.
        image_size: Image dimension (square).

    Returns:
        target: Tensor [S, S, 5 + C].
    """
    target = torch.zeros(S, S, 5 + C)

    for ann in annotations:
        x1, y1, x2, y2 = ann["box"]
        class_id = ann["class_id"]

        # Step 1: Compute normalized box center and size
        # All values normalized to [0, 1] relative to image size
        cx = (x1 + x2) / 2.0 / image_size  # center x, normalized
        cy = (y1 + y2) / 2.0 / image_size  # center y, normalized
        w = (x2 - x1) / image_size          # width, normalized
        h = (y2 - y1) / image_size          # height, normalized

        # Step 2: Find which grid cell the center falls into
        # If cx = 0.35 and S = 7: cell_col = int(0.35 * 7) = int(2.45) = 2
        cell_col = int(cx * S)  # which column (j)
        cell_row = int(cy * S)  # which row (i)

        # Clamp to valid range (edge case: cx or cy = 1.0)
        cell_col = min(cell_col, S - 1)
        cell_row = min(cell_row, S - 1)

        # Step 3: Compute cell-relative offsets
        # x_cell = fractional position within the cell
        # If cx = 0.35, S = 7, cell_col = 2:
        #   x_cell = 0.35 * 7 - 2 = 2.45 - 2 = 0.45
        x_cell = cx * S - cell_col  # offset within cell (0 to 1)
        y_cell = cy * S - cell_row

        # Step 4: Fill in the target tensor
        # Layout: [x, y, w, h, obj, class_0, ..., class_{C-1}]
        target[cell_row, cell_col, 0] = x_cell  # x offset in cell
        target[cell_row, cell_col, 1] = y_cell  # y offset in cell
        target[cell_row, cell_col, 2] = w        # width (image-relative)
        target[cell_row, cell_col, 3] = h        # height (image-relative)
        target[cell_row, cell_col, 4] = 1.0      # object present flag

        # One-hot class encoding
        target[cell_row, cell_col, 5:] = 0.0     # reset (in case of overlap)
        target[cell_row, cell_col, 5 + class_id] = 1.0

    return target


def create_synthetic_dataset(
    num_images: int = 32,
) -> Tuple[Tensor, Tensor]:
    """Create a batch of synthetic images and their YOLO-encoded targets.

    Returns:
        images: Tensor [num_images, 3, 448, 448]
        targets: Tensor [num_images, S, S, 5 + C]
    """
    images = []
    targets = []

    for _ in range(num_images):
        img, anns = generate_synthetic_sample(IMAGE_SIZE, C)
        target = encode_targets(anns, S, B, C, IMAGE_SIZE)
        images.append(img)
        targets.append(target)

    return torch.stack(images), torch.stack(targets)


# =============================================================================
# Training Loop
# =============================================================================

def train():
    """Run the full training demo."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Grid: {S}×{S}, Boxes/cell: {B}, Classes: {C}")
    print(f"Output per cell: {B * 5 + C} values")
    print(f"Total output: {S}×{S}×{B * 5 + C} = {S * S * (B * 5 + C)} values\n")

    # ── Create model and loss ──
    model = YOLOv1(S=S, B=B, C=C, image_size=IMAGE_SIZE).to(device)
    criterion = YOLOv1Loss(S=S, B=B, C=C)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Generate synthetic dataset ──
    print("Generating synthetic dataset...")
    images, targets = create_synthetic_dataset(NUM_IMAGES)
    images = images.to(device)
    targets = targets.to(device)
    print(f"Dataset: {images.shape[0]} images of size {images.shape[2]}×{images.shape[3]}")
    print(f"Target shape: {targets.shape}")
    print()

    # ── Training loop ──
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_losses = {
            "loss_coord": 0.0,
            "loss_conf_obj": 0.0,
            "loss_conf_noobj": 0.0,
            "loss_class": 0.0,
            "loss_total": 0.0,
        }
        num_batches = 0

        # Simple batching (no DataLoader for simplicity)
        for i in range(0, NUM_IMAGES, BATCH_SIZE):
            batch_images = images[i : i + BATCH_SIZE]
            batch_targets = targets[i : i + BATCH_SIZE]

            # Forward pass
            predictions = model(batch_images)  # [B, S, S, B*5+C]

            # Compute loss
            losses = criterion(predictions, batch_targets)

            # Backward pass
            optimizer.zero_grad()
            losses["loss_total"].backward()
            optimizer.step()

            # Accumulate losses for logging
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            num_batches += 1

        # Print epoch summary
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg = {k: v / num_batches for k, v in epoch_losses.items()}
            print(
                f"Epoch [{epoch + 1:3d}/{NUM_EPOCHS}] "
                f"Total: {avg['loss_total']:.4f}  "
                f"Coord: {avg['loss_coord']:.4f}  "
                f"Conf_obj: {avg['loss_conf_obj']:.4f}  "
                f"Conf_noobj: {avg['loss_conf_noobj']:.4f}  "
                f"Class: {avg['loss_class']:.4f}"
            )

    # ── Inference demo ──
    print("\n" + "=" * 60)
    print("INFERENCE DEMO")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        # Take the first image
        test_image = images[:1]  # [1, 3, 448, 448]
        output = model(test_image)  # [1, S, S, B*5+C]

        detections = model.decode_predictions(output, confidence_threshold=0.3)

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

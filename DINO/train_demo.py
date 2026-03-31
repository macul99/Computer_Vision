"""Minimal DINO training demo with synthetic data.

This script demonstrates the full DINO training pipeline:
  1. Generate synthetic images with random colored rectangles
  2. Prepare targets in DINO format (labels + normalized boxes)
  3. Train the DINO model with all loss components
  4. Run inference and display predictions

This is intentionally simple — meant for understanding, not production use.

Usage:
    python -m DINO.train_demo
"""

from __future__ import annotations

import random
from typing import Tuple, List, Dict

import torch
import torch.optim as optim
from torch import Tensor

from .model import DINO, DINOConfig
from .loss import DINOLoss, LossConfig


# =============================================================================
# Configuration
# =============================================================================
IMAGE_SIZE = 256       # Input image size (smaller for fast demo)
NUM_CLASSES = 5        # Number of object classes
NUM_QUERIES = 50       # Object queries (smaller for fast demo)
NUM_IMAGES = 32        # Synthetic dataset size
BATCH_SIZE = 4         # Training batch size
NUM_EPOCHS = 30        # Training epochs
LR = 1e-4              # Learning rate
WEIGHT_DECAY = 1e-4    # AdamW weight decay

CLASS_NAMES = ["circle", "square", "triangle", "star", "diamond"]


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_sample(
    image_size: int = 256,
    num_classes: int = 5,
    max_objects: int = 3,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Generate a synthetic training image with random colored rectangles.

    Each image contains 1 to max_objects randomly placed rectangles.
    Each rectangle represents an "object" with a random class label.

    Unlike YOLO which encodes targets into a grid, DINO takes targets
    as a list of (class_label, box) pairs per image. Boxes are in
    normalized (cx, cy, w, h) format with values in [0, 1].

    Args:
        image_size: Width and height of the square image.
        num_classes: Number of possible object classes.
        max_objects: Maximum number of objects per image.

    Returns:
        image:  [3, H, W] synthetic image tensor.
        target: Dict with:
                  "labels": [N] LongTensor of class indices
                  "boxes":  [N, 4] FloatTensor of (cx, cy, w, h) normalized
    """
    # Start with a random background
    image = torch.rand(3, image_size, image_size) * 0.3  # Dark background

    num_objects = random.randint(1, max_objects)
    labels = []
    boxes = []

    for _ in range(num_objects):
        # Random class
        cls = random.randint(0, num_classes - 1)

        # Random box in pixel coordinates
        min_size = image_size // 8
        max_size = image_size // 3
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        x1 = random.randint(0, image_size - w)
        y1 = random.randint(0, image_size - h)

        # Draw colored rectangle on the image
        # Each class gets a distinct color for visual debugging
        color = torch.zeros(3)
        color[cls % 3] = 0.7 + 0.3 * random.random()
        if cls >= 3:
            color[(cls + 1) % 3] = 0.5
        image[:, y1:y1 + h, x1:x1 + w] = color.view(3, 1, 1)

        # Convert to normalized (cx, cy, w, h)
        cx = (x1 + w / 2) / image_size
        cy = (y1 + h / 2) / image_size
        nw = w / image_size
        nh = h / image_size

        labels.append(cls)
        boxes.append([cx, cy, nw, nh])

    target = {
        "labels": torch.tensor(labels, dtype=torch.long),
        "boxes": torch.tensor(boxes, dtype=torch.float32),
    }

    return image, target


def generate_dataset(
    num_images: int,
    image_size: int = 256,
    num_classes: int = 5,
) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    """Generate a complete synthetic dataset.

    Args:
        num_images: Number of images to generate.
        image_size: Size of each image.
        num_classes: Number of object classes.

    Returns:
        images:  [N, 3, H, W] all images stacked.
        targets: List of N target dicts.
    """
    images = []
    targets = []
    for _ in range(num_images):
        img, tgt = generate_synthetic_sample(image_size, num_classes)
        images.append(img)
        targets.append(tgt)

    return torch.stack(images), targets


# =============================================================================
# Training Loop
# =============================================================================

def main() -> None:
    """Run the complete DINO training demo."""

    # ─── Device selection ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ─── Generate synthetic data ───
    print(f"Generating {NUM_IMAGES} synthetic images ({IMAGE_SIZE}×{IMAGE_SIZE})...")
    images, targets = generate_dataset(NUM_IMAGES, IMAGE_SIZE, NUM_CLASSES)

    # ─── Create model ───
    config = DINOConfig(
        num_classes=NUM_CLASSES,
        num_queries=NUM_QUERIES,
        hidden_dim=128,        # Smaller for fast training
        ffn_dim=512,
        enc_layers=2,
        dec_layers=3,
        enc_heads=4,
        dec_heads=4,
        dn_number=3,           # 3 denoising groups
        backbone_channels=[32, 64, 128],  # Smaller backbone
    )
    model = DINO(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {num_params:,} parameters")

    # ─── Create loss function ───
    loss_config = LossConfig()
    criterion = DINOLoss(NUM_CLASSES, loss_config)

    # ─── Optimizer ───
    # AdamW is standard for transformer-based detectors.
    # Different learning rates for backbone vs transformer is common
    # in full DINO, but we use a single rate here for simplicity.
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ─── Training loop ───
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    print("=" * 70)

    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_losses = {}

        # Simple batching (no DataLoader for simplicity)
        indices = list(range(NUM_IMAGES))
        random.shuffle(indices)

        for batch_start in range(0, NUM_IMAGES, BATCH_SIZE):
            batch_idx = indices[batch_start:batch_start + BATCH_SIZE]
            batch_images = images[batch_idx].to(device)
            batch_targets = [
                {k: v.to(device) for k, v in targets[i].items()}
                for i in batch_idx
            ]

            # ─── Forward pass ───
            outputs = model(batch_images, batch_targets)

            # ─── Compute losses ───
            losses = criterion(outputs, batch_targets)

            # ─── Total loss ───
            total_loss = sum(losses.values())

            # ─── Backward pass ───
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping (standard in DETR-style training)
            # Prevents exploding gradients, especially early in training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

            optimizer.step()

            # Accumulate losses for reporting
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

        # ─── Print epoch summary ───
        if (epoch + 1) % 5 == 0 or epoch == 0:
            num_batches = (NUM_IMAGES + BATCH_SIZE - 1) // BATCH_SIZE
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            total = sum(avg_losses.values())

            # Show the main loss components
            cls_loss = avg_losses.get("loss_cls", 0.0)
            bbox_loss = avg_losses.get("loss_bbox", 0.0)
            giou_loss = avg_losses.get("loss_giou", 0.0)
            dn_cls = avg_losses.get("loss_cls_dn", 0.0)
            dn_bbox = avg_losses.get("loss_bbox_dn", 0.0)

            print(
                f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} | "
                f"Total: {total:.4f} | "
                f"cls: {cls_loss:.4f} | "
                f"bbox: {bbox_loss:.4f} | "
                f"giou: {giou_loss:.4f} | "
                f"dn_cls: {dn_cls:.4f} | "
                f"dn_bbox: {dn_bbox:.4f}"
            )

    print("=" * 70)
    print("Training complete!")

    # =================================================================
    # Inference Demo
    # =================================================================
    print("\n" + "=" * 70)
    print("INFERENCE DEMO")
    print("=" * 70)

    model.eval()

    # Generate a test image
    test_image, test_target = generate_synthetic_sample(IMAGE_SIZE, NUM_CLASSES)
    test_image = test_image.unsqueeze(0).to(device)  # [1, 3, H, W]

    # Run prediction
    results = model.predict(test_image, score_threshold=0.2)

    print(f"\nGround truth ({len(test_target['labels'])} objects):")
    for i in range(len(test_target['labels'])):
        cls = test_target['labels'][i].item()
        box = test_target['boxes'][i].tolist()
        print(f"  {CLASS_NAMES[cls]:10s} | box: ({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f})")

    print(f"\nPredictions ({len(results[0]['scores'])} detections):")
    if len(results[0]['scores']) == 0:
        print("  No detections above threshold.")
        print("  (This is normal for a small demo — try more epochs or lower threshold)")
    else:
        for i in range(len(results[0]['scores'])):
            score = results[0]['scores'][i].item()
            cls = results[0]['labels'][i].item()
            box = results[0]['boxes'][i].tolist()
            name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"
            print(f"  {name:10s} | score: {score:.3f} | box: ({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f})")

    # =================================================================
    # Architecture Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("ARCHITECTURE SUMMARY")
    print("=" * 70)
    print(f"  Backbone:          SimpleBackbone (3 stages, channels={config.backbone_channels})")
    print(f"  Feature levels:    {config.num_feature_levels}")
    print(f"  Hidden dim:        {config.hidden_dim}")
    print(f"  Encoder layers:    {config.enc_layers} (standard multi-head attention)")
    print(f"  Decoder layers:    {config.dec_layers} (self-attn + cross-attn + FFN)")
    print(f"  Object queries:    {config.num_queries}")
    print(f"  Denoising groups:  {config.dn_number}")
    print(f"  Classes:           {config.num_classes}")
    print(f"  Parameters:        {num_params:,}")
    print()
    print("Key DINO innovations demonstrated:")
    print("  ✓ Multi-scale features (2 levels)")
    print("  ✓ Anchor-based queries (learned reference points)")
    print("  ✓ Iterative box refinement across decoder layers")
    print("  ✓ Denoising training (noisy GT → reconstruction)")
    print("  ✓ Auxiliary losses at each decoder layer")
    print("  ✓ Hungarian matching (bipartite, no NMS)")
    print()
    print("Simplifications vs full DINO:")
    print("  ✗ Standard attention instead of deformable attention")
    print("  ✗ Small CNN backbone instead of ResNet/Swin")
    print("  ✗ Learned positional encoding instead of sinusoidal")
    print("  ✗ 2 feature scales instead of 4")
    print("  ✗ Simplified denoising attention masking")


if __name__ == "__main__":
    main()

"""Tiny training/inference demo for the minimal Faster R-CNN model.

This script uses synthetic random data so you can verify that:
1) forward + loss path works,
2) backward pass runs,
3) inference returns detections.

Replace the synthetic dataset with a real dataset once the flow is understood.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor

from .model import MinimalFasterRCNN


def make_synthetic_batch(
    batch_size: int,
    image_size: Tuple[int, int],
    num_classes: int,
    max_boxes_per_image: int = 5,
    device: str = "cpu",
) -> Tuple[List[Tensor], List[Dict[str, Tensor]]]:
    """Create random images and random GT boxes for smoke testing.

    Boxes are generated as valid xyxy rectangles with minimum side length.
    Labels are in [1, num_classes-1], where 0 is reserved for background.
    """
    h, w = image_size
    images: List[Tensor] = []
    targets: List[Dict[str, Tensor]] = []

    for _ in range(batch_size):
        img = torch.rand(3, h, w, device=device)
        images.append(img)

        n = int(torch.randint(1, max_boxes_per_image + 1, size=(1,)).item())

        x1 = torch.rand(n, device=device) * (w * 0.7)
        y1 = torch.rand(n, device=device) * (h * 0.7)

        # Make widths/heights strictly positive and not too tiny.
        bw = torch.rand(n, device=device) * (w * 0.25) + 10.0
        bh = torch.rand(n, device=device) * (h * 0.25) + 10.0

        x2 = (x1 + bw).clamp(max=w - 1)
        y2 = (y1 + bh).clamp(max=h - 1)

        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        labels = torch.randint(1, num_classes, size=(n,), device=device)

        targets.append({"boxes": boxes, "labels": labels})

    return images, targets


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = 4  # class ids: 0=background, 1..3=foreground classes
    image_size = (512, 512)

    model = MinimalFasterRCNN(num_classes=num_classes, image_size=image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for step in range(3):
        images, targets = make_synthetic_batch(
            batch_size=2,
            image_size=image_size,
            num_classes=num_classes,
            device=device,
        )

        losses = model(images, targets)
        total_loss = sum(losses.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        printable = {k: float(v.detach().cpu()) for k, v in losses.items()}
        print(f"step={step} losses={printable}")

    model.eval()
    with torch.no_grad():
        images, _ = make_synthetic_batch(
            batch_size=2,
            image_size=image_size,
            num_classes=num_classes,
            device=device,
        )
        detections = model(images)

    for i, det in enumerate(detections):
        print(
            f"image={i} detections={det['boxes'].shape[0]} "
            f"top_score={float(det['scores'].max().cpu()) if det['scores'].numel() else 0.0:.4f}"
        )


if __name__ == "__main__":
    main()

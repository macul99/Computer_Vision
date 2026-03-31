"""Minimal YOLOv1 architecture implemented from scratch in PyTorch.

==========================================================================
HIGH-LEVEL DESIGN (YOLOv1)
==========================================================================

YOLOv1 treats object detection as a single regression problem:
  Image pixels  →  CNN  →  S×S×(B*5 + C) tensor  →  Bounding boxes + classes

Key parameters:
  S = grid size (e.g., 7) — the image is divided into an S×S grid
  B = number of bounding boxes predicted per grid cell (e.g., 2)
  C = number of object classes (e.g., 20 for VOC)

Each grid cell predicts:
  - B bounding boxes, each with 5 values: (x, y, w, h, confidence)
      x, y  = center of box relative to the grid cell (0 to 1)
      w, h  = width and height relative to the full image (0 to 1)
      confidence = P(Object) × IoU(pred, truth)
  - C class probabilities: P(Class_i | Object)

So the final output tensor has shape: [batch, S, S, B*5 + C]

At inference time, the class-specific confidence for each box is:
  P(Class_i) × IoU = confidence × P(Class_i | Object)

==========================================================================
COMPARISON WITH FASTER R-CNN
==========================================================================

In Faster R-CNN (see ../faster_rcnn_min/):
  1. Backbone extracts features
  2. RPN proposes ~300 candidate regions
  3. ROI head classifies each region separately

In YOLO:
  1. Backbone extracts features
  2. A single dense layer directly predicts ALL boxes and classes
  → No region proposal step! This is why YOLO is faster.
  → But each grid cell can only predict B boxes with 1 class,
    limiting detection of small/dense objects.

==========================================================================
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TinyYOLOBackbone(nn.Module):
    """A small CNN backbone for feature extraction.

    Architecture overview:
    ┌──────────────┐
    │  Input Image  │  [B, 3, 448, 448]
    └──────┬───────┘
           ↓  Conv 3→64, stride 2, padding 1   (224×224)
           ↓  Conv 64→64, stride 1, padding 1
           ↓  MaxPool 2×2                       (112×112)
           ↓  Conv 64→128, stride 1, padding 1
           ↓  Conv 128→128, stride 1, padding 1
           ↓  MaxPool 2×2                       (56×56)
           ↓  Conv 128→256, stride 1, padding 1
           ↓  Conv 256→256, stride 1, padding 1
           ↓  MaxPool 2×2                       (28×28)
           ↓  Conv 256→512, stride 1, padding 1
           ↓  Conv 512→512, stride 1, padding 1
           ↓  MaxPool 2×2                       (14×14)
           ↓  Conv 512→512, stride 1, padding 1
           ↓  Conv 512→512, stride 1, padding 1
           ↓  MaxPool 2×2                       (7×7)
    ┌──────┴───────┐
    │ Feature Map   │  [B, 512, 7, 7]
    └──────────────┘

    The original YOLOv1 used a much larger backbone (24 conv layers inspired by GoogLeNet).
    This simplified version uses a VGG-style stack of 3×3 convolutions with max pooling,
    achieving the same 7×7 spatial output from a 448×448 input (total stride = 64).

    Why stride 64?
        448 / 64 = 7, which gives us our S=7 grid directly from the backbone!
        Each spatial location in the 7×7 feature map corresponds to a 64×64 region
        in the original image (its "receptive field").
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Each block: convolutions + optional maxpool to downsample by 2
        self.features = nn.Sequential(
            # --- Block 1: Downsample 448→224 via stride-2 conv, then 224→112 via maxpool ---
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),  # [B,64,224,224]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),           # [B,64,224,224]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),                            # [B,64,112,112]

            # --- Block 2: 112→56 via maxpool ---
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),          # [B,128,112,112]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),         # [B,128,112,112]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),                            # [B,128,56,56]

            # --- Block 3: 56→28 via maxpool ---
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),         # [B,256,56,56]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),         # [B,256,56,56]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),                            # [B,256,28,28]

            # --- Block 4: 28→14 via maxpool ---
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),         # [B,512,28,28]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),         # [B,512,28,28]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),                            # [B,512,14,14]

            # --- Block 5: 14→7 via maxpool ---
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),         # [B,512,14,14]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),         # [B,512,14,14]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),                            # [B,512,7,7]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input images [B, 3, 448, 448]
        Returns:
            Feature map [B, 512, 7, 7]
        """
        return self.features(x)


class YOLOv1(nn.Module):
    """Complete YOLOv1 detection network.

    =======================================================================
    ARCHITECTURE FLOW
    =======================================================================

    Input: [B, 3, 448, 448]          ← RGB image, fixed size
           ↓
    Backbone: TinyYOLOBackbone
           ↓
    Feature map: [B, 512, 7, 7]      ← 512 channels, 7×7 spatial grid
           ↓
    Flatten: [B, 512*7*7] = [B, 25088]
           ↓
    FC layer 1: [B, 4096]            ← with LeakyReLU + Dropout
           ↓
    FC layer 2: [B, S*S*(B*5+C)]     ← final prediction vector
           ↓
    Reshape: [B, S, S, B*5+C]        ← structured output grid

    =======================================================================
    OUTPUT TENSOR LAYOUT
    =======================================================================

    For each of the S×S grid cells, the output vector has (B*5 + C) values:

    With B=2 and C=20:
    Index:  [0    1    2    3    4   |  5    6    7    8    9   | 10 ... 29]
    Meaning:[x1   y1   w1   h1   c1 |  x2   y2   w2   h2   c2 | class_0 ... class_19]
             ↑ Box 1 (5 values)     ↑ Box 2 (5 values)        ↑ Class probs (C values)

    Where:
      x, y = center of predicted box relative to grid cell origin (sigmoid → 0 to 1)
      w, h = width and height relative to image size (sigmoid → 0 to 1)
             (original paper used sqrt(w), sqrt(h) for regression targets)
      c    = objectness confidence = P(object) × IoU(pred, truth)
    """

    def __init__(
        self,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        image_size: int = 448,
    ):
        """
        Args:
            S: Grid size. Image is divided into S×S cells.
               Original YOLOv1 uses S=7.
            B: Number of bounding boxes predicted per cell.
               Original YOLOv1 uses B=2.
            C: Number of object classes.
               VOC dataset has C=20, COCO has C=80.
            image_size: Expected input image size (square).
        """
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.image_size = image_size

        # ── Backbone: extracts spatial features ──
        self.backbone = TinyYOLOBackbone(in_channels=3)

        # ── Detection head: converts features → predictions ──
        # The backbone outputs [B, 512, 7, 7] → flatten to [B, 25088]
        # Then FC layers map to the final prediction vector.
        #
        # Why fully connected layers?
        #   In YOLOv1, the FC layers allow every grid cell's prediction to
        #   depend on the ENTIRE image context (global reasoning).
        #   Later versions (v2+) replaced FC with 1×1 convolutions for efficiency.
        backbone_output_dim = 512 * S * S  # = 512 * 7 * 7 = 25088

        # Total predictions per cell: B boxes × 5 values + C class probs
        self.output_per_cell = B * 5 + C   # e.g., 2*5 + 20 = 30

        self.head = nn.Sequential(
            # FC layer 1: compress feature vector
            nn.Flatten(),                                        # [B, 25088]
            nn.Linear(backbone_output_dim, 4096),                # [B, 4096]
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),  # Regularization — prevents overfitting

            # FC layer 2: output predictions for all grid cells
            nn.Linear(4096, S * S * self.output_per_cell),       # [B, S*S*30]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the full YOLOv1 forward pass.

        Args:
            x: Input images, tensor of shape [batch, 3, 448, 448].

        Returns:
            predictions: Tensor of shape [batch, S, S, B*5 + C].
                For each grid cell:
                  - First B*5 values are B bounding boxes (x, y, w, h, conf each)
                  - Last C values are class probabilities

        Forward pass steps:
            1. Extract features via backbone CNN
            2. Flatten spatial features to a vector
            3. Pass through FC layers to get raw predictions
            4. Reshape into the S×S grid structure
            5. Apply sigmoid to box coordinates and confidence
               (classes left as logits for loss computation)
        """
        # Step 1: Feature extraction
        features = self.backbone(x)        # [batch, 512, 7, 7]

        # Step 2-3: Flatten and predict via FC head
        raw_output = self.head(features)   # [batch, S*S*(B*5+C)]

        # Step 4: Reshape into grid structure
        # Each spatial position (i, j) contains predictions for that grid cell
        output = raw_output.view(-1, self.S, self.S, self.output_per_cell)
        # output shape: [batch, S, S, B*5+C] e.g., [batch, 7, 7, 30]

        # Step 5: Apply activations to appropriate outputs
        output = self._apply_activations(output)

        return output

    def _apply_activations(self, output: Tensor) -> Tensor:
        """Apply sigmoid to box coordinates and confidence scores.

        Why sigmoid?
          - x, y should be in [0, 1] (offset within grid cell)
          - w, h should be in [0, 1] (fraction of image size)
          - confidence should be in [0, 1] (probability × IoU)

        Class probabilities are left as raw logits because:
          - During training, nn.CrossEntropyLoss expects raw logits
          - During inference, we apply softmax explicitly

        Layout reminder (per cell, B=2, C=20):
          [x1, y1, w1, h1, c1, x2, y2, w2, h2, c2, cls0, cls1, ..., cls19]
           0    1   2   3   4   5   6   7   8   9   10         ...    29
        """
        activated = output.clone()

        for b in range(self.B):
            # Box coordinate indices for box b: [5*b, 5*b+1, 5*b+2, 5*b+3]
            # Confidence index for box b: 5*b+4
            base = 5 * b

            # x, y, w, h → sigmoid to constrain to [0, 1]
            activated[..., base:base + 4] = torch.sigmoid(output[..., base:base + 4])

            # confidence → sigmoid to constrain to [0, 1]
            activated[..., base + 4] = torch.sigmoid(output[..., base + 4])

        # Class probabilities (indices B*5 to B*5+C) are left as logits
        # They will be passed to CrossEntropyLoss during training
        # or softmax during inference

        return activated

    def decode_predictions(
        self,
        output: Tensor,
        confidence_threshold: float = 0.5,
    ) -> list[list[dict]]:
        """Convert raw grid output to final bounding box detections.

        This is the POST-PROCESSING step that happens at inference time.

        For each grid cell (i, j) and each box b:
          1. Compute absolute box coordinates in image space
          2. Compute class-specific confidence = box_conf × class_prob
          3. Filter by confidence threshold
          4. Apply Non-Maximum Suppression (NMS) to remove duplicates

        Coordinate conversion (grid-relative → image-absolute):
          The predicted (x, y) are offsets within the grid cell [0, 1].
          To get absolute image coordinates:
            abs_x = (j + x) / S × image_width
            abs_y = (i + y) / S × image_height
          The predicted (w, h) are fractions of image size:
            abs_w = w × image_width
            abs_h = h × image_height

        Args:
            output: Model output [batch, S, S, B*5+C] with activations applied.
            confidence_threshold: Minimum confidence to keep a detection.

        Returns:
            List of detections per image. Each detection is a dict with:
              "box": [x1, y1, x2, y2] in absolute image coordinates
              "confidence": float
              "class_id": int
              "class_name_score": float (class probability)
        """
        batch_size = output.shape[0]
        all_detections = []

        for b_idx in range(batch_size):
            detections = []
            grid = output[b_idx]  # [S, S, B*5+C]

            for i in range(self.S):       # row (y-axis)
                for j in range(self.S):   # col (x-axis)
                    cell = grid[i, j]     # [B*5+C]

                    # Extract class probabilities (shared across all B boxes in this cell)
                    class_logits = cell[self.B * 5:]        # [C]
                    class_probs = F.softmax(class_logits, dim=0)  # [C]
                    best_class_prob, best_class_id = class_probs.max(0)

                    # Check each bounding box
                    for box_idx in range(self.B):
                        base = 5 * box_idx
                        # Extract box predictions
                        x = cell[base + 0]     # center x (relative to cell, 0~1)
                        y = cell[base + 1]     # center y (relative to cell, 0~1)
                        w = cell[base + 2]     # width (relative to image, 0~1)
                        h = cell[base + 3]     # height (relative to image, 0~1)
                        conf = cell[base + 4]  # objectness confidence

                        # Class-specific confidence:
                        #   P(Class_i | Object) × P(Object) × IoU(pred, truth)
                        #   = class_prob × box_confidence
                        class_conf = conf * best_class_prob

                        if class_conf < confidence_threshold:
                            continue

                        # Convert to absolute image coordinates
                        abs_cx = (j + x.item()) / self.S * self.image_size
                        abs_cy = (i + y.item()) / self.S * self.image_size
                        abs_w = w.item() * self.image_size
                        abs_h = h.item() * self.image_size

                        # Convert center format to corner format
                        x1 = abs_cx - abs_w / 2
                        y1 = abs_cy - abs_h / 2
                        x2 = abs_cx + abs_w / 2
                        y2 = abs_cy + abs_h / 2

                        detections.append({
                            "box": [x1, y1, x2, y2],
                            "confidence": class_conf.item(),
                            "class_id": best_class_id.item(),
                            "class_prob": best_class_prob.item(),
                        })

            # Apply NMS per class to remove duplicate detections
            detections = self._nms(detections, iou_threshold=0.5)
            all_detections.append(detections)

        return all_detections

    @staticmethod
    def _nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
        """Non-Maximum Suppression (NMS) — remove redundant overlapping boxes.

        NMS algorithm:
        1. Group detections by class
        2. For each class, sort by confidence (highest first)
        3. Greedily keep the best box; remove all boxes that overlap
           with it above the IoU threshold
        4. Repeat until no boxes remain

        Why NMS is needed:
          Multiple grid cells may detect the same object, producing
          overlapping boxes. NMS keeps only the most confident one.

        Args:
            detections: List of detection dicts with "box", "confidence", "class_id".
            iou_threshold: IoU above which a lower-confidence box is suppressed.

        Returns:
            Filtered list of detections after NMS.
        """
        if not detections:
            return []

        # Group by class
        from collections import defaultdict
        by_class: dict[int, list] = defaultdict(list)
        for det in detections:
            by_class[det["class_id"]].append(det)

        kept = []
        for class_id, class_dets in by_class.items():
            # Sort by confidence, highest first
            class_dets.sort(key=lambda d: d["confidence"], reverse=True)

            while class_dets:
                # Keep the most confident detection
                best = class_dets.pop(0)
                kept.append(best)

                # Remove all remaining detections that overlap too much with best
                remaining = []
                for det in class_dets:
                    iou = YOLOv1._compute_iou_single(best["box"], det["box"])
                    if iou < iou_threshold:
                        remaining.append(det)
                class_dets = remaining

        return kept

    @staticmethod
    def _compute_iou_single(box1: list, box2: list) -> float:
        """Compute IoU between two individual boxes (Python lists).

        Used in NMS post-processing (not in the training loop).

        Args:
            box1, box2: Each is [x1, y1, x2, y2].
        Returns:
            IoU value as float.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter_area

        return inter_area / (union + 1e-6)

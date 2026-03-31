"""Box operation utilities for DINO.

This module provides helper functions for bounding-box manipulation used
throughout the DINO detection pipeline:

  - Format conversion between (cx, cy, w, h) and (x1, y1, x2, y2)
  - Pairwise IoU computation (standard and generalized)

DINO works in **normalized center format** (cx, cy, w, h) where all values
are in [0, 1] relative to the image dimensions. This is the same convention
as the original DETR.

All functions operate on PyTorch tensors and are autograd-compatible.
"""

from __future__ import annotations

import torch
from torch import Tensor


# =========================================================================
# Format Conversions
# =========================================================================

def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).

    DINO internally predicts boxes in center format, but IoU computation
    and GIoU loss require corner format.

    Args:
        boxes: Tensor of shape [..., 4] where the last dimension is
               (cx, cy, w, h).
               cx, cy = center of the box (normalized 0–1)
               w, h   = width and height (normalized 0–1)

    Returns:
        Tensor of same shape with (x1, y1, x2, y2) format.
        x1, y1 = top-left corner
        x2, y2 = bottom-right corner

    Math:
        x1 = cx - w/2       x2 = cx + w/2
        y1 = cy - h/2       y2 = cy + h/2
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """Convert boxes from corner format (x1, y1, x2, y2) to center format (cx, cy, w, h).

    Args:
        boxes: Tensor of shape [..., 4] with (x1, y1, x2, y2).

    Returns:
        Tensor of same shape with (cx, cy, w, h).

    Math:
        cx = (x1 + x2) / 2      w = x2 - x1
        cy = (y1 + y2) / 2      h = y2 - y1
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


# =========================================================================
# IoU Computation
# =========================================================================

def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute pairwise IoU between two sets of boxes (corner format).

    IoU (Intersection over Union) measures overlap between boxes:

        IoU(A, B) = |A ∩ B| / |A ∪ B|
                  = |A ∩ B| / (|A| + |B| - |A ∩ B|)

    Args:
        boxes1: [N, 4] in (x1, y1, x2, y2) format.
        boxes2: [M, 4] in (x1, y1, x2, y2) format.

    Returns:
        iou: [N, M] pairwise IoU matrix. iou[i, j] = IoU between
             boxes1[i] and boxes2[j].
    """
    # Area of each box: width × height
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

    # Intersection coordinates:
    #   The intersection of two axis-aligned boxes is itself a box whose
    #   top-left corner is the element-wise max of the two top-left corners,
    #   and whose bottom-right corner is the element-wise min of the two
    #   bottom-right corners.
    #
    #   boxes1[:, None, :2] has shape [N, 1, 2]  (top-left corners of set 1)
    #   boxes2[None, :, :2] has shape [1, M, 2]  (top-left corners of set 2)
    #   Broadcasting gives [N, M, 2]
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    # Intersection area: clamp to zero if boxes don't overlap
    wh = (rb - lt).clamp(min=0)        # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Union = area1 + area2 - intersection
    union = area1[:, None] + area2[None, :] - inter  # [N, M]

    # IoU with small epsilon to avoid division by zero
    iou = inter / (union + 1e-8)  # [N, M]
    return iou


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute pairwise Generalized IoU (GIoU) between two sets of boxes.

    GIoU extends IoU by adding a penalty for the unused area of the
    smallest enclosing box. This addresses a weakness of IoU: when two
    boxes do not overlap at all, IoU = 0 regardless of how far apart
    they are, providing no gradient signal. GIoU fixes this.

    Formula:
        GIoU(A, B) = IoU(A, B) - |C \\ (A ∪ B)| / |C|

    Where C is the smallest axis-aligned box enclosing both A and B.

    Properties:
        - GIoU ∈ [-1, 1]
        - GIoU = IoU when the enclosing box equals the union
        - GIoU < 0 when boxes are far apart (provides gradient even with no overlap)
        - GIoU = -1 when boxes are infinitely far apart

    DINO uses GIoU for both the Hungarian matching cost and the box
    regression loss. GIoU is preferred over L1 alone because it is
    scale-invariant: a 1-pixel error matters more for a 10×10 box
    than for a 100×100 box.

    Args:
        boxes1: [N, 4] in (x1, y1, x2, y2) format.
        boxes2: [M, 4] in (x1, y1, x2, y2) format.

    Returns:
        giou: [N, M] pairwise GIoU values.
    """
    # Step 1: Compute standard IoU components
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)        # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    union = area1[:, None] + area2[None, :] - inter  # [N, M]

    iou = inter / (union + 1e-8)  # [N, M]

    # Step 2: Compute the smallest enclosing box C
    #   Top-left of C = element-wise min of top-left corners
    #   Bottom-right of C = element-wise max of bottom-right corners
    enclose_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    enclose_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)                        # [N, M, 2]
    enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1]                   # [N, M]

    # Step 3: GIoU = IoU - (|C| - |A ∪ B|) / |C|
    #   The term (|C| - union) / |C| penalizes the wasted space in the
    #   enclosing box that neither A nor B occupies.
    giou = iou - (enclose_area - union) / (enclose_area + 1e-8)  # [N, M]
    return giou

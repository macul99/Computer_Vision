"""Box operation utilities for RTMDet.

==========================================================================
KEY DIFFERENCES FROM YOLO BOX OPS
==========================================================================

RTMDet is an *anchor-free, point-based* detector. Instead of predicting
(x_center, y_center, w, h) like YOLO, RTMDet predicts distances from
each feature-map point to the four sides of the bounding box:

    (left, top, right, bottom) — called "ltrb" or "distance" format

This is more natural for anchor-free detectors because:
  - Each point on the feature map has a known (x, y) position on the image.
  - To describe a box from that point, we just say how far each edge is.
  - No need for anchor priors or center offset tricks.

Conversion:
    Given a point at (px, py) and prediction (l, t, r, b):
        x1 = px - l
        y1 = py - t
        x2 = px + r
        y2 = py + b

This module also provides GIoU (Generalized IoU), which RTMDet uses
as its box regression loss instead of the plain IoU or L1 loss used
in YOLO / Faster R-CNN.
==========================================================================
"""

import torch
from torch import Tensor


# =========================================================================
# Box format conversions
# =========================================================================

def xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2).

    Args:
        boxes: [..., 4] in (cx, cy, w, h) format.

    Returns:
        [..., 4] in (x1, y1, x2, y2) format.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_xywh(boxes: Tensor) -> Tensor:
    """Convert (x1, y1, x2, y2) → (cx, cy, w, h).

    Args:
        boxes: [..., 4] in (x1, y1, x2, y2) format.

    Returns:
        [..., 4] in (cx, cy, w, h) format.
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def distance2bbox(points: Tensor, distances: Tensor) -> Tensor:
    """Convert (point, ltrb-distance) → (x1, y1, x2, y2).

    This is the CORE decoding function for anchor-free detectors like RTMDet.

    Each point on the feature map has a coordinate (px, py) on the image.
    The model predicts distances from that point to each edge of the box:
        l = distance to the LEFT edge
        t = distance to the TOP edge
        r = distance to the RIGHT edge
        b = distance to the BOTTOM edge

    Conversion:
        x1 = px - l
        y1 = py - t
        x2 = px + r
        y2 = py + b

    Visual:
             t
             ↑
        l ←  ● → r      ● = feature-map point at (px, py)
             ↓
             b

        Result box:
        (px-l, py-t) ────────── (px+r, py-t)
             │                       │
             │         ●             │
             │      (px, py)         │
             │                       │
        (px-l, py+b) ────────── (px+r, py+b)

    Args:
        points: [N, 2] — (px, py) coordinates of each feature-map point.
        distances: [N, 4] — (left, top, right, bottom) predicted distances.

    Returns:
        boxes: [N, 4] — (x1, y1, x2, y2) in absolute image coordinates.
    """
    x1 = points[:, 0] - distances[:, 0]  # px - l
    y1 = points[:, 1] - distances[:, 1]  # py - t
    x2 = points[:, 0] + distances[:, 2]  # px + r
    y2 = points[:, 1] + distances[:, 3]  # py + b
    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox2distance(points: Tensor, boxes: Tensor, max_dist: float = -1.0) -> Tensor:
    """Convert (point, xyxy-box) → ltrb distances.

    This is the INVERSE of distance2bbox. Used to compute regression targets:
    given a point and a ground-truth box, compute what distances the model
    should predict.

    Args:
        points: [N, 2] — (px, py) for each point.
        boxes: [N, 4] — (x1, y1, x2, y2) ground-truth boxes (one per point).
        max_dist: If > 0, clamp distances to this maximum value.
                  Points far from the box edges get their distance clamped.

    Returns:
        distances: [N, 4] — (left, top, right, bottom) target distances.
    """
    l = points[:, 0] - boxes[:, 0]  # px - x1
    t = points[:, 1] - boxes[:, 1]  # py - y1
    r = boxes[:, 2] - points[:, 0]  # x2 - px
    b = boxes[:, 3] - points[:, 1]  # y2 - py

    distances = torch.stack([l, t, r, b], dim=-1)

    if max_dist > 0:
        distances = distances.clamp(min=0, max=max_dist)

    return distances


# =========================================================================
# IoU computations
# =========================================================================

def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute pairwise IoU between two sets of xyxy boxes.

    IoU = Intersection / Union

    Args:
        boxes1: [N, 4] in (x1, y1, x2, y2).
        boxes2: [M, 4] in (x1, y1, x2, y2).

    Returns:
        iou: [N, M] pairwise IoU matrix.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

    # Intersection: top-left = max of corners, bottom-right = min of corners
    inter_tl = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    inter_br = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    inter_wh = (inter_br - inter_tl).clamp(min=0)  # [N, M, 2]
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2[None, :] - inter_area  # [N, M]

    return inter_area / (union + 1e-7)


def box_iou_flat(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute element-wise IoU between paired boxes (same count).

    Unlike box_iou which computes N×M pairwise, this computes N element-wise IoUs
    where boxes1[i] is compared only to boxes2[i].

    Args:
        boxes1: [N, 4] in (x1, y1, x2, y2).
        boxes2: [N, 4] in (x1, y1, x2, y2).

    Returns:
        iou: [N] element-wise IoU.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_tl = torch.max(boxes1[:, :2], boxes2[:, :2])
    inter_br = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    inter_wh = (inter_br - inter_tl).clamp(min=0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]

    union = area1 + area2 - inter_area
    return inter_area / (union + 1e-7)


def giou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute element-wise Generalized IoU (GIoU) between paired boxes.

    =======================================================================
    WHY GIoU INSTEAD OF PLAIN IoU?
    =======================================================================

    Plain IoU has a problem as a loss: when two boxes don't overlap at all,
    IoU = 0 everywhere, so the gradient is zero and the model can't learn
    which direction to move the box.

    GIoU fixes this by considering the smallest enclosing box:

        GIoU = IoU - (Area_enclosing - Area_union) / Area_enclosing

    Range: GIoU ∈ [-1, 1]
      - GIoU =  1 when boxes are identical
      - GIoU = -1 when boxes are infinitely far apart
      - GIoU > 0 when boxes overlap
      - GIoU < 0 when boxes don't overlap (but still provides gradient!)

    =======================================================================
    FORMULA
    =======================================================================

    Given two boxes A and B:

    1. Compute IoU(A, B) as usual
    2. Find the smallest enclosing box C that contains both A and B
    3. GIoU = IoU - (|C| - |A ∪ B|) / |C|

    Where:
      |C|     = area of the enclosing box
      |A ∪ B| = area of the union = |A| + |B| - |A ∩ B|

    The penalty term (|C| - |A ∪ B|) / |C| measures how much dead space
    there is inside the enclosing box but outside the union. This penalizes
    boxes that are far apart even when IoU = 0.

    =======================================================================

    Args:
        boxes1: [N, 4] in (x1, y1, x2, y2).
        boxes2: [N, 4] in (x1, y1, x2, y2).

    Returns:
        giou_values: [N] element-wise GIoU, range [-1, 1].
    """
    # Step 1: Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Step 2: Intersection
    inter_tl = torch.max(boxes1[:, :2], boxes2[:, :2])
    inter_br = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    inter_wh = (inter_br - inter_tl).clamp(min=0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]

    # Step 3: Union
    union = area1 + area2 - inter_area

    # Step 4: IoU
    iou = inter_area / (union + 1e-7)

    # Step 5: Smallest enclosing box
    # Top-left = min of both top-lefts, bottom-right = max of both bottom-rights
    enclose_tl = torch.min(boxes1[:, :2], boxes2[:, :2])
    enclose_br = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    enclose_wh = (enclose_br - enclose_tl).clamp(min=0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]

    # Step 6: GIoU = IoU - (enclose_area - union) / enclose_area
    giou_values = iou - (enclose_area - union) / (enclose_area + 1e-7)

    return giou_values

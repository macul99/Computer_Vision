"""Box operation utilities for YOLO.

This module provides helper functions used throughout the YOLO pipeline:
- Converting between box formats (center-based vs corner-based)
- Computing Intersection over Union (IoU)

All functions operate on PyTorch tensors so they can participate in
autograd computation graphs when needed.
"""

import torch
from torch import Tensor


def xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert boxes from center format (x_center, y_center, w, h) to corner format (x1, y1, x2, y2).

    This is needed because IoU computation is easiest in corner format,
    but YOLO predicts in center format.

    Args:
        boxes: Tensor of shape [..., 4] where last dim is (cx, cy, w, h).
               cx, cy = center coordinates
               w, h   = width and height

    Returns:
        Tensor of same shape with (x1, y1, x2, y2) format.
        x1, y1 = top-left corner
        x2, y2 = bottom-right corner

    Math:
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_xywh(boxes: Tensor) -> Tensor:
    """Convert boxes from corner format (x1, y1, x2, y2) to center format (cx, cy, w, h).

    Args:
        boxes: Tensor of shape [..., 4] where last dim is (x1, y1, x2, y2).

    Returns:
        Tensor of same shape with (cx, cy, w, h) format.

    Math:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute pairwise Intersection over Union (IoU) between two sets of boxes.

    Both inputs must be in corner format (x1, y1, x2, y2).

    IoU is the key metric for evaluating how well a predicted box overlaps
    with a ground truth box:

        IoU = Area_of_Intersection / Area_of_Union

    Where:
        Area_of_Union = Area_box1 + Area_box2 - Area_of_Intersection

    IoU ranges from 0 (no overlap) to 1 (perfect overlap).

    Args:
        boxes1: Tensor of shape [N, 4] in (x1, y1, x2, y2) format.
        boxes2: Tensor of shape [M, 4] in (x1, y1, x2, y2) format.

    Returns:
        iou: Tensor of shape [N, M] where iou[i, j] = IoU(boxes1[i], boxes2[j]).
    """
    # Step 1: Compute areas of each box.
    # Area = width * height = (x2 - x1) * (y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

    # Step 2: Compute the intersection rectangle for each pair.
    # The intersection top-left corner is the MAX of the two top-lefts.
    # The intersection bottom-right corner is the MIN of the two bottom-rights.
    # We use broadcasting: boxes1[:, None, :2] has shape [N, 1, 2],
    #                      boxes2[None, :, :2] has shape [1, M, 2].
    inter_top_left = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])      # [N, M, 2]
    inter_bottom_right = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    # Step 3: Clamp to zero — if boxes don't overlap, intersection dims are negative.
    inter_wh = (inter_bottom_right - inter_top_left).clamp(min=0)  # [N, M, 2]
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]            # [N, M]

    # Step 4: Union = Area1 + Area2 - Intersection (inclusion-exclusion principle).
    union = area1[:, None] + area2[None, :] - inter_area  # [N, M]

    # Step 5: IoU. Add small epsilon to avoid division by zero.
    iou = inter_area / (union + 1e-6)  # [N, M]
    return iou


def box_iou_flat(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute element-wise IoU between two aligned sets of boxes.

    Unlike box_iou() which computes all N×M pairs, this computes IoU
    element-wise: iou[i] = IoU(boxes1[i], boxes2[i]).

    This is used in the YOLO loss to compare each predicted box against
    its corresponding ground truth box.

    Args:
        boxes1: Tensor of shape [N, 4] in (x1, y1, x2, y2) format.
        boxes2: Tensor of shape [N, 4] in (x1, y1, x2, y2) format.

    Returns:
        iou: Tensor of shape [N] with element-wise IoU values.
    """
    # Intersection
    inter_top_left = torch.max(boxes1[:, :2], boxes2[:, :2])      # [N, 2]
    inter_bottom_right = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N, 2]
    inter_wh = (inter_bottom_right - inter_top_left).clamp(min=0) # [N, 2]
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]                 # [N]

    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [N]

    # Union and IoU
    union = area1 + area2 - inter_area
    return inter_area / (union + 1e-6)

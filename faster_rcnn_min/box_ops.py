"""Box geometry utilities used by the minimal Faster R-CNN implementation.

The functions in this file are intentionally explicit and heavily commented so
that each transformation can be studied line by line.
"""

from __future__ import annotations

import torch
from torch import Tensor


def box_area(boxes: Tensor) -> Tensor:
    """Compute area for each box in (x1, y1, x2, y2) format.

    Boxes are expected to satisfy x2 >= x1 and y2 >= y1.
    """
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return widths * heights


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise IoU matrix between two sets of boxes.

    Args:
        boxes1: [N, 4] in xyxy format.
        boxes2: [M, 4] in xyxy format.

    Returns:
        iou: [N, M], where iou[i, j] is IoU(boxes1[i], boxes2[j]).
    """
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    # For each pair (i, j), intersection top-left is max of top-left corners.
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    # For each pair (i, j), intersection bottom-right is min of bottom-right corners.
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # negative means no overlap, clamp to 0.
    inter = wh[..., 0] * wh[..., 1]  # [N, M]

    union = area1[:, None] + area2[None, :] - inter
    # Small epsilon protects against division by zero in degenerate cases.
    eps = torch.finfo(union.dtype).eps
    return inter / (union + eps)


def clip_boxes_to_image(boxes: Tensor, height: int, width: int) -> Tensor:
    """Clip boxes so they stay inside image boundaries."""
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=width - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height - 1)
    return boxes


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """Return indices of boxes with both width and height >= min_size."""
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = torch.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def encode_boxes(reference_boxes: Tensor, proposals: Tensor) -> Tensor:
    """Encode reference boxes relative to proposals using Faster R-CNN deltas.

    This computes the target tuple (tx, ty, tw, th) used for box regression:
      tx = (x_gt - x_p) / w_p
      ty = (y_gt - y_p) / h_p
      tw = log(w_gt / w_p)
      th = log(h_gt / h_p)

    Args:
        reference_boxes: ground-truth boxes [N, 4].
        proposals: anchors/proposals [N, 4].
    """
    # Proposal centers and sizes.
    px = (proposals[:, 0] + proposals[:, 2]) * 0.5
    py = (proposals[:, 1] + proposals[:, 3]) * 0.5
    pw = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
    ph = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)

    # Ground-truth centers and sizes.
    gx = (reference_boxes[:, 0] + reference_boxes[:, 2]) * 0.5
    gy = (reference_boxes[:, 1] + reference_boxes[:, 3]) * 0.5
    gw = (reference_boxes[:, 2] - reference_boxes[:, 0]).clamp(min=1e-6)
    gh = (reference_boxes[:, 3] - reference_boxes[:, 1]).clamp(min=1e-6)

    tx = (gx - px) / pw
    ty = (gy - py) / ph
    tw = torch.log(gw / pw)
    th = torch.log(gh / ph)
    return torch.stack((tx, ty, tw, th), dim=1)


def decode_boxes(deltas: Tensor, proposals: Tensor) -> Tensor:
    """Decode predicted deltas back into absolute xyxy boxes.

    Inverse transform of encode_boxes.
    """
    px = (proposals[:, 0] + proposals[:, 2]) * 0.5
    py = (proposals[:, 1] + proposals[:, 3]) * 0.5
    pw = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
    ph = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)

    dx, dy, dw, dh = deltas.unbind(dim=1)

    # Clamp scale deltas to avoid exploding exp() for extreme predictions.
    dw = dw.clamp(max=4.0)
    dh = dh.clamp(max=4.0)

    gx = px + dx * pw
    gy = py + dy * ph
    gw = pw * torch.exp(dw)
    gh = ph * torch.exp(dh)

    x1 = gx - 0.5 * gw
    y1 = gy - 0.5 * gh
    x2 = gx + 0.5 * gw
    y2 = gy + 0.5 * gh
    return torch.stack((x1, y1, x2, y2), dim=1)

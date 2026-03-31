"""YOLOv1 Loss Function — Detailed Implementation.

==========================================================================
YOLO LOSS OVERVIEW
==========================================================================

The YOLOv1 loss is a SINGLE multi-part loss that combines:

  L_total = λ_coord × L_localization
          + L_confidence_obj
          + λ_noobj × L_confidence_noobj
          + L_classification

Where:
  λ_coord = 5.0   (weight for box coordinate errors — we want precise boxes)
  λ_noobj = 0.5   (weight for no-object confidence — most cells have no object,
                    so we downweight to prevent them from dominating the loss)

==========================================================================
LOSS COMPONENTS (from the YOLOv1 paper)
==========================================================================

1. LOCALIZATION LOSS (only for the "responsible" predictor):

   L_loc = Σ_cells Σ_boxes 𝟙_obj [
       (x_pred - x_true)² + (y_pred - y_true)²
       + (√w_pred - √w_true)² + (√h_pred - √h_true)²
   ]

   Why square root of w, h?
     Large boxes can afford bigger absolute errors than small boxes.
     Taking √ compresses the scale, making the loss more sensitive
     to errors in small boxes. Example:
       √0.1 - √0.2 ≈ 0.13  (small box: 13% change causes big loss)
       √0.8 - √0.9 ≈ 0.06  (large box: same 10% change causes small loss)

2. CONFIDENCE LOSS (object present):

   L_conf_obj = Σ_cells Σ_boxes 𝟙_obj (conf_pred - IoU_pred_truth)²

   The target for confidence is the actual IoU between the predicted
   box and the ground truth box. This teaches the network to output
   a confidence that reflects how well the box actually fits.

3. CONFIDENCE LOSS (no object):

   L_conf_noobj = Σ_cells Σ_boxes 𝟙_noobj (conf_pred - 0)²

   For cells without objects, confidence should be 0. This is
   downweighted by λ_noobj because most cells are empty.

4. CLASSIFICATION LOSS (only for cells containing objects):

   L_cls = Σ_cells 𝟙_obj Σ_classes (p_pred(c) - p_true(c))²

   Original paper uses MSE for classification. Modern implementations
   often use cross-entropy instead (we provide both options).

==========================================================================
RESPONSIBLE PREDICTOR (𝟙_obj)
==========================================================================

Each cell predicts B boxes, but only ONE is "responsible" for detecting
the ground truth object in that cell. The responsible predictor is the
one whose current prediction has the HIGHEST IoU with the ground truth.

This is a key design choice: it encourages specialization — different
predictors in the same cell learn to detect different aspect ratios
or sizes of objects.

==========================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .box_ops import xywh_to_xyxy, box_iou_flat


class YOLOv1Loss(nn.Module):
    """Compute the YOLOv1 multi-part loss.

    This loss expects:
      - predictions: [batch, S, S, B*5 + C]  (model output with activations)
      - targets:     [batch, S, S, 5 + C]    (ground truth grid encoding)

    Target encoding (per cell):
      [x, y, w, h, obj_flag, class_0, class_1, ..., class_{C-1}]
      - x, y: center relative to cell (0~1)
      - w, h: size relative to image (0~1)
      - obj_flag: 1.0 if an object center falls in this cell, 0.0 otherwise
      - class_{i}: 1.0 for the true class, 0.0 otherwise (one-hot)
    """

    def __init__(
        self,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    ):
        """
        Args:
            S: Grid size (must match model).
            B: Number of boxes per cell (must match model).
            C: Number of classes (must match model).
            lambda_coord: Weight for localization loss.
                          Set high (5.0) because we want precise boxes.
            lambda_noobj: Weight for no-object confidence loss.
                          Set low (0.5) because most cells are empty —
                          without this, the model would learn to always
                          predict 0 confidence (trivial solution).
        """
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions: Tensor, targets: Tensor) -> dict[str, Tensor]:
        """Compute YOLOv1 loss.

        Args:
            predictions: [batch, S, S, B*5 + C] — model output.
            targets:     [batch, S, S, 5 + C]   — ground truth encoding.

        Returns:
            Dictionary with individual loss components and total loss:
              "loss_coord": localization loss (xy + wh)
              "loss_conf_obj": confidence loss for cells with objects
              "loss_conf_noobj": confidence loss for cells without objects
              "loss_class": classification loss
              "loss_total": weighted sum of all components
        """
        batch_size = predictions.shape[0]

        # =====================================================================
        # Step 1: Extract ground truth components from target tensor
        # =====================================================================
        # Target layout per cell: [x, y, w, h, obj_flag, class_one_hot...]
        gt_boxes = targets[..., :4]       # [batch, S, S, 4] — (x, y, w, h)
        obj_mask = targets[..., 4]        # [batch, S, S]    — 1 if object present
        gt_classes = targets[..., 5:]     # [batch, S, S, C] — one-hot class vector

        # Boolean mask: which cells contain an object?
        has_obj = obj_mask > 0            # [batch, S, S], boolean
        no_obj = ~has_obj                 # [batch, S, S], boolean

        # =====================================================================
        # Step 2: Find the "responsible" predictor for each cell
        # =====================================================================
        # For cells with objects, compute IoU between each predicted box and
        # the ground truth box. The box with highest IoU is "responsible".
        #
        # We need to convert predicted (x,y,w,h) to (x1,y1,x2,y2) for IoU.
        # But x,y are cell-relative, so we first convert to image-relative.

        # Collect IoUs for each of the B predicted boxes
        ious = []  # will be [B] tensors, each of shape [batch, S, S]
        pred_boxes_list = []  # [B] tensors, each [batch, S, S, 4]

        for b in range(self.B):
            base = 5 * b
            # Extract this predictor's box: (x, y, w, h) all in [0, 1]
            pred_box = predictions[..., base:base + 4]  # [batch, S, S, 4]
            pred_boxes_list.append(pred_box)

            # Convert both pred and gt to absolute coordinates for IoU
            pred_abs = self._to_absolute_xyxy(pred_box)  # [batch, S, S, 4]
            gt_abs = self._to_absolute_xyxy(gt_boxes)    # [batch, S, S, 4]

            # Element-wise IoU (flatten for box_iou_flat, then reshape)
            pred_flat = pred_abs.reshape(-1, 4)           # [batch*S*S, 4]
            gt_flat = gt_abs.reshape(-1, 4)               # [batch*S*S, 4]
            iou = box_iou_flat(pred_flat, gt_flat)        # [batch*S*S]
            iou = iou.reshape(batch_size, self.S, self.S) # [batch, S, S]
            ious.append(iou)

        # Stack IoUs: [batch, S, S, B] and find the best predictor
        ious_stacked = torch.stack(ious, dim=-1)         # [batch, S, S, B]
        best_box_idx = ious_stacked.argmax(dim=-1)       # [batch, S, S] — index of responsible box

        # =====================================================================
        # Step 3: Gather the responsible predictor's outputs
        # =====================================================================
        # We need to select the box (x,y,w,h,conf) from the responsible predictor.
        # best_box_idx tells us which of the B boxes to use at each cell.

        # Gather predicted box coordinates from the responsible predictor
        pred_boxes_stacked = torch.stack(pred_boxes_list, dim=-2)  # [batch, S, S, B, 4]
        idx_expanded = best_box_idx.unsqueeze(-1).unsqueeze(-1)    # [batch, S, S, 1, 1]
        idx_expanded = idx_expanded.expand(-1, -1, -1, -1, 4)     # [batch, S, S, 1, 4]
        resp_pred_box = pred_boxes_stacked.gather(dim=3, index=idx_expanded).squeeze(3)
        # resp_pred_box: [batch, S, S, 4] — the responsible predictor's (x, y, w, h)

        # Gather confidence from the responsible predictor
        pred_confs = []
        for b in range(self.B):
            pred_confs.append(predictions[..., 5 * b + 4])  # [batch, S, S]
        pred_confs_stacked = torch.stack(pred_confs, dim=-1)  # [batch, S, S, B]
        resp_conf = pred_confs_stacked.gather(
            dim=-1, index=best_box_idx.unsqueeze(-1)
        ).squeeze(-1)  # [batch, S, S]

        # Gather IoU for the responsible predictor (used as confidence target)
        resp_iou = ious_stacked.gather(
            dim=-1, index=best_box_idx.unsqueeze(-1)
        ).squeeze(-1)  # [batch, S, S]

        # =====================================================================
        # Step 4: Compute LOCALIZATION LOSS (only for cells with objects)
        # =====================================================================
        # L_loc = Σ [(x_pred - x_gt)² + (y_pred - y_gt)²
        #           + (√w_pred - √w_gt)² + (√h_pred - √h_gt)²]
        #
        # Using square root of w, h to be more sensitive to small box errors.
        # We add a small epsilon before sqrt to avoid NaN gradients at 0.

        # Only compute for cells where obj_mask == 1
        obj_mask_bool = has_obj  # [batch, S, S]

        if obj_mask_bool.any():
            # XY loss: squared error on center coordinates
            xy_loss = F.mse_loss(
                resp_pred_box[obj_mask_bool][..., :2],  # predicted (x, y)
                gt_boxes[obj_mask_bool][..., :2],       # ground truth (x, y)
                reduction="sum",
            )

            # WH loss: squared error on √w, √h
            # Adding epsilon to avoid sqrt(0) gradient issues
            pred_wh = resp_pred_box[obj_mask_bool][..., 2:4]
            gt_wh = gt_boxes[obj_mask_bool][..., 2:4]
            wh_loss = F.mse_loss(
                torch.sqrt(pred_wh + 1e-6),
                torch.sqrt(gt_wh + 1e-6),
                reduction="sum",
            )

            loss_coord = xy_loss + wh_loss
        else:
            loss_coord = torch.tensor(0.0, device=predictions.device)

        # =====================================================================
        # Step 5: Compute CONFIDENCE LOSS (object cells)
        # =====================================================================
        # Target confidence = IoU(predicted_box, ground_truth_box)
        # This teaches the network: "your confidence should reflect
        # how well your box actually matches the ground truth."

        if obj_mask_bool.any():
            loss_conf_obj = F.mse_loss(
                resp_conf[obj_mask_bool],
                resp_iou[obj_mask_bool].detach(),  # detach: IoU is a target, not a gradient source
                reduction="sum",
            )
        else:
            loss_conf_obj = torch.tensor(0.0, device=predictions.device)

        # =====================================================================
        # Step 6: Compute CONFIDENCE LOSS (no-object cells)
        # =====================================================================
        # For ALL B predictors in cells without objects, confidence target = 0.
        # Also for non-responsible predictors in object cells.

        loss_conf_noobj = torch.tensor(0.0, device=predictions.device)
        for b in range(self.B):
            pred_conf_b = predictions[..., 5 * b + 4]  # [batch, S, S]

            # Create mask for non-responsible predictions:
            # In no-object cells: ALL predictors contribute to noobj loss
            # In object cells: only non-responsible predictors contribute
            is_responsible = (best_box_idx == b) & has_obj  # [batch, S, S]
            noobj_mask = ~is_responsible  # Everything that is NOT the responsible predictor in an obj cell

            loss_conf_noobj = loss_conf_noobj + F.mse_loss(
                pred_conf_b[noobj_mask],
                torch.zeros_like(pred_conf_b[noobj_mask]),
                reduction="sum",
            )

        # =====================================================================
        # Step 7: Compute CLASSIFICATION LOSS (only for cells with objects)
        # =====================================================================
        # Using cross-entropy loss (more modern than the paper's MSE).
        # Class probabilities are shared across all B boxes in a cell.

        pred_class_logits = predictions[..., self.B * 5:]  # [batch, S, S, C]

        if obj_mask_bool.any():
            # Get class logits and targets for object cells only
            obj_logits = pred_class_logits[obj_mask_bool]   # [N_obj, C]
            obj_gt_classes = gt_classes[obj_mask_bool]       # [N_obj, C] one-hot

            # Convert one-hot to class indices for cross-entropy
            gt_class_ids = obj_gt_classes.argmax(dim=-1)     # [N_obj]

            loss_class = F.cross_entropy(obj_logits, gt_class_ids, reduction="sum")
        else:
            loss_class = torch.tensor(0.0, device=predictions.device)

        # =====================================================================
        # Step 8: Combine all losses with weights
        # =====================================================================
        # Normalize by batch size for stable gradients across different batch sizes
        N = batch_size

        loss_total = (
            self.lambda_coord * loss_coord / N       # Weight=5.0: precise boxes matter
            + loss_conf_obj / N                       # Weight=1.0: object confidence
            + self.lambda_noobj * loss_conf_noobj / N # Weight=0.5: downweight empty cells
            + loss_class / N                          # Weight=1.0: correct classification
        )

        return {
            "loss_coord": loss_coord / N,
            "loss_conf_obj": loss_conf_obj / N,
            "loss_conf_noobj": loss_conf_noobj / N,
            "loss_class": loss_class / N,
            "loss_total": loss_total,
        }

    def _to_absolute_xyxy(self, boxes: Tensor) -> Tensor:
        """Convert cell-relative (x, y, w, h) to absolute (x1, y1, x2, y2).

        The predicted x, y are offsets within a grid cell (range 0~1).
        To get image-absolute coordinates:
          abs_cx = (col + x) / S    (normalized to image width)
          abs_cy = (row + y) / S    (normalized to image height)
          abs_w  = w                (already normalized to image size)
          abs_h  = h                (already normalized to image size)

        Then convert from center format to corner format.

        Args:
            boxes: [batch, S, S, 4] with (x, y, w, h) cell-relative coordinates.

        Returns:
            [batch, S, S, 4] with (x1, y1, x2, y2) in normalized image coordinates.
        """
        device = boxes.device
        batch_size = boxes.shape[0]

        # Create grid of cell indices (like a meshgrid)
        # col_offset[i, j] = j, row_offset[i, j] = i
        col_offset = torch.arange(self.S, device=device).float().unsqueeze(0).expand(self.S, -1)
        row_offset = torch.arange(self.S, device=device).float().unsqueeze(1).expand(-1, self.S)
        # Both: [S, S], broadcast to [batch, S, S]

        x = boxes[..., 0]  # [batch, S, S]
        y = boxes[..., 1]
        w = boxes[..., 2]
        h = boxes[..., 3]

        # Convert cell-relative center to image-normalized center
        abs_cx = (col_offset + x) / self.S  # [batch, S, S]
        abs_cy = (row_offset + y) / self.S

        # Convert to corners (x1, y1, x2, y2)
        x1 = abs_cx - w / 2
        y1 = abs_cy - h / 2
        x2 = abs_cx + w / 2
        y2 = abs_cy + h / 2

        return torch.stack([x1, y1, x2, y2], dim=-1)  # [batch, S, S, 4]

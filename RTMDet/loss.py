"""RTMDet Loss Functions — Dynamic Label Assignment + Quality Focal Loss + GIoU.

==========================================================================
RTMDet LOSS OVERVIEW
==========================================================================

RTMDet's loss has three major components:

  L_total = L_classification + λ_reg × L_box_regression

Where:
  - L_classification uses Quality Focal Loss (QFL)
  - L_box_regression uses GIoU loss
  - λ_reg = 2.0 typically

But BEFORE computing these losses, RTMDet must decide:
  "Which ground-truth box should each feature-map point be responsible for?"

This is the LABEL ASSIGNMENT problem, and RTMDet uses SimOTA
(Simplified Optimal Transport Assignment) — a DYNAMIC assignment
strategy that adapts during training.

==========================================================================
COMPARISON WITH YOLO ASSIGNMENT
==========================================================================

YOLOv1 assignment:
  - Static: the grid cell containing the object center "owns" that object
  - Simple but rigid: only one cell can learn each object
  - Doesn't consider how well the model currently predicts

RTMDet / SimOTA assignment:
  - Dynamic: considers the model's current predictions
  - Multiple points can be assigned to the same ground-truth box
  - Points that already predict well are more likely to be assigned
  - This creates a positive feedback loop: good predictions get reinforced

==========================================================================
QUALITY FOCAL LOSS (QFL)
==========================================================================

Standard Focal Loss:
  FL(p) = -α (1-p)^γ log(p)    for positive samples
  FL(p) = -(1-α) p^γ log(1-p)  for negative samples

The target is binary: 0 or 1.

Quality Focal Loss (QFL) generalizes this:
  - Instead of hard 0/1 targets, use IoU quality as the target
  - If a point's predicted box has IoU=0.8 with the ground truth,
    the classification target becomes 0.8 (not 1.0)
  - This teaches the model: "your confidence should match your box quality"

  QFL(σ) = -|y - σ|^β × [y × log(σ) + (1-y) × log(1-σ)]

Where:
  σ = predicted probability (sigmoid of logit)
  y = quality target (IoU for positives, 0 for negatives)
  β = focusing parameter (typically 2.0)

This is important because it unifies classification and localization quality
into a single score, making post-processing more reliable.

==========================================================================
GIoU LOSS
==========================================================================

L_giou = 1 - GIoU(predicted_box, gt_box)

Range: [0, 2]
  - L_giou = 0 when boxes are identical (GIoU = 1)
  - L_giou = 2 when boxes are infinitely far apart (GIoU = -1)

GIoU is defined in box_ops.py. The key advantage over L1 loss:
  - Scale-invariant: treats small and large boxes equally
  - Always provides gradient even when boxes don't overlap

==========================================================================
SimOTA (Simplified Optimal Transport Assignment)
==========================================================================

SimOTA decides which feature-map points are positive (assigned to a GT box)
and which are negative (background). Steps:

1. CANDIDATE SELECTION:
   For each GT box, find points inside or near the box (center region).

2. COST MATRIX:
   For candidate points, compute assignment cost based on:
     cost = L_cls + λ × L_reg
   Lower cost = more suitable assignment.

3. DYNAMIC k SELECTION:
   For each GT box, determine how many positive points to assign.
   This is based on the sum of IoUs between the top candidates and
   the GT box. More IoU = more positives.

4. TOP-k SELECTION:
   For each GT box, select the k lowest-cost candidates as positives.

5. CONFLICT RESOLUTION:
   If a point is assigned to multiple GT boxes, keep the assignment
   with the lower cost.

This is "simplified" OT because full Optimal Transport (used in OTA)
requires solving a Sinkhorn iteration, which is slower.

==========================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple

from .box_ops import box_iou, giou, distance2bbox, bbox2distance


class QualityFocalLoss(nn.Module):
    """Quality Focal Loss for classification.

    QFL generalizes binary cross-entropy with soft targets:

      QFL(σ, y) = -|y - σ|^β × BCE(σ, y)

    Where:
      - σ = sigmoid(logit) = predicted probability
      - y = quality target (IoU for positives, 0 for negatives)
      - β = focusing parameter (default 2.0)

    The |y - σ|^β term is the "focusing" factor:
      - When the prediction is close to the target (|y - σ| small),
        the loss is down-weighted → easy examples contribute less
      - When the prediction is far from the target (|y - σ| large),
        the loss has full weight → hard examples contribute more

    This is the same motivation as Focal Loss, but with continuous
    targets instead of binary ones.
    """

    def __init__(self, beta: float = 2.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred_logits: Tensor, targets: Tensor) -> Tensor:
        """Compute Quality Focal Loss.

        Args:
            pred_logits: [N, C] raw logits (before sigmoid).
            targets: [N, C] soft targets (IoU values for positives, 0 for negatives).

        Returns:
            Scalar loss (mean-reduced).
        """
        pred_sigmoid = pred_logits.sigmoid()

        # Binary cross-entropy (computed manually for numerical stability)
        # BCE(σ, y) = -[y × log(σ) + (1-y) × log(1-σ)]
        # Using logsigmoid for numerical stability:
        #   log(σ) = log_sigmoid(logit)
        #   log(1-σ) = log_sigmoid(-logit)
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, targets, reduction="none"
        )

        # Focusing factor: |y - σ|^β
        # This modulates each element's contribution to the loss
        scale = (targets - pred_sigmoid).abs().pow(self.beta)

        loss = scale * bce

        return loss.sum() / max(1, (targets > 0).sum().item())


class GIoULoss(nn.Module):
    """GIoU-based regression loss.

    L_giou = 1 - GIoU(pred, target)

    GIoU ranges from [-1, 1], so this loss ranges from [0, 2].
    See box_ops.py for the full GIoU formula.
    """

    def forward(self, pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
        """Compute GIoU loss between predicted and target boxes.

        Args:
            pred_boxes: [N, 4] in xyxy format.
            target_boxes: [N, 4] in xyxy format.

        Returns:
            Scalar loss (mean-reduced).
        """
        if pred_boxes.numel() == 0:
            return pred_boxes.sum() * 0.0  # Return 0 loss with gradient

        giou_values = giou(pred_boxes, target_boxes)  # [N], range [-1, 1]
        loss = 1.0 - giou_values  # [N], range [0, 2]
        return loss.mean()


class SimOTAAssigner:
    """SimOTA dynamic label assigner.

    =======================================================================
    STEP-BY-STEP ALGORITHM
    =======================================================================

    Given:
      - N feature-map points with their predictions (cls scores, boxes)
      - M ground-truth boxes with their class labels

    Step 1: CANDIDATE SELECTION
      For each GT box, find points that are inside the box OR inside
      a fixed-size center region around the box center.
      This limits the candidates to points near the GT.

    Step 2: COST COMPUTATION
      For each (candidate point, GT box) pair:
        cost = cls_cost + λ × reg_cost
      Where:
        cls_cost = BCE between predicted class score and 1.0 for the GT class
        reg_cost = -log(IoU between predicted box and GT box)

    Step 3: DYNAMIC k SELECTION
      For each GT box j:
        - Compute IoU between all candidate points and GT box j
        - Take the top-k IoUs (k is a hyperparameter, e.g., 13)
        - Sum them: k_j = clamp(sum(top_k_ious), min=1)
        - Round to integer: this is the number of positives for GT j

    Step 4: SELECT LOWEST-COST CANDIDATES
      For each GT box j:
        - From its candidates, select the k_j with lowest cost
        - These become the positive samples for GT j

    Step 5: CONFLICT RESOLUTION
      If a point is positive for multiple GT boxes:
        - Assign it to the GT box with the lowest cost
        - It's negative for all other GT boxes it was assigned to

    =======================================================================
    """

    def __init__(
        self,
        center_radius: float = 2.5,
        candidate_topk: int = 13,
        iou_weight: float = 3.0,
    ):
        """
        Args:
            center_radius: Radius (in stride units) for the center sampling region.
                Points within this radius from GT center are candidates.
            candidate_topk: Max number of IoU values to use for dynamic k.
            iou_weight: Weight for the regression cost in the cost matrix.
        """
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight

    @torch.no_grad()
    def assign(
        self,
        pred_scores: Tensor,
        pred_boxes: Tensor,
        points: Tensor,
        strides: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        num_classes: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform SimOTA assignment.

        Args:
            pred_scores: [N, num_classes] predicted class probs (after sigmoid).
            pred_boxes: [N, 4] predicted boxes in xyxy format.
            points: [N, 2] point coordinates (x, y) in image space.
            strides: [N] stride for each point.
            gt_boxes: [M, 4] ground-truth boxes in xyxy format.
            gt_labels: [M] ground-truth class labels (integer).
            num_classes: Number of classes.

        Returns:
            assigned_labels: [N] class label for each point (num_classes = background).
            assigned_boxes: [N, 4] target box for each point (zeros for background).
            assigned_scores: [N, num_classes] soft classification targets.
        """
        num_points = points.shape[0]
        num_gts = gt_boxes.shape[0]
        device = points.device

        # Default: everything is background
        assigned_labels = torch.full((num_points,), num_classes, dtype=torch.long, device=device)
        assigned_boxes = torch.zeros((num_points, 4), device=device)
        assigned_scores = torch.zeros((num_points, num_classes), device=device)

        if num_gts == 0:
            return assigned_labels, assigned_boxes, assigned_scores

        # =================================================================
        # Step 1: Candidate selection — find points near each GT box
        # =================================================================
        # A point is a candidate for GT box j if it is inside the box
        # OR within center_radius × stride of the box center.

        # Check if points are inside GT boxes
        # For each point (px, py), check if  x1 < px < x2 and y1 < py < y2
        # points[:, None, :] is [N, 1, 2], gt_boxes[None, :, :] is [1, M, 4]
        px = points[:, 0]  # [N]
        py = points[:, 1]  # [N]

        # In-box check: [N, M] boolean
        in_box = (
            (px[:, None] > gt_boxes[None, :, 0]) &  # px > x1
            (py[:, None] > gt_boxes[None, :, 1]) &  # py > y1
            (px[:, None] < gt_boxes[None, :, 2]) &  # px < x2
            (py[:, None] < gt_boxes[None, :, 3])     # py < y2
        )

        # Center region check
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2  # [M]
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2  # [M]
        # Center region: a square around GT center with side = 2 * radius * stride
        center_radius_pixels = self.center_radius * strides[:, None]  # [N, 1]
        in_center = (
            (px[:, None] > gt_cx[None, :] - center_radius_pixels) &
            (py[:, None] > gt_cy[None, :] - center_radius_pixels) &
            (px[:, None] < gt_cx[None, :] + center_radius_pixels) &
            (py[:, None] < gt_cy[None, :] + center_radius_pixels)
        )  # [N, M]

        # A point is a candidate if it's in the box OR in the center region
        is_candidate = in_box | in_center  # [N, M]

        # Points that are candidate for at least one GT box
        candidate_mask = is_candidate.any(dim=1)  # [N]

        if not candidate_mask.any():
            return assigned_labels, assigned_boxes, assigned_scores

        # =================================================================
        # Step 2: Compute cost matrix for candidates
        # =================================================================
        candidate_idxs = candidate_mask.nonzero(as_tuple=False).squeeze(1)  # [K]
        num_candidates = candidate_idxs.shape[0]

        # Get predictions for candidate points only
        cand_scores = pred_scores[candidate_idxs]  # [K, num_classes]
        cand_boxes = pred_boxes[candidate_idxs]     # [K, 4]

        # IoU between candidate predictions and GT boxes: [K, M]
        pairwise_iou = box_iou(cand_boxes, gt_boxes)  # [K, M]

        # Classification cost: BCE(pred_score_for_gt_class, 1.0) for each pair
        # We want the cost of assigning candidate i to GT j
        # Target = 1.0 for the GT class, computed via BCE
        gt_onehot = F.one_hot(gt_labels, num_classes).float()  # [M, num_classes]
        # Expand for pairwise: [K, M, num_classes]
        cls_target = gt_onehot[None, :, :].expand(num_candidates, -1, -1)
        cls_pred = cand_scores[:, None, :].expand(-1, num_gts, -1)

        # BCE cost (lower = better match)
        cls_cost = F.binary_cross_entropy(
            cls_pred.clamp(1e-6, 1 - 1e-6),
            cls_target,
            reduction="none",
        ).sum(dim=-1)  # [K, M]

        # Regression cost: -log(IoU) — high IoU = low cost
        reg_cost = -torch.log(pairwise_iou + 1e-7)  # [K, M]

        # Total cost
        cost = cls_cost + self.iou_weight * reg_cost  # [K, M]

        # Mask out non-candidate pairs (set to very high cost)
        candidate_is_valid = is_candidate[candidate_idxs]  # [K, M]
        cost[~candidate_is_valid] = 1e8

        # =================================================================
        # Step 3: Dynamic k selection
        # =================================================================
        # For each GT, determine how many positive points to assign
        # Based on the sum of top-k IoUs (more overlap = more positives)
        topk = min(self.candidate_topk, num_candidates)
        topk_ious, _ = pairwise_iou.topk(topk, dim=0)  # [topk, M]
        dynamic_ks = topk_ious.sum(dim=0).int().clamp(min=1)  # [M]

        # =================================================================
        # Step 4: Select lowest-cost candidates for each GT
        # =================================================================
        # matching_matrix[i, j] = 1 if candidate i is assigned to GT j
        matching_matrix = torch.zeros_like(cost, dtype=torch.bool)  # [K, M]

        for gt_idx in range(num_gts):
            k = dynamic_ks[gt_idx].item()
            k = min(k, candidate_is_valid[:, gt_idx].sum().item())
            if k == 0:
                continue

            # Get valid candidates for this GT
            valid_mask = candidate_is_valid[:, gt_idx]
            gt_cost = cost[:, gt_idx].clone()
            gt_cost[~valid_mask] = 1e8

            # Select top-k lowest cost candidates
            _, topk_indices = gt_cost.topk(k, largest=False)
            matching_matrix[topk_indices, gt_idx] = True

        # =================================================================
        # Step 5: Conflict resolution
        # =================================================================
        # If a candidate is matched to multiple GTs, keep lowest cost
        multi_match = matching_matrix.sum(dim=1) > 1  # [K]
        if multi_match.any():
            # For multi-matched candidates, keep only the lowest-cost GT
            multi_cost = cost[multi_match]  # [num_multi, M]
            multi_cost[~matching_matrix[multi_match]] = 1e8
            best_gt = multi_cost.argmin(dim=1)  # [num_multi]
            # Reset and assign only to best GT
            matching_matrix[multi_match] = False
            matching_matrix[multi_match, best_gt] = True

        # =================================================================
        # Convert matching matrix to assignments
        # =================================================================
        matched_gts = matching_matrix.any(dim=1)  # [K] — which candidates are positive
        gt_indices = matching_matrix.float().argmax(dim=1)  # [K] — assigned GT for each candidate

        # Fill in assignments for positive candidates
        pos_candidate_idxs = candidate_idxs[matched_gts]
        pos_gt_idxs = gt_indices[matched_gts].long()

        assigned_labels[pos_candidate_idxs] = gt_labels[pos_gt_idxs]
        assigned_boxes[pos_candidate_idxs] = gt_boxes[pos_gt_idxs]

        # Soft target = IoU between prediction and assigned GT (quality-aware)
        pos_ious = pairwise_iou[matched_gts, pos_gt_idxs]  # [num_pos]
        for i, (cand_idx, gt_idx, iou_val) in enumerate(
            zip(pos_candidate_idxs, pos_gt_idxs, pos_ious)
        ):
            assigned_scores[cand_idx, gt_labels[gt_idx]] = iou_val

        return assigned_labels, assigned_boxes, assigned_scores


class RTMDetLoss(nn.Module):
    """Complete RTMDet loss: SimOTA assignment + QFL + GIoU.

    =======================================================================
    LOSS COMPUTATION FLOW
    =======================================================================

    1. Flatten multi-scale predictions:
       cls_scores: [[B,C,40,40], [B,C,20,20], [B,C,10,10]]
       → [B, 2100, C]  (40*40 + 20*20 + 10*10 = 2100 total points)

    2. Generate grid points for all levels

    3. For each image in the batch:
       a. Decode predicted boxes (point + ltrb → xyxy)
       b. Run SimOTA to assign GT boxes to points
       c. Compute classification loss (QFL) on ALL points
       d. Compute regression loss (GIoU) on POSITIVE points only

    4. Return total loss = L_cls + λ_reg × L_reg

    =======================================================================
    """

    def __init__(
        self,
        num_classes: int = 20,
        strides: Tuple[int, ...] = (8, 16, 32),
        reg_weight: float = 2.0,
    ):
        """
        Args:
            num_classes: Number of object classes.
            strides: Feature map strides (must match model).
            reg_weight: Weight for the GIoU regression loss.
        """
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.reg_weight = reg_weight

        self.cls_loss_fn = QualityFocalLoss(beta=2.0)
        self.reg_loss_fn = GIoULoss()
        self.assigner = SimOTAAssigner()

    def forward(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        gt_boxes_list: List[Tensor],
        gt_labels_list: List[Tensor],
    ) -> dict[str, Tensor]:
        """Compute RTMDet loss.

        Args:
            cls_scores: List of [B, C, H_l, W_l] per level.
            bbox_preds: List of [B, 4, H_l, W_l] per level.
            gt_boxes_list: List of [M_i, 4] GT boxes per image (xyxy).
            gt_labels_list: List of [M_i] GT class labels per image.

        Returns:
            Dict with:
              "loss_cls": classification loss (QFL)
              "loss_reg": box regression loss (GIoU)
              "loss_total": weighted sum
        """
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]

        # ── Flatten multi-scale predictions ──
        # Convert from per-level [B, C, H, W] to [B, total_points, C]
        all_cls_scores = []  # [B, N, num_classes]
        all_bbox_preds = []  # [B, N, 4]
        all_points = []      # [N, 2]
        all_strides = []     # [N]

        for lvl, (cls_score, bbox_pred) in enumerate(zip(cls_scores, bbox_preds)):
            B, C, H, W = cls_score.shape
            stride = self.strides[lvl]

            # Reshape: [B, C, H, W] → [B, H*W, C]
            cls_flat = cls_score.permute(0, 2, 3, 1).reshape(B, H * W, C)
            reg_flat = bbox_pred.permute(0, 2, 3, 1).reshape(B, H * W, 4)

            # Scale reg predictions by stride to get image-pixel distances
            reg_flat = reg_flat * stride

            all_cls_scores.append(cls_flat)
            all_bbox_preds.append(reg_flat)

            # Generate point coordinates for this level
            shift_x = (torch.arange(W, device=device) + 0.5) * stride
            shift_y = (torch.arange(H, device=device) + 0.5) * stride
            grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
            all_points.append(points)
            all_strides.append(torch.full((H * W,), stride, device=device))

        # Concatenate across levels
        flat_cls = torch.cat(all_cls_scores, dim=1)    # [B, N, num_classes]
        flat_reg = torch.cat(all_bbox_preds, dim=1)    # [B, N, 4]
        flat_points = torch.cat(all_points, dim=0)      # [N, 2]
        flat_strides = torch.cat(all_strides, dim=0)    # [N]

        num_points = flat_points.shape[0]

        # ── Per-image loss computation ──
        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        num_pos_total = 0

        for img_idx in range(batch_size):
            # Get predictions for this image
            img_cls = flat_cls[img_idx].sigmoid()  # [N, num_classes], probabilities
            img_reg = flat_reg[img_idx]             # [N, 4], ltrb distances
            img_cls_logits = flat_cls[img_idx]      # [N, num_classes], raw logits

            # Decode predicted boxes: point + ltrb → xyxy
            pred_boxes = distance2bbox(flat_points, img_reg)  # [N, 4]

            # Get ground truth for this image
            gt_boxes = gt_boxes_list[img_idx]    # [M, 4]
            gt_labels = gt_labels_list[img_idx]  # [M]

            # ── Run SimOTA assignment ──
            assigned_labels, assigned_boxes, assigned_scores = self.assigner.assign(
                pred_scores=img_cls,
                pred_boxes=pred_boxes,
                points=flat_points,
                strides=flat_strides,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                num_classes=self.num_classes,
            )

            # ── Classification loss (all points) ──
            # Target is the soft score from SimOTA (IoU for positives, 0 for negatives)
            cls_loss = self.cls_loss_fn(img_cls_logits, assigned_scores)
            total_cls_loss = total_cls_loss + cls_loss

            # ── Regression loss (positive points only) ──
            pos_mask = assigned_labels < self.num_classes  # Not background
            num_pos = pos_mask.sum().item()
            num_pos_total += num_pos

            if num_pos > 0:
                pos_pred_boxes = pred_boxes[pos_mask]       # [num_pos, 4]
                pos_gt_boxes = assigned_boxes[pos_mask]     # [num_pos, 4]
                reg_loss = self.reg_loss_fn(pos_pred_boxes, pos_gt_boxes)
                total_reg_loss = total_reg_loss + reg_loss

        # Average over batch
        total_cls_loss = total_cls_loss / batch_size
        total_reg_loss = total_reg_loss / max(1, batch_size)

        total_loss = total_cls_loss + self.reg_weight * total_reg_loss

        return {
            "loss_cls": total_cls_loss,
            "loss_reg": total_reg_loss,
            "loss_total": total_loss,
        }

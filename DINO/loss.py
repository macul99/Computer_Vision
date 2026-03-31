"""DINO Loss — Hungarian Matching + Detection Losses + Denoising Loss.

==========================================================================
LOSS OVERVIEW
==========================================================================

DINO training uses a **set-based loss** that treats detection as a
bipartite matching problem. Unlike YOLO which assigns targets to grid
cells, DINO finds the optimal one-to-one assignment between predicted
queries and ground-truth objects.

Total loss for one decoder layer:

    L = λ_cls × L_classification
      + λ_L1  × L_box_L1
      + λ_giou × L_box_GIoU

This loss is applied to:
  1. The final decoder output  (main loss)
  2. Every intermediate decoder layer output  (auxiliary losses)
  3. The denoising predictions  (denoising loss — no matching needed)

==========================================================================
HUNGARIAN MATCHING
==========================================================================

Given N_q predictions and N_gt ground truth objects (N_gt ≤ N_q), we
need to find the best one-to-one assignment using the Hungarian algorithm.

For each possible assignment σ mapping predictions to ground truths,
compute a total cost:

    C(σ) = Σ_{i=1}^{N_gt} [
        λ_cls × C_cls(i, σ(i))
      + λ_L1  × C_L1(i, σ(i))
      + λ_giou × C_giou(i, σ(i))
    ]

The Hungarian algorithm finds σ* = argmin_σ C(σ) in O(n³) time.

Cost components:
  - C_cls:  -P(class_i | query_σ(i))  (negative probability → matching
            prefers high-confidence predictions)
  - C_L1:   ||b_σ(i) - b̂_i||₁  (L1 distance between predicted and GT boxes)
  - C_giou: -GIoU(b_σ(i), b̂_i) (negative GIoU → matching prefers overlapping boxes)

After matching, the loss is computed only on matched pairs.

==========================================================================
DENOISING LOSS
==========================================================================

Denoising queries are constructed by adding noise to ground-truth boxes
and labels. Since we already know which GT each denoising query should
reconstruct, NO matching is needed — the loss is computed directly.

This serves as a "shortcut" training signal that helps the decoder
learn localization and classification much faster.

==========================================================================
AUXILIARY LOSSES
==========================================================================

The same detection loss is applied at the output of every decoder layer
(not just the final one). Each intermediate layer gets its own Hungarian
matching. This provides gradient to all layers and improves convergence.

==========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from scipy.optimize import linear_sum_assignment

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class LossConfig:
    """Loss hyperparameters for DINO.

    These follow the standard DETR/DINO defaults.
    """
    # Cost weights for Hungarian matching (used to build the cost matrix)
    cost_class: float = 2.0    # Weight for classification cost in matching
    cost_bbox: float = 5.0     # Weight for L1 box cost in matching
    cost_giou: float = 2.0     # Weight for GIoU cost in matching

    # Loss weights (used to weight the final loss terms)
    loss_class: float = 1.0    # Weight for focal loss
    loss_bbox: float = 5.0     # Weight for L1 regression loss
    loss_giou: float = 2.0     # Weight for GIoU loss

    # Focal loss parameters
    focal_alpha: float = 0.25  # Balances positive vs negative examples
    focal_gamma: float = 2.0   # Focuses on hard examples


# =========================================================================
# Hungarian Matcher
# =========================================================================

class HungarianMatcher(nn.Module):
    """Bipartite matching between predictions and ground truth using the
    Hungarian algorithm.

    This is the core mechanism that makes DETR-style detectors work without
    NMS. Instead of assigning targets based on spatial location (like anchors
    in Faster R-CNN or grid cells in YOLO), we find the globally optimal
    one-to-one assignment.

    The matching cost matrix is:
        C[i, j] = cost_class × C_cls(pred_i, gt_j)
                + cost_bbox  × C_L1(pred_i, gt_j)
                + cost_giou  × C_giou(pred_i, gt_j)

    The Hungarian algorithm solves the assignment problem:
        σ* = argmin_σ Σ_j C[σ(j), j]

    Complexity: O(max(N_q, N_gt)³) per image.

    Why is this important?
    ─────────────────────
    Without matching, the network has no way to know which query should
    predict which object. If query #5 happens to predict the cat and
    query #12 happens to predict the dog, we need matching to figure
    this out before we can compute the loss.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.cost_class = config.cost_class
        self.cost_bbox = config.cost_bbox
        self.cost_giou = config.cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        targets: List[Dict[str, Tensor]],
    ) -> List[Tuple[Tensor, Tensor]]:
        """Perform Hungarian matching for a batch of predictions.

        Args:
            pred_logits: [batch_size, num_queries, num_classes]
                         Raw class logits from the decoder.
            pred_boxes:  [batch_size, num_queries, 4]
                         Predicted boxes in (cx, cy, w, h) normalized format.
            targets:     List of dicts (one per image), each with:
                         - "labels": [num_gt] class indices (LongTensor)
                         - "boxes":  [num_gt, 4] GT boxes in (cx, cy, w, h)

        Returns:
            List of (pred_indices, gt_indices) tuples, one per image.
            pred_indices[k] is the index of the prediction matched to
            gt_indices[k]-th ground truth.
        """
        batch_size, num_queries, _ = pred_logits.shape

        # ─── Step 1: Flatten predictions across the batch ───
        # This makes it easier to compute the cost matrix in one go.
        # We'll split back per image later.
        out_prob = pred_logits.flatten(0, 1).sigmoid()  # [B*Q, C]
        out_bbox = pred_boxes.flatten(0, 1)              # [B*Q, 4]

        # ─── Step 2: Concatenate all GT labels and boxes ───
        tgt_ids = torch.cat([t["labels"] for t in targets])   # [total_gt]
        tgt_bbox = torch.cat([t["boxes"] for t in targets])   # [total_gt, 4]

        if tgt_ids.numel() == 0:
            # No ground truth objects in this batch — return empty matches
            return [(torch.tensor([], dtype=torch.long),
                     torch.tensor([], dtype=torch.long)) for _ in range(batch_size)]

        # ─── Step 3: Classification cost ───
        # For focal-loss-based matching, the cost is:
        #   C_cls = -α × (1-p)^γ × p   for positive matches
        # Simplified: we just use -p(target_class) as the cost.
        # Lower cost = better match = higher predicted probability for the GT class.
        cost_class = -out_prob[:, tgt_ids]  # [B*Q, total_gt]

        # ─── Step 4: L1 box cost ───
        # Absolute difference between predicted and GT box coordinates.
        # This penalizes predictions that are spatially far from the GT.
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # [B*Q, total_gt]

        # ─── Step 5: GIoU cost ───
        # Negative GIoU because we want to minimize cost (maximize overlap).
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox),
        )  # [B*Q, total_gt]

        # ─── Step 6: Combined cost matrix ───
        cost_matrix = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )  # [B*Q, total_gt]

        # Reshape back to per-image: [B, Q, total_gt]
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1)

        # ─── Step 7: Split GT counts per image and run Hungarian matching ───
        sizes = [len(t["labels"]) for t in targets]
        indices = []
        offset = 0
        for i, num_gt in enumerate(sizes):
            if num_gt == 0:
                indices.append((torch.tensor([], dtype=torch.long),
                                torch.tensor([], dtype=torch.long)))
            else:
                # Extract cost sub-matrix for image i
                c = cost_matrix[i, :, offset:offset + num_gt]  # [Q, num_gt_i]
                # scipy's linear_sum_assignment solves the assignment problem
                # It returns (row_indices, col_indices) for the optimal assignment
                pred_idx, gt_idx = linear_sum_assignment(c.cpu().numpy())
                indices.append((
                    torch.as_tensor(pred_idx, dtype=torch.long),
                    torch.as_tensor(gt_idx, dtype=torch.long),
                ))
            offset += num_gt

        return indices


# =========================================================================
# Focal Loss Helper
# =========================================================================

def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Sigmoid focal loss for classification.

    Focal loss down-weights well-classified examples to focus training
    on hard negatives. This is critical because in detection, most queries
    match nothing (negative), and a few match objects (positive).

    Standard BCE:
        L = -y log(p) - (1-y) log(1-p)

    Focal loss adds a modulating factor:
        L = -α × (1-p)^γ × y × log(p) - (1-α) × p^γ × (1-y) × log(1-p)

    Where:
        p = sigmoid(logit)
        α = balance factor for pos/neg (default 0.25)
        γ = focusing parameter (default 2.0)

    When γ=0, this reduces to standard weighted BCE.
    When γ>0, easy examples (p close to correct target) get down-weighted.

    Example with γ=2:
        If p=0.9 for a positive: (1-0.9)² = 0.01 → loss reduced 100×
        If p=0.1 for a positive: (1-0.1)² = 0.81 → nearly full loss

    Args:
        inputs:  [N, C] raw logits (before sigmoid).
        targets: [N, C] binary targets (one-hot encoded).
        alpha:   Balance factor.
        gamma:   Focusing parameter.

    Returns:
        Scalar loss (mean over all elements).
    """
    prob = inputs.sigmoid()
    # Binary cross entropy (element-wise, no reduction)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # Modulating factor: (1-p_t)^γ where p_t = p if target=1, else (1-p)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    modulating = (1 - p_t) ** gamma

    # Alpha weighting
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    loss = alpha_t * modulating * ce_loss
    return loss.mean()


# =========================================================================
# DINO Loss
# =========================================================================

class DINOLoss(nn.Module):
    """Complete loss computation for DINO.

    This module handles:
      1. Hungarian matching between predictions and ground truth
      2. Classification loss (focal loss)
      3. Box regression losses (L1 + GIoU)
      4. Auxiliary losses on intermediate decoder layers
      5. Denoising losses (when denoising queries are present)

    The loss is computed independently for each decoder layer's output.
    This is called "auxiliary loss" or "deep supervision" and helps
    train all layers of the decoder, not just the last one.

    Target format:
        List of dicts, one per image:
        {
            "labels": LongTensor [num_gt],      # class indices
            "boxes":  FloatTensor [num_gt, 4],   # (cx, cy, w, h) normalized
        }
    """

    def __init__(self, num_classes: int, config: LossConfig | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.config = config or LossConfig()
        self.matcher = HungarianMatcher(self.config)

    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Compute all loss components.

        Args:
            outputs: Dict with keys:
                - "pred_logits": [B, Q, C] class logits from final layer
                - "pred_boxes":  [B, Q, 4] predicted boxes from final layer
                - "aux_outputs": list of {"pred_logits", "pred_boxes"} from
                                 intermediate decoder layers
                - "dn_pred_logits" (optional): [B, Q_dn, C] denoising logits
                - "dn_pred_boxes"  (optional): [B, Q_dn, 4] denoising boxes
                - "dn_targets"     (optional): list of dicts with denoising GT
            targets: List of ground-truth dicts per image.

        Returns:
            Dict of named loss tensors, e.g.:
                {"loss_cls": ..., "loss_bbox": ..., "loss_giou": ...,
                 "loss_cls_0": ..., ... (auxiliary layer losses),
                 "loss_cls_dn": ..., ... (denoising losses)}
        """
        losses = {}

        # ─── Main loss (final decoder layer) ───
        main_losses = self._compute_layer_loss(
            outputs["pred_logits"],
            outputs["pred_boxes"],
            targets,
        )
        losses.update(main_losses)

        # ─── Auxiliary losses (intermediate decoder layers) ───
        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux_losses = self._compute_layer_loss(
                    aux["pred_logits"],
                    aux["pred_boxes"],
                    targets,
                )
                # Suffix with layer index: loss_cls_0, loss_cls_1, ...
                losses.update({f"{k}_{i}": v for k, v in aux_losses.items()})

        # ─── Denoising loss (no matching needed) ───
        if "dn_pred_logits" in outputs and "dn_targets" in outputs:
            dn_losses = self._compute_denoising_loss(
                outputs["dn_pred_logits"],
                outputs["dn_pred_boxes"],
                outputs["dn_targets"],
            )
            losses.update(dn_losses)

        return losses

    def _compute_layer_loss(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Compute classification + box losses for one decoder layer.

        Steps:
          1. Run Hungarian matching to get optimal pred↔GT assignment
          2. Compute focal loss on matched + unmatched predictions
          3. Compute L1 and GIoU losses on matched predictions only

        Args:
            pred_logits: [B, Q, C] raw class logits.
            pred_boxes:  [B, Q, 4] predicted boxes (cx, cy, w, h).
            targets:     Ground truth list.

        Returns:
            Dict with "loss_cls", "loss_bbox", "loss_giou".
        """
        batch_size, num_queries, num_classes = pred_logits.shape
        device = pred_logits.device

        # ─── Step 1: Hungarian matching ───
        indices = self.matcher(pred_logits, pred_boxes, targets)

        # ─── Step 2: Classification loss (focal loss) ───
        # Build target tensor: [B, Q, C] with one-hot for matched queries,
        # all zeros for unmatched queries (background).
        target_classes = torch.zeros(
            batch_size, num_queries, num_classes,
            dtype=torch.float32, device=device,
        )
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                # Set the correct class to 1.0 for matched queries
                target_classes[b, pred_idx, targets[b]["labels"][gt_idx]] = 1.0

        loss_cls = sigmoid_focal_loss(
            pred_logits.flatten(0, 1),        # [B*Q, C]
            target_classes.flatten(0, 1),      # [B*Q, C]
            alpha=self.config.focal_alpha,
            gamma=self.config.focal_gamma,
        )

        # ─── Step 3: Box regression losses (only on matched pairs) ───
        # Gather matched predictions and GT boxes
        src_boxes_list = []
        tgt_boxes_list = []
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                src_boxes_list.append(pred_boxes[b, pred_idx])     # [K, 4]
                tgt_boxes_list.append(targets[b]["boxes"][gt_idx]) # [K, 4]

        if len(src_boxes_list) > 0:
            src_boxes = torch.cat(src_boxes_list, dim=0)  # [total_matched, 4]
            tgt_boxes = torch.cat(tgt_boxes_list, dim=0)  # [total_matched, 4]
            num_matched = src_boxes.shape[0]

            # L1 loss: penalizes absolute coordinate differences
            # Each box has 4 coordinates, so we normalize by num_matched
            loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / max(num_matched, 1)

            # GIoU loss: penalizes poor overlap (scale-invariant)
            # GIoU ∈ [-1, 1], so 1 - GIoU ∈ [0, 2]
            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(tgt_boxes),
            )  # [total_matched, total_matched]
            # We only want the diagonal (each pred matched with its own GT)
            loss_giou = (1 - giou.diag()).sum() / max(num_matched, 1)
        else:
            # No matches in this batch (all images have 0 GT objects)
            loss_bbox = pred_boxes.sum() * 0.0
            loss_giou = pred_boxes.sum() * 0.0

        return {
            "loss_cls": self.config.loss_class * loss_cls,
            "loss_bbox": self.config.loss_bbox * loss_bbox,
            "loss_giou": self.config.loss_giou * loss_giou,
        }

    def _compute_denoising_loss(
        self,
        dn_pred_logits: Tensor,
        dn_pred_boxes: Tensor,
        dn_targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Compute loss for denoising queries.

        Denoising queries are corrupted versions of ground-truth boxes/labels.
        Since we *know* which GT each denoising query corresponds to, we
        don't need Hungarian matching — the assignment is already known.

        This provides a strong auxiliary training signal because:
          1. The decoder gets easy-to-match queries (near GT boxes)
          2. Gradients flow directly without matching ambiguity
          3. This dramatically speeds up convergence

        Args:
            dn_pred_logits: [B, Q_dn, C] logits for denoising queries.
            dn_pred_boxes:  [B, Q_dn, 4] predicted boxes for denoising queries.
            dn_targets:     List of dicts, each with:
                            - "labels": [Q_dn] GT class for each dn query
                            - "boxes":  [Q_dn, 4] GT box for each dn query

        Returns:
            Dict with "loss_cls_dn", "loss_bbox_dn", "loss_giou_dn".
        """
        device = dn_pred_logits.device
        batch_size, num_dn, num_classes = dn_pred_logits.shape

        # Build one-hot targets for focal loss
        target_classes = torch.zeros_like(dn_pred_logits)
        all_src_boxes = []
        all_tgt_boxes = []

        for b in range(batch_size):
            labels = dn_targets[b]["labels"]  # [Q_dn]
            boxes = dn_targets[b]["boxes"]    # [Q_dn, 4]

            # Valid denoising queries (class >= 0; padding uses -1)
            valid = labels >= 0
            if valid.any():
                valid_idx = valid.nonzero(as_tuple=True)[0]
                target_classes[b, valid_idx, labels[valid_idx]] = 1.0
                all_src_boxes.append(dn_pred_boxes[b, valid_idx])
                all_tgt_boxes.append(boxes[valid_idx])

        # Classification loss (focal)
        loss_cls_dn = sigmoid_focal_loss(
            dn_pred_logits.flatten(0, 1),
            target_classes.flatten(0, 1),
            alpha=self.config.focal_alpha,
            gamma=self.config.focal_gamma,
        )

        # Box losses
        if len(all_src_boxes) > 0:
            src_boxes = torch.cat(all_src_boxes, dim=0)
            tgt_boxes = torch.cat(all_tgt_boxes, dim=0)
            num_dn_valid = src_boxes.shape[0]

            loss_bbox_dn = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / max(num_dn_valid, 1)

            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(tgt_boxes),
            )
            loss_giou_dn = (1 - giou.diag()).sum() / max(num_dn_valid, 1)
        else:
            loss_bbox_dn = dn_pred_boxes.sum() * 0.0
            loss_giou_dn = dn_pred_boxes.sum() * 0.0

        return {
            "loss_cls_dn": self.config.loss_class * loss_cls_dn,
            "loss_bbox_dn": self.config.loss_bbox * loss_bbox_dn,
            "loss_giou_dn": self.config.loss_giou * loss_giou_dn,
        }

"""Minimal DINO (DETR with Improved deNoising anchOr boxes) from scratch.

==========================================================================
HIGH-LEVEL ARCHITECTURE
==========================================================================

DINO is an end-to-end object detector based on the DETR family. It predicts
a fixed set of object queries and uses bipartite matching (Hungarian algorithm)
instead of NMS to deduplicate predictions.

    Image ──► Backbone ──► Multi-Scale Features ──► Transformer Encoder
                                                          │
                                                          ▼
                               Object Queries ──► Transformer Decoder
                                                          │
                                                  ┌───────┴───────┐
                                                  ▼               ▼
                                            Class Heads      Box Heads
                                           (per layer)      (per layer)

==========================================================================
WHAT MAKES DINO SPECIAL (vs plain DETR)
==========================================================================

1. MULTI-SCALE DEFORMABLE ATTENTION
   - Plain DETR uses standard attention over all pixel features → O(N²)
   - DINO uses deformable attention: each query attends to only K
     learned sampling points near a reference point → O(N×K)
   - Multi-scale: features from multiple backbone levels (e.g., 4 scales)
   - Here we simplify to standard attention for clarity, but document
     where deformable attention would go.

2. ANCHOR-BASED QUERIES (not content-only)
   - Plain DETR: queries are pure learned embeddings (no spatial prior)
   - DINO: each query has a learned spatial anchor (reference point)
     that provides an initial box estimate
   - The decoder then refines this anchor iteratively

3. ITERATIVE BOX REFINEMENT
   - Each decoder layer predicts a box *update* (delta) from the
     previous layer's box, not an absolute box from scratch
   - Layer 0: box₀ = anchor + Δ₀
   - Layer k: box_k = box_{k-1} + Δ_k
   - This makes it progressively easier for deeper layers

4. DENOISING TRAINING
   - During training, we create extra "denoising queries" by adding
     noise to ground-truth boxes and labels
   - These bypass Hungarian matching (assignment is known)
   - The decoder learns to reconstruct clean boxes from noisy ones
   - This dramatically accelerates convergence

5. MIXED QUERY SELECTION
   - Top encoder features are selected to initialize some queries
   - Remaining queries come from learned embeddings
   - This gives the decoder a head start by seeding it with promising
     encoder outputs

==========================================================================
COMPARISON WITH PREVIOUS IMPLEMENTATIONS
==========================================================================

vs Faster R-CNN (../faster_rcnn_min/):
  - No RPN, no anchors grid, no NMS
  - Set prediction with matching
  - End-to-end training

vs YOLO (../yolo/):
  - No grid-based assignment
  - Attention-based feature aggregation (not just conv)
  - One-to-one matching instead of one-to-many

==========================================================================
SIMPLIFICATIONS IN THIS IMPLEMENTATION
==========================================================================

To keep the code readable and study-focused:
  1. We use standard multi-head attention instead of deformable attention
  2. We use a small CNN backbone instead of ResNet/Swin
  3. We use 2 feature scales instead of 4
  4. We use fewer queries (100 instead of 900)
  5. We use a simplified denoising scheme
  6. Positional encodings are learned instead of sinusoidal

Each simplification is documented where it appears.

==========================================================================
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class DINOConfig:
    """All hyperparameters for the DINO model.

    Grouped by component for clarity.
    """
    # --- Backbone ---
    in_channels: int = 3          # Input image channels (RGB)
    backbone_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    # Output channels at each backbone stage. We extract multi-scale
    # features from the last two stages.

    # --- Feature Projection ---
    hidden_dim: int = 256         # Transformer hidden dimension (d_model)
    num_feature_levels: int = 2   # Number of multi-scale feature levels

    # --- Transformer Encoder ---
    enc_layers: int = 3           # Number of encoder layers
    enc_heads: int = 8            # Attention heads in encoder

    # --- Transformer Decoder ---
    dec_layers: int = 3           # Number of decoder layers
    dec_heads: int = 8            # Attention heads in decoder

    # --- Queries ---
    num_queries: int = 100        # Number of object queries
    # In the full DINO, this is ~900. We use 100 for simplicity.

    # --- Detection Head ---
    num_classes: int = 5          # Number of object categories

    # --- Denoising ---
    dn_number: int = 5            # Number of denoising groups during training
    dn_label_noise_ratio: float = 0.5   # Probability of flipping a GT label
    dn_box_noise_scale: float = 0.4     # Scale of noise added to GT boxes

    # --- General ---
    dropout: float = 0.0         # Dropout rate (0 for small-scale experiments)
    ffn_dim: int = 1024          # Feed-forward network intermediate dimension


# =========================================================================
# Backbone: Simple Multi-Scale CNN
# =========================================================================

class SimpleBackbone(nn.Module):
    """A minimal CNN backbone that outputs multi-scale feature maps.

    Architecture (3 stages, each downsamples by 2):

        Stage 0:  [B, 3, H, W]    →  [B, 64, H/2, W/2]
        Stage 1:  [B, 64, H/2, W/2]  →  [B, 128, H/4, W/4]
        Stage 2:  [B, 128, H/4, W/4] →  [B, 256, H/8, W/8]

    We extract features from the last two stages (scale 1 and scale 2)
    to create multi-scale features, similar to how the full DINO uses
    features from ResNet/Swin stages {C3, C4, C5} plus a downsampled C6.

    In the full DINO:
    ─────────────────
    - Backbone: ResNet-50 or Swin Transformer
    - Feature levels: 4 scales from FPN
    - Deformable attention samples across all 4 scales
    Here we use 2 scales with standard attention for simplicity.
    """

    def __init__(self, config: DINOConfig):
        super().__init__()
        channels = [config.in_channels] + config.backbone_channels

        self.stages = nn.ModuleList()
        for i in range(len(config.backbone_channels)):
            self.stages.append(nn.Sequential(
                # Stride-2 conv for downsampling
                nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
                # Additional conv at same resolution for more capacity
                nn.Conv2d(channels[i + 1], channels[i + 1], 3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ))

    def forward(self, x: Tensor) -> List[Tensor]:
        """Extract multi-scale features.

        Args:
            x: [B, 3, H, W] input image.

        Returns:
            List of feature maps from selected stages.
            We return the last `num_feature_levels` stages.
            E.g., for 3 stages and 2 levels:
                [stage1_output, stage2_output]
                = [[B, 128, H/4, W/4], [B, 256, H/8, W/8]]
        """
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


# =========================================================================
# Positional Encoding (Learned 2D)
# =========================================================================

class LearnedPositionalEncoding2D(nn.Module):
    """Learned 2D positional encoding for spatial feature maps.

    Since attention is permutation-invariant, we need positional encodings
    to tell the transformer where each feature comes from spatially.

    We create separate learned embeddings for row and column positions,
    then concatenate them. This is simpler than sinusoidal encodings
    but works well for small-scale experiments.

    In the full DINO:
    ─────────────────
    - Uses sinusoidal 2D positional encodings (sine + cosine at different
      frequencies for x and y coordinates)
    - Temperature-based frequency scaling
    - More robust to different input sizes
    """

    def __init__(self, hidden_dim: int, max_size: int = 128):
        super().__init__()
        # We allocate max_size positions for both rows and columns.
        # Each position gets hidden_dim//2 dimensions.
        self.row_embed = nn.Embedding(max_size, hidden_dim // 2)
        self.col_embed = nn.Embedding(max_size, hidden_dim // 2)

    def forward(self, x: Tensor) -> Tensor:
        """Generate positional encoding matching the spatial size of x.

        Args:
            x: [B, C, H, W] feature map.

        Returns:
            [1, hidden_dim, H, W] positional encoding (broadcast over batch).
        """
        H, W = x.shape[-2:]
        device = x.device

        rows = torch.arange(H, device=device)  # [H]
        cols = torch.arange(W, device=device)   # [W]

        row_emb = self.row_embed(rows)  # [H, D/2]
        col_emb = self.col_embed(cols)  # [W, D/2]

        # Expand to [H, W, D]:
        #   row_emb[:, None, :] → [H, 1, D/2], broadcast to [H, W, D/2]
        #   col_emb[None, :, :] → [1, W, D/2], broadcast to [H, W, D/2]
        pos = torch.cat([
            row_emb[:, None, :].expand(-1, W, -1),  # [H, W, D/2]
            col_emb[None, :, :].expand(H, -1, -1),   # [H, W, D/2]
        ], dim=-1)  # [H, W, D]

        # Rearrange to [1, D, H, W] for easy addition to feature maps
        pos = pos.permute(2, 0, 1).unsqueeze(0)  # [1, D, H, W]
        return pos


# =========================================================================
# Transformer Encoder Layer
# =========================================================================

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer: self-attention + FFN.

    The encoder processes the flattened multi-scale image features to
    build global feature representations. Each spatial position attends
    to all other positions across all scale levels.

    Architecture:
        Input ──► Self-Attention ──► Add & Norm ──► FFN ──► Add & Norm ──► Output

    In the full DINO:
    ─────────────────
    - Uses multi-scale deformable attention instead of standard attention
    - Each position only attends to K=4 learned sampling points per level
    - Complexity: O(N × L × K) instead of O(N²)
      where N=total tokens, L=num_levels, K=num_points
    - This makes it practical to process high-resolution features
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Feed-forward network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        # Layer normalization (pre-norm style)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, pos: Tensor) -> Tensor:
        """Process features through one encoder layer.

        Positional encoding is added to queries and keys (but not values)
        in self-attention. This is because position should influence
        *which* tokens attend to each other, but not *what* information
        is passed. This is standard practice in DETR-style models.

        Args:
            src: [B, N, D] flattened multi-scale features.
            pos: [B, N, D] positional encoding for each feature.

        Returns:
            [B, N, D] features after self-attention + FFN.
        """
        # Self-attention with positional encoding added to Q and K
        q = k = src + pos       # Position-aware queries and keys
        v = src                  # Values without position (content only)
        src2 = self.self_attn(q, k, v)[0]  # [0] = attention output, [1] = weights
        src = src + src2         # Residual connection
        src = self.norm1(src)    # Layer norm

        # Feed-forward network
        src = src + self.ffn(src)  # Residual connection
        src = self.norm2(src)       # Layer norm

        return src


# =========================================================================
# Transformer Decoder Layer
# =========================================================================

class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer: self-attn + cross-attn + FFN.

    The decoder takes object queries and refines them by attending to
    the encoder output (image features). Each decoder layer:
      1. Self-attention among queries (so queries communicate)
      2. Cross-attention from queries to encoder features (gather image info)
      3. FFN to process the gathered information

    Architecture:
        Queries ──► Self-Attention ──► Add & Norm
                                          │
             Encoder Output ──────► Cross-Attention ──► Add & Norm
                                                           │
                                                     FFN ──► Add & Norm ──► Output

    In the full DINO:
    ─────────────────
    - Cross-attention uses multi-scale deformable attention
    - Each query attends to K=4 sampling points per feature level
    - Sampling points are predicted relative to the query's reference point
    - This is what enables efficient attention over high-resolution features
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        # Self-attention among object queries
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Cross-attention: queries attend to encoder features
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        query_pos: Tensor,
        memory_pos: Tensor,
    ) -> Tensor:
        """Process queries through one decoder layer.

        Args:
            tgt:        [B, Q, D] object queries (learnable content).
            memory:     [B, N, D] encoder output features.
            query_pos:  [B, Q, D] positional encoding for queries.
            memory_pos: [B, N, D] positional encoding for encoder features.

        Returns:
            [B, Q, D] refined query features.
        """
        # ─── Self-attention (queries talk to each other) ───
        # Positional encoding added to Q and K so queries know their
        # relative positions (which helps avoid duplicate predictions).
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        # ─── Cross-attention (queries gather information from image) ───
        # Q = queries + query_pos → where to look
        # K = memory + memory_pos → position-aware image features
        # V = memory → raw image features (content)
        tgt2 = self.cross_attn(
            query=tgt + query_pos,
            key=memory + memory_pos,
            value=memory,
        )[0]
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        # ─── FFN ───
        tgt = tgt + self.ffn(tgt)
        tgt = self.norm3(tgt)

        return tgt


# =========================================================================
# Detection Head (shared across decoder layers)
# =========================================================================

class DetectionHead(nn.Module):
    """Prediction head that converts query features into class logits and boxes.

    Each decoder layer has its own detection head (or shares weights).
    In DINO, the heads produce:
      - Class logits: [B, Q, num_classes]
      - Box deltas:   [B, Q, 4] in (Δcx, Δcy, Δw, Δh)

    The box deltas are added to the current reference point to get
    the predicted box (iterative refinement).

    Why deltas instead of absolute coordinates?
    ───────────────────────────────────────────
    Predicting residuals relative to an initial anchor is easier than
    predicting absolute coordinates from scratch. The anchor provides
    a good initialization, and the head only needs to predict small
    corrections. This is the same insight behind Faster R-CNN's
    box regression, but applied across decoder layers.
    """

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        # Classification: hidden_dim → num_classes
        self.class_head = nn.Linear(hidden_dim, num_classes)
        # Box regression: hidden_dim → 4 (Δcx, Δcy, Δw, Δh)
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        query_feat: Tensor,
        reference_points: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Predict classes and boxes from query features.

        Args:
            query_feat:       [B, Q, D] features from a decoder layer.
            reference_points: [B, Q, 4] current reference boxes
                              (cx, cy, w, h) in normalized coordinates.

        Returns:
            logits:     [B, Q, num_classes] raw class logits.
            pred_boxes: [B, Q, 4] predicted boxes (cx, cy, w, h) normalized,
                        after sigmoid to ensure boxes stay in [0, 1].
        """
        logits = self.class_head(query_feat)  # [B, Q, num_classes]

        # Box regression: predict deltas relative to reference
        box_delta = self.bbox_head(query_feat)  # [B, Q, 4]

        # ─── Iterative refinement ───
        # The reference point is in inverse-sigmoid space for numerical
        # stability. We add the predicted delta in that space, then
        # apply sigmoid to get the final box in [0, 1].
        #
        # Why inverse sigmoid?
        #   Sigmoid maps ℝ → (0, 1). Working in inverse-sigmoid (logit)
        #   space lets us do unconstrained addition of deltas, then
        #   sigmoid maps the result back to valid [0, 1] coordinates.
        #
        # ref_logit = inverse_sigmoid(reference_point)
        # pred_box = sigmoid(ref_logit + delta)
        ref_logit = inverse_sigmoid(reference_points)  # [B, Q, 4]
        pred_boxes = (ref_logit + box_delta).sigmoid()  # [B, Q, 4]

        return logits, pred_boxes


# =========================================================================
# Denoising Query Generator
# =========================================================================

class DenoisingGenerator(nn.Module):
    """Generate denoising queries from ground truth during training.

    =======================================================================
    HOW DENOISING WORKS IN DINO
    =======================================================================

    During training only, we create extra queries by corrupting GT info:

    1. Take each ground truth box and label
    2. Add random noise to the box coordinates (shift and scale)
    3. Randomly flip the class label with some probability
    4. Create multiple noisy copies ("denoising groups")
    5. Feed these as extra queries to the decoder

    Since we know exactly which GT each denoising query corresponds to,
    the loss doesn't need Hungarian matching — it's computed directly.

    Why does this help?
    ───────────────────
    - Standard DETR queries start with no spatial prior → slow convergence
    - Denoising queries start near GT boxes → easy reconstruction task
    - This teaches the decoder to refine boxes, which transfers to normal queries
    - Each denoising group creates an independent set of queries, so using
      multiple groups (dn_number) multiplies the training signal

    Attention mask:
    ───────────────
    To prevent information leaking between denoising groups and regular
    queries, DINO uses an attention mask in the decoder's self-attention:
    - Denoising queries in group i can attend to each other
    - Denoising queries cannot attend to normal queries
    - Normal queries cannot attend to denoising queries
    - Different denoising groups cannot attend to each other

    This ensures that the denoising path doesn't "cheat" by looking at
    normal predictions, and normal queries don't get free answers from
    denoising queries.

    In this simplified implementation, we concatenate denoising queries
    with normal queries and return them separately after the decoder.
    We use a simpler masking approach for clarity.
    """

    def __init__(self, config: DINOConfig):
        super().__init__()
        self.num_classes = config.num_classes
        self.hidden_dim = config.hidden_dim
        self.dn_number = config.dn_number
        self.label_noise_ratio = config.dn_label_noise_ratio
        self.box_noise_scale = config.dn_box_noise_scale

        # Embedding for class labels (used to create denoising query content)
        self.label_embedding = nn.Embedding(config.num_classes, config.hidden_dim)

    def forward(
        self,
        targets: List[Dict[str, Tensor]],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]], Optional[Tensor]]:
        """Generate denoising queries and their targets.

        Args:
            targets: List of dicts per image with "labels" and "boxes".
            device: Device for tensor creation.

        Returns:
            dn_query:     [B, Q_dn, D] denoising query content embeddings.
            dn_ref:       [B, Q_dn, 4] denoising reference points (noisy GT boxes).
            dn_query_pos: [B, Q_dn, D] positional embeddings for dn queries.
            dn_targets:   List of dicts with GT labels/boxes for each dn query.
            dn_attn_mask: [Q_dn + Q, Q_dn + Q] attention mask (optional).
                          True = masked (cannot attend), False = can attend.
        """
        batch_size = len(targets)

        # Count GT objects per image to determine the size of dn queries
        num_gts = [len(t["labels"]) for t in targets]
        max_gt = max(num_gts) if num_gts else 0

        if max_gt == 0:
            # No GT objects → no denoising queries
            return None, None, None, None, None

        # Total denoising queries per image = dn_number groups × max_gt per group
        # We pad images with fewer GTs to keep tensor sizes uniform
        total_dn = self.dn_number * max_gt

        # ─── Step 1: Create noisy labels ───
        # Start with GT labels repeated dn_number times
        dn_labels = torch.full((batch_size, total_dn), -1, dtype=torch.long, device=device)
        dn_boxes = torch.zeros(batch_size, total_dn, 4, device=device)

        for b in range(batch_size):
            n = num_gts[b]
            if n == 0:
                continue

            gt_labels = targets[b]["labels"]  # [n]
            gt_boxes = targets[b]["boxes"]    # [n, 4]

            # Repeat GT dn_number times: [dn_number * n]
            rep_labels = gt_labels.repeat(self.dn_number)
            rep_boxes = gt_boxes.repeat(self.dn_number, 1)

            # Add label noise: randomly flip some labels to a different class
            if self.label_noise_ratio > 0:
                noise_mask = torch.rand(len(rep_labels), device=device) < self.label_noise_ratio
                random_labels = torch.randint(0, self.num_classes, (len(rep_labels),), device=device)
                rep_labels = torch.where(noise_mask, random_labels, rep_labels)

            # Add box noise: shift center and scale width/height
            # The noise is proportional to the box size (larger boxes get more noise)
            if self.box_noise_scale > 0:
                box_noise = torch.rand_like(rep_boxes) * 2 - 1  # uniform [-1, 1]
                box_noise = box_noise * self.box_noise_scale

                # Noise on center: shift by fraction of box size
                noisy_cx = rep_boxes[:, 0] + box_noise[:, 0] * rep_boxes[:, 2]
                noisy_cy = rep_boxes[:, 1] + box_noise[:, 1] * rep_boxes[:, 3]
                # Noise on size: scale w and h
                noisy_w = rep_boxes[:, 2] * (1 + box_noise[:, 2])
                noisy_h = rep_boxes[:, 3] * (1 + box_noise[:, 3])

                rep_boxes = torch.stack([noisy_cx, noisy_cy, noisy_w, noisy_h], dim=-1)
                # Clamp to [0, 1] to keep boxes valid
                rep_boxes = rep_boxes.clamp(min=0.0, max=1.0)

            # Place into padded tensor (first dn_number*n slots)
            dn_count = self.dn_number * n
            dn_labels[b, :dn_count] = rep_labels
            dn_boxes[b, :dn_count] = rep_boxes

        # ─── Step 2: Create query embeddings from noisy labels ───
        # For valid queries (label >= 0), use label embedding as content
        # For padding queries (label == -1), use zeros
        valid_mask = dn_labels >= 0  # [B, total_dn]
        safe_labels = dn_labels.clamp(min=0)  # Clamp for embedding lookup (padding uses 0)
        dn_query = self.label_embedding(safe_labels)  # [B, total_dn, D]
        # Zero out padding queries
        dn_query = dn_query * valid_mask.unsqueeze(-1).float()

        # ─── Step 3: Reference points = noisy GT boxes ───
        dn_ref = dn_boxes  # [B, total_dn, 4]

        # ─── Step 4: Positional encoding for dn queries ───
        # We derive positional embeddings from the reference points.
        # A simple MLP projects the 4D box coords to hidden_dim.
        # (In the full DINO, this uses sinusoidal encoding of the reference point.)
        # Here we just use the reference points as-is and let the decoder
        # handle positional information through query_pos (set to zeros
        # and let the cross-attention learn position from the reference).
        dn_query_pos = torch.zeros_like(dn_query)  # [B, total_dn, D]

        # ─── Step 5: Build denoising targets ───
        # These are the *original* (un-noised) GT labels and boxes,
        # repeated to match the denoising queries.
        dn_targets = []
        for b in range(batch_size):
            n = num_gts[b]
            if n == 0:
                dn_targets.append({
                    "labels": torch.full((total_dn,), -1, dtype=torch.long, device=device),
                    "boxes": torch.zeros(total_dn, 4, device=device),
                })
            else:
                # Original GT repeated dn_number times, padded to total_dn
                rep_gt_labels = targets[b]["labels"].repeat(self.dn_number)
                rep_gt_boxes = targets[b]["boxes"].repeat(self.dn_number, 1)
                # Pad remaining slots
                pad_labels = torch.full((total_dn - self.dn_number * n,),
                                        -1, dtype=torch.long, device=device)
                pad_boxes = torch.zeros(total_dn - self.dn_number * n, 4, device=device)
                dn_targets.append({
                    "labels": torch.cat([rep_gt_labels, pad_labels]),
                    "boxes": torch.cat([rep_gt_boxes, pad_boxes]),
                })

        # ─── Step 6: Attention mask ───
        # We need to prevent denoising queries from seeing normal queries
        # and vice versa. Also different denoising groups shouldn't see
        # each other.
        #
        # For simplicity, we handle this by processing dn queries and
        # normal queries separately through the decoder, then combining
        # the results. This avoids complex attention masking.
        #
        # The full DINO uses a block-diagonal attention mask. We return
        # None here because our decoder handles the separation internally.
        dn_attn_mask = None

        return dn_query, dn_ref, dn_query_pos, dn_targets, dn_attn_mask


# =========================================================================
# Helper Functions
# =========================================================================

def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Compute the inverse of sigmoid (logit function).

    sigmoid(y) = x  →  y = log(x / (1 - x))

    Used in iterative box refinement to convert reference points
    from [0, 1] space to unbounded logit space, where we can freely
    add predicted deltas.

    Args:
        x: Tensor with values in (0, 1).
        eps: Small epsilon for numerical stability.

    Returns:
        Logit values (inverse sigmoid of x).
    """
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


def _clone_module(module: nn.Module, n: int) -> nn.ModuleList:
    """Create n independent copies of a module (deep copy).

    Each copy has its own parameters. This is used to create
    multiple decoder layers or detection heads that share architecture
    but not weights.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# =========================================================================
# Main DINO Model
# =========================================================================

class DINO(nn.Module):
    """DINO: DETR with Improved deNoising anchOr boxes.

    =======================================================================
    FORWARD PASS OVERVIEW
    =======================================================================

    Training:
        Image → Backbone → Multi-Scale Features → Encoder →
        [Denoising Queries + Normal Queries] → Decoder (iterative) →
        Detection Heads → Losses (main + auxiliary + denoising)

    Inference:
        Image → Backbone → Multi-Scale Features → Encoder →
        Normal Queries → Decoder (iterative) →
        Detection Heads → Final predictions (no denoising)

    =======================================================================
    INPUT / OUTPUT FORMAT
    =======================================================================

    Training input:
        images:  [B, 3, H, W] normalized images
        targets: List[Dict] with "labels" [N_gt] and "boxes" [N_gt, 4]
                 Boxes in (cx, cy, w, h) normalized to [0, 1]

    Training output:
        Dict with:
            "pred_logits":     [B, Q, C] final layer class logits
            "pred_boxes":      [B, Q, 4] final layer predicted boxes
            "aux_outputs":     List of {"pred_logits", "pred_boxes"} per layer
            "dn_pred_logits":  [B, Q_dn, C] denoising logits (if targets given)
            "dn_pred_boxes":   [B, Q_dn, 4] denoising boxes (if targets given)
            "dn_targets":      denoising GT (if targets given)

    Inference output:
        Dict with:
            "pred_logits": [B, Q, C]
            "pred_boxes":  [B, Q, 4]
    """

    def __init__(self, config: DINOConfig | None = None):
        super().__init__()
        self.config = config or DINOConfig()
        c = self.config

        # ─── Backbone ───
        self.backbone = SimpleBackbone(c)

        # ─── Feature projection ───
        # Project backbone features to hidden_dim for the transformer.
        # One projection per feature level.
        self.input_proj = nn.ModuleList()
        backbone_out_channels = c.backbone_channels[-c.num_feature_levels:]
        for ch in backbone_out_channels:
            self.input_proj.append(nn.Sequential(
                nn.Conv2d(ch, c.hidden_dim, 1),   # 1×1 conv to project channels
                nn.GroupNorm(32, c.hidden_dim),     # GroupNorm is standard in DETR
            ))

        # ─── Positional encoding ───
        self.pos_encoder = LearnedPositionalEncoding2D(c.hidden_dim)

        # ─── Level embedding ───
        # Each feature level gets a learned embedding added to distinguish
        # which scale a token comes from. This is important because after
        # flattening, the transformer doesn't know which level each token is from.
        self.level_embed = nn.Parameter(torch.randn(c.num_feature_levels, c.hidden_dim))

        # ─── Transformer Encoder ───
        encoder_layer = TransformerEncoderLayer(
            c.hidden_dim, c.enc_heads, c.ffn_dim, c.dropout,
        )
        self.encoder_layers = _clone_module(encoder_layer, c.enc_layers)

        # ─── Transformer Decoder ───
        decoder_layer = TransformerDecoderLayer(
            c.hidden_dim, c.dec_heads, c.ffn_dim, c.dropout,
        )
        self.decoder_layers = _clone_module(decoder_layer, c.dec_layers)

        # ─── Object queries ───
        # Content part: learned embeddings that represent "what to detect"
        self.query_embed = nn.Embedding(c.num_queries, c.hidden_dim)
        # Anchor part: learned reference points (initial box guesses)
        # Each query gets a 4D anchor (cx, cy, w, h) in sigmoid space
        self.reference_points = nn.Embedding(c.num_queries, 4)
        # Initialize reference points uniformly across the image
        nn.init.uniform_(self.reference_points.weight, 0.0, 1.0)

        # ─── Detection heads (one per decoder layer for iterative refinement) ───
        # Each layer predicts its own class logits and box deltas
        det_head = DetectionHead(c.hidden_dim, c.num_classes)
        self.det_heads = _clone_module(det_head, c.dec_layers)

        # ─── Denoising generator (training only) ───
        self.dn_generator = DenoisingGenerator(c)

        # ─── Initialize weights ───
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights with sensible defaults.

        - Linear/Conv layers: Xavier uniform
        - Biases: zero
        - Classification head: bias initialized so that initial predictions
          have low probability (prevents confident wrong predictions early)
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize classification bias to predict low probability
        # This is important: without this, the model starts by predicting
        # high confidence for random classes, causing unstable training.
        # bias = -log((1-π)/π) where π ≈ 0.01 (prior probability of object)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for head in self.det_heads:
            nn.init.constant_(head.class_head.bias, bias_value)

    def forward(
        self,
        images: Tensor,
        targets: List[Dict[str, Tensor]] | None = None,
    ) -> Dict[str, Tensor]:
        """Full DINO forward pass.

        Args:
            images:  [B, 3, H, W] input images (normalized).
            targets: Ground truth (required for training, None for inference).

        Returns:
            Dict of predictions and optionally denoising outputs.
        """
        device = images.device
        batch_size = images.shape[0]

        # =================================================================
        # Step 1: Backbone — Extract multi-scale features
        # =================================================================
        # Backbone produces features at multiple resolutions.
        # We take the last num_feature_levels levels.
        backbone_features = self.backbone(images)
        # Select the relevant levels (e.g., last 2 of 3 stages)
        multi_scale_features = backbone_features[-self.config.num_feature_levels:]

        # =================================================================
        # Step 2: Project features to hidden_dim and add positional encoding
        # =================================================================
        src_flatten = []
        pos_flatten = []

        for lvl, (feat, proj) in enumerate(zip(multi_scale_features, self.input_proj)):
            # Project to hidden_dim: [B, hidden_dim, H_l, W_l]
            src = proj(feat)

            # Positional encoding: [1, hidden_dim, H_l, W_l]
            pos = self.pos_encoder(src)

            # Add level embedding to distinguish scales
            # level_embed[lvl] is [D], reshape to [1, D, 1, 1] for broadcasting
            src = src + self.level_embed[lvl].view(1, -1, 1, 1)

            # Flatten spatial dimensions: [B, hidden_dim, H_l, W_l] → [B, H_l*W_l, hidden_dim]
            B, D, H, W = src.shape
            src_flatten.append(src.flatten(2).transpose(1, 2))    # [B, H*W, D]
            pos_flatten.append(pos.flatten(2).transpose(1, 2).expand(B, -1, -1))  # [B, H*W, D]

        # Concatenate all levels: [B, N_total, D]
        # N_total = Σ_l (H_l × W_l) — total number of spatial tokens
        src = torch.cat(src_flatten, dim=1)   # [B, N, D]
        pos = torch.cat(pos_flatten, dim=1)   # [B, N, D]

        # =================================================================
        # Step 3: Transformer Encoder — Refine features with self-attention
        # =================================================================
        # The encoder lets each spatial location attend to all others,
        # building a global representation. After encoding, each token
        # has information about the entire image.
        memory = src
        for enc_layer in self.encoder_layers:
            memory = enc_layer(memory, pos)    # [B, N, D]

        # =================================================================
        # Step 4: Prepare object queries
        # =================================================================
        # Content queries: learned "what to detect" embeddings
        query_content = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        # [B, Q, D]

        # Reference points: learned anchor boxes (sigmoid to ensure [0, 1])
        ref_points = self.reference_points.weight.sigmoid().unsqueeze(0).expand(batch_size, -1, -1)
        # [B, Q, 4] — (cx, cy, w, h) anchors

        # Query positional encoding = content query embeddings
        # (In full DINO, this is derived from the reference points using a sinusoidal encoding.
        #  Here we use the learned query embeddings directly as positional encoding.)
        query_pos = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        # [B, Q, D]

        # =================================================================
        # Step 5: Generate denoising queries (training only)
        # =================================================================
        dn_query = dn_ref = dn_query_pos = dn_targets = None
        if self.training and targets is not None:
            dn_query, dn_ref, dn_query_pos, dn_targets, _ = \
                self.dn_generator(targets, device)

        # =================================================================
        # Step 6: Transformer Decoder — Iterative box refinement
        # =================================================================
        # Process normal queries through the decoder layers, collecting
        # predictions from each layer for auxiliary losses.
        all_logits, all_boxes = self._run_decoder(
            query_content, ref_points, query_pos, memory, pos,
        )
        # all_logits: list of [B, Q, C], one per decoder layer
        # all_boxes:  list of [B, Q, 4], one per decoder layer

        # Process denoising queries if present
        dn_all_logits = None
        dn_all_boxes = None
        if dn_query is not None:
            dn_all_logits, dn_all_boxes = self._run_decoder(
                dn_query, dn_ref, dn_query_pos, memory, pos,
            )

        # =================================================================
        # Step 7: Assemble output
        # =================================================================
        outputs = {
            "pred_logits": all_logits[-1],   # Final layer predictions
            "pred_boxes": all_boxes[-1],
        }

        # Auxiliary outputs (intermediate decoder layers)
        if len(all_logits) > 1:
            outputs["aux_outputs"] = [
                {"pred_logits": logits, "pred_boxes": boxes}
                for logits, boxes in zip(all_logits[:-1], all_boxes[:-1])
            ]

        # Denoising outputs
        if dn_all_logits is not None:
            outputs["dn_pred_logits"] = dn_all_logits[-1]
            outputs["dn_pred_boxes"] = dn_all_boxes[-1]
            outputs["dn_targets"] = dn_targets

        return outputs

    def _run_decoder(
        self,
        query_content: Tensor,
        reference_points: Tensor,
        query_pos: Tensor,
        memory: Tensor,
        memory_pos: Tensor,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Run queries through all decoder layers with iterative refinement.

        This implements the core DINO decoder loop:
          For each layer k:
            1. Run decoder layer to update query features
            2. Predict class logits and box deltas
            3. Update reference points using predicted boxes
               (detached from gradient to prevent gradient shortcut)

        Detaching reference points between layers:
        ──────────────────────────────────────────
        When updating reference points for the next layer, we detach
        them from the computation graph. Without detaching, gradients
        would flow through the reference points of all previous layers,
        which creates optimization instabilities. Each layer should
        learn to refine based on its input reference, not backprop
        through all previous refinements.

        Args:
            query_content:   [B, Q, D] initial query content features.
            reference_points:[B, Q, 4] initial reference boxes (cx, cy, w, h).
            query_pos:       [B, Q, D] positional encoding for queries.
            memory:          [B, N, D] encoder output.
            memory_pos:      [B, N, D] encoder positional encoding.

        Returns:
            all_logits: List of [B, Q, C] class logits, one per decoder layer.
            all_boxes:  List of [B, Q, 4] pred boxes, one per decoder layer.
        """
        all_logits = []
        all_boxes = []

        tgt = query_content  # Working query features
        ref = reference_points  # Working reference points

        for layer_idx, (dec_layer, det_head) in enumerate(
            zip(self.decoder_layers, self.det_heads)
        ):
            # ─── Decoder layer: update query features ───
            tgt = dec_layer(tgt, memory, query_pos, memory_pos)
            # [B, Q, D]

            # ─── Detection head: predict classes and boxes ───
            logits, pred_boxes = det_head(tgt, ref)
            # logits:     [B, Q, C]
            # pred_boxes: [B, Q, 4]

            all_logits.append(logits)
            all_boxes.append(pred_boxes)

            # ─── Update reference points for next layer ───
            # Detach to prevent gradient flow through previous layers.
            # The next decoder layer uses the current prediction as its
            # starting point, but each layer optimizes independently.
            ref = pred_boxes.detach()

        return all_logits, all_boxes

    @torch.no_grad()
    def predict(
        self,
        images: Tensor,
        score_threshold: float = 0.3,
    ) -> List[Dict[str, Tensor]]:
        """Run inference and return filtered predictions.

        This is a convenience method for evaluation/inference that:
          1. Runs the forward pass
          2. Applies sigmoid to get class probabilities
          3. Filters by confidence threshold
          4. Returns per-image results

        Args:
            images: [B, 3, H, W] input images.
            score_threshold: Minimum confidence to keep a detection.

        Returns:
            List of dicts (one per image), each with:
                "scores": [K] confidence scores
                "labels": [K] class indices
                "boxes":  [K, 4] boxes in (cx, cy, w, h) normalized format
        """
        self.eval()
        outputs = self.forward(images, targets=None)

        pred_logits = outputs["pred_logits"]  # [B, Q, C]
        pred_boxes = outputs["pred_boxes"]    # [B, Q, 4]

        # Convert logits to probabilities
        probs = pred_logits.sigmoid()  # [B, Q, C]

        results = []
        for b in range(pred_logits.shape[0]):
            # For each query, take the max class probability
            scores, labels = probs[b].max(dim=-1)  # [Q], [Q]
            boxes = pred_boxes[b]  # [Q, 4]

            # Filter by threshold
            keep = scores > score_threshold
            results.append({
                "scores": scores[keep],
                "labels": labels[keep],
                "boxes": boxes[keep],
            })

        return results

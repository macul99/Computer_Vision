"""Minimal RTMDet architecture implemented from scratch in PyTorch.

==========================================================================
HIGH-LEVEL DESIGN (RTMDet)
==========================================================================

RTMDet (Real-Time Models for object DETection) is a modern anchor-free,
single-stage object detector from OpenMMLab (2022). It achieves excellent
speed-accuracy tradeoff by combining several key design choices:

Key innovations over classic YOLO:
  1. CSPNeXt backbone — efficient CSP blocks with large-kernel depthwise convs
  2. PAFPN neck — bidirectional feature pyramid (top-down + bottom-up)
  3. Decoupled head — separate branches for classification and box regression
  4. Anchor-free — predicts distances from points, not offsets from anchors
  5. Dynamic label assignment (SimOTA) — assigns labels during training
  6. Quality Focal Loss — quality-aware classification loss

==========================================================================
ARCHITECTURE OVERVIEW
==========================================================================

Input Image [B, 3, 320, 320]
     │
     ▼
┌─────────────────────┐
│   CSPNeXt Backbone   │    Extracts multi-scale features
│                      │
│  Stem ──► Stage1 ──► Stage2 ──► Stage3 ──► Stage4
│           stride4     stride8    stride16   stride32
│                       feat_s8    feat_s16   feat_s32
└──────┬────────┬────────┬─────────┘
       │        │        │
       ▼        ▼        ▼
┌─────────────────────┐
│    CSPAFPN Neck      │    Fuses features across scales
│                      │
│  Top-Down Path:      │    High-level → low-level
│    s32 ──► s16 ──► s8│
│                      │
│  Bottom-Up Path:     │    Low-level → high-level
│    s8 ──► s16 ──► s32│
└──────┬────────┬────────┬─────────┘
       │        │        │
       ▼        ▼        ▼
┌─────────────────────┐
│   RTMDet Head        │    Predicts boxes and classes
│                      │
│  For each scale:     │
│    cls_branch → cls_scores [B, C, H, W]
│    reg_branch → ltrb_dists [B, 4, H, W]
└─────────────────────┘

Output: Multi-scale predictions at strides 8, 16, 32
  - Stride 8:  40×40 = 1600 predictions (small objects)
  - Stride 16: 20×20 = 400  predictions (medium objects)
  - Stride 32: 10×10 = 100  predictions (large objects)
  - Total: 2100 candidate detections per image

==========================================================================
COMPARISON WITH YOLOv1 (see ../yolo/)
==========================================================================

| Aspect          | YOLOv1                          | RTMDet                              |
|-----------------|----------------------------------|--------------------------------------|
| Feature levels  | Single scale (7×7)               | Multi-scale (3 levels via FPN)       |
| Box format      | (x, y, w, h) per grid cell       | (l, t, r, b) distances from point    |
| Anchors         | B fixed boxes per cell            | Anchor-free, point-based             |
| Head            | Shared FC layers                 | Decoupled conv heads (cls + reg)     |
| Assignment      | Grid cell ownership              | Dynamic (SimOTA)                     |
| NMS             | Yes                              | Yes                                  |
| Cls loss        | MSE or CrossEntropy              | Quality Focal Loss                   |
| Box loss        | MSE on coords                    | GIoU loss                            |

==========================================================================
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =========================================================================
# Building blocks
# =========================================================================

class ConvBNSiLU(nn.Module):
    """Conv2d + BatchNorm + SiLU (Swish) activation.

    This is the basic building block used everywhere in RTMDet.

    Why SiLU instead of ReLU or LeakyReLU?
      SiLU(x) = x * sigmoid(x)  — smooth, non-monotonic activation.
      It tends to perform better in modern architectures and allows
      small negative values through, which can help gradient flow.
      It was popularized by EfficientNet and adopted widely since.

    Why BatchNorm?
      Normalizes activations across the batch, stabilizing training
      and allowing higher learning rates.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=groups, bias=False,  # No bias when using BatchNorm
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise conv + pointwise conv.

    =======================================================================
    WHY DEPTHWISE SEPARABLE?
    =======================================================================

    A standard 3×3 conv with C_in → C_out channels has:
        params = C_in × C_out × 3 × 3

    A depthwise separable conv splits this into two steps:
      1. Depthwise: one 3×3 filter per input channel (groups=C_in)
         params = C_in × 1 × 3 × 3
      2. Pointwise: 1×1 conv to mix channels
         params = C_in × C_out × 1 × 1

    Total params ≈ C_in × (9 + C_out) vs C_in × C_out × 9
    This is roughly C_out / 9 times cheaper for typical C_out values.

    RTMDet uses large-kernel (5×5) depthwise convolutions in its CSP blocks,
    which gives a larger receptive field without the cost of a large standard conv.
    =======================================================================
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
    ):
        super().__init__()
        # Step 1: Depthwise — each channel gets its own spatial filter
        # groups=in_channels means each channel is convolved independently
        self.depthwise = ConvBNSiLU(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=in_channels,  # <-- THIS makes it depthwise
        )
        # Step 2: Pointwise — 1×1 conv to mix information across channels
        self.pointwise = ConvBNSiLU(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class CSPNeXtBlock(nn.Module):
    """A single CSPNeXt block with large-kernel depthwise convolutions.

    =======================================================================
    WHAT IS A CSP (Cross Stage Partial) BLOCK?
    =======================================================================

    CSP splits the input channels into two parts:
      Part 1: Passes through a sequence of bottleneck layers
      Part 2: Skips directly (identity shortcut)
    Then both parts are concatenated and mixed.

    This design:
      - Reduces computation (only part of channels go through heavy layers)
      - Preserves gradient flow (shortcut path)
      - Encourages feature reuse

    Original CSP: split → process part1 → concat(part1, part2) → mix
    CSPNeXt: same idea but uses depthwise separable convolutions
             with large kernels (5×5) for the bottleneck.

    =======================================================================
    BLOCK STRUCTURE
    =======================================================================

    Input [C_in]
        │
        ├──── main_conv (1×1) ─── DW-Sep blocks ──── [C_out//2]
        │                                                │
        └──── short_conv (1×1) ──────────────────── [C_out//2]
                                                         │
                                                    concat [C_out]
                                                         │
                                                    final_conv (1×1)
                                                         │
                                                    Output [C_out]
    =======================================================================
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        expand_ratio: float = 0.5,
    ):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)

        # Main path: 1×1 conv to reduce channels, then N depthwise-sep blocks
        self.main_conv = ConvBNSiLU(in_channels, mid_channels, kernel_size=1, padding=0)
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                # Each sub-block: 3×3 standard conv + 5×5 depthwise separable conv
                # The 3×3 conv mixes channels, the 5×5 DW conv captures spatial context
                ConvBNSiLU(mid_channels, mid_channels, kernel_size=3, padding=1),
                DepthwiseSeparableConv(mid_channels, mid_channels, kernel_size=5, padding=2),
            )
            for _ in range(num_blocks)
        ])

        # Short path: 1×1 conv for the skip connection branch
        self.short_conv = ConvBNSiLU(in_channels, mid_channels, kernel_size=1, padding=0)

        # Final mixing: combine both paths back to desired channel count
        self.final_conv = ConvBNSiLU(mid_channels * 2, out_channels, kernel_size=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        # Main path: reduce channels → bottleneck blocks
        main = self.main_conv(x)
        main = self.blocks(main)

        # Short path: reduce channels via 1×1 conv (skip the heavy blocks)
        short = self.short_conv(x)

        # Concatenate along channel dim and mix
        return self.final_conv(torch.cat([main, short], dim=1))


# =========================================================================
# Backbone: CSPNeXt
# =========================================================================

class CSPNeXtBackbone(nn.Module):
    """CSPNeXt backbone — extracts multi-scale features from the input image.

    =======================================================================
    DESIGN PHILOSOPHY
    =======================================================================

    The backbone progressively downsamples the image while increasing
    channel depth, producing feature maps at multiple scales.

    RTMDet uses features from the last 3 stages (stride 8, 16, 32)
    because:
      - Stride 8: high resolution, good for small objects
      - Stride 16: medium resolution, good for medium objects
      - Stride 32: low resolution but rich semantics, good for large objects

    This multi-scale design is a major advantage over YOLOv1, which
    only uses a single 7×7 feature map (stride 64).

    =======================================================================
    ARCHITECTURE (Tiny version for study)
    =======================================================================

    Input: [B, 3, 320, 320]

    Stem (stride 4):
      Conv 3→32, stride=2    → [B, 32, 160, 160]
      Conv 32→64, stride=2   → [B, 64, 80, 80]

    Stage 1 (stride 8):          ← Output level P3
      Conv 64→128, stride=2  → [B, 128, 40, 40]
      CSPNeXtBlock            → [B, 128, 40, 40]

    Stage 2 (stride 16):         ← Output level P4
      Conv 128→256, stride=2 → [B, 256, 20, 20]
      CSPNeXtBlock            → [B, 256, 20, 20]

    Stage 3 (stride 32):         ← Output level P5
      Conv 256→512, stride=2 → [B, 512, 10, 10]
      CSPNeXtBlock            → [B, 512, 10, 10]

    =======================================================================
    """

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 64,
        stage_channels: Tuple[int, ...] = (128, 256, 512),
        num_blocks_per_stage: Tuple[int, ...] = (1, 1, 1),
    ):
        """
        Args:
            in_channels: Input image channels (3 for RGB).
            stem_channels: Output channels of the stem.
            stage_channels: Output channels for each of the 3 stages.
                These correspond to P3, P4, P5 feature levels.
            num_blocks_per_stage: Number of CSPNeXt blocks in each stage.
        """
        super().__init__()

        # ── Stem: rapidly downsample from input resolution ──
        # Two stride-2 convolutions: total stride = 4
        # This is more efficient than starting with stride-1 convolutions
        # on the full resolution image.
        self.stem = nn.Sequential(
            ConvBNSiLU(in_channels, stem_channels // 2, kernel_size=3, stride=2, padding=1),
            # [B, 32, H/2, W/2]
            ConvBNSiLU(stem_channels // 2, stem_channels, kernel_size=3, stride=2, padding=1),
            # [B, 64, H/4, W/4]
        )

        # ── Stages: each Stage = downsample + CSP block ──
        # Each stage doubles the stride and typically doubles the channels.
        self.stages = nn.ModuleList()
        prev_channels = stem_channels

        for i, (out_ch, n_blocks) in enumerate(
            zip(stage_channels, num_blocks_per_stage)
        ):
            stage = nn.Sequential(
                # Stride-2 conv for spatial downsampling
                ConvBNSiLU(prev_channels, out_ch, kernel_size=3, stride=2, padding=1),
                # CSPNeXt block for feature extraction
                CSPNeXtBlock(out_ch, out_ch, num_blocks=n_blocks),
            )
            self.stages.append(stage)
            prev_channels = out_ch

        # Store output channels for use by the neck
        self.out_channels = stage_channels

    def forward(self, x: Tensor) -> List[Tensor]:
        """Extract multi-scale features.

        Args:
            x: [B, 3, H, W] input image.

        Returns:
            List of 3 feature tensors at strides [8, 16, 32]:
              [0]: [B, 128, H/8,  W/8 ]  — P3, small objects
              [1]: [B, 256, H/16, W/16]  — P4, medium objects
              [2]: [B, 512, H/32, W/32]  — P5, large objects
        """
        x = self.stem(x)  # [B, 64, H/4, W/4]

        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)

        return outputs


# =========================================================================
# Neck: CSPAFPN (CSP-based Path Aggregation Feature Pyramid Network)
# =========================================================================

class CSPAFPN(nn.Module):
    """CSP-based Path Aggregation FPN for multi-scale feature fusion.

    =======================================================================
    WHY DO WE NEED A NECK?
    =======================================================================

    The backbone produces features at different scales:
      P3 (stride 8):  Good spatial detail, weak semantics
      P4 (stride 16): Medium detail, medium semantics
      P5 (stride 32): Weak spatial detail, strong semantics

    "Semantics" = high-level understanding (is this a cat? a car?)
    "Spatial detail" = fine-grained localization (exact edges)

    The neck FUSES features across scales so that each level gets BOTH
    good spatial detail AND strong semantics.

    =======================================================================
    FPN vs PAN vs PAFPN
    =======================================================================

    FPN (Feature Pyramid Network):
      Top-down only: P5 → P4 → P3
      High-level semantics flow DOWN to lower levels.
      Problem: low-level detail doesn't flow UP.

    PAN (Path Aggregation Network):
      Adds a bottom-up path: P3 → P4 → P5
      Now detail also flows UP.

    PAFPN = FPN + PAN (both directions):
      Top-down:  P5 → P4 → P3     (semantics flow down)
      Bottom-up: P3 → P4 → P5     (detail flows up)

    RTMDet uses CSP blocks in the PAFPN for efficiency.

    =======================================================================
    DATAFLOW (our tiny version with 3 levels)
    =======================================================================

    From backbone:
      P3 [128, 40, 40]   P4 [256, 20, 20]   P5 [512, 10, 10]

    Step 1: Reduce all to same channel count via 1×1 convs
      P3 [128, 40, 40]   P4 [128, 20, 20]   P5 [128, 10, 10]

    Step 2: Top-down path (high-level semantics flow to lower levels)
      P5 ──upsample──► fuse with P4 ──upsample──► fuse with P3
      Result: N3 [128, 40, 40]   N4 [128, 20, 20]

    Step 3: Bottom-up path (spatial detail flows to higher levels)
      N3 ──downsample──► fuse with N4 ──downsample──► fuse with P5
      Result: O3 [128, 40, 40]   O4 [128, 20, 20]   O5 [128, 10, 10]

    =======================================================================
    """

    def __init__(
        self,
        in_channels: Tuple[int, ...] = (128, 256, 512),
        out_channels: int = 128,
        num_csp_blocks: int = 1,
    ):
        """
        Args:
            in_channels: Channel counts from backbone for each level.
            out_channels: Uniform channel count after the neck.
            num_csp_blocks: Number of CSP blocks at each fusion point.
        """
        super().__init__()
        self.out_channels = out_channels
        num_levels = len(in_channels)

        # ── Lateral convs: reduce backbone channels to uniform out_channels ──
        # These 1×1 convs align all feature levels to the same channel count
        # so they can be added/concatenated during fusion.
        self.reduce_convs = nn.ModuleList([
            ConvBNSiLU(in_ch, out_channels, kernel_size=1, padding=0)
            for in_ch in in_channels
        ])

        # ── Top-down path (P5 → P4 → P3) ──
        # For each level except the highest, we have:
        #   1. Upsample the higher-level feature
        #   2. Concatenate with the current level
        #   3. CSP block to fuse the concatenation
        self.top_down_blocks = nn.ModuleList([
            CSPNeXtBlock(out_channels * 2, out_channels, num_blocks=num_csp_blocks)
            for _ in range(num_levels - 1)
        ])

        # ── Bottom-up path (P3 → P4 → P5) ──
        # For each level except the lowest, we have:
        #   1. Downsample the lower-level feature (stride-2 conv)
        #   2. Concatenate with the current level
        #   3. CSP block to fuse the concatenation
        self.downsample_convs = nn.ModuleList([
            ConvBNSiLU(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            for _ in range(num_levels - 1)
        ])
        self.bottom_up_blocks = nn.ModuleList([
            CSPNeXtBlock(out_channels * 2, out_channels, num_blocks=num_csp_blocks)
            for _ in range(num_levels - 1)
        ])

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """Fuse multi-scale features through top-down + bottom-up paths.

        Args:
            features: List of 3 tensors from backbone:
              [0]: [B, C3, H/8,  W/8 ]
              [1]: [B, C4, H/16, W/16]
              [2]: [B, C5, H/32, W/32]

        Returns:
            List of 3 fused tensors, all with out_channels channels:
              [0]: [B, out_ch, H/8,  W/8 ]
              [1]: [B, out_ch, H/16, W/16]
              [2]: [B, out_ch, H/32, W/32]
        """
        # Step 1: Reduce channel counts to uniform out_channels
        reduced = [
            self.reduce_convs[i](features[i])
            for i in range(len(features))
        ]
        # reduced[0]: [B, 128, 40, 40]
        # reduced[1]: [B, 128, 20, 20]
        # reduced[2]: [B, 128, 10, 10]

        # Step 2: Top-down path
        # Process from highest level (smallest spatial) to lowest (largest spatial)
        # P5 is kept as-is, then we fuse downward.
        top_down_feats = [None] * len(reduced)
        top_down_feats[-1] = reduced[-1]  # Start with P5 (unchanged)

        for i in range(len(reduced) - 2, -1, -1):
            # Upsample the feature from one level above
            # F.interpolate doubles the spatial resolution to match the current level
            upsampled = F.interpolate(
                top_down_feats[i + 1],
                size=reduced[i].shape[2:],  # Match spatial size of current level
                mode="nearest",  # Nearest-neighbor upsampling (fast, no learnable params)
            )
            # Concatenate with current level's reduced features
            # This combines high-level semantics (from above) with local detail (current)
            fused = torch.cat([reduced[i], upsampled], dim=1)  # [B, 2*out_ch, H, W]
            # CSP block to learn the fusion
            top_down_feats[i] = self.top_down_blocks[i](fused)  # [B, out_ch, H, W]

        # Step 3: Bottom-up path
        # Process from lowest level (largest spatial) to highest (smallest spatial)
        outputs = [None] * len(reduced)
        outputs[0] = top_down_feats[0]  # Start with the finest level

        for i in range(1, len(reduced)):
            # Downsample the feature from one level below
            # Stride-2 conv halves the spatial resolution
            downsampled = self.downsample_convs[i - 1](outputs[i - 1])
            # Concatenate with the current level from top-down path
            fused = torch.cat([top_down_feats[i], downsampled], dim=1)
            # CSP block to learn the fusion
            outputs[i] = self.bottom_up_blocks[i - 1](fused)

        return outputs


# =========================================================================
# Detection Head: RTMDetHead
# =========================================================================

class RTMDetHead(nn.Module):
    """Decoupled detection head for classification and box regression.

    =======================================================================
    DECOUPLED HEAD DESIGN
    =======================================================================

    Early detectors (YOLOv1, RetinaNet) used a SHARED head that predicts
    both class and box from the same features. This is suboptimal because:

      - Classification needs features that distinguish object categories
        (texture, color, shape patterns)
      - Box regression needs features that precisely locate edges
        (spatial gradients, boundary information)

    YOLOX and RTMDet use a DECOUPLED head: two separate branches that
    each specialize in their own task:

      Shared stem → cls_branch → classification scores
                  → reg_branch → box regression (ltrb distances)

    =======================================================================
    ANCHOR-FREE POINT-BASED PREDICTION
    =======================================================================

    Unlike anchor-based detectors (e.g., RetinaNet, original YOLO with
    predefined anchor boxes), RTMDet is anchor-free:

    - Each spatial location on the feature map is a "point"
    - Each point has a known position on the image (the center of its
      receptive field)
    - The model predicts (l, t, r, b) distances from that point to each
      edge of the bounding box

    This eliminates the need for hand-designed anchor boxes and makes
    the model simpler and more flexible.

    Grid points at stride 8 with 320×320 input:
      Feature map size = 40×40
      Point positions = (4, 4), (12, 4), (20, 4), ..., (316, 316)
      (Start at stride/2, step by stride)

    =======================================================================
    ARCHITECTURE (per level, weights shared across levels)
    =======================================================================

    Feature [B, C, H, W]
         │
         ▼
    Shared stem: ConvBNSiLU [B, C, H, W]
         │
         ├── cls_branch: 2× ConvBNSiLU → Conv → [B, num_classes, H, W]
         │
         └── reg_branch: 2× ConvBNSiLU → Conv → [B, 4, H, W]

    =======================================================================
    """

    def __init__(
        self,
        in_channels: int = 128,
        feat_channels: int = 128,
        num_classes: int = 20,
        stacked_convs: int = 2,
        strides: Tuple[int, ...] = (8, 16, 32),
    ):
        """
        Args:
            in_channels: Input feature channels (from neck).
            feat_channels: Hidden channels in the head branches.
            num_classes: Number of object classes.
            stacked_convs: Number of conv layers in each branch.
            strides: Feature map strides for each level.
                Used to generate point coordinates on the image.
        """
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        # Shared stem: light processing before branching
        self.shared_stem = ConvBNSiLU(in_channels, feat_channels, kernel_size=3, padding=1)

        # Classification branch: predicts class scores
        cls_layers = []
        for _ in range(stacked_convs):
            cls_layers.append(
                ConvBNSiLU(feat_channels, feat_channels, kernel_size=3, padding=1)
            )
        self.cls_branch = nn.Sequential(*cls_layers)
        # Final classification conv: outputs raw logits for each class
        # Why no activation? Loss function (BCEWithLogitsLoss / QFL) expects logits
        self.cls_pred = nn.Conv2d(feat_channels, num_classes, kernel_size=1)

        # Regression branch: predicts (left, top, right, bottom) distances
        reg_layers = []
        for _ in range(stacked_convs):
            reg_layers.append(
                ConvBNSiLU(feat_channels, feat_channels, kernel_size=3, padding=1)
            )
        self.reg_branch = nn.Sequential(*reg_layers)
        # Final regression conv: outputs 4 distance values per point
        self.reg_pred = nn.Conv2d(feat_channels, 4, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with best practices for detection heads.

        Classification bias is set to a negative value so that the model
        starts by predicting "no object" for all locations. This prevents
        the massive positive loss at the start of training when most
        locations are background.

        Formula: bias = -log((1 - prior) / prior)
        With prior = 0.01: bias ≈ -4.6

        This means sigmoid(-4.6) ≈ 0.01, so initial predictions are ~1%
        probability of being any class — a reasonable starting point.
        """
        import math
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Focal/quality loss bias trick
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)

    def forward(
        self, features: List[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Run the detection head on multi-scale features.

        The SAME head weights are applied to ALL feature levels.
        This is called "weight sharing" and helps because:
          - Fewer parameters (don't need separate heads per scale)
          - Each level sees the same learned patterns
          - The head learns to handle different scales naturally

        Args:
            features: List of L tensors, each [B, C, H_l, W_l].

        Returns:
            cls_scores: List of L tensors, each [B, num_classes, H_l, W_l].
                Raw logits (no sigmoid applied yet).
            bbox_preds: List of L tensors, each [B, 4, H_l, W_l].
                Predicted (left, top, right, bottom) distances.
                ReLU is applied to ensure distances are non-negative.
        """
        cls_scores = []
        bbox_preds = []

        for feat in features:
            # Shared stem processing
            shared = self.shared_stem(feat)  # [B, C, H, W]

            # Classification branch
            cls_feat = self.cls_branch(shared)
            cls_score = self.cls_pred(cls_feat)  # [B, num_classes, H, W]
            cls_scores.append(cls_score)

            # Regression branch
            reg_feat = self.reg_branch(shared)
            bbox_pred = self.reg_pred(reg_feat)  # [B, 4, H, W]
            # ReLU ensures predicted distances are non-negative
            # (a box edge can't be behind the prediction point)
            bbox_pred = F.relu(bbox_pred)
            bbox_preds.append(bbox_pred)

        return cls_scores, bbox_preds

    def get_points(
        self,
        featmap_sizes: List[Tuple[int, int]],
        device: torch.device,
    ) -> List[Tensor]:
        """Generate grid point coordinates for the feature maps.

        Each point represents the center of a "cell" on the feature map,
        mapped back to image coordinates using the stride.

        For stride=8, feature map 40×40 from input 320×320:
          Points: (4, 4), (12, 4), (20, 4), ..., (316, 316)
          Spacing: 8 pixels
          First point: at stride/2 = 4 (center of the first cell)

        Why stride/2?
          Each feature map cell maps to a stride×stride region on the image.
          The "point" is at the center of that region.
          For stride=8, the first cell covers pixels [0, 8), center = 4.

        Args:
            featmap_sizes: List of (H, W) for each feature level.
            device: Device for the tensors.

        Returns:
            List of tensors, each [H*W, 2] with (x, y) image coordinates.
        """
        all_points = []
        for (h, w), stride in zip(featmap_sizes, self.strides):
            # Generate grid of (x, y) coordinates
            # shift_x: [0, stride, 2*stride, ...] + stride/2
            shift_x = (torch.arange(w, device=device) + 0.5) * stride
            shift_y = (torch.arange(h, device=device) + 0.5) * stride

            # meshgrid creates all combinations of x and y
            grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            # Flatten to [H*W] and stack to [H*W, 2]
            points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
            all_points.append(points)

        return all_points


# =========================================================================
# Full RTMDet Model
# =========================================================================

class RTMDet(nn.Module):
    """Complete RTMDet detection model: backbone + neck + head.

    =======================================================================
    FULL PIPELINE
    =======================================================================

    Training:
        image → backbone → neck → head → (cls_scores, bbox_preds)
        These are passed to RTMDetLoss along with ground truth.

    Inference:
        image → backbone → neck → head → decode → NMS → final detections

    =======================================================================
    """

    def __init__(
        self,
        num_classes: int = 20,
        input_size: int = 320,
        backbone_channels: Tuple[int, ...] = (128, 256, 512),
        neck_out_channels: int = 128,
        strides: Tuple[int, ...] = (8, 16, 32),
    ):
        """
        Args:
            num_classes: Number of object classes.
            input_size: Expected input image size (square).
            backbone_channels: Output channels for each backbone stage.
            neck_out_channels: Channel count after FPN fusion.
            strides: Feature map strides.
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.strides = strides

        # ── Backbone: extracts multi-scale features ──
        self.backbone = CSPNeXtBackbone(
            in_channels=3,
            stem_channels=64,
            stage_channels=backbone_channels,
        )

        # ── Neck: fuses features across scales ──
        self.neck = CSPAFPN(
            in_channels=backbone_channels,
            out_channels=neck_out_channels,
        )

        # ── Head: predicts classes and boxes ──
        self.head = RTMDetHead(
            in_channels=neck_out_channels,
            feat_channels=neck_out_channels,
            num_classes=num_classes,
            strides=strides,
        )

    def forward(
        self, x: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Full forward pass.

        Args:
            x: [B, 3, H, W] input images.

        Returns:
            cls_scores: List of [B, num_classes, H_l, W_l] per level.
            bbox_preds: List of [B, 4, H_l, W_l] per level.
        """
        # Extract multi-scale features from backbone
        features = self.backbone(x)
        # features: [[B,128,40,40], [B,256,20,20], [B,512,10,10]]

        # Fuse features in the neck
        fused = self.neck(features)
        # fused: [[B,128,40,40], [B,128,20,20], [B,128,10,10]]

        # Predict classes and boxes
        cls_scores, bbox_preds = self.head(fused)
        # cls_scores: [[B,C,40,40], [B,C,20,20], [B,C,10,10]]
        # bbox_preds: [[B,4,40,40], [B,4,20,20], [B,4,10,10]]

        return cls_scores, bbox_preds

    @torch.no_grad()
    def decode_predictions(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        score_threshold: float = 0.3,
        nms_iou_threshold: float = 0.5,
    ) -> List[List[dict]]:
        """Decode raw predictions into final detections with NMS.

        =======================================================================
        POST-PROCESSING PIPELINE
        =======================================================================

        For each image:
          1. Generate point coordinates for each feature level
          2. Convert (point + ltrb distances) → (x1, y1, x2, y2) boxes
          3. Apply sigmoid to get class probabilities
          4. Filter by score threshold
          5. Apply class-wise NMS to remove duplicate detections

        =======================================================================

        Args:
            cls_scores: List of [B, num_classes, H_l, W_l] per level.
            bbox_preds: List of [B, 4, H_l, W_l] per level.
            score_threshold: Minimum confidence to keep.
            nms_iou_threshold: IoU threshold for NMS.

        Returns:
            List of detection lists per image. Each detection is a dict:
              "box": [x1, y1, x2, y2]
              "confidence": float
              "class_id": int
        """
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]

        # Get feature map sizes and generate grid points
        featmap_sizes = [(s.shape[2], s.shape[3]) for s in cls_scores]
        all_points = self.head.get_points(featmap_sizes, device)

        all_detections = []

        for b_idx in range(batch_size):
            all_boxes = []
            all_scores = []
            all_class_ids = []

            for lvl, (cls_score, bbox_pred, points, stride) in enumerate(
                zip(cls_scores, bbox_preds, all_points, self.strides)
            ):
                # Get predictions for this image at this level
                # cls_score[b_idx]: [num_classes, H, W] → reshape to [H*W, num_classes]
                cls = cls_score[b_idx].permute(1, 2, 0).reshape(-1, self.num_classes)
                cls = cls.sigmoid()  # Convert logits to probabilities

                # bbox_pred[b_idx]: [4, H, W] → reshape to [H*W, 4]
                reg = bbox_pred[b_idx].permute(1, 2, 0).reshape(-1, 4)

                # Scale distances by stride
                # The model predicts distances in feature-map units.
                # Multiply by stride to get image-pixel distances.
                reg = reg * stride

                # Get max class score and class ID per point
                max_scores, class_ids = cls.max(dim=-1)  # [H*W]

                # Filter by score threshold
                keep = max_scores > score_threshold
                if keep.sum() == 0:
                    continue

                kept_scores = max_scores[keep]
                kept_class_ids = class_ids[keep]
                kept_reg = reg[keep]
                kept_points = points[keep]

                # Decode ltrb distances to xyxy boxes
                from .box_ops import distance2bbox
                kept_boxes = distance2bbox(kept_points, kept_reg)

                all_boxes.append(kept_boxes)
                all_scores.append(kept_scores)
                all_class_ids.append(kept_class_ids)

            if len(all_boxes) == 0:
                all_detections.append([])
                continue

            # Concatenate predictions from all levels
            all_boxes = torch.cat(all_boxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            all_class_ids = torch.cat(all_class_ids, dim=0)

            # Apply class-wise NMS
            # NMS removes overlapping boxes that likely detect the same object.
            # We do it per-class to avoid suppressing different-class detections.
            final_detections = []
            for cls_id in all_class_ids.unique():
                cls_mask = all_class_ids == cls_id
                cls_boxes = all_boxes[cls_mask]
                cls_scores_c = all_scores[cls_mask]

                # Simple NMS implementation
                keep_indices = self._nms(cls_boxes, cls_scores_c, nms_iou_threshold)

                for idx in keep_indices:
                    final_detections.append({
                        "box": cls_boxes[idx].cpu().tolist(),
                        "confidence": cls_scores_c[idx].item(),
                        "class_id": cls_id.item(),
                    })

            # Sort by confidence (highest first)
            final_detections.sort(key=lambda d: d["confidence"], reverse=True)
            all_detections.append(final_detections)

        return all_detections

    @staticmethod
    def _nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> List[int]:
        """Greedy Non-Maximum Suppression.

        NMS removes redundant detections for the same object:
          1. Pick the box with the highest score
          2. Remove all other boxes that overlap with it above iou_threshold
          3. Repeat until no boxes remain

        Args:
            boxes: [N, 4] in xyxy format.
            scores: [N] confidence scores.
            iou_threshold: Overlapping boxes above this threshold are suppressed.

        Returns:
            List of kept indices.
        """
        if boxes.numel() == 0:
            return []

        # Sort by score descending
        order = scores.argsort(descending=True)
        keep = []

        while order.numel() > 0:
            # Pick the highest-scoring box
            i = order[0].item()
            keep.append(i)

            if order.numel() == 1:
                break

            # Compute IoU of this box with all remaining boxes
            remaining = order[1:]
            from .box_ops import box_iou_flat
            ious = box_iou_flat(
                boxes[i].unsqueeze(0).expand(remaining.numel(), -1),
                boxes[remaining],
            )

            # Keep only boxes with IoU below threshold
            mask = ious < iou_threshold
            order = remaining[mask]

        return keep

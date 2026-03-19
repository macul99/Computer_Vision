"""A minimal Faster R-CNN implementation in PyTorch.

This is intentionally written for learning:
- clear modules,
- explicit tensor shape comments,
- verbose training target assignment,
- minimal hidden magic.

It is not optimized for production speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import nms, roi_align

from .box_ops import (
    box_iou,
    clip_boxes_to_image,
    decode_boxes,
    encode_boxes,
    remove_small_boxes,
)


@dataclass
class RPNConfig:
    """Hyperparameters for the Region Proposal Network (RPN)."""

    pre_nms_topk_train: int = 1200
    post_nms_topk_train: int = 300
    pre_nms_topk_test: int = 600
    post_nms_topk_test: int = 100
    nms_thresh: float = 0.7
    min_size: float = 4.0
    fg_iou_thresh: float = 0.7
    bg_iou_thresh: float = 0.3
    batch_size_per_image: int = 256
    positive_fraction: float = 0.5


@dataclass
class ROIConfig:
    """Hyperparameters for ROI stage."""

    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    detections_per_img: int = 100
    fg_iou_thresh: float = 0.5
    bg_iou_thresh: float = 0.5
    batch_size_per_image: int = 128
    positive_fraction: float = 0.25
    roi_output_size: int = 7


class TinyBackbone(nn.Module):
    """A very small CNN backbone that outputs one feature map.

    Stride is 16 overall (four downsampling stages by factor 2).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class AnchorGenerator(nn.Module):
    """Generate anchors for each spatial position on the feature map.

    With one feature level and K anchor templates per location:
    total anchors per image = H_feat * W_feat * K.
    """

    def __init__(
        self,
        sizes: Tuple[int, ...] = (32, 64, 128),
        aspect_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
        stride: int = 16,
    ):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.stride = stride

        base_anchors = self._generate_base_anchors()
        self.register_buffer("base_anchors", base_anchors, persistent=False)

    def _generate_base_anchors(self) -> Tensor:
        """Create anchor templates centered at (0, 0)."""
        anchors = []
        for size in self.sizes:
            for ratio in self.aspect_ratios:
                # Solve for width/height given area=size^2 and h/w=ratio.
                w = size / (ratio**0.5)
                h = size * (ratio**0.5)
                anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.tensor(anchors, dtype=torch.float32)

    @property
    def num_anchors_per_location(self) -> int:
        return len(self.sizes) * len(self.aspect_ratios)

    def forward(self, feature: Tensor) -> Tensor:
        """Generate all anchors for the given feature map.

        Args:
            feature: [B, C, Hf, Wf]
        Returns:
            anchors: [Hf*Wf*K, 4] shared across images if image sizes match.
        """
        _, _, h_feat, w_feat = feature.shape
        device = feature.device
        dtype = feature.dtype

        # Feature cell centers mapped to image coordinates.
        shifts_x = (torch.arange(w_feat, device=device, dtype=dtype) + 0.5) * self.stride
        shifts_y = (torch.arange(h_feat, device=device, dtype=dtype) + 0.5) * self.stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        # [Hf*Wf, 4], each row is [cx, cy, cx, cy].
        shifts = torch.stack((shift_x.reshape(-1), shift_y.reshape(-1), shift_x.reshape(-1), shift_y.reshape(-1)), dim=1)

        # Broadcast: each shift gets all K base anchors.
        anchors = shifts[:, None, :] + self.base_anchors.to(device=device, dtype=dtype)[None, :, :]
        return anchors.reshape(-1, 4)


class RPNHead(nn.Module):
    """RPN conv head that predicts objectness + box deltas for each anchor."""

    def __init__(self, in_channels: int, num_anchors_per_location: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors_per_location, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors_per_location * 4, 1)

        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, feat: Tensor) -> Tuple[Tensor, Tensor]:
        # feat: [B, C, Hf, Wf]
        t = F.relu(self.conv(feat))
        logits = self.cls_logits(t)  # [B, K, Hf, Wf]
        deltas = self.bbox_pred(t)   # [B, 4K, Hf, Wf]
        return logits, deltas


class FastRCNNHead(nn.Module):
    """ROI feature head + final classifier and box regressor."""

    def __init__(self, in_channels: int, num_classes: int, roi_size: int = 7):
        super().__init__()
        hidden = 1024
        flat_dim = in_channels * roi_size * roi_size
        self.fc1 = nn.Linear(flat_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.cls_score = nn.Linear(hidden, num_classes)  # includes background class 0
        self.bbox_pred = nn.Linear(hidden, num_classes * 4)

        for layer in [self.fc1, self.fc2, self.cls_score, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x: [R, C, 7, 7] where R is total sampled RoIs across batch.
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        cls_logits = self.cls_score(x)       # [R, num_classes]
        bbox_deltas = self.bbox_pred(x)      # [R, num_classes*4]
        return cls_logits, bbox_deltas


class MinimalFasterRCNN(nn.Module):
    """End-to-end minimal Faster R-CNN.

    Expected targets format during training:
      targets[i] = {
        "boxes": FloatTensor [Ni, 4] in xyxy, absolute image coordinates,
        "labels": LongTensor [Ni], class ids in [1, num_classes-1]
      }
    Class id 0 is reserved for background.
    """

    def __init__(
        self,
        num_classes: int,
        image_size: Tuple[int, int] = (512, 512),
        backbone_out_channels: int = 256,
        rpn_cfg: Optional[RPNConfig] = None,
        roi_cfg: Optional[ROIConfig] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        self.backbone = TinyBackbone(out_channels=backbone_out_channels)
        self.anchor_gen = AnchorGenerator(stride=16)
        self.rpn_head = RPNHead(backbone_out_channels, self.anchor_gen.num_anchors_per_location)
        self.rpn_cfg = rpn_cfg or RPNConfig()
        self.roi_cfg = roi_cfg or ROIConfig()
        self.roi_head = FastRCNNHead(
            in_channels=backbone_out_channels,
            num_classes=num_classes,
            roi_size=self.roi_cfg.roi_output_size,
        )

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Dict[str, Tensor] | List[Dict[str, Tensor]]:
        if self.training and targets is None:
            raise ValueError("targets must be provided in training mode")

        # For a "minimum" implementation, we keep one fixed image size.
        # This lets us stack directly and keep anchor math easy to read.
        images_tensor = torch.stack(images, dim=0)  # [B, 3, H, W]
        bsz, _, h_img, w_img = images_tensor.shape

        feat = self.backbone(images_tensor)  # [B, C, Hf, Wf]

        anchors = self.anchor_gen(feat)  # [A, 4], A=Hf*Wf*K

        rpn_logits_map, rpn_deltas_map = self.rpn_head(feat)
        # Flatten predictions so axis aligns with anchors:
        # objectness: [B, A], deltas: [B, A, 4]
        objectness, rpn_deltas = self._reshape_rpn_outputs(rpn_logits_map, rpn_deltas_map)

        proposals, rpn_losses = self._rpn_forward(
            objectness=objectness,
            rpn_deltas=rpn_deltas,
            anchors=anchors,
            targets=targets,
            image_size=(h_img, w_img),
        )

        if self.training:
            assert targets is not None
            roi_losses = self._roi_forward_train(feat, proposals, targets, (h_img, w_img))
            losses = {}
            losses.update(rpn_losses)
            losses.update(roi_losses)
            return losses

        detections = self._roi_forward_infer(feat, proposals, (h_img, w_img))
        return detections

    def _reshape_rpn_outputs(self, logits: Tensor, deltas: Tensor) -> Tuple[Tensor, Tensor]:
        """Convert RPN maps to per-anchor tensors.

        logits: [B, K, Hf, Wf] -> [B, A]
        deltas: [B, 4K, Hf, Wf] -> [B, A, 4]
        """
        bsz, k, h_feat, w_feat = logits.shape

        # Objectness layout conversion:
        # [B, K, Hf, Wf] -> [B, Hf, Wf, K] -> [B, A]
        objectness = logits.permute(0, 2, 3, 1).reshape(bsz, -1)

        # Box delta layout conversion:
        # [B, 4K, Hf, Wf] -> [B, Hf, Wf, K, 4] -> [B, A, 4]
        deltas = deltas.view(bsz, k, 4, h_feat, w_feat)
        deltas = deltas.permute(0, 3, 4, 1, 2).reshape(bsz, -1, 4)
        return objectness, deltas

    def _rpn_forward(
        self,
        objectness: Tensor,
        rpn_deltas: Tensor,
        anchors: Tensor,
        targets: Optional[List[Dict[str, Tensor]]],
        image_size: Tuple[int, int],
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        bsz = objectness.shape[0]
        device = objectness.device

        # Decode anchors + predicted deltas into proposal boxes.
        proposals = []
        for i in range(bsz):
            props_i = decode_boxes(rpn_deltas[i], anchors)
            props_i = clip_boxes_to_image(props_i, image_size[0], image_size[1])
            proposals.append(props_i)

        # Convert objectness logits to probabilities for ranking proposals.
        scores = torch.sigmoid(objectness)

        # NMS and top-k selection per image.
        final_props = []
        for i in range(bsz):
            final_props.append(self._filter_proposals(proposals[i], scores[i], image_size))

        losses: Dict[str, Tensor] = {}
        if self.training:
            assert targets is not None
            obj_loss, reg_loss = self._rpn_losses(objectness, rpn_deltas, anchors, targets)
            losses["loss_rpn_objectness"] = obj_loss
            losses["loss_rpn_box_reg"] = reg_loss

        return final_props, losses

    def _filter_proposals(self, boxes: Tensor, scores: Tensor, image_size: Tuple[int, int]) -> Tensor:
        """Keep strongest, valid proposals with NMS."""
        cfg = self.rpn_cfg
        pre_nms_topk = cfg.pre_nms_topk_train if self.training else cfg.pre_nms_topk_test
        post_nms_topk = cfg.post_nms_topk_train if self.training else cfg.post_nms_topk_test

        num_topk = min(pre_nms_topk, boxes.shape[0])
        topk_scores, topk_idx = scores.topk(num_topk)
        boxes = boxes[topk_idx]

        keep = remove_small_boxes(boxes, cfg.min_size)
        boxes = boxes[keep]
        topk_scores = topk_scores[keep]

        keep = nms(boxes, topk_scores, cfg.nms_thresh)
        keep = keep[:post_nms_topk]
        return boxes[keep]

    def _rpn_losses(
        self,
        objectness: Tensor,
        rpn_deltas: Tensor,
        anchors: Tensor,
        targets: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor]:
        """Compute RPN classification and regression losses."""
        cfg = self.rpn_cfg
        bsz = objectness.shape[0]

        all_obj_losses = []
        all_reg_losses = []

        for i in range(bsz):
            gt_boxes = targets[i]["boxes"]

            # Match each anchor to best GT by IoU.
            match_labels, matched_gt_idx = self._match_anchors_to_gt(
                anchors,
                gt_boxes,
                fg_thresh=cfg.fg_iou_thresh,
                bg_thresh=cfg.bg_iou_thresh,
            )

            sampled_idx = self._sample_labels(
                labels=match_labels,
                batch_size=cfg.batch_size_per_image,
                positive_fraction=cfg.positive_fraction,
            )

            sampled_labels = match_labels[sampled_idx]
            sampled_logits = objectness[i, sampled_idx]

            # Binary objectness target: positive=1, negative=0.
            obj_target = (sampled_labels == 1).float()
            obj_loss = F.binary_cross_entropy_with_logits(sampled_logits, obj_target)
            all_obj_losses.append(obj_loss)

            pos_idx = torch.where(match_labels == 1)[0]
            if len(pos_idx) == 0:
                # Keep graph connected by creating a differentiable zero.
                all_reg_losses.append(rpn_deltas[i].sum() * 0.0)
                continue

            matched_gt = gt_boxes[matched_gt_idx[pos_idx]]
            reg_target = encode_boxes(matched_gt, anchors[pos_idx])
            reg_pred = rpn_deltas[i, pos_idx]

            reg_loss = F.smooth_l1_loss(reg_pred, reg_target, beta=1.0 / 9, reduction="sum")
            reg_loss = reg_loss / max(1, cfg.batch_size_per_image)
            all_reg_losses.append(reg_loss)

        return torch.stack(all_obj_losses).mean(), torch.stack(all_reg_losses).mean()

    def _roi_forward_train(
        self,
        feat: Tensor,
        proposals: List[Tensor],
        targets: List[Dict[str, Tensor]],
        image_size: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        """Train ROI stage by sampling proposals and computing Fast R-CNN losses."""
        cfg = self.roi_cfg
        device = feat.device

        sampled_props: List[Tensor] = []
        sampled_labels: List[Tensor] = []
        sampled_reg_targets: List[Tensor] = []

        for props, tgt in zip(proposals, targets):
            gt_boxes = tgt["boxes"]
            gt_labels = tgt["labels"]

            # Add GT boxes to proposals as in the original paper;
            # this stabilizes early training by guaranteeing positives.
            props = torch.cat([props, gt_boxes], dim=0)

            labels, matched_gt_idx = self._match_proposals_to_gt(
                props,
                gt_boxes,
                gt_labels,
                fg_thresh=cfg.fg_iou_thresh,
                bg_thresh=cfg.bg_iou_thresh,
            )

            sample_idx = self._sample_labels(
                labels,
                batch_size=cfg.batch_size_per_image,
                positive_fraction=cfg.positive_fraction,
            )

            props = props[sample_idx]
            labels = labels[sample_idx]
            matched_gt_idx = matched_gt_idx[sample_idx]

            # Regression target is defined for foreground samples only.
            reg_targets = torch.zeros((props.shape[0], 4), device=device, dtype=props.dtype)
            fg = labels > 0
            if fg.any():
                reg_targets[fg] = encode_boxes(gt_boxes[matched_gt_idx[fg]], props[fg])

            sampled_props.append(props)
            sampled_labels.append(labels)
            sampled_reg_targets.append(reg_targets)

        # Build RoI tensor for roi_align: [R, 5] with (batch_idx, x1, y1, x2, y2)
        rois = []
        for batch_idx, props in enumerate(sampled_props):
            batch_col = torch.full((props.shape[0], 1), float(batch_idx), device=device, dtype=props.dtype)
            rois.append(torch.cat([batch_col, props], dim=1))
        rois = torch.cat(rois, dim=0)

        # Because our feature map has stride 16, convert image coordinates to
        # feature coordinates by spatial_scale=1/16.
        pooled = roi_align(
            feat,
            rois,
            output_size=(cfg.roi_output_size, cfg.roi_output_size),
            spatial_scale=1.0 / 16.0,
            sampling_ratio=2,
            aligned=True,
        )

        cls_logits, bbox_deltas = self.roi_head(pooled)

        labels_all = torch.cat(sampled_labels, dim=0)
        reg_targets_all = torch.cat(sampled_reg_targets, dim=0)

        cls_loss = F.cross_entropy(cls_logits, labels_all)

        # bbox_deltas has class-specific box outputs: [R, C*4].
        # For each foreground RoI, pick the 4 numbers for its target class.
        bbox_deltas = bbox_deltas.view(-1, self.num_classes, 4)
        fg_idx = torch.where(labels_all > 0)[0]

        if len(fg_idx) == 0:
            box_loss = bbox_deltas.sum() * 0.0
        else:
            fg_labels = labels_all[fg_idx]
            pred = bbox_deltas[fg_idx, fg_labels]
            target = reg_targets_all[fg_idx]
            box_loss = F.smooth_l1_loss(pred, target, beta=1.0, reduction="sum")
            box_loss = box_loss / max(1, labels_all.numel())

        return {
            "loss_roi_classifier": cls_loss,
            "loss_roi_box_reg": box_loss,
        }

    @torch.no_grad()
    def _roi_forward_infer(
        self,
        feat: Tensor,
        proposals: List[Tensor],
        image_size: Tuple[int, int],
    ) -> List[Dict[str, Tensor]]:
        """Inference for Fast R-CNN head: class scores + box decoding + per-class NMS."""
        cfg = self.roi_cfg
        device = feat.device

        # Pack all proposals across batch for one roi_align call.
        rois = []
        counts = []
        for batch_idx, props in enumerate(proposals):
            counts.append(props.shape[0])
            batch_col = torch.full((props.shape[0], 1), float(batch_idx), device=device, dtype=props.dtype)
            rois.append(torch.cat([batch_col, props], dim=1))

        rois = torch.cat(rois, dim=0)

        pooled = roi_align(
            feat,
            rois,
            output_size=(cfg.roi_output_size, cfg.roi_output_size),
            spatial_scale=1.0 / 16.0,
            sampling_ratio=2,
            aligned=True,
        )

        cls_logits, bbox_deltas = self.roi_head(pooled)
        probs = F.softmax(cls_logits, dim=1)  # [R, C]
        bbox_deltas = bbox_deltas.view(-1, self.num_classes, 4)

        detections: List[Dict[str, Tensor]] = []
        start = 0
        for num in counts:
            end = start + num

            props_i = rois[start:end, 1:]
            probs_i = probs[start:end]
            deltas_i = bbox_deltas[start:end]

            boxes_all = []
            scores_all = []
            labels_all = []

            # Class 0 is background; start from class 1.
            for cls in range(1, self.num_classes):
                scores = probs_i[:, cls]
                keep = torch.where(scores > cfg.score_thresh)[0]
                if keep.numel() == 0:
                    continue

                scores = scores[keep]
                cls_deltas = deltas_i[keep, cls]
                cls_props = props_i[keep]

                boxes = decode_boxes(cls_deltas, cls_props)
                boxes = clip_boxes_to_image(boxes, image_size[0], image_size[1])

                keep_nms = nms(boxes, scores, cfg.nms_thresh)
                boxes = boxes[keep_nms]
                scores = scores[keep_nms]
                labels = torch.full((boxes.shape[0],), cls, dtype=torch.long, device=device)

                boxes_all.append(boxes)
                scores_all.append(scores)
                labels_all.append(labels)

            if len(boxes_all) == 0:
                detections.append(
                    {
                        "boxes": torch.empty((0, 4), device=device),
                        "scores": torch.empty((0,), device=device),
                        "labels": torch.empty((0,), dtype=torch.long, device=device),
                    }
                )
            else:
                boxes = torch.cat(boxes_all, dim=0)
                scores = torch.cat(scores_all, dim=0)
                labels = torch.cat(labels_all, dim=0)

                # Final global top-k detections.
                keep = scores.argsort(descending=True)[: cfg.detections_per_img]
                detections.append(
                    {
                        "boxes": boxes[keep],
                        "scores": scores[keep],
                        "labels": labels[keep],
                    }
                )

            start = end

        return detections

    def _match_anchors_to_gt(
        self,
        anchors: Tensor,
        gt_boxes: Tensor,
        fg_thresh: float,
        bg_thresh: float,
    ) -> Tuple[Tensor, Tensor]:
        """Match anchors to GT and return labels + GT indices.

        labels:
          1 = foreground,
          0 = background,
         -1 = ignore.
        """
        device = anchors.device

        if gt_boxes.numel() == 0:
            labels = torch.zeros((anchors.shape[0],), device=device, dtype=torch.long)
            matched_idx = torch.zeros((anchors.shape[0],), device=device, dtype=torch.long)
            return labels, matched_idx

        iou = box_iou(anchors, gt_boxes)  # [A, G]
        max_iou, matched_idx = iou.max(dim=1)

        labels = torch.full((anchors.shape[0],), -1, dtype=torch.long, device=device)
        labels[max_iou < bg_thresh] = 0
        labels[max_iou >= fg_thresh] = 1

        # Guarantee every GT has at least one positive anchor.
        gt_best_iou, gt_best_anchor = iou.max(dim=0)
        labels[gt_best_anchor] = 1

        return labels, matched_idx

    def _match_proposals_to_gt(
        self,
        proposals: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        fg_thresh: float,
        bg_thresh: float,
    ) -> Tuple[Tensor, Tensor]:
        """Match proposals to GT for ROI training.

        Output labels are class ids for foreground and 0 for background.
        """
        device = proposals.device

        if gt_boxes.numel() == 0:
            labels = torch.zeros((proposals.shape[0],), device=device, dtype=torch.long)
            matched_idx = torch.zeros((proposals.shape[0],), device=device, dtype=torch.long)
            return labels, matched_idx

        iou = box_iou(proposals, gt_boxes)  # [P, G]
        max_iou, matched_idx = iou.max(dim=1)

        labels = torch.full((proposals.shape[0],), -1, dtype=torch.long, device=device)

        bg = max_iou < bg_thresh
        fg = max_iou >= fg_thresh

        labels[bg] = 0
        labels[fg] = gt_labels[matched_idx[fg]]

        # Between thresholds is ignore for sampling.
        return labels, matched_idx

    def _sample_labels(self, labels: Tensor, batch_size: int, positive_fraction: float) -> Tensor:
        """Balanced subsampling for positives and negatives.

        labels can contain:
          >0 / ==1 : positive,
          ==0      : negative,
          ==-1     : ignore.
        """
        positive = torch.where(labels > 0)[0]
        negative = torch.where(labels == 0)[0]

        num_pos = int(batch_size * positive_fraction)
        num_pos = min(num_pos, positive.numel())
        num_neg = batch_size - num_pos
        num_neg = min(num_neg, negative.numel())

        # Random subset for SGD mini-batch style training.
        perm_pos = torch.randperm(positive.numel(), device=labels.device)[:num_pos]
        perm_neg = torch.randperm(negative.numel(), device=labels.device)[:num_neg]

        idx = torch.cat([positive[perm_pos], negative[perm_neg]], dim=0)
        return idx

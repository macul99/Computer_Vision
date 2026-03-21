# Faster R-CNN: Algorithm and Mathematics (Detailed Study Notes)

This document explains the algorithmic steps and core equations implemented in this folder.

## 1. High-Level Pipeline

Faster R-CNN is a two-stage detector:

1. Backbone CNN converts image to feature map.
2. Region Proposal Network (RPN) scans feature map with anchors and proposes candidate boxes.
3. ROI stage (Fast R-CNN head) classifies each proposal and refines its box.

So, prediction is:
- first stage answers "where might objects be?"
- second stage answers "what class is this region, and what is the precise box?"

## 2. Coordinate and Box Conventions

Boxes use:
- `(x1, y1, x2, y2)`
- top-left corner `(x1, y1)`
- bottom-right corner `(x2, y2)`

Box width/height:
- $w = x_2 - x_1$
- $h = y_2 - y_1$

Center:
- $x = (x_1 + x_2)/2$
- $y = (y_1 + y_2)/2$

Area:
- $A = \max(0, w) \cdot \max(0, h)$

## 3. IoU (Intersection over Union)

For two boxes $B_a, B_b$:

$$
\mathrm{IoU}(B_a, B_b) = \frac{|B_a \cap B_b|}{|B_a \cup B_b|}
$$

Intersection uses coordinate overlap:
- left-top = max of left-top corners
- right-bottom = min of right-bottom corners
- if width/height become negative, intersection is 0.

IoU is used for:
- assigning positives/negatives to anchors in RPN,
- assigning positives/negatives to proposals in ROI stage,
- NMS suppression decisions.

## 4. Anchors

At each feature-map location, several anchor templates are placed.

For size $s$ and aspect ratio $r = h/w$:
- $s = \sqrt{w*h}$
- $w = s / \sqrt{r}$
- $h = s \sqrt{r}$

Anchor centered at $(c_x, c_y)$:
- $(c_x - w/2, c_y - h/2, c_x + w/2, c_y + h/2)$

If feature map is $H_f \times W_f$ and anchors per location are $K$, total anchors:

$$
A = H_f W_f K
$$

## 5. Bounding Box Regression Parameterization

This is a critical part of Faster R-CNN.

Given proposal/anchor $P$ and target GT box $G$:
- proposal center/size: $(x_p, y_p, w_p, h_p)$
- GT center/size: $(x_g, y_g, w_g, h_g)$

### Encode (target deltas)

$$
t_x = \frac{x_g - x_p}{w_p}, \quad
 t_y = \frac{y_g - y_p}{h_p}, \quad
 t_w = \log\frac{w_g}{w_p}, \quad
 t_h = \log\frac{h_g}{h_p}
$$

### Decode (predicted deltas back to box)

Given predicted $(d_x, d_y, d_w, d_h)$:

$$
\hat{x} = x_p + d_x w_p, \quad
\hat{y} = y_p + d_y h_p, \quad
\hat{w} = w_p e^{d_w}, \quad
\hat{h} = h_p e^{d_h}
$$

Then convert center-size back to corners:

$$
x_1 = \hat{x} - \hat{w}/2,\;
 y_1 = \hat{y} - \hat{h}/2,\;
 x_2 = \hat{x} + \hat{w}/2,\;
 y_2 = \hat{y} + \hat{h}/2
$$

This transform makes learning easier because translation is normalized by scale and scale change is logarithmic.

## 6. Stage 1: RPN Details

RPN predicts per anchor:
- objectness logit (object vs background),
- box delta (4 values).

### Anchor Label Assignment (Training)

For each anchor, compute max IoU with all GT boxes.
- positive: IoU >= fg threshold (e.g. 0.7)
- negative: IoU < bg threshold (e.g. 0.3)
- ignore: in between

Also force at least one positive anchor for each GT by marking the best-IoU anchor positive.

### RPN Classification Loss

Binary cross-entropy with logits:

$$
\mathcal{L}_{\mathrm{rpn\_cls}} = \mathrm{BCEWithLogits}(o, y)
$$

where $o$ is objectness logit and $y \in \{0,1\}$.

### RPN Box Regression Loss

Only for positive anchors:

$$
\mathcal{L}_{\mathrm{rpn\_reg}} = \frac{1}{N} \sum_{i \in \mathrm{pos}} \mathrm{SmoothL1}(\Delta_i, t_i)
$$

where $\Delta_i$ is predicted delta and $t_i$ is encoded GT delta.

### Proposal Filtering

1. Decode anchors to proposal boxes.
2. Clip boxes to image boundaries.
3. Remove tiny boxes.
4. Keep top-k by objectness score (pre-NMS top-k).
5. Apply NMS.
6. Keep post-NMS top-k proposals.

## 7. Stage 2: ROI Head Details

ROI stage takes proposals and feature map.

### ROI Align

Each proposal is pooled to fixed spatial size (e.g. 7x7), producing uniform feature tensors regardless of original proposal size.

Then a small MLP predicts:
- class logits over C classes (including background class 0),
- class-specific box deltas (C x 4 values).

### Proposal Label Assignment (Training)

For each proposal, match to GT box by IoU:
- foreground if IoU >= 0.5
- background if IoU < 0.5
- ignore optional between thresholds (depends on exact setup)

Foreground label is GT class id; background label is 0.

### ROI Classification Loss

Cross entropy on all sampled RoIs:

$$
\mathcal{L}_{\mathrm{roi\_cls}} = \mathrm{CE}(p, y)
$$

where $p$ are class logits and $y \in \{0,\ldots,C-1\}$.

### ROI Box Regression Loss

Only foreground RoIs contribute:

$$
\mathcal{L}_{\mathrm{roi\_reg}} = \frac{1}{N} \sum_{i \in \mathrm{fg}} \mathrm{SmoothL1}(\Delta^{(y_i)}_i, t_i)
$$

$\Delta^{(y_i)}_i$ means selecting the 4 deltas corresponding to the true class for that RoI.

## 8. Multi-Task Loss

Total loss (training):

$$
\mathcal{L} = \mathcal{L}_{\mathrm{rpn\_cls}} + \mathcal{L}_{\mathrm{rpn\_reg}} + \mathcal{L}_{\mathrm{roi\_cls}} + \mathcal{L}_{\mathrm{roi\_reg}}
$$

In many implementations each term can be weighted, but this minimal version uses equal weight 1.0.

## 9. Inference Steps

1. Run backbone and RPN to get proposals.
2. ROI Align proposals and compute class scores + class box deltas.
3. For each class (except background):
- decode class-specific boxes,
- threshold by score,
- apply class-wise NMS.
4. Merge all classes and keep top detections per image.

## 10. Why This Minimal Version Is Useful

This code intentionally avoids many production features so that logic is easy to inspect:
- single feature level,
- fixed image size,
- straightforward sampling and matching,
- explicit tensor reshaping and indexing.

Once understood, extend with:
- FPN multi-level features,
- multi-scale training,
- stronger backbone (ResNet),
- better sampler and assigner rules,
- mixed precision and distributed training.

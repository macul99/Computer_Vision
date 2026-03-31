# YOLOv1: Algorithm and Mathematics (Detailed Study Notes)

This document explains the mathematical foundations of the YOLOv1 detector
implemented in this folder. Compare with `../faster_rcnn_min/MATH_AND_ALGO_README.md`
for the two-stage approach.

---

## 1. High-Level Pipeline

YOLOv1 is a **single-stage** detector:

```
Image [3, 448, 448]
    → CNN Backbone
    → Feature Map [512, 7, 7]
    → Flatten + FC layers
    → Reshape to [S, S, B×5 + C]
    → Decode → Final Detections
```

Contrast with Faster R-CNN's two-stage pipeline:
```
Image → Backbone → RPN (stage 1) → ROI Head (stage 2) → Detections
```

---

## 2. Grid Division

The image is divided into an **S × S grid** (S=7 by default).

Each grid cell is responsible for detecting objects whose **center** falls within that cell.

Cell size in pixels: $\frac{448}{7} = 64$ pixels.

If an object's center is at pixel $(224, 160)$:
- Normalized: $(224/448, 160/448) = (0.5, 0.357)$
- Grid cell: $(⌊0.5 × 7⌋, ⌊0.357 × 7⌋) = (3, 2)$
- Cell-relative offset: $(0.5 × 7 - 3, 0.357 × 7 - 2) = (0.5, 0.5)$

---

## 3. Output Tensor Structure

Each cell predicts:

$$\text{Output per cell} = B \times 5 + C$$

With $B=2, C=20$: $2 \times 5 + 20 = 30$ values per cell.

Total output: $S \times S \times (B \times 5 + C) = 7 \times 7 \times 30 = 1470$ values.

### Per-box values (×B):
| Value | Symbol | Range | Meaning |
|-------|--------|-------|---------|
| x | $\hat{x}$ | [0, 1] | Center x offset within cell |
| y | $\hat{y}$ | [0, 1] | Center y offset within cell |
| w | $\hat{w}$ | [0, 1] | Width relative to image |
| h | $\hat{h}$ | [0, 1] | Height relative to image |
| conf | $\hat{C}$ | [0, 1] | Objectness confidence |

### Per-cell values (×1):
| Value | Symbol | Range | Meaning |
|-------|--------|-------|---------|
| class_i | $P(c_i | \text{Object})$ | [0, 1] | Class conditional probability |

---

## 4. Coordinate Conversions

### Cell-relative → Image-absolute (for inference):

$$x_{\text{abs}} = \frac{j + \hat{x}}{S} \times W_{\text{img}}$$

$$y_{\text{abs}} = \frac{i + \hat{y}}{S} \times H_{\text{img}}$$

$$w_{\text{abs}} = \hat{w} \times W_{\text{img}}$$

$$h_{\text{abs}} = \hat{h} \times H_{\text{img}}$$

Where $(i, j)$ is the grid cell row and column.

### Image-absolute → Cell-relative (for target encoding):

$$\hat{x} = \frac{x_{\text{center}}}{W_{\text{img}}} \times S - j$$

$$\hat{y} = \frac{y_{\text{center}}}{H_{\text{img}}} \times S - i$$

$$\hat{w} = \frac{w_{\text{box}}}{W_{\text{img}}}$$

$$\hat{h} = \frac{h_{\text{box}}}{H_{\text{img}}}$$

---

## 5. Confidence Score

The confidence score encodes two things simultaneously:

$$C = P(\text{Object}) \times \text{IoU}(\text{pred}, \text{truth})$$

- **During training**: The target confidence is the actual IoU between the predicted box and ground truth.
- **During inference**: It represents the model's belief that an object exists AND the box is accurate.

### Class-specific confidence (at inference):

$$\text{Score}(c_i) = C \times P(c_i | \text{Object})$$

This gives us $P(c_i) \times \text{IoU}$ — the probability of class $c_i$ weighted by box quality.

---

## 6. Responsible Predictor Selection

Each cell has $B$ predictors. Only the **responsible** one gets trained for localization.

The responsible predictor is chosen as:

$$b^* = \arg\max_{b \in \{1, ..., B\}} \text{IoU}(\text{pred}_b, \text{GT})$$

This means the predictor whose current box has the **highest IoU** with the ground truth
is the one that gets the localization gradient. Over time, different predictors
specialize in different object shapes.

---

## 7. Loss Function

### Complete formulation:

$$\mathcal{L} = \lambda_{\text{coord}} \mathcal{L}_{\text{loc}} + \mathcal{L}_{\text{conf\_obj}} + \lambda_{\text{noobj}} \mathcal{L}_{\text{conf\_noobj}} + \mathcal{L}_{\text{class}}$$

### 7.1 Localization Loss

$$\mathcal{L}_{\text{loc}} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]$$

Where $\mathbb{1}_{ij}^{\text{obj}}$ = 1 if predictor $j$ in cell $i$ is responsible.

**Why square root for width/height?**

Without sqrt, a 10-pixel error on a 200-pixel box and a 10-pixel error on a 20-pixel box
produce the same loss. With sqrt:

$$(\sqrt{0.5} - \sqrt{0.45})^2 \approx 0.0006 \quad \text{(large box, small relative error)}$$
$$(\sqrt{0.05} - \sqrt{0.0})^2 \approx 0.05 \quad \text{(small box, same absolute error → much bigger loss)}$$

This makes the model more careful with small objects.

### 7.2 Confidence Loss (Object)

$$\mathcal{L}_{\text{conf\_obj}} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2$$

Target $C_i = \text{IoU}(\text{pred}, \text{GT})$, so the model learns to predict its own accuracy.

### 7.3 Confidence Loss (No Object)

$$\mathcal{L}_{\text{conf\_noobj}} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (0 - \hat{C}_i)^2$$

Weighted by $\lambda_{\text{noobj}} = 0.5$. Most cells are empty, so without downweighting,
the gradient from empty cells would overwhelm the signal from object cells.

### 7.4 Classification Loss

$$\mathcal{L}_{\text{class}} = \sum_{i=0}^{S^2} \mathbb{1}_{i}^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2$$

Original paper uses MSE. Our implementation uses cross-entropy (standard modern practice).

### Loss weights:

| Weight | Value | Reason |
|--------|-------|--------|
| $\lambda_{\text{coord}}$ | 5.0 | Precise localization is critical |
| $\lambda_{\text{noobj}}$ | 0.5 | Prevent empty cells from dominating |
| $\lambda_{\text{conf\_obj}}$ | 1.0 | Default weight |
| $\lambda_{\text{class}}$ | 1.0 | Default weight |

---

## 8. Non-Maximum Suppression (NMS)

After the model produces $S \times S \times B$ candidate boxes, many overlap.
NMS removes redundant detections:

```
Algorithm NMS(detections, iou_threshold):
    Sort detections by confidence (descending)
    kept = []
    while detections not empty:
        best = detections.pop(0)
        kept.append(best)
        detections = [d for d in detections if IoU(best, d) < iou_threshold]
    return kept
```

Typically applied per-class with $\text{IoU threshold} = 0.5$.

---

## 9. Key Limitations of YOLOv1

1. **One object per cell**: Each cell can only predict one class. If two objects
   have centers in the same cell, one is lost.

2. **Coarse grid**: S=7 gives only 49 cells. Small nearby objects are hard to detect.

3. **Fixed aspect ratios**: Unlike Faster R-CNN's anchors, YOLOv1 learns box shapes
   from scratch with only B=2 predictors per cell.

4. **Localization errors**: The fully-connected layers lose spatial precision compared
   to convolutional approaches used in later versions.

5. **No multi-scale detection**: Only one feature map resolution. Small objects are
   particularly challenging (addressed in YOLOv3+ with FPN-like multi-scale heads).

---

## 10. Comparison: YOLO vs Faster R-CNN Mathematics

| Aspect | Faster R-CNN | YOLOv1 |
|--------|-------------|--------|
| **Box parameterization** | Deltas relative to anchors: $t_x = (x_g - x_a)/w_a$ | Direct prediction: $(x, y, w, h)$ relative to cell/image |
| **Confidence target** | Binary (object/not-object) per anchor | IoU between predicted box and GT |
| **Classification** | Per-proposal softmax after ROI pooling | Per-cell class probabilities shared across B boxes |
| **Training samples** | Mini-batch of anchors/proposals (256 RPN, 128 ROI) | All S×S cells simultaneously |
| **Loss function** | Separate RPN loss + ROI loss | Single combined loss |
| **Box regression** | Log-space encoding (log(w_g/w_a)) | Direct regression with √w, √h |

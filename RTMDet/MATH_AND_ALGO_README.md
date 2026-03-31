# RTMDet: Algorithm and Mathematics (Detailed Study Notes)

This document explains the mathematical foundations of the RTMDet detector
implemented in this folder. Compare with `../yolo/MATH_AND_ALGO_README.md`
for the classic single-stage approach and `../faster_rcnn_min/MATH_AND_ALGO_README.md`
for the two-stage approach.

---

## 1. High-Level Pipeline

RTMDet is a **modern single-stage, anchor-free** detector:

```
Image [3, 320, 320]
    → CSPNeXt Backbone → Multi-scale features [P3, P4, P5]
    → CSPAFPN Neck     → Fused features [N3, N4, N5]
    → Decoupled Head   → cls_scores + bbox_preds per level
    → SimOTA Assignment (training) / NMS Decode (inference)
    → Final Detections
```

Contrast with YOLOv1:
```
Image → Backbone → FC layers → Single 7×7 grid → Detections
```

Contrast with Faster R-CNN:
```
Image → Backbone → RPN (stage 1) → ROI Head (stage 2) → Detections
```

---

## 2. Multi-Scale Feature Maps

Unlike YOLOv1 which uses a single 7×7 feature map, RTMDet uses 3 levels:

| Level | Stride | Feature Size (320 input) | Points | Best For |
|-------|--------|--------------------------|--------|----------|
| P3 | 8 | 40×40 | 1600 | Small objects |
| P4 | 16 | 20×20 | 400 | Medium objects |
| P5 | 32 | 10×10 | 100 | Large objects |
| **Total** | | | **2100** | |

Each point at stride $s$ covers an $s \times s$ region on the image.
The point's coordinate is the center of that region:

$$p_x = (j + 0.5) \times s, \quad p_y = (i + 0.5) \times s$$

where $(i, j)$ are the row and column indices on the feature map.

---

## 3. Anchor-Free Box Representation (LTRB Distances)

### How RTMDet represents boxes

For each point $(p_x, p_y)$, the model predicts 4 distances:

$$d = (l, t, r, b)$$

where:
- $l$ = distance to the **left** edge of the box
- $t$ = distance to the **top** edge of the box
- $r$ = distance to the **right** edge of the box
- $b$ = distance to the **bottom** edge of the box

### Decoding: distance → box

$$x_1 = p_x - l, \quad y_1 = p_y - t$$
$$x_2 = p_x + r, \quad y_2 = p_y + b$$

### Encoding: box → distance (for computing targets)

Given a ground-truth box $(x_1^{gt}, y_1^{gt}, x_2^{gt}, y_2^{gt})$:

$$l = p_x - x_1^{gt}, \quad t = p_y - y_1^{gt}$$
$$r = x_2^{gt} - p_x, \quad b = y_2^{gt} - p_y$$

### Why LTRB instead of (x, y, w, h)?

1. **No anchor dependency**: (x, y, w, h) in anchor-based detectors are offsets from anchor boxes. LTRB needs only the point position.
2. **Natural for points**: each point directly describes its box.
3. **Positive constraint**: all distances should be non-negative (the point should be inside its box). This is enforced by ReLU on predictions.

### Comparison

| Format | Used by | Predicted values | Needs anchors? |
|--------|---------|-----------------|----------------|
| $(x, y, w, h)$ | YOLOv1 | Center + size | No (grid-relative) |
| $(\delta x, \delta y, \delta w, \delta h)$ | Faster R-CNN | Offsets from anchors | Yes |
| $(l, t, r, b)$ | RTMDet, FCOS | Distances from point | No |

---

## 4. CSP (Cross Stage Partial) Architecture

### Basic Idea

Given input features $X$ with $C$ channels:

$$X_{\text{main}} = f_{\text{reduce}}(X) \quad [\frac{C}{2} \text{ channels}]$$
$$X_{\text{short}} = g_{\text{reduce}}(X) \quad [\frac{C}{2} \text{ channels}]$$
$$X_{\text{main}}' = \text{BottleneckBlocks}(X_{\text{main}})$$
$$\text{Output} = h_{\text{mix}}(\text{Concat}(X_{\text{main}}', X_{\text{short}}))$$

Where:
- $f_{\text{reduce}}$, $g_{\text{reduce}}$: 1×1 convolutions to reduce channels
- BottleneckBlocks: sequence of conv + depthwise separable conv
- $h_{\text{mix}}$: 1×1 convolution to project back to desired channels

### Why CSP?

- Only half the channels go through the expensive bottleneck blocks
- The shortcut path preserves original features and gradient flow
- Reduces FLOPs while maintaining representation power

### CSPNeXt vs CSPDarknet

CSPNeXt (used in RTMDet) replaces the standard 3×3 conv in the bottleneck with a **5×5 depthwise separable convolution**:

Standard: $\text{Conv}_{3\times3}(C \to C)$ → params = $9C^2$

Depthwise separable: $\text{DWConv}_{5\times5}(C) + \text{Conv}_{1\times1}(C \to C)$ → params = $25C + C^2$

For $C = 128$: standard = 147,456; depthwise separable = 19,584 → **7.5× fewer parameters**

The larger 5×5 kernel also increases receptive field, helping capture broader context.

---

## 5. PAFPN (Path Aggregation Feature Pyramid Network)

### Why feature fusion is needed

High-level features (P5, stride 32) have strong semantics but poor spatial resolution.
Low-level features (P3, stride 8) have fine spatial detail but weak semantics.

FPN fuses them:

### Top-down path (semantics flow down)

$$N_5 = \text{Reduce}(P_5)$$
$$N_4 = \text{CSP}(\text{Concat}(\text{Reduce}(P_4), \text{Upsample}(N_5)))$$
$$N_3 = \text{CSP}(\text{Concat}(\text{Reduce}(P_3), \text{Upsample}(N_4)))$$

### Bottom-up path (detail flows up)

$$O_3 = N_3$$
$$O_4 = \text{CSP}(\text{Concat}(N_4, \text{Downsample}(O_3)))$$
$$O_5 = \text{CSP}(\text{Concat}(N_5, \text{Downsample}(O_4)))$$

After fusion, each level has both strong semantics AND fine spatial detail.

---

## 6. Decoupled Head

### Why decouple classification and regression?

Classification and regression have different optimal feature representations:

- **Classification** benefits from features that capture texture, color, and shape patterns (semantic features)
- **Regression** benefits from features that capture precise edge locations (spatial features)

Sharing a single head for both forces a compromise. Decoupling lets each branch specialize.

### Architecture per level

$$x_{\text{shared}} = \text{SiLU}(\text{BN}(\text{Conv}_{3\times3}(F_l)))$$
$$\text{cls} = \text{Conv}_{1\times1}(\text{ConvBN}^{(N)}(x_{\text{shared}})) \quad [C \text{ values}]$$
$$\text{reg} = \text{ReLU}(\text{Conv}_{1\times1}(\text{ConvBN}^{(N)}(x_{\text{shared}}))) \quad [4 \text{ values}]$$

Note: ReLU on regression ensures distances are non-negative.

### Weight sharing across levels

The same head is applied to all feature levels. This is possible because:
1. Features are normalized by the neck to the same channel count
2. Regression predictions are in stride-relative units (multiplied by stride before use)

---

## 7. Quality Focal Loss (QFL)

### Motivation

Standard binary cross-entropy uses hard targets:
- Positive sample: target = 1.0
- Negative sample: target = 0.0

But this doesn't reflect box quality. A positive sample with a poor box (IoU = 0.3) gets the same target as one with a great box (IoU = 0.9).

QFL uses **soft targets**: the target is the IoU between the predicted box and the assigned GT box.

### Formula

$$\text{QFL}(\sigma, y) = -|y - \sigma|^\beta \cdot \left[ y \log(\sigma) + (1-y) \log(1-\sigma) \right]$$

Where:
- $\sigma = \text{sigmoid}(\text{logit})$ = predicted probability
- $y$ = quality target (IoU for positives, 0 for negatives)
- $\beta$ = focusing parameter (typically 2.0)

### Understanding the focusing factor $|y - \sigma|^\beta$

| Scenario | $|y - \sigma|$ | Weight | Interpretation |
|----------|--------------|--------|----------------|
| Easy negative: pred≈0, target=0 | ≈0 | Low | Already correct, don't learn |
| Hard negative: pred≈0.8, target=0 | ≈0.8 | High | Wrong! Learn hard |
| Good positive: pred≈0.7, target=0.8 | ≈0.1 | Low | Close enough |
| Bad positive: pred≈0.1, target=0.8 | ≈0.7 | High | Wrong! Learn hard |

### Comparison with standard losses

| Loss | Positive target | Negative target | Handles quality? |
|------|----------------|-----------------|-----------------|
| BCE | 1.0 | 0.0 | No |
| Focal Loss | 1.0 | 0.0 | No (but handles class imbalance) |
| Quality FL | IoU (0 to 1) | 0.0 | Yes |

---

## 8. GIoU Loss

### Standard IoU

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

**Problem**: When $|A \cap B| = 0$ (no overlap), IoU = 0 and $\nabla \text{IoU} = 0$. The model gets no signal about which direction to move the box.

### Generalized IoU

$$\text{GIoU}(A, B) = \text{IoU}(A, B) - \frac{|C \setminus (A \cup B)|}{|C|}$$

Where $C$ is the smallest enclosing box containing both $A$ and $B$.

The penalty term $\frac{|C \setminus (A \cup B)|}{|C|}$ measures the fraction of the enclosing box that is NOT covered by the union. This provides gradient even when boxes don't overlap.

### Properties

| Property | IoU | GIoU |
|----------|-----|------|
| Range | [0, 1] | [-1, 1] |
| Gradient when no overlap | 0 | Non-zero |
| Perfect overlap | 1 | 1 |
| Loss range ($1 - \cdot$) | [0, 1] | [0, 2] |

### Example

Two non-overlapping boxes:
- Box A: (0, 0, 10, 10), Box B: (20, 0, 30, 10)
- IoU = 0 → loss = 1, gradient = 0 (stuck!)
- Enclosing box C: (0, 0, 30, 10), area = 300
- Union area: 100 + 100 = 200
- GIoU = 0 - (300 - 200)/300 = -0.333 → loss = 1.333, gradient ≠ 0 (learning!)

---

## 9. SimOTA (Simplified Optimal Transport Assignment)

### The label assignment problem

With 2100 prediction points per image, how do we decide which points should learn from which ground-truth objects?

### Static assignment (YOLOv1 style)

Assign the grid cell containing the object center. Problems:
- Only ONE cell can learn each object
- Doesn't adapt to the model's current ability
- Edge cases: objects near cell boundaries

### Dynamic assignment (SimOTA)

The key insight: **the model's own predictions should influence the assignment**.

### Algorithm

**Input**: $N$ prediction points, $M$ ground-truth boxes

**Step 1: Candidate filtering**

For each GT box $j$, find candidate points:
$$\mathcal{C}_j = \{i : p_i \in \text{box}_j \text{ OR } \|p_i - \text{center}_j\| < r \times s_i\}$$

Where $r$ is the center radius and $s_i$ is the stride for point $i$.

**Step 2: Cost matrix**

For each candidate $(i, j)$:
$$\text{cost}_{ij} = \text{BCE}(\hat{c}_i, 1) + \lambda \cdot (-\log(\text{IoU}(\hat{b}_i, b_j^{gt})))$$

Where $\hat{c}_i$ is the predicted class score and $\hat{b}_i$ is the predicted box.

**Step 3: Dynamic k**

For each GT box $j$:
$$k_j = \lceil \sum_{i \in \text{top-}K} \text{IoU}(\hat{b}_i, b_j^{gt}) \rceil$$

Clamped to $[1, K]$ where $K$ is a fixed hyperparameter (13).

Intuition: Objects that the model already predicts well (high IoU) get more positive samples, reinforcing good predictions.

**Step 4: Assignment**

For each GT box $j$, select the $k_j$ candidates with lowest cost.

**Step 5: Resolve conflicts**

If point $i$ is assigned to multiple GT boxes, keep only the one with lowest cost.

### Why is this better?

1. **Adaptive**: Easy objects get more positives; hard objects get fewer
2. **Quality-aware**: Points with good predictions are preferred
3. **Self-reinforcing**: Good predictions → more positive samples → better predictions

---

## 10. Loss Summary

### Training loss

$$L_{\text{total}} = L_{\text{cls}} + \lambda_{\text{reg}} \cdot L_{\text{reg}}$$

Where $\lambda_{\text{reg}} = 2.0$.

**Classification** (computed on ALL points):
$$L_{\text{cls}} = \frac{1}{N_{\text{pos}}} \sum_{i=1}^{N} \text{QFL}(\sigma_i, y_i)$$

**Box regression** (computed on POSITIVE points only):
$$L_{\text{reg}} = \frac{1}{N_{\text{pos}}} \sum_{i \in \text{pos}} (1 - \text{GIoU}(\hat{b}_i, b_i^{gt}))$$

### Comparison with YOLOv1 loss

| Component | YOLOv1 | RTMDet |
|-----------|--------|--------|
| Localization | $\lambda(x-\hat x)^2 + (\sqrt w - \sqrt{\hat w})^2$ | GIoU loss |
| Confidence (obj) | $(C - \text{IoU})^2$ | Merged into QFL |
| Confidence (noobj) | $\lambda_{\text{noobj}} C^2$ | Merged into QFL |
| Classification | Cross-entropy | Quality Focal Loss |
| Target assignment | Static (grid cell) | Dynamic (SimOTA) |

RTMDet unifies objectness and classification into a single quality-aware score, simplifying the loss while improving performance.

---

## 11. Inference Pipeline

```
For each image:
  1. Forward pass → cls_scores, bbox_preds at 3 levels
  2. For each level:
     a. Generate grid points
     b. Decode: point + (l,t,r,b) × stride → (x1,y1,x2,y2)
     c. Sigmoid(cls_logits) → class probabilities
     d. Filter by score threshold
  3. Concatenate predictions from all levels
  4. Per-class NMS to remove duplicate detections
  5. Return final boxes, scores, class IDs
```

### NMS (Non-Maximum Suppression)

Same as in YOLOv1:
1. Sort by score descending
2. Pick highest-scoring box
3. Remove all boxes overlapping with it above threshold
4. Repeat

---

## 12. Depthwise Separable Convolution Math

### Standard convolution

Input: $C_{\text{in}} \times H \times W$, kernel: $C_{\text{out}} \times C_{\text{in}} \times k \times k$

Parameters: $C_{\text{in}} \times C_{\text{out}} \times k^2$

### Depthwise separable

**Step 1 (depthwise)**: $C_{\text{in}}$ independent $k \times k$ filters
- Parameters: $C_{\text{in}} \times k^2$

**Step 2 (pointwise)**: $1 \times 1$ convolution $C_{\text{in}} \to C_{\text{out}}$
- Parameters: $C_{\text{in}} \times C_{\text{out}}$

**Total**: $C_{\text{in}} \times (k^2 + C_{\text{out}})$

**Savings ratio**: $\frac{k^2 + C_{\text{out}}}{C_{\text{out}} \times k^2} \approx \frac{1}{C_{\text{out}}} + \frac{1}{k^2}$

For $C_{\text{out}} = 128, k = 5$: savings ≈ $\frac{1}{128} + \frac{1}{25} \approx 4.8\%$ of original cost.

---

## 13. Classification Bias Initialization

The head's classification conv bias is initialized to:

$$b = -\log\left(\frac{1 - \pi}{\pi}\right)$$

Where $\pi = 0.01$ is the prior probability.

This means $\text{sigmoid}(b) = \pi = 0.01$, so the model initially predicts ~1% probability for every class at every location — a very conservative "no object" prior.

**Why?** Without this, initial random predictions would trigger massive focal loss on the thousands of background locations, causing unstable early training.

---

## 14. SiLU (Swish) Activation

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

Properties:
- Smooth and non-monotonic
- $\text{SiLU}(0) = 0$
- For large $x$: $\text{SiLU}(x) \approx x$ (like ReLU)
- For large negative $x$: $\text{SiLU}(x) \approx 0$ (like ReLU)
- Allows small negative values, which can help gradient flow
- Derivative: $\text{SiLU}'(x) = \sigma(x)(1 + x(1 - \sigma(x)))$

Compared to ReLU:
- No dead neurons (gradient is never exactly 0 for finite x)
- Smoother optimization landscape
- Slightly more expensive to compute

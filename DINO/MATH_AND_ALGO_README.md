# DINO — Math and Algorithm Deep Dive

## High-Level Pipeline

```
Input Image [B, 3, H, W]
       │
       ▼
┌─────────────────────┐
│     Backbone CNN     │  Extract multi-scale features
│  Stage 0 → [B, 64, H/2, W/2]
│  Stage 1 → [B, 128, H/4, W/4]   ← Feature Level 0
│  Stage 2 → [B, 256, H/8, W/8]   ← Feature Level 1
└─────────┬───────────┘
          │
          ▼  Project to hidden_dim + positional encoding + level embedding
┌─────────────────────┐
│ Flatten & Concatenate│  [B, N_total, D]  where N_total = H₀W₀ + H₁W₁
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Transformer Encoder  │  Self-attention across all spatial tokens
│  (L_enc layers)      │  Each token attends to all others
└─────────┬───────────┘
          │ memory: [B, N, D]
          │
          ├───────────────────────────────────┐
          ▼                                   ▼
┌─────────────────────┐          ┌─────────────────────────┐
│   Normal Queries     │          │   Denoising Queries      │
│   Q learned anchors  │          │   (training only)        │
│   [B, Q, D]          │          │   Noisy GT → reconstruct │
└─────────┬───────────┘          └─────────┬───────────────┘
          │                                 │
          ▼                                 ▼
┌────────────────────────────────────────────────────────────┐
│              Transformer Decoder (L_dec layers)             │
│                                                            │
│  For each layer k:                                         │
│    1. Self-attention among queries                         │
│    2. Cross-attention: queries → encoder memory            │
│    3. FFN                                                  │
│    4. Detection Head → logits_k, boxes_k                   │
│    5. boxes_k becomes reference point for layer k+1        │
│       (iterative refinement, detached gradient)            │
│                                                            │
└────────────────┬───────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────┐
│    Output            │
│  pred_logits [B,Q,C] │  ← class logits (final layer)
│  pred_boxes  [B,Q,4] │  ← boxes (cx, cy, w, h) normalized
│  aux_outputs         │  ← intermediate layer predictions
│  dn_outputs          │  ← denoising predictions (training)
└─────────────────────┘
```

---

## 1. Multi-Scale Feature Extraction

### Why Multiple Scales?

Objects vary enormously in size. A person might occupy 80% of the image while a
distant car occupies 2%. Features at different resolutions capture different
object sizes:

| Feature Level | Resolution | Stride | Best For |
|---|---|---|---|
| Level 0 (fine) | H/4 × W/4 | 4 | Small objects |
| Level 1 (coarse) | H/8 × W/8 | 8 | Medium/large objects |

In the full DINO, 4 levels are used (strides 8, 16, 32, 64).

### Feature Projection

Each backbone level has different channel dimensions. We project all levels to
a common `hidden_dim` using 1×1 convolutions:

$$
\text{src}_l = \text{Conv1×1}(\text{backbone}_l) + \text{level\_embed}_l
$$

The level embedding is a learned vector added to all tokens from level $l$, so
the transformer knows which scale each token comes from.

### Flattening

After projection, spatial dimensions are flattened and all levels are concatenated:

$$
\text{src} = [\text{flatten}(\text{src}_0); \text{flatten}(\text{src}_1)] \in \mathbb{R}^{B \times N \times D}
$$

where $N = H_0 W_0 + H_1 W_1$.

---

## 2. Positional Encoding

Since attention is permutation-invariant, we add positional information:

$$
q = k = \text{src} + \text{pos}
$$

In our simplified version, we use **learned 2D positional encoding**:
- Separate row and column embeddings, each of dimension $D/2$
- Concatenated to form a $D$-dimensional position vector per spatial location

In the full DINO: **sinusoidal 2D encoding** with temperature scaling.

---

## 3. Transformer Encoder

The encoder refines features through self-attention. Each layer:

$$
\text{src}' = \text{LayerNorm}(\text{src} + \text{MultiHeadAttn}(\text{src}+\text{pos}, \text{src}+\text{pos}, \text{src}))
$$
$$
\text{src}'' = \text{LayerNorm}(\text{src}' + \text{FFN}(\text{src}'))
$$

**Key design choice:** Positional encoding is added to queries Q and keys K
but NOT to values V. This is because position should influence _which_ tokens
attend to each other (via Q·K similarity), but the _content_ passed through
(V) should be pure feature information.

### Deformable Attention (Full DINO)

Standard attention has $O(N^2)$ complexity. For high-resolution features this is
prohibitive. The full DINO uses **multi-scale deformable attention**:

For each query at position $p$, instead of attending to all $N$ tokens:
1. Predict $K$ sampling offsets $\Delta p_k$ relative to $p$
2. Attend only to features at positions $p + \Delta p_k$ across all levels
3. Complexity: $O(N \times L \times K)$ where $L$ = levels, $K$ = points per level (typically 4)

This is similar to deformable convolutions but in the attention framework.

---

## 4. Object Queries and Anchors

### Plain DETR Queries

In the original DETR, object queries are pure learned embeddings with no spatial
prior. The decoder must learn both _where_ to look and _what_ to detect from
scratch → slow convergence (~500 epochs).

### DINO's Anchor-Based Queries

DINO queries have two components:

1. **Content query** $q_c \in \mathbb{R}^D$: learned embedding for "what to detect"
2. **Reference point** $r \in [0,1]^4$: learned anchor $(c_x, c_y, w, h)$ for "where to look"

The reference point gives the decoder a spatial prior, so it only needs to learn
small refinements → much faster convergence (~12-24 epochs).

$$
\text{query} = q_c, \quad \text{anchor} = \sigma(\text{learned\_param}) \in [0,1]^4
$$

---

## 5. Iterative Box Refinement

Each decoder layer predicts a **delta** (correction) relative to the current
reference point, not an absolute box:

$$
\hat{b}_k = \sigma\left(\sigma^{-1}(r_k) + \Delta_k\right)
$$

Where:
- $r_k$ = reference point for layer $k$
- $\Delta_k$ = predicted box delta from layer $k$'s detection head
- $\sigma$ = sigmoid function (ensures output in [0,1])
- $\sigma^{-1}$ = inverse sigmoid (logit function)

The updated reference for the next layer:

$$
r_{k+1} = \text{detach}(\hat{b}_k)
$$

**Why detach?** Without detaching, gradients would flow through all previous
layers' reference points, creating optimization instabilities. Each layer should
learn to refine independently from its input reference.

**Why inverse sigmoid space?** Working in logit space $\sigma^{-1}(x) = \log\frac{x}{1-x}$
allows unconstrained addition of deltas, with sigmoid mapping the result back
to valid [0,1] coordinates.

### Refinement Visualization

```
Layer 0: anchor ──[+Δ₀]──► box₀  (rough localization)
                            │ detach
Layer 1: box₀  ──[+Δ₁]──► box₁  (finer adjustment)
                            │ detach
Layer 2: box₁  ──[+Δ₂]──► box₂  (precise localization)
                                   ↑ final output
```

---

## 6. Denoising Training

### Motivation

The Hungarian matching in DETR creates an unstable training dynamic:
- Early in training, all queries predict poorly
- Matching is essentially random
- The decoder gets noisy training signals → slow convergence

Denoising gives the decoder "easy" queries that are already near the answer.

### Denoising Query Construction

For each ground truth object $(c_i, b_i)$:

1. **Label noise:** With probability $p_{noise}$, replace $c_i$ with a random class:
$$
\tilde{c}_i = \begin{cases} c_{\text{random}} & \text{with prob } p_{noise} \\ c_i & \text{otherwise} \end{cases}
$$

2. **Box noise:** Add random perturbation to the box:
$$
\tilde{cx} = cx + \epsilon_x \cdot w, \quad \tilde{cy} = cy + \epsilon_y \cdot h
$$
$$
\tilde{w} = w \cdot (1 + \epsilon_w), \quad \tilde{h} = h \cdot (1 + \epsilon_h)
$$
where $\epsilon \sim \text{Uniform}(-\lambda, \lambda)$ and $\lambda$ is the noise scale.

3. **Multiple groups:** Repeat this $G$ times to create $G$ independent noise realizations per GT object.

### Denoising Loss

Since each denoising query has a known GT target, the loss is computed directly:

$$
L_{dn} = L_{cls}(\tilde{c}, c) + L_{box}(\tilde{b}, b) + L_{giou}(\tilde{b}, b)
$$

No Hungarian matching needed for denoising queries.

### Attention Masking

In the full DINO, the decoder uses an attention mask to prevent information leaking:

| | Normal Queries | DN Group 1 | DN Group 2 |
|---|---|---|---|
| **Normal Queries** | ✓ attend | ✗ masked | ✗ masked |
| **DN Group 1** | ✗ masked | ✓ attend | ✗ masked |
| **DN Group 2** | ✗ masked | ✗ masked | ✓ attend |

This ensures:
- Normal queries don't get "free answers" from denoising queries
- Each denoising group is independent
- The denoising pathway provides training signal without contaminating inference

---

## 7. Hungarian Matching

### The Assignment Problem

Given:
- $N_q$ predicted queries → $(p_j, \hat{b}_j)$ for $j = 1, \ldots, N_q$
- $N_{gt}$ ground truth objects → $(c_i, b_i)$ for $i = 1, \ldots, N_{gt}$

Find the optimal one-to-one assignment $\sigma^*$:

$$
\sigma^* = \arg\min_{\sigma \in \mathfrak{S}} \sum_{i=1}^{N_{gt}} \mathcal{C}_{\text{match}}(i, \sigma(i))
$$

where $\mathfrak{S}$ is the set of all permutations.

### Matching Cost

$$
\mathcal{C}_{\text{match}}(i, j) = \lambda_{cls} \cdot \mathcal{C}_{cls}(i,j) + \lambda_{L1} \cdot \mathcal{C}_{L1}(i,j) + \lambda_{giou} \cdot \mathcal{C}_{giou}(i,j)
$$

Where:
- $\mathcal{C}_{cls}(i,j) = -p_j(c_i)$ — negative predicted probability for the true class
- $\mathcal{C}_{L1}(i,j) = \|b_j - b_i\|_1$ — L1 distance between boxes
- $\mathcal{C}_{giou}(i,j) = -\text{GIoU}(b_j, b_i)$ — negative generalized IoU

### Why Hungarian Matching?

| Approach | NMS needed? | Assignment type | Pro | Con |
|---|---|---|---|---|
| Anchor-based (Faster R-CNN) | Yes | Many-to-one (IoU threshold) | Simple, fast | Duplicates, NMS tuning |
| Grid-based (YOLO) | Yes | Grid cell overlap | Very fast | Limited per-cell capacity |
| Hungarian (DETR/DINO) | No | One-to-one optimal | Clean, end-to-end | O(n³) matching |

---

## 8. Loss Functions

### Classification: Focal Loss

Standard cross-entropy treats all examples equally. In detection, most queries
match nothing (background), so a naive loss would:
- Be dominated by easy negatives
- Not focus enough on the few positive matches

Focal loss modulates the standard binary cross-entropy:

$$
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

Where:
- $p_t = p$ if $y=1$, else $p_t = 1-p$
- $\alpha$ = balance factor (typically 0.25)
- $\gamma$ = focusing parameter (typically 2.0)

Effect of $\gamma$:

| $p_t$ (correct prob) | $(1-p_t)^2$ factor | Interpretation |
|---|---|---|
| 0.9 (easy) | 0.01 | Loss reduced 100× |
| 0.5 (moderate) | 0.25 | Loss reduced 4× |
| 0.1 (hard) | 0.81 | Nearly full loss |

### Box Regression: L1 Loss

$$
L_{L1} = \frac{1}{N_{\text{matched}}} \sum_{(i,j) \in \sigma^*} \|\hat{b}_j - b_i\|_1
$$

L1 loss (absolute difference) on the normalized box coordinates.

### Box Regression: GIoU Loss

$$
L_{GIoU} = \frac{1}{N_{\text{matched}}} \sum_{(i,j) \in \sigma^*} \left(1 - \text{GIoU}(\hat{b}_j, b_i)\right)
$$

GIoU (Generalized IoU) is preferred over L1 alone because:

1. **Scale invariance:** A 0.01 L1 error matters more for a tiny box than a large one.
   GIoU normalizes by the box size implicitly.
2. **Gradient when no overlap:** If predicted and GT boxes don't overlap, IoU = 0
   and provides no gradient. GIoU can be negative and still provides signal.

**GIoU formula:**

$$
\text{GIoU}(A, B) = \text{IoU}(A, B) - \frac{|C \setminus (A \cup B)|}{|C|}
$$

Where $C$ is the smallest enclosing box of $A$ and $B$.

### Auxiliary Losses

The same loss (matching + classification + box) is applied at every intermediate
decoder layer output:

$$
L_{\text{total}} = L_{\text{final}} + \sum_{k=0}^{K-2} L_{\text{layer}_k}
$$

Each intermediate layer performs its own **independent** Hungarian matching.
This is important: different layers may match queries to objects differently
as predictions improve through the layers.

### Why Auxiliary Losses Help

Without auxiliary losses, only the final decoder layer receives direct gradient
from the detection loss. Earlier layers receive gradient only through backprop,
which gets weaker for deeper architectures. Auxiliary losses ensure that every
layer receives strong supervision, similar to "deep supervision" in segmentation
networks.

---

## 9. Comparison: DINO vs DETR vs Deformable DETR

| Feature | DETR | Deformable DETR | DINO |
|---|---|---|---|
| Attention | Standard | Deformable | Deformable |
| Queries | Content only | Content + ref point | Content + anchor box |
| Box prediction | Absolute | Absolute | Iterative delta |
| Denoising | No | No | Yes |
| Convergence | ~500 epochs | ~50 epochs | ~12-24 epochs |
| COCO AP | ~42 | ~47 | ~49-51 |
| NMS | Not needed | Not needed | Not needed |

### Evolution of improvements:

1. **DETR → Deformable DETR:** Replaced O(N²) attention with O(NK) deformable, added multi-scale, added reference points
2. **Deformable DETR → DAB-DETR:** Made reference points into full 4D anchors (cx, cy, w, h)
3. **DAB-DETR → DN-DETR:** Added denoising training
4. **DN-DETR → DINO:** Combined all above + mixed query selection + better anchor init + iterative refinement

---

## 10. Key Equations Summary

**Inverse sigmoid (logit):**
$$\sigma^{-1}(x) = \log\left(\frac{x}{1-x}\right)$$

**Iterative box refinement:**
$$\hat{b}_k = \sigma\left(\sigma^{-1}(r_k) + \Delta_k\right)$$

**Hungarian matching:**
$$\sigma^* = \arg\min_\sigma \sum_i \left[\lambda_1 C_{cls}(i, \sigma(i)) + \lambda_2 \|\hat{b}_{\sigma(i)} - b_i\|_1 + \lambda_3 (1 - \text{GIoU}(\hat{b}_{\sigma(i)}, b_i))\right]$$

**Focal loss:**
$$\text{FL}(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

**GIoU:**
$$\text{GIoU}(A,B) = \frac{|A \cap B|}{|A \cup B|} - \frac{|C \setminus (A \cup B)|}{|C|}$$

**Total training loss:**
$$L = \sum_{k=0}^{K-1} \left(\lambda_{cls} L_{cls}^{(k)} + \lambda_{L1} L_{L1}^{(k)} + \lambda_{giou} L_{giou}^{(k)}\right) + L_{dn}$$

---

## 11. What to Study in Code

### model.py
- `SimpleBackbone` → multi-scale feature extraction
- `LearnedPositionalEncoding2D` → position information for attention
- `TransformerEncoderLayer` → how Q/K/V are constructed, where position is added
- `TransformerDecoderLayer` → self-attention + cross-attention pattern
- `DetectionHead` → iterative refinement with inverse sigmoid
- `DenoisingGenerator` → how noisy queries are constructed
- `DINO._run_decoder()` → the iterative refinement loop with detached references

### loss.py
- `HungarianMatcher` → cost matrix construction + scipy assignment
- `sigmoid_focal_loss` → focal modulation for hard example mining
- `DINOLoss._compute_layer_loss()` → matching → classification + box losses
- `DINOLoss._compute_denoising_loss()` → direct loss without matching

### box_ops.py
- `generalized_box_iou` → enclosing box computation and GIoU formula
- `box_cxcywh_to_xyxy` → format conversion needed for IoU computation

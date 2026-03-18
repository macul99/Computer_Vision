# Counting And Tracking SOTA (March 2026)

There is no single best model for counting or tracking. The right answer depends on the task definition:

- Counting:
  - open-world or promptable object counting
  - few-shot or zero-shot exemplar counting
  - dense crowd counting
- Tracking:
  - multi-object tracking (MOT): track all people/vehicles in a video
  - single-object tracking (SOT): track one target given in the first frame

For study and practice, the most useful models are:

- Counting frontier: CountGD++, CountGD, LOCA, CounTR
- MOT frontier: MeMOTR, MOTR
- Practical MOT baselines: ByteTrack, BoT-SORT, OC-SORT
- SOT baseline to understand transformer tracking: OSTrack

## 1. Counting

### 1.1 Open-world counting: CountGD++

Paper/repo:
- https://github.com/niki-amini-naieni/CountGDPlusPlus

Why it matters:
- This is the clearest 2026 frontier model for open-world counting.
- It supports text prompts, positive visual exemplars, negative visual exemplars, and pseudo-exemplars.
- It is designed for the real problem: counting arbitrary categories instead of a fixed label set.

Core structure:
- Base model: GroundingDINO-style open-vocabulary detector.
- Text branch: encodes a positive text description of what to count.
- Exemplar branch: encodes one or more positive examples and, in CountGD++, negative examples.
- Cross-modal grounding: image features, text tokens, and exemplar features interact so the model can localize only the intended instances.
- Detection-based counting: the count is obtained from grounded detections rather than from integrating a density map.
- CountGD++ adds:
  - negative prompts to say what should not be counted
  - pseudo-exemplars automatically extracted from text-only input
  - adaptive cropping / test-time refinements in some evaluation settings

How to think about it:
- Older counting models usually predict a density map.
- CountGD++ instead reframes counting as open-vocabulary grounded detection plus filtering.
- That is why it scales better to arbitrary categories.

Typical losses:
- Because the model is detection-grounding based, the training objective is dominated by GroundingDINO-style losses:
  - bipartite matching / Hungarian matching
  - classification or token-alignment loss
  - box regression loss
  - GIoU loss
- In practice, the loss looks conceptually like:

$$
\mathcal{L} = \mathcal{L}_{align/cls} + \lambda_1 \mathcal{L}_{L1-box} + \lambda_2 \mathcal{L}_{GIoU} + \mathcal{L}_{aux}
$$

- CountGD++ also benefits from prompting logic that improves discrimination at inference time, especially with negative exemplars and pseudo-exemplars.

Important practical note:
- The repo states full training and inference code is scheduled for release by June 3, 2026.
- So this is the right model to know, but not yet the easiest full training codebase to learn from line-by-line.

Best use cases:
- open-world counting
- text-conditioned counting
- counting with exemplars
- ambiguous categories where negative exemplars help

### 1.2 Open-world counting: CountGD

Paper/repo:
- https://github.com/niki-amini-naieni/CountGD

Why it matters:
- This is the strongest fully available practical codebase for modern open-world counting.
- It introduced the core CountGD idea before CountGD++ generalized prompting further.

Core structure:
- GroundingDINO backbone for open-vocabulary region grounding.
- Text prompt encoder so the object category can be described in natural language.
- Visual exemplar branch so the target can also be specified by example patches.
- Joint prompt conditioning so text and exemplars can reinforce each other.
- Optional SAM-assisted test-time normalization and cropping in the released inference pipeline.

Losses:
- Detection and grounding losses inherited from the open-vocabulary detector base:
  - matching cost for assigning predictions to targets
  - text-region classification/alignment loss
  - box L1 loss
  - GIoU loss
  - auxiliary decoder losses

Why this is different from density-map counting:
- Density models learn count via a per-pixel map whose integral gives the count.
- CountGD learns to localize countable instances directly, so it often transfers better to unseen categories.

What to study in code:
- prompt handling
- GroundingDINO integration
- exemplar filtering
- inference thresholds
- SAM-based test-time normalization and cropping

### 1.3 Low-shot exemplar counting: LOCA

Paper/repo:
- https://github.com/djukicn/loca

Why it matters:
- LOCA is one of the cleanest low-shot counting models to study.
- It is much easier to understand than the newer multimodal grounding models.

Core structure:
- CNN backbone, typically ResNet-50 with self-supervised pretraining.
- Exemplar features are converted into prototypes.
- Iterative prototype adaptation updates those prototypes based on the query image.
- The adapted prototypes guide the network toward regions that match the target category.
- The final output is density-oriented rather than detection-oriented.

How it works conceptually:
- Start with a few example objects.
- Build a prototype of what the target looks like.
- Refine that prototype by looking at the full image.
- Predict a density/response map from which count is derived.

Losses:
- The repo workflow explicitly generates density maps, so the core supervision is density-map regression.
- In practice, this is typically an L2 or MSE-style loss between predicted and ground-truth density maps:

$$
\mathcal{L}_{count} = \lVert D_{pred} - D_{gt} \rVert_2^2
$$

- Some of the performance gain comes from the iterative prototype adaptation rather than from inventing a radically new loss.

Best use cases:
- few-shot object counting
- zero-shot and low-shot FSC-147 style benchmarks
- learning the exemplar-counting pipeline without the complexity of a detector foundation model

### 1.4 Transformer exemplar counting: CounTR

Paper/repo:
- https://github.com/Verg-Avesta/CounTR

Why it matters:
- CounTR is one of the most important transformer baselines for generalized visual counting.
- It is historically important because it made transformer-based exemplar counting competitive and easier to reason about.

Core structure:
- Image patches and exemplar patches are processed jointly.
- Transformer attention models similarity between the scene and the provided exemplars.
- Two-stage training:
  - self-supervised pretraining
  - supervised fine-tuning for counting
- Synthetic image generation is used during training to force the model to rely on exemplars.
- Output is density-style, not pure box detection.

Losses:
- Density map regression is the core supervision.
- In practice this is usually an L2/MSE-style objective over the density map, with the total count computed by summing the map.

Why still study it:
- Cleaner than CountGD if you want to understand exemplar-based counting.
- Good stepping stone before reading CountGD.

### 1.5 If you mean dense crowd counting specifically

For dense crowd counting, the strongest ideas are still density-map or localization-density hybrids rather than open-vocabulary detectors.

Representative families to know:
- DM-Count
- CLTR
- TopoCount

But if your goal is general CV practice, CountGD and LOCA are higher value to study first because they connect better to modern foundation-model thinking.

## 2. Tracking

Tracking splits into two different problems:

- MOT: detect and maintain identities for all objects across frames
- SOT: track one target given an initial box

### 2.1 End-to-end MOT: MeMOTR

Paper/repo:
- https://github.com/MCG-NJU/MeMOTR

Why it matters:
- MeMOTR is one of the strongest end-to-end transformer MOT repos with official code.
- It improves on MOTR by explicitly injecting long-term memory.
- It is particularly relevant on datasets like DanceTrack where association quality matters a lot.

Core structure:
- Deformable-DETR style visual encoder/decoder.
- Track queries, where each query is intended to follow one object across time.
- Memory-augmented attention so information from earlier frames is reused when later frames are ambiguous.
- Decoder predicts class, box, and identity-consistent track states frame by frame.

Why this is different from tracking-by-detection:
- Traditional pipelines detect every frame, then associate detections with heuristics.
- MeMOTR learns the temporal association inside the transformer using trainable track queries and memory.

Losses:
- DETR-style matching loss with one-to-one assignment between predicted track queries and ground-truth trajectories.
- Classification loss.
- Box L1 loss.
- GIoU loss.
- Auxiliary decoder-layer losses.
- Temporal identity consistency is encouraged by keeping the same track query attached to the same target over time through the matching scheme.

A useful mental model is:

$$
\mathcal{L} = \mathcal{L}_{cls} + \lambda_1 \mathcal{L}_{L1-box} + \lambda_2 \mathcal{L}_{GIoU} + \mathcal{L}_{aux} + \mathcal{L}_{temporal}
$$

Best use cases:
- learning end-to-end MOT
- studying transformer tracking
- understanding memory in video models

### 2.2 End-to-end MOT baseline: MOTR

Paper/repo:
- https://github.com/megvii-research/MOTR

Why it matters:
- MOTR introduced the key concept of track queries for MOT.
- Even if newer models outperform it, this is still the repo to read first if you want to understand the family.

Core structure:
- Built on Deformable DETR.
- Track queries persist through frames.
- Temporal aggregation network improves long-range reasoning.
- Tracklet-aware label assignment keeps learning consistent across time.

Losses:
- Hungarian matching over track queries
- class loss
- L1 box loss
- GIoU loss
- auxiliary decoder losses
- collective average loss / temporal training tricks described in the paper

Why study it before newer trackers:
- MeMOTR makes more sense once you understand MOTR.

### 2.3 Practical online MOT: ByteTrack

Paper/repo:
- https://github.com/FoundationVision/ByteTrack

Why it matters:
- This is still one of the most important practical MOT systems.
- It is simple, fast, and widely reused.
- Many later trackers either compare against it or build on its association logic.

Core structure:
- Detector: usually YOLOX in the official repo.
- Motion model: Kalman filter track state.
- Association: two-stage matching that keeps low-confidence detections instead of throwing them away immediately.
- Post-processing and interpolation for smoother trajectories.

Key idea:
- Most trackers drop low-score boxes too early.
- ByteTrack first matches high-score detections, then tries to recover occluded or weak detections using the low-score set.

Losses:
- ByteTrack itself is not trained end-to-end as a transformer tracker.
- The main learned losses come from the detector:
  - detection classification/objectness losses
  - box regression losses
- The association logic is algorithmic rather than learned from a dedicated tracking loss.

Why this matters:
- If you want the best system to hack on quickly, ByteTrack is easier than MOTR-style models.
- If you want to learn tracking losses, ByteTrack is less interesting than MeMOTR or MOTR.

### 2.4 Practical online MOT with stronger association: BoT-SORT

Paper/repo:
- https://github.com/NirAharon/BoT-SORT

Why it matters:
- BoT-SORT is a strong tracking-by-detection system with camera motion compensation and optional ReID.
- It is very good for understanding what makes practical MOT robust.

Core structure:
- Detector backbone from YOLOX or YOLOv7 in the repo.
- Kalman filter state for motion prediction.
- Appearance embedding branch via ReID.
- Camera motion compensation.
- Data association combining motion and appearance.

Losses:
- Detector losses come from the chosen detector.
- ReID network uses metric-learning / identity classification losses during its own training.
- The tracker itself is largely heuristic/algorithmic at inference time.

When to study it:
- if you want the strongest practical MOT recipe
- if your videos have camera motion
- if ID switches are the main failure mode

### 2.5 Motion-first online MOT: OC-SORT

Paper/repo:
- https://github.com/noahcao/OC_SORT

Why it matters:
- OC-SORT shows how far strong motion modeling can go, even without heavy appearance models.
- It is a good contrast with BoT-SORT and ByteTrack.

Core structure:
- SORT-style motion tracking
- observation-centric re-update to better handle non-linear motion and occlusion
- flexible association cost choices
- optional deeper variants like Deep-OC-SORT add appearance cues

Losses:
- Like ByteTrack, the tracking logic itself is mostly algorithmic.
- The detector contributes the learned loss.

When to study it:
- if you want a simpler tracker with strong motion reasoning
- if you want a very readable online MOT codebase

## 3. Single-object tracking: OSTrack

Paper/repo:
- https://github.com/botaoye/OSTrack

Why it matters:
- OSTrack is still one of the clearest transformer SOT baselines to study.
- It replaced older two-stream Siamese thinking with a one-stream transformer formulation.

Core structure:
- Template patch from the first frame
- Search region from the current frame
- One-stream transformer that concatenates template and search tokens
- Self-attention jointly does feature learning and relation modeling
- Candidate elimination module removes unlikely tokens early to save compute
- Box head predicts the target location in the current frame

Why it is elegant:
- Earlier trackers often had separate template/search branches and hand-designed fusion.
- OSTrack lets a single transformer do both representation learning and matching.

Typical losses:
- SOT models usually use a mix of:
  - target classification / foreground confidence loss
  - box regression loss
  - IoU or GIoU-style overlap loss
- The repo builds on STARK/PyTracking conventions, so think of it as transformer box prediction with standard localization supervision.

Best use cases:
- learn modern SOT
- understand template-search token interaction
- build a custom tracker for a single object category or robotic target

## 4. What to study first

If your goal is understanding modern ideas, study in this order:

1. CounTR
   - easiest path into exemplar-based counting
2. LOCA
   - clean low-shot counting design
3. CountGD
   - modern open-world counting with multimodal prompts
4. MOTR
   - first track-query MOT model to really understand
5. MeMOTR
   - stronger memory-augmented end-to-end MOT
6. ByteTrack
   - practical MOT system you can train and deploy faster
7. OSTrack
   - modern transformer baseline for single-object tracking

If your goal is building something usable fast, study in this order:

1. ByteTrack
2. BoT-SORT
3. CountGD
4. OSTrack

## 5. Repo shortlist for code study and practice

### Counting
- CountGD: https://github.com/niki-amini-naieni/CountGD
- CountGD++: https://github.com/niki-amini-naieni/CountGDPlusPlus
- LOCA: https://github.com/djukicn/loca
- CounTR: https://github.com/Verg-Avesta/CounTR

### Multi-object tracking
- MeMOTR: https://github.com/MCG-NJU/MeMOTR
- MOTR: https://github.com/megvii-research/MOTR
- ByteTrack: https://github.com/FoundationVision/ByteTrack
- BoT-SORT: https://github.com/NirAharon/BoT-SORT
- OC-SORT: https://github.com/noahcao/OC_SORT

### Single-object tracking
- OSTrack: https://github.com/botaoye/OSTrack

## 6. Bottom line

If you want one short answer:

- Best modern counting family: CountGD / CountGD++
- Best clean low-shot counting baseline: LOCA
- Best research-style end-to-end MOT family: MeMOTR / MOTR
- Best practical MOT baseline: ByteTrack
- Best practical MOT with stronger association engineering: BoT-SORT
- Best single-object transformer tracker to study: OSTrack

If you only have time for four repos, use these:

1. https://github.com/niki-amini-naieni/CountGD
2. https://github.com/MCG-NJU/MeMOTR
3. https://github.com/FoundationVision/ByteTrack
4. https://github.com/botaoye/OSTrack
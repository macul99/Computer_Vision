# Minimal Faster R-CNN (From Scratch, Study-Oriented)

This folder contains a compact, fully readable Faster R-CNN implementation in PyTorch for learning purposes.

## Files

- `model.py`: end-to-end model (backbone + RPN + ROI head).
- `box_ops.py`: box geometry helpers (IoU, encode/decode, clipping, filtering).
- `train_demo.py`: synthetic data smoke test for train/inference flow.
- `MATH_AND_ALGO_README.md`: detailed algorithm and mathematical explanation.

## Install

```bash
pip install torch torchvision
```

## Run Demo

From workspace root:

```bash
python -m faster_rcnn_min.train_demo
```

You should see:
- training loss dictionaries for a few iterations,
- inference summary with number of detections per image.

## Expected Inputs and Outputs

### Training Mode

```python
model.train()
losses = model(images, targets)
```

- `images`: list of tensors, each `[3, H, W]` with same `H, W`.
- `targets`: list of dicts:
  - `boxes`: float tensor `[N, 4]` in `(x1, y1, x2, y2)`.
  - `labels`: long tensor `[N]`, class ids in `[1, num_classes - 1]`.

Returns dict with:
- `loss_rpn_objectness`
- `loss_rpn_box_reg`
- `loss_roi_classifier`
- `loss_roi_box_reg`

### Inference Mode

```python
model.eval()
with torch.no_grad():
    detections = model(images)
```

Returns list (one per image):
- `boxes`: `[M, 4]`
- `scores`: `[M]`
- `labels`: `[M]`

## Important Notes

- This implementation uses a fixed single feature map and fixed image size for clarity.
- It is intentionally explicit and not heavily optimized.
- It is ideal for studying logic and tensor flow before moving to production-grade code (e.g. FPN, multi-scale training, mixed precision, distributed training).

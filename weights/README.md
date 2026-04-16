# Weights Directory

## Purpose

This directory stores generated weight files used by the HLS implementation of YOLO26n.

The current repository does **not** directly include full pretrained weights in C++ header form because:

- pretrained model weights are first exported through ONNX
- ONNX initializers must be parsed
- floating-point values should be converted to fixed-point format
- generated arrays can become very large for GitHub commits

So this folder is used to hold generated artifacts such as:

- `yolo26n_params.npz`
- `weights.hpp`
- optional per-layer weight text files
- quantized parameter dumps

---

## Planned Flow

1. Export pretrained model
   - `scripts/export_yolo26n.py`

2. Parse ONNX weights
   - `scripts/parse_onnx_weights.py`

3. Convert float weights to fixed-point arrays
   - future script: `scripts/generate_hls_weights.py`

4. Save generated HLS-compatible headers here
   - example: `weights/weights.hpp`

---

## Suggested Files

Example future contents:

```text
weights/
├── README.md
├── yolo26n_params.npz
├── weights.hpp
├── conv1_weights.txt
├── conv1_bias.txt

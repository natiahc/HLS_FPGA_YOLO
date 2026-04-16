# HLS Implementation of YOLO26n for FPGA-Based Edge Inference

## Overview

This project implements a **High-Level Synthesis (HLS)** representation of the YOLO26n object detection model for FPGA deployment.

The goal is to design a **low-latency, hardware-efficient inference pipeline** suitable for edge devices using:

* Fixed-point arithmetic
* Parallel computation
* Memory-efficient dataflow

---

## Motivation

Modern object detection models like YOLO26 are optimized for edge deployment:

* End-to-end detection (no NMS required)
* Efficient architecture
* Suitable for real-time systems

However, deploying them on FPGA requires:

* Mapping neural ops → hardware blocks
* Optimizing memory + compute
* Using HLS tools like Vitis

---

## Features

* HLS-based implementation (Vitis compatible)
* Fixed-point computation (`ap_fixed`)
* Modular architecture:

  * Convolution blocks
  * Residual blocks (C3k2-style)
  * SPPF block
  * Upsampling + concatenation
  * Detection head
* ONNX export + weight extraction pipeline

---

## Project Architecture

YOLO26n Pipeline (Simplified):

Input Image (640x640x3)
→ Conv (Stride 2)
→ Conv (Stride 2)
→ C3k2 Block
→ Feature Map
→ Detection Head
→ Output (Max 300 detections)

---

## Tech Stack

* C++ (HLS)
* Xilinx Vitis HLS
* Python (model export)
* ONNX

---

## Build & Run (HLS Simulation)

### Step 1: Open Vitis HLS

```
vitis_hls
```

### Step 2: Create Project

* Add files from `/hls`
* Set top function: `yolo26n_hls_top`

### Step 3: Run Simulation

* C Simulation
* C Synthesis

---

## Model Export

Run:

```
cd scripts
python export_yolo26n.py
```

This generates:

* `yolo26n.onnx`

---

## Weight Extraction

```
python parse_onnx_weights.py
```

---

## Future Work

* Integrate full YOLO26 architecture
* Implement attention (C2PSA)
* Add quantization pipeline
* Optimize BRAM/DSP usage
* Deploy on FPGA (U200 / ZCU102)

---

## License

MIT License

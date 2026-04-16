# Project Details: YOLO26n HLS FPGA Implementation

## Overview

This project implements a **hardware-oriented version of the YOLO26n object detection model** using High-Level Synthesis (HLS) for FPGA deployment.

The implementation focuses on:
- Mapping neural network operations to hardware-friendly modules
- Using fixed-point arithmetic for efficiency
- Designing a modular and scalable pipeline

---

## Repository Structure

```text
yolo26-hls-fpga/
│
├── README.md
├── report/
│   ├── report.md
│   └── details.md
│
├── hls/
│   ├── yolo26_hls.hpp
│   ├── yolo26_hls.cpp
│   ├── tb_yolo26_hls.cpp
│
├── scripts/
│   ├── export_yolo26n.py
│   ├── parse_onnx_weights.py
│
├── weights/
│   └── README.md

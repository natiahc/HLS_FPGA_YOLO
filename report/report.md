# HLS Implementation of YOLO26n for FPGA-Based Edge Inference

## 1. Introduction

Object detection has become a fundamental task in computer vision, widely used in applications such as autonomous driving, surveillance, robotics, and smart cities. Modern deep learning models like YOLO (You Only Look Once) provide real-time detection capabilities with high accuracy.

The latest evolution, **YOLO26**, introduces improvements such as:

* End-to-end detection (eliminating Non-Maximum Suppression)
* Efficient architecture for edge deployment
* Faster inference performance

However, deploying such models on resource-constrained edge devices remains challenging due to:

* High computational complexity
* Memory bandwidth limitations
* Power constraints

This project addresses these challenges by implementing a **hardware-efficient representation of YOLO26n using High-Level Synthesis (HLS)** targeting FPGA platforms.

---

## 2. Problem Statement

The objective of this project is:

To design and implement an FPGA-optimized HLS representation of the YOLO26n object detection model that achieves low latency and efficient resource utilization for real-time edge inference.

Key challenges include:

* Mapping deep learning operators to hardware
* Optimizing memory access patterns
* Managing parallelism and pipelining
* Reducing precision without significant accuracy loss

---

## 3. Objectives

1. Study the architecture of YOLO26n and identify key computational blocks
2. Design an HLS-based pipeline for inference
3. Implement core operations:

   * Convolution layers
   * Residual blocks (C3k2-style)
   * SPPF block
   * Upsampling and concatenation
   * Detection head
4. Apply hardware optimizations:

   * Loop unrolling
   * Pipelining
   * Array partitioning
5. Evaluate performance in terms of:

   * Latency
   * Throughput
   * Resource utilization (LUT, DSP, BRAM)

---

## 4. Background

### 4.1 YOLO26 Architecture

YOLO26 is an evolution of the YOLO family designed for:

* Efficient edge deployment
* Reduced post-processing
* Simplified detection pipeline

Key components:

* Backbone (feature extraction)
* Neck (feature fusion)
* Detection head (bounding box prediction)

Unlike earlier YOLO models, YOLO26:

* Produces final detections directly
* Eliminates the need for NMS
* Outputs a fixed number of detections (e.g., 300)

---

### 4.2 FPGA and HLS

Field Programmable Gate Arrays (FPGAs) provide:

* Massive parallelism
* Custom data paths
* Energy-efficient computation

High-Level Synthesis (HLS):

* Converts C/C++ code → hardware (RTL)
* Enables faster hardware design
* Supports optimizations via pragmas

---

## 5. Methodology

### 5.1 System Flow

1. Train or load pretrained YOLO26n model
2. Export model to ONNX format
3. Extract weights and parameters
4. Convert floating-point weights to fixed-point
5. Implement HLS modules
6. Integrate modules into pipeline
7. Simulate and synthesize using Vitis HLS

---

### 5.2 HLS Design

The implementation consists of the following modules:

#### 5.2.1 Convolution Block

* Performs 2D convolution
* Uses fixed-point arithmetic
* Includes activation (SiLU)

#### 5.2.2 Residual Block (C3k2-style)

* Two convolution layers
* Skip connection
* Improves feature learning

#### 5.2.3 SPPF Block

* Spatial pyramid pooling
* Uses repeated max pooling
* Expands receptive field

#### 5.2.4 Upsampling

* Nearest neighbor interpolation
* Used for feature fusion

#### 5.2.5 Concatenation

* Channel-wise merge of feature maps

#### 5.2.6 Detection Head

* Produces bounding boxes and scores
* Simplified for initial implementation

---

### 5.3 Hardware Optimizations

* **Loop Pipelining**
  Reduces latency by overlapping operations

* **Loop Unrolling**
  Increases parallelism

* **Array Partitioning**
  Improves memory bandwidth

* **Fixed-Point Arithmetic**
  Reduces resource usage compared to floating-point

---

## 6. Implementation Details

* Language: C++ (HLS compatible)
* Tool: Xilinx Vitis HLS
* Data Type: `ap_fixed<16,6>`
* Input Size: 640 × 640 × 3
* Output: Up to 300 detections

The implementation is modular, enabling:

* Easy replacement of blocks
* Incremental optimization
* Scalability

---

## 7. Results (Expected)

The expected outcomes include:

* Functional HLS simulation of YOLO26n pipeline
* Reduced latency compared to CPU inference
* Efficient FPGA resource utilization

Metrics to evaluate:

* Latency (cycles)
* Throughput (FPS)
* Resource usage (LUTs, DSPs, BRAM)

---

## 8. Limitations

* Full YOLO26 architecture not completely implemented
* Simplified detection head
* No quantization-aware training
* No attention module (C2PSA) yet

---

## 9. Future Work

* Implement full YOLO26 architecture
* Add attention modules (C2PSA)
* Integrate quantization pipeline
* Optimize memory hierarchy
* Deploy on FPGA hardware (e.g., Xilinx U200)
* Compare with GPU/CPU implementations

---

## 10. Conclusion

This project demonstrates the feasibility of mapping modern object detection models like YOLO26n to FPGA using HLS.

The modular design and hardware-aware optimizations provide a strong foundation for:

* Real-time edge inference
* Efficient AI acceleration
* Future research in FPGA-based deep learning systems

---

## References

1. Ultralytics YOLO Documentation
2. Xilinx Vitis HLS User Guide
3. ONNX Model Format Documentation
4. Research papers on FPGA-based CNN acceleration

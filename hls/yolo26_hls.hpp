#ifndef YOLO26_HLS_HPP
#define YOLO26_HLS_HPP

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

// ============================================================
// Global Configuration
// ============================================================

static constexpr int IMG_H = 640;
static constexpr int IMG_W = 640;
static constexpr int IMG_C = 3;

static constexpr int NUM_CLASSES = 80;
static constexpr int MAX_DET = 300;

// Fixed-point configuration
typedef ap_fixed<16, 6> data_t;
typedef ap_fixed<16, 6> weight_t;
typedef ap_fixed<24, 10> acc_t;

// ============================================================
// Tensor Container
// ============================================================

template<int C, int H, int W>
struct Tensor3D {
    data_t data[C][H][W];
};

// ============================================================
// Detection Output Structure
// ============================================================

struct Detection {
    data_t x1;
    data_t y1;
    data_t x2;
    data_t y2;
    data_t score;
    ap_uint<16> cls;
};

// ============================================================
// Utility Functions
// ============================================================

data_t silu(data_t x);

// ============================================================
// Core Operator Prototypes
// ============================================================

// Conv + bias + SiLU activation
template<
    int IN_C, int OUT_C,
    int IN_H, int IN_W,
    int K, int STRIDE, int PAD,
    int OUT_H, int OUT_W
>
void conv2d_silu(
    Tensor3D<IN_C, IN_H, IN_W>& in,
    Tensor3D<OUT_C, OUT_H, OUT_W>& out,
    weight_t weights[OUT_C][IN_C][K][K],
    weight_t bias[OUT_C]
);

// Conv + bias without activation
template<
    int IN_C, int OUT_C,
    int IN_H, int IN_W,
    int K, int STRIDE, int PAD,
    int OUT_H, int OUT_W
>
void conv2d_linear(
    Tensor3D<IN_C, IN_H, IN_W>& in,
    Tensor3D<OUT_C, OUT_H, OUT_W>& out,
    weight_t weights[OUT_C][IN_C][K][K],
    weight_t bias[OUT_C]
);

// Nearest-neighbor upsampling by 2x
template<int C, int IN_H, int IN_W, int OUT_H, int OUT_W>
void upsample2x(
    Tensor3D<C, IN_H, IN_W>& in,
    Tensor3D<C, OUT_H, OUT_W>& out
);

// Channel concatenation
template<int C1, int C2, int H, int W>
void concat_ch(
    Tensor3D<C1, H, W>& a,
    Tensor3D<C2, H, W>& b,
    Tensor3D<C1 + C2, H, W>& out
);

// 5x5 maxpool with same padding
template<int C, int H, int W>
void maxpool5x5_same(
    Tensor3D<C, H, W>& in,
    Tensor3D<C, H, W>& out
);

// SPPF-style concat block
template<int C, int H, int W>
void sppf_concat(
    Tensor3D<C, H, W>& in,
    Tensor3D<4 * C, H, W>& out
);

// Simplified C3k2-style residual block
template<int C, int H, int W>
void c3k2_block(
    Tensor3D<C, H, W>& in,
    Tensor3D<C, H, W>& out,
    weight_t w1[C][C][3][3],
    weight_t b1[C],
    weight_t w2[C][C][3][3],
    weight_t b2[C]
);

// Simplified detection head
template<int C, int H, int W>
void detect_head_simple(
    Tensor3D<C, H, W>& feat,
    Detection dets[MAX_DET],
    int& det_count
);

// ============================================================
// Top-Level YOLO26n HLS Function
// ============================================================

void yolo26n_hls_top(
    data_t image[IMG_C][IMG_H][IMG_W],
    Detection dets[MAX_DET],
    int& det_count
);

#endif

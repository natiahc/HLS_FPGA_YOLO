#include "yolo26_hls.hpp"
#include <math.h>

// ============================================================
// Utility Functions
// ============================================================

data_t silu(data_t x) {
    float xf = (float)x;
    float y = xf / (1.0f + expf(-xf));
    return (data_t)y;
}

// ============================================================
// Convolution + SiLU
// ============================================================

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
) {
#pragma HLS INLINE off

    for (int oc = 0; oc < OUT_C; oc++) {
        for (int oh = 0; oh < OUT_H; oh++) {
            for (int ow = 0; ow < OUT_W; ow++) {
#pragma HLS PIPELINE II=1
                acc_t sum = bias[oc];

                for (int ic = 0; ic < IN_C; ic++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int ih = oh * STRIDE + kh - PAD;
                            int iw = ow * STRIDE + kw - PAD;

                            if (ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W) {
                                sum += in.data[ic][ih][iw] * weights[oc][ic][kh][kw];
                            }
                        }
                    }
                }

                out.data[oc][oh][ow] = silu((data_t)sum);
            }
        }
    }
}

// ============================================================
// Convolution + Linear
// ============================================================

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
) {
#pragma HLS INLINE off

    for (int oc = 0; oc < OUT_C; oc++) {
        for (int oh = 0; oh < OUT_H; oh++) {
            for (int ow = 0; ow < OUT_W; ow++) {
#pragma HLS PIPELINE II=1
                acc_t sum = bias[oc];

                for (int ic = 0; ic < IN_C; ic++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int ih = oh * STRIDE + kh - PAD;
                            int iw = ow * STRIDE + kw - PAD;

                            if (ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W) {
                                sum += in.data[ic][ih][iw] * weights[oc][ic][kh][kw];
                            }
                        }
                    }
                }

                out.data[oc][oh][ow] = (data_t)sum;
            }
        }
    }
}

// ============================================================
// Upsample 2x
// ============================================================

template<int C, int IN_H, int IN_W, int OUT_H, int OUT_W>
void upsample2x(
    Tensor3D<C, IN_H, IN_W>& in,
    Tensor3D<C, OUT_H, OUT_W>& out
) {
#pragma HLS INLINE off

    for (int c = 0; c < C; c++) {
        for (int h = 0; h < OUT_H; h++) {
            for (int w = 0; w < OUT_W; w++) {
#pragma HLS PIPELINE II=1
                out.data[c][h][w] = in.data[c][h / 2][w / 2];
            }
        }
    }
}

// ============================================================
// Channel Concatenation
// ============================================================

template<int C1, int C2, int H, int W>
void concat_ch(
    Tensor3D<C1, H, W>& a,
    Tensor3D<C2, H, W>& b,
    Tensor3D<C1 + C2, H, W>& out
) {
#pragma HLS INLINE off

    for (int c = 0; c < C1; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
#pragma HLS PIPELINE II=1
                out.data[c][h][w] = a.data[c][h][w];
            }
        }
    }

    for (int c = 0; c < C2; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
#pragma HLS PIPELINE II=1
                out.data[C1 + c][h][w] = b.data[c][h][w];
            }
        }
    }
}

// ============================================================
// MaxPool 5x5 Same
// ============================================================

template<int C, int H, int W>
void maxpool5x5_same(
    Tensor3D<C, H, W>& in,
    Tensor3D<C, H, W>& out
) {
#pragma HLS INLINE off

    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
#pragma HLS PIPELINE II=1
                data_t maxv = in.data[c][h][w];

                for (int kh = -2; kh <= 2; kh++) {
                    for (int kw = -2; kw <= 2; kw++) {
                        int ih = h + kh;
                        int iw = w + kw;

                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            if (in.data[c][ih][iw] > maxv) {
                                maxv = in.data[c][ih][iw];
                            }
                        }
                    }
                }

                out.data[c][h][w] = maxv;
            }
        }
    }
}

// ============================================================
// SPPF Concat
// ============================================================

template<int C, int H, int W>
void sppf_concat(
    Tensor3D<C, H, W>& in,
    Tensor3D<4 * C, H, W>& out
) {
#pragma HLS INLINE off

    Tensor3D<C, H, W> p1;
    Tensor3D<C, H, W> p2;
    Tensor3D<C, H, W> p3;

    maxpool5x5_same(in, p1);
    maxpool5x5_same(p1, p2);
    maxpool5x5_same(p2, p3);

    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
#pragma HLS PIPELINE II=1
                out.data[c][h][w] = in.data[c][h][w];
                out.data[C + c][h][w] = p1.data[c][h][w];
                out.data[2 * C + c][h][w] = p2.data[c][h][w];
                out.data[3 * C + c][h][w] = p3.data[c][h][w];
            }
        }
    }
}

// ============================================================
// Simplified C3k2 Block
// ============================================================

template<int C, int H, int W>
void c3k2_block(
    Tensor3D<C, H, W>& in,
    Tensor3D<C, H, W>& out,
    weight_t w1[C][C][3][3],
    weight_t b1[C],
    weight_t w2[C][C][3][3],
    weight_t b2[C]
) {
#pragma HLS INLINE off

    Tensor3D<C, H, W> t1;
    Tensor3D<C, H, W> t2;

    conv2d_silu<C, C, H, W, 3, 1, 1, H, W>(in, t1, w1, b1);
    conv2d_silu<C, C, H, W, 3, 1, 1, H, W>(t1, t2, w2, b2);

    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
#pragma HLS PIPELINE II=1
                out.data[c][h][w] = in.data[c][h][w] + t2.data[c][h][w];
            }
        }
    }
}

// ============================================================
// Simplified Detection Head
// ============================================================

template<int C, int H, int W>
void detect_head_simple(
    Tensor3D<C, H, W>& feat,
    Detection dets[MAX_DET],
    int& det_count
) {
#pragma HLS INLINE off

    det_count = 0;

    for (int h = 0; h < H && det_count < MAX_DET; h++) {
        for (int w = 0; w < W && det_count < MAX_DET; w++) {
#pragma HLS PIPELINE II=1
            data_t obj = feat.data[0][h][w];

            if (obj > (data_t)0.5) {
                Detection d;
                d.x1 = (data_t)(w * 4);
                d.y1 = (data_t)(h * 4);
                d.x2 = (data_t)(w * 4 + 32);
                d.y2 = (data_t)(h * 4 + 32);
                d.score = obj;
                d.cls = 0;
                dets[det_count++] = d;
            }
        }
    }
}

// ============================================================
// Top Function
// ============================================================

void yolo26n_hls_top(
    data_t image[IMG_C][IMG_H][IMG_W],
    Detection dets[MAX_DET],
    int& det_count
) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=dets offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=det_count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    Tensor3D<3, 640, 640> in;
    Tensor3D<16, 320, 320> x1;
    Tensor3D<32, 160, 160> x2;
    Tensor3D<32, 160, 160> x3;

    static weight_t w1[16][3][3][3];
    static weight_t b1[16];
    static weight_t w2[32][16][3][3];
    static weight_t b2[32];
    static weight_t wc31[32][32][3][3];
    static weight_t bc31[32];
    static weight_t wc32[32][32][3][3];
    static weight_t bc32[32];

#pragma HLS ARRAY_PARTITION variable=w1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=w2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=wc31 complete dim=1
#pragma HLS ARRAY_PARTITION variable=wc32 complete dim=1

    for (int c = 0; c < IMG_C; c++) {
        for (int h = 0; h < IMG_H; h++) {
            for (int w = 0; w < IMG_W; w++) {
#pragma HLS PIPELINE II=1
                in.data[c][h][w] = image[c][h][w];
            }
        }
    }

    // Stem
    conv2d_silu<3, 16, 640, 640, 3, 2, 1, 320, 320>(in, x1, w1, b1);

    // Downsample
    conv2d_silu<16, 32, 320, 320, 3, 2, 1, 160, 160>(x1, x2, w2, b2);

    // Simplified C3k2 block
    c3k2_block<32, 160, 160>(x2, x3, wc31, bc31, wc32, bc32);

    // Simplified detection
    detect_head_simple<32, 160, 160>(x3, dets, det_count);
}

// ============================================================
// Explicit Template Instantiations
// ============================================================

template void conv2d_silu<3, 16, 640, 640, 3, 2, 1, 320, 320>(
    Tensor3D<3, 640, 640>&,
    Tensor3D<16, 320, 320>&,
    weight_t[16][3][3][3],
    weight_t[16]
);

template void conv2d_silu<16, 32, 320, 320, 3, 2, 1, 160, 160>(
    Tensor3D<16, 320, 320>&,
    Tensor3D<32, 160, 160>&,
    weight_t[32][16][3][3],
    weight_t[32]
);

template void conv2d_linear<3, 16, 640, 640, 3, 2, 1, 320, 320>(
    Tensor3D<3, 640, 640>&,
    Tensor3D<16, 320, 320>&,
    weight_t[16][3][3][3],
    weight_t[16]
);

template void upsample2x<32, 160, 160, 320, 320>(
    Tensor3D<32, 160, 160>&,
    Tensor3D<32, 320, 320>&
);

template void concat_ch<16, 16, 320, 320>(
    Tensor3D<16, 320, 320>&,
    Tensor3D<16, 320, 320>&,
    Tensor3D<32, 320, 320>&
);

template void maxpool5x5_same<32, 160, 160>(
    Tensor3D<32, 160, 160>&,
    Tensor3D<32, 160, 160>&
);

template void sppf_concat<32, 160, 160>(
    Tensor3D<32, 160, 160>&,
    Tensor3D<128, 160, 160>&
);

template void c3k2_block<32, 160, 160>(
    Tensor3D<32, 160, 160>&,
    Tensor3D<32, 160, 160>&,
    weight_t[32][32][3][3],
    weight_t[32],
    weight_t[32][32][3][3],
    weight_t[32]
);

template void detect_head_simple<32, 160, 160>(
    Tensor3D<32, 160, 160>&,
    Detection[MAX_DET],
    int&
);

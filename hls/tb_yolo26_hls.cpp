#include "yolo26_hls.hpp"
#include <iostream>

int main() {
    static data_t image[IMG_C][IMG_H][IMG_W];
    static Detection dets[MAX_DET];
    int det_count = 0;

    // ------------------------------------------------------------
    // Initialize input image with dummy pattern
    // ------------------------------------------------------------
    for (int c = 0; c < IMG_C; c++) {
        for (int h = 0; h < IMG_H; h++) {
            for (int w = 0; w < IMG_W; w++) {
                image[c][h][w] = (data_t)(((c + h + w) % 256) / 255.0);
            }
        }
    }

    // ------------------------------------------------------------
    // Call top-level HLS function
    // ------------------------------------------------------------
    yolo26n_hls_top(image, dets, det_count);

    // ------------------------------------------------------------
    // Print results
    // ------------------------------------------------------------
    std::cout << "Detection Count: " << det_count << std::endl;

    int print_count = (det_count < 10) ? det_count : 10;
    for (int i = 0; i < print_count; i++) {
        std::cout
            << "det[" << i << "] "
            << "x1=" << dets[i].x1 << ", "
            << "y1=" << dets[i].y1 << ", "
            << "x2=" << dets[i].x2 << ", "
            << "y2=" << dets[i].y2 << ", "
            << "score=" << dets[i].score << ", "
            << "cls=" << dets[i].cls
            << std::endl;
    }

    return 0;
}

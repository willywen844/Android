// grid_sample_bilinear_nhwc
// 等效 F.grid_sample(value, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
//
//   value_nhwc : [N, H, W, C]   float32，注意是 NHWC（通道連續），從 PyTorch 的 NCHW 轉一次即可
//   grid       : [N, P, 2]      float32，(x, y) 值域 [-1, 1]，P = Ho*Wo
//   out        : [N, P, C]      float32（要回到 NCHW/[N,C,Ho,Wo] 再轉一次）
//
// C 目前固定為 16（你的 head dim），NEON 路徑一次處理 4 個 float，16 通道 = 4 個 q 暫存器。
#include <stdint.h>
#include <stddef.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define CH 16

// 比 floorf 快、不依賴 libm（A7 沒有 vector floor 指令）
static inline int fast_floor(float v) { int i = (int)v; return i - (v < (float)i); }

static inline void tap_accumulate(const float *src, float w,
#ifdef __ARM_NEON
                                  float32x4_t acc[4]
#else
                                  float acc[CH]
#endif
) {
#ifdef __ARM_NEON
    acc[0] = vmlaq_n_f32(acc[0], vld1q_f32(src + 0), w);
    acc[1] = vmlaq_n_f32(acc[1], vld1q_f32(src + 4), w);
    acc[2] = vmlaq_n_f32(acc[2], vld1q_f32(src + 8), w);
    acc[3] = vmlaq_n_f32(acc[3], vld1q_f32(src + 12), w);
#else
    for (int c = 0; c < CH; ++c) acc[c] += src[c] * w;
#endif
}

void grid_sample_bilinear_nhwc(const float *value_nhwc, const float *grid, float *out,
                               int N, int H, int W, int P)
{
    const float sx = 0.5f * (float)W, bx = 0.5f * (float)(W - 1);   // align_corners=False
    const float sy = 0.5f * (float)H, by = 0.5f * (float)(H - 1);

    for (int n = 0; n < N; ++n) {
        const float *img = value_nhwc + (size_t)n * H * W * CH;
        const float *g   = grid + (size_t)n * P * 2;
        float       *o   = out  + (size_t)n * P * CH;

        for (int p = 0; p < P; ++p) {
            float x = g[2 * p + 0] * sx + bx;         // 像素座標
            float y = g[2 * p + 1] * sy + by;
            int x0 = fast_floor(x), y0 = fast_floor(y);
            int x1 = x0 + 1,        y1 = y0 + 1;
            float fx = x - (float)x0, fy = y - (float)y0;
            float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy);
            float w01 = (1 - fx) * fy,       w11 = fx * fy;

#ifdef __ARM_NEON
            float32x4_t acc[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
#else
            float acc[CH] = {0};
#endif
            // zeros padding：出界的 tap 直接跳過
            int xin0 = (x0 >= 0 && x0 < W), xin1 = (x1 >= 0 && x1 < W);
            int yin0 = (y0 >= 0 && y0 < H), yin1 = (y1 >= 0 && y1 < H);
            if (yin0 && xin0) tap_accumulate(img + ((size_t)y0 * W + x0) * CH, w00, acc);
            if (yin0 && xin1) tap_accumulate(img + ((size_t)y0 * W + x1) * CH, w10, acc);
            if (yin1 && xin0) tap_accumulate(img + ((size_t)y1 * W + x0) * CH, w01, acc);
            if (yin1 && xin1) tap_accumulate(img + ((size_t)y1 * W + x1) * CH, w11, acc);

#ifdef __ARM_NEON
            vst1q_f32(o + p * CH + 0,  acc[0]);
            vst1q_f32(o + p * CH + 4,  acc[1]);
            vst1q_f32(o + p * CH + 8,  acc[2]);
            vst1q_f32(o + p * CH + 12, acc[3]);
#else
            for (int c = 0; c < CH; ++c) o[p * CH + c] = acc[c];
#endif
        }
    }
}

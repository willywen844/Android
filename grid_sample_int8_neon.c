// grid_sample_bilinear_nhwc_int8
// int8 版本，等效 F.grid_sample(value, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
//
//   value_nhwc : [N, H, W, C]  int8，對稱量化：real = q * scale
//   grid       : [N, P, 2]     float32，(x, y) 值域 [-1, 1]
//   out        : [N, P, C]     int8，和 value 同一個 scale
//
// 權重量化成 Q6（0..64），四個 tap 權重和恰為 64，所以 int16 累加不會溢位：
//   sum(w_q6 * v_q) <= 64 * 127 = 8128
// 輸出 = round(sum >> 6)，落回 int8。
//
// NEON 路徑用 vmull.s8 / vmlal.s8（int8 x int8 -> int16，一次 8 lane），
// 這在 Cortex-A7 上是單 cycle 指令，比 vmla.f32 q（2 cycle、4 lane）划算。
#include <stdint.h>
#include <stddef.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define CH 16
#define WQ 64   // 權重定點：Q6

static inline int fast_floor(float v) { int i = (int)v; return i - (v < (float)i); }

#ifdef __ARM_NEON
static inline void tap_accumulate(const int8_t *src, int8_t w, int16x8_t acc[2]) {
    int8x8_t wv = vdup_n_s8(w);
    acc[0] = vmlal_s8(acc[0], vld1_s8(src + 0), wv);
    acc[1] = vmlal_s8(acc[1], vld1_s8(src + 8), wv);
}
#else
static inline void tap_accumulate(const int8_t *src, int8_t w, int16_t acc[CH]) {
    for (int c = 0; c < CH; ++c) acc[c] += (int16_t)src[c] * (int16_t)w;
}
#endif

void grid_sample_bilinear_nhwc_int8(const int8_t *value_nhwc, const float *grid, int8_t *out,
                                    int N, int H, int W, int P)
{
    const float sx = 0.5f * (float)W, bx = 0.5f * (float)(W - 1);   // align_corners=False
    const float sy = 0.5f * (float)H, by = 0.5f * (float)(H - 1);

    for (int n = 0; n < N; ++n) {
        const int8_t *img = value_nhwc + (size_t)n * H * W * CH;
        const float  *g   = grid + (size_t)n * P * 2;
        int8_t       *o   = out  + (size_t)n * P * CH;

        for (int p = 0; p < P; ++p) {
            float x = g[2 * p + 0] * sx + bx;
            float y = g[2 * p + 1] * sy + by;
            int x0 = fast_floor(x), y0 = fast_floor(y);
            int x1 = x0 + 1,        y1 = y0 + 1;

            // 小數部分量化成 Q6，四個權重用整數算，保證和為 64
            int fx = (int)((x - (float)x0) * WQ + 0.5f);
            int fy = (int)((y - (float)y0) * WQ + 0.5f);
            int gx = WQ - fx, gy = WQ - fy;
            int8_t w00 = (int8_t)((gx * gy + WQ / 2) / WQ);
            int8_t w10 = (int8_t)((fx * gy + WQ / 2) / WQ);
            int8_t w01 = (int8_t)((gx * fy + WQ / 2) / WQ);
            int8_t w11 = (int8_t)(WQ - w00 - w10 - w01);

#ifdef __ARM_NEON
            int16x8_t acc[2] = { vdupq_n_s16(0), vdupq_n_s16(0) };
#else
            int16_t acc[CH] = {0};
#endif
            int xin0 = (x0 >= 0 && x0 < W), xin1 = (x1 >= 0 && x1 < W);
            int yin0 = (y0 >= 0 && y0 < H), yin1 = (y1 >= 0 && y1 < H);
            if (yin0 && xin0) tap_accumulate(img + ((size_t)y0 * W + x0) * CH, w00, acc);
            if (yin0 && xin1) tap_accumulate(img + ((size_t)y0 * W + x1) * CH, w10, acc);
            if (yin1 && xin0) tap_accumulate(img + ((size_t)y1 * W + x0) * CH, w01, acc);
            if (yin1 && xin1) tap_accumulate(img + ((size_t)y1 * W + x1) * CH, w11, acc);

#ifdef __ARM_NEON
            // 帶 rounding 的右移 6 位並飽和窄化回 int8
            vst1_s8(o + p * CH + 0, vqrshrn_n_s16(acc[0], 6));
            vst1_s8(o + p * CH + 8, vqrshrn_n_s16(acc[1], 6));
#else
            for (int c = 0; c < CH; ++c) {
                int v = (acc[c] + (1 << 5)) >> 6;
                o[p * CH + c] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
            }
#endif
        }
    }
}

// ---------------------------------------------------------------------------
// 高精度變體：value 仍是 int8，但權重用 Q8（0..256）int16，int32 累加。
// 每個 tap 多了 vmovl.s8 的加寬，並用 4 條 vmlal.s16 取代 2 條 vmlal.s8，
// MAC 部分約慢 1.5–2 倍，換來位置精度 1/256 像素、誤差約 1 LSB 以內。
// ---------------------------------------------------------------------------
#define WQ8 256

#ifdef __ARM_NEON
static inline void tap_accumulate_q8(const int8_t *src, int16_t w, int32x4_t acc[4]) {
    int16x8_t lo = vmovl_s8(vld1_s8(src + 0));
    int16x8_t hi = vmovl_s8(vld1_s8(src + 8));
    acc[0] = vmlal_n_s16(acc[0], vget_low_s16(lo),  w);
    acc[1] = vmlal_n_s16(acc[1], vget_high_s16(lo), w);
    acc[2] = vmlal_n_s16(acc[2], vget_low_s16(hi),  w);
    acc[3] = vmlal_n_s16(acc[3], vget_high_s16(hi), w);
}
#else
static inline void tap_accumulate_q8(const int8_t *src, int16_t w, int32_t acc[CH]) {
    for (int c = 0; c < CH; ++c) acc[c] += (int32_t)src[c] * (int32_t)w;
}
#endif

void grid_sample_bilinear_nhwc_int8_q8(const int8_t *value_nhwc, const float *grid, int8_t *out,
                                       int N, int H, int W, int P)
{
    const float sx = 0.5f * (float)W, bx = 0.5f * (float)(W - 1);
    const float sy = 0.5f * (float)H, by = 0.5f * (float)(H - 1);

    for (int n = 0; n < N; ++n) {
        const int8_t *img = value_nhwc + (size_t)n * H * W * CH;
        const float  *g   = grid + (size_t)n * P * 2;
        int8_t       *o   = out  + (size_t)n * P * CH;

        for (int p = 0; p < P; ++p) {
            float x = g[2 * p + 0] * sx + bx;
            float y = g[2 * p + 1] * sy + by;
            int x0 = fast_floor(x), y0 = fast_floor(y);
            int x1 = x0 + 1,        y1 = y0 + 1;

            int fx = (int)((x - (float)x0) * WQ8 + 0.5f);
            int fy = (int)((y - (float)y0) * WQ8 + 0.5f);
            int gx = WQ8 - fx, gy = WQ8 - fy;
            int16_t w00 = (int16_t)((gx * gy + WQ8 / 2) / WQ8);
            int16_t w10 = (int16_t)((fx * gy + WQ8 / 2) / WQ8);
            int16_t w01 = (int16_t)((gx * fy + WQ8 / 2) / WQ8);
            int16_t w11 = (int16_t)(WQ8 - w00 - w10 - w01);

#ifdef __ARM_NEON
            int32x4_t acc[4] = { vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0) };
#else
            int32_t acc[CH] = {0};
#endif
            int xin0 = (x0 >= 0 && x0 < W), xin1 = (x1 >= 0 && x1 < W);
            int yin0 = (y0 >= 0 && y0 < H), yin1 = (y1 >= 0 && y1 < H);
            if (yin0 && xin0) tap_accumulate_q8(img + ((size_t)y0 * W + x0) * CH, w00, acc);
            if (yin0 && xin1) tap_accumulate_q8(img + ((size_t)y0 * W + x1) * CH, w10, acc);
            if (yin1 && xin0) tap_accumulate_q8(img + ((size_t)y1 * W + x0) * CH, w01, acc);
            if (yin1 && xin1) tap_accumulate_q8(img + ((size_t)y1 * W + x1) * CH, w11, acc);

#ifdef __ARM_NEON
            // int32 -> int16（帶 rounding 右移 8）-> int8（飽和）
            int16x8_t lo = vcombine_s16(vrshrn_n_s32(acc[0], 8), vrshrn_n_s32(acc[1], 8));
            int16x8_t hi = vcombine_s16(vrshrn_n_s32(acc[2], 8), vrshrn_n_s32(acc[3], 8));
            vst1_s8(o + p * CH + 0, vqmovn_s16(lo));
            vst1_s8(o + p * CH + 8, vqmovn_s16(hi));
#else
            for (int c = 0; c < CH; ++c) {
                int v = (acc[c] + (1 << 7)) >> 8;
                o[p * CH + c] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
            }
#endif
        }
    }
}

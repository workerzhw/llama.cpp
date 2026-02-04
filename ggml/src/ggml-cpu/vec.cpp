#include "vec.h"

// ---------------------------------------------------------------------------
// FP8(E4M3) block-quant simulation implementation
//
// 约定：
//  - FP8: 1 sign, 4 exp (bias=7), 3 mantissa
//  - 不生成 NaN/Inf（输入若超范围 -> 饱和到 max finite）
//  - 支持 subnormal；小于最小 subnormal 才刷 0
//  - 舍入：RNE (round-to-nearest-even)
//
// 这里实现的是“量化后立刻解量化”的回放，用于评估数值损失。
// ---------------------------------------------------------------------------

#if GGML_SIM_FP8E4M3

static inline int ggml_rne_int(float x) {
    // round-to-nearest-even for non-negative x
    // x is assumed finite and >=0
    const float f = floorf(x);
    const float r = x - f;
    int i = (int) f;
    if (r > 0.5f) {
        return i + 1;
    } else if (r < 0.5f) {
        return i;
    } else {
        // tie -> even
        return (i & 1) ? (i + 1) : i;
    }
}

static inline float ggml_fp8e4m3_max_finite(void) {
    // max finite for e4m3: exp=14 (1110), mant=7 => (1 + 7/8)*2^(14-7) = 1.875*128 = 240
    return 240.0f;
}

static inline float ggml_fp8e4m3_min_subnormal(void) {
    // exp=0, mant=1 => 2^(1-bias) * (mant/8) = 2^(-6) * (1/8) = 2^-9
    return ldexpf(1.0f, -9);
}

static inline float ggml_fp8e4m3_quant_dequant_one(float x) {
    // Applies: saturate, subnormal support, RNE. No NaN/Inf handling.
    if (x == 0.0f) {
        return 0.0f;
    }
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fabsf(x);

    // underflow to 0 below min subnormal
    const float min_sub = ggml_fp8e4m3_min_subnormal();
    if (ax < min_sub) {
        return 0.0f;
    }

    // saturate overflow
    const float max_f = ggml_fp8e4m3_max_finite();
    if (ax > max_f) {
        return sign * max_f;
    }

    // normalized threshold: exp==1, mant==0 => 1.0 * 2^(1-bias) = 2^-6
    const float min_norm = ldexpf(1.0f, -6);
    if (ax >= min_norm) {
        int e;
        float m = frexpf(ax, &e); // ax = m*2^e, m in [0.5,1)
        // normalize to [1,2)
        m *= 2.0f;
        e -= 1;
        // exp field = e + bias
        int ef = e + 7;
        if (ef <= 0) {
            // should not happen because ax>=min_norm
            ef = 0;
        }
        if (ef >= 15) {
            return sign * max_f;
        }
        // mantissa bits: 3, represent fractional part of m in [1,2): frac=(m-1) in [0,1)
        float frac = m - 1.0f;
        int q = ggml_rne_int(frac * 8.0f); // 0..8
        if (q == 8) {
            // carry into exponent
            q = 0;
            ef += 1;
            if (ef >= 15) {
                return sign * max_f;
            }
        }
        const float dq = 1.0f + (float)q / 8.0f;
        const int de = ef - 7;
        return sign * ldexpf(dq, de);
    } else {
        // subnormal: exp field=0, value = 2^(1-bias) * (mant/8)
        // so mant = round(ax / 2^(1-bias) * 8)
        const float scale = ldexpf(1.0f, 6) * 8.0f; // 2^6*8 = 512
        int mant = ggml_rne_int(ax * scale);
        if (mant <= 0) {
            return 0.0f;
        }
        if (mant > 7) {
            // would become normalized; clamp to max subnormal (mant=7)
            mant = 7;
        }
        return sign * ldexpf((float)mant, -9); // mant/512
    }
}

static inline int8_t ggml_choose_k_for_block(const float * in, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(in[i]);
        if (ax > amax) amax = ax;
    }
    if (amax == 0.0f) {
        return 0;
    }
    // need ax / 2^k <= max_finite (240)
    const float max_f = ggml_fp8e4m3_max_finite();
    float ratio = amax / max_f;
    int k = 0;
    if (ratio > 1.0f) {
        // k = ceil(log2(ratio))
        k = (int)ceilf(log2f(ratio));
    }
    if (k < -128) k = -128;
    if (k > 127)  k = 127;
    return (int8_t)k;
}

extern "C" void ggml_sim_fp8e4m3_block_quant_dequant_f32(
        const float * in,
        float       * out,
        int           n,
        int           block,
        int8_t      * scales_out) {
    if (block <= 0) {
        block = 16;
    }
    for (int i = 0; i < n; i += block) {
        const int len = (i + block <= n) ? block : (n - i);
        const int8_t k = ggml_choose_k_for_block(in + i, len);
        if (scales_out) {
            scales_out[i / block] = k;
        }
        const float inv = ldexpf(1.0f, -k); // 2^{-k}
        const float mul = ldexpf(1.0f,  k); // 2^{k}
        for (int j = 0; j < len; ++j) {
            const float q = ggml_fp8e4m3_quant_dequant_one(in[i + j] * inv);
            out[i + j] = q * mul;
        }
    }
}

extern "C" void ggml_sim_fp8e4m3_block_quant_dequant_f32_to_bf16(
        const float     * in,
        ggml_bf16_t     * out,
        int               n,
        int               block,
        int8_t          * scales_out) {
    if (block <= 0) {
        block = 16;
    }
    for (int i = 0; i < n; i += block) {
        const int len = (i + block <= n) ? block : (n - i);
        const int8_t k = ggml_choose_k_for_block(in + i, len);
        if (scales_out) {
            scales_out[i / block] = k;
        }
        const float inv = ldexpf(1.0f, -k);
        const float mul = ldexpf(1.0f,  k);
        for (int j = 0; j < len; ++j) {
            const float q = ggml_fp8e4m3_quant_dequant_one(in[i + j] * inv);
            out[i + j] = GGML_FP32_TO_BF16(q * mul);
        }
    }
}

#endif // GGML_SIM_FP8E4M3

#include <cassert>

// precomputed gelu table for f16 (128 KB)
ggml_fp16_t ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16];

void ggml_vec_dot_f32(int n, float * GGML_RESTRICT s, size_t bs, const float * GGML_RESTRICT x, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc) {
   assert(nrc == 1);
   GGML_UNUSED(nrc);
   GGML_UNUSED(bx);
   GGML_UNUSED(by);
   GGML_UNUSED(bs);

#if defined(GGML_SIMD)
    float sumf = 0.0f;

    #if defined(__ARM_FEATURE_SVE)
        const int sve_register_length = ggml_cpu_get_sve_cnt() * 8;
        const int ggml_f32_epr = sve_register_length / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
        const int ggml_f32_step = 8 * ggml_f32_epr; // choose 8 SVE registers

        const int np = (n & ~(ggml_f32_step - 1));
        svfloat32_t sum1 = svdup_n_f32(0.0f);
        svfloat32_t sum2 = svdup_n_f32(0.0f);
        svfloat32_t sum3 = svdup_n_f32(0.0f);
        svfloat32_t sum4 = svdup_n_f32(0.0f);
        svfloat32_t sum5 = svdup_n_f32(0.0f);
        svfloat32_t sum6 = svdup_n_f32(0.0f);
        svfloat32_t sum7 = svdup_n_f32(0.0f);
        svfloat32_t sum8 = svdup_n_f32(0.0f);
        svfloat32_t ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8;
        svfloat32_t ay1,ay2,ay3,ay4,ay5,ay6,ay7,ay8;
        for (int i = 0; i < np; i += ggml_f32_step) {
            ax1 = GGML_F32_VEC_LOAD(x + i);
            ay1 = GGML_F32_VEC_LOAD(y + i);
            sum1 = GGML_F32_VEC_FMA(sum1, ax1, ay1);

            ax2 = GGML_F32_VEC_LOAD(x + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_LOAD(y + i + 1*ggml_f32_epr);
            sum2 = GGML_F32_VEC_FMA(sum2, ax2, ay2);

            ax3 = GGML_F32_VEC_LOAD(x + i + 2*ggml_f32_epr);
            ay3 = GGML_F32_VEC_LOAD(y + i + 2*ggml_f32_epr);
            sum3 = GGML_F32_VEC_FMA(sum3, ax3, ay3);

            ax4 = GGML_F32_VEC_LOAD(x + i + 3*ggml_f32_epr);
            ay4 = GGML_F32_VEC_LOAD(y + i + 3*ggml_f32_epr);
            sum4 = GGML_F32_VEC_FMA(sum4, ax4, ay4);

            ax5 = GGML_F32_VEC_LOAD(x + i + 4*ggml_f32_epr);
            ay5 = GGML_F32_VEC_LOAD(y + i + 4*ggml_f32_epr);
            sum5 = GGML_F32_VEC_FMA(sum5, ax5, ay5);

            ax6 = GGML_F32_VEC_LOAD(x + i + 5*ggml_f32_epr);
            ay6 = GGML_F32_VEC_LOAD(y + i + 5*ggml_f32_epr);
            sum6 = GGML_F32_VEC_FMA(sum6, ax6, ay6);

            ax7 = GGML_F32_VEC_LOAD(x + i + 6*ggml_f32_epr);
            ay7 = GGML_F32_VEC_LOAD(y + i + 6*ggml_f32_epr);
            sum7 = GGML_F32_VEC_FMA(sum7, ax7, ay7);

            ax8 = GGML_F32_VEC_LOAD(x + i + 7*ggml_f32_epr);
            ay8 = GGML_F32_VEC_LOAD(y + i + 7*ggml_f32_epr);
            sum8 = GGML_F32_VEC_FMA(sum8, ax8, ay8);
        }
        // leftovers
        // Since 8 unrolls are done in above loop, leftovers lie in range [0, ggml_f32_step] which is handled in below loop
        const int np2 = (n & ~(ggml_f32_epr - 1));
        for (int i = np; i < np2; i += ggml_f32_epr) {
            ax1 = GGML_F32_VEC_LOAD(x + i);
            ay1 = GGML_F32_VEC_LOAD(y + i);
            sum1 = GGML_F32_VEC_FMA(sum1, ax1, ay1);
        }
        // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmad on available elements only
        if (np2 < n) {
            svbool_t pg = svwhilelt_b32(np2, n);
            ax1 = svld1_f32(pg, x + np2);
            ay1 = svld1_f32(pg, y + np2);
            sum1 = svmad_f32_m(pg, ax1, ay1, sum1);
        }
        // reduce sum1,sum2 to sum1
        GGML_F32_VEC_REDUCE(sumf, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8);
    #elif defined(__riscv_v_intrinsic)
        int vl = __riscv_vsetvlmax_e32m8();
        vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m8_t vsum;
        vfloat32m8_t ax;
        vfloat32m8_t ay;
        vsum = __riscv_vfmv_v_f_f32m8_tu(vsum, 0.0f, vl);
        for (int i = 0; i < n; i += vl) {
            vl = __riscv_vsetvl_e32m8(n - i);
            ax = __riscv_vle32_v_f32m8_tu(ax, &x[i], vl);
            ay = __riscv_vle32_v_f32m8_tu(ay, &y[i], vl);
            vsum = __riscv_vfmacc_vv_f32m8_tu(vsum, ax, ay, vl);
        }
        vl = __riscv_vsetvlmax_e32m8();
        vs = __riscv_vfredusum_vs_f32m8_f32m1(vsum, vs, vl);
        sumf += __riscv_vfmv_f_s_f32m1_f32(vs);
    #else
        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

        GGML_F32_VEC ax[GGML_F32_ARR];
        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);

                sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], ay[j]);
            }
        }

        // reduce sum0..sum3 to sum0
        GGML_F32_VEC_REDUCE(sumf, sum);

        // leftovers
        for (int i = np; i < n; ++i) {
            sumf += x[i]*y[i];
        }
    #endif
#else
    // scalar
    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(x[i]*y[i]);
    }
#endif

    *s = sumf;
}

void ggml_vec_dot_bf16(int n, float * GGML_RESTRICT s, size_t bs, ggml_bf16_t * GGML_RESTRICT x, size_t bx, ggml_bf16_t * GGML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    int i = 0;
    ggml_float sumf = 0;

#if defined(__AVX512BF16__)
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 64 <= n; i += 64) {
        c1 = _mm512_dpbf16_ps(c1, m512bh(_mm512_loadu_si512((x + i))),
                             m512bh(_mm512_loadu_si512((y + i))));
        c2 = _mm512_dpbf16_ps(c2, m512bh(_mm512_loadu_si512((x + i + 32))),
                             m512bh(_mm512_loadu_si512((y + i + 32))));
    }
    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#elif defined(__AVX512F__)
#define LOAD(p) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(p))), 16))
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c2);
    }
    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#undef LOAD
#elif defined(__AVX2__) || defined(__AVX__)
#if defined(__AVX2__)
#define LOAD(p) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16))
#else
#define LOAD(p) _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16)), (_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_bsrli_si128(_mm_loadu_si128((const __m128i *)(p)), 8)), 16)), 1))
#endif
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 8), LOAD(y + i + 8)), c2);
        c3 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c3);
        c4 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 24), LOAD(y + i + 24)), c4);
    }
    __m128 g;
    c1 = _mm256_add_ps(_mm256_add_ps(c1, c3),
                       _mm256_add_ps(c2, c4));
    g = _mm_add_ps(_mm256_extractf128_ps(c1, 1),
                   _mm256_castps256_ps128(c1));
    g = _mm_add_ps(g, _mm_movehl_ps(g, g));
    g = _mm_add_ss(g, _mm_movehdup_ps(g));
    sumf += (ggml_float)_mm_cvtss_f32(g);

#undef LOAD
#endif

    for (; i < n; ++i) {
        sumf += (ggml_float)(GGML_BF16_TO_FP32(x[i]) *
                             GGML_BF16_TO_FP32(y[i]));
    }
    *s = sumf;
}

void ggml_vec_dot_bf16_trunc4(
    int n,
    float * GGML_RESTRICT s, size_t bs,
    ggml_bf16_t * GGML_RESTRICT x, size_t bx,
    ggml_bf16_t * GGML_RESTRICT y, size_t by,
    int nrc) {

    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    int i = 0;
    ggml_float sumf = 0;

#if defined(__AVX512BF16__)
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();

    const __m512i mask = _mm512_set1_epi16((short)GGML_BF16_TRUNC4_MASK);
    const __m512i bias = _mm512_set1_epi16((short)GGML_BF16_RNA_BIAS);

    for (; i + 64 <= n; i += 64) {
        __m512i vx0 = _mm512_loadu_si512((const void *)(&x[i].bits));
        vx0 = _mm512_add_epi16(vx0, bias);
        vx0 = _mm512_and_si512(vx0, mask);

        __m512i vy0 = _mm512_loadu_si512((const void *)(&y[i].bits));
#if GGML_MULMAT_TRUNC4_SRC1
        vy0 = _mm512_add_epi16(vy0, bias);
        vy0 = _mm512_and_si512(vy0, mask);
#endif
        c1 = _mm512_dpbf16_ps(c1, m512bh(vx0), m512bh(vy0));

        __m512i vx1 = _mm512_loadu_si512((const void *)(&x[i + 32].bits));
        vx1 = _mm512_add_epi16(vx1, bias);
        vx1 = _mm512_and_si512(vx1, mask);

        __m512i vy1 = _mm512_loadu_si512((const void *)(&y[i + 32].bits));
#if GGML_MULMAT_TRUNC4_SRC1
        vy1 = _mm512_add_epi16(vy1, bias);
        vy1 = _mm512_and_si512(vy1, mask);
#endif
        c2 = _mm512_dpbf16_ps(c2, m512bh(vx1), m512bh(vy1));
    }

    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#elif defined(__AVX512F__)
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();

    const __m256i mask256 = _mm256_set1_epi16((short)GGML_BF16_TRUNC4_MASK);
    const __m256i bias256 = _mm256_set1_epi16((short)GGML_BF16_RNA_BIAS);

    for (; i + 32 <= n; i += 32) {
        __m256i x16a = _mm256_loadu_si256((const __m256i *)(&x[i].bits));
        x16a = _mm256_add_epi16(x16a, bias256);
        x16a = _mm256_and_si256(x16a, mask256);

        __m256i y16a = _mm256_loadu_si256((const __m256i *)(&y[i].bits));
#if GGML_MULMAT_TRUNC4_SRC1
        y16a = _mm256_add_epi16(y16a, bias256);
        y16a = _mm256_and_si256(y16a, mask256);
#endif

        __m512 xf_a = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_castsi256_si128(x16a)), 16));
        __m512 yf_a = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_castsi256_si128(y16a)), 16));
        c1 = _mm512_add_ps(_mm512_mul_ps(xf_a, yf_a), c1);

        __m256i x16b = _mm256_loadu_si256((const __m256i *)(&x[i + 16].bits));
        x16b = _mm256_add_epi16(x16b, bias256);
        x16b = _mm256_and_si256(x16b, mask256);

        __m256i y16b = _mm256_loadu_si256((const __m256i *)(&y[i + 16].bits));
#if GGML_MULMAT_TRUNC4_SRC1
        y16b = _mm256_add_epi16(y16b, bias256);
        y16b = _mm256_and_si256(y16b, mask256);
#endif

        __m512 xf_b = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_castsi256_si128(x16b)), 16));
        __m512 yf_b = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_castsi256_si128(y16b)), 16));
        c2 = _mm512_add_ps(_mm512_mul_ps(xf_b, yf_b), c2);
    }

    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#elif defined(__AVX2__)
    #define LOAD_TRUNC_RNA_8(p_bits_ptr, do_trunc) \
        _mm256_castsi256_ps( \
            _mm256_slli_epi32( \
                _mm256_cvtepu16_epi32( \
                    (do_trunc) ? \
                    _mm_and_si128( \
                        _mm_add_epi16( \
                            _mm_loadu_si128((const __m128i *)(p_bits_ptr)), \
                            _mm_set1_epi16((short)GGML_BF16_RNA_BIAS) \
                        ), \
                        _mm_set1_epi16((short)GGML_BF16_TRUNC4_MASK) \
                    ) : \
                    _mm_loadu_si128((const __m128i *)(p_bits_ptr)) \
                ), \
                16 \
            ) \
        )

    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();

    for (; i + 32 <= n; i += 32) {
        c1 = _mm256_add_ps(_mm256_mul_ps(
            LOAD_TRUNC_RNA_8(&x[i +  0].bits, 1),
            LOAD_TRUNC_RNA_8(&y[i +  0].bits, GGML_MULMAT_TRUNC4_SRC1)), c1);

        c2 = _mm256_add_ps(_mm256_mul_ps(
            LOAD_TRUNC_RNA_8(&x[i +  8].bits, 1),
            LOAD_TRUNC_RNA_8(&y[i +  8].bits, GGML_MULMAT_TRUNC4_SRC1)), c2);

        c3 = _mm256_add_ps(_mm256_mul_ps(
            LOAD_TRUNC_RNA_8(&x[i + 16].bits, 1),
            LOAD_TRUNC_RNA_8(&y[i + 16].bits, GGML_MULMAT_TRUNC4_SRC1)), c3);

        c4 = _mm256_add_ps(_mm256_mul_ps(
            LOAD_TRUNC_RNA_8(&x[i + 24].bits, 1),
            LOAD_TRUNC_RNA_8(&y[i + 24].bits, GGML_MULMAT_TRUNC4_SRC1)), c4);
    }

    __m128 g;
    __m256 acc = _mm256_add_ps(_mm256_add_ps(c1, c3), _mm256_add_ps(c2, c4));
    g = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc));
    g = _mm_add_ps(g, _mm_movehl_ps(g, g));
    g = _mm_add_ss(g, _mm_movehdup_ps(g));
    sumf += (ggml_float)_mm_cvtss_f32(g);

    #undef LOAD_TRUNC_RNA_8
#endif

    for (; i < n; ++i) {
        ggml_bf16_t xt = ggml_bf16_rna_trunc4(x[i]);
#if GGML_MULMAT_TRUNC4_SRC1
        ggml_bf16_t yt = ggml_bf16_rna_trunc4(y[i]);
#else
        ggml_bf16_t yt = y[i];
#endif
        sumf += (ggml_float)(GGML_BF16_TO_FP32(xt) * GGML_BF16_TO_FP32(yt));
    }

    *s = (float)sumf;
}




void ggml_vec_dot_f16(int n, float * GGML_RESTRICT s, size_t bs, ggml_fp16_t * GGML_RESTRICT x, size_t bx, ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    ggml_float sumf = 0.0;


#if defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)
        const int sve_register_length = svcntb() * 8; //get vector length
        const int ggml_f16_epr = sve_register_length / 16; // running when 16
        const int ggml_f16_step = 8 * ggml_f16_epr; // choose 8 SVE registers

        const int np= (n & ~(ggml_f16_step - 1));
        svfloat16_t sum1 = svdup_n_f16(0.0f);
        svfloat16_t sum2 = svdup_n_f16(0.0f);
        svfloat16_t sum3 = svdup_n_f16(0.0f);
        svfloat16_t sum4 = svdup_n_f16(0.0f);

        svfloat16_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
        svfloat16_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;
        for (int i = 0; i < np; i += ggml_f16_step) {
            ax1 = GGML_F16x_VEC_LOAD(x + i + 0 * ggml_f16_epr, 0);
            ay1 = GGML_F16x_VEC_LOAD(y + i + 0 * ggml_f16_epr, 0);
            sum1 = GGML_F16x_VEC_FMA(sum1, ax1, ay1);

            ax2 = GGML_F16x_VEC_LOAD(x + i + 1 * ggml_f16_epr, 1);
            ay2 = GGML_F16x_VEC_LOAD(y + i + 1 * ggml_f16_epr, 1);
            sum2 = GGML_F16x_VEC_FMA(sum2, ax2, ay2);

            ax3 = GGML_F16x_VEC_LOAD(x + i + 2 * ggml_f16_epr, 2);
            ay3 = GGML_F16x_VEC_LOAD(y + i + 2 * ggml_f16_epr, 2);
            sum3 = GGML_F16x_VEC_FMA(sum3, ax3, ay3);

            ax4 = GGML_F16x_VEC_LOAD(x + i + 3 * ggml_f16_epr, 3);
            ay4 = GGML_F16x_VEC_LOAD(y + i + 3 * ggml_f16_epr, 3);
            sum4 = GGML_F16x_VEC_FMA(sum4, ax4, ay4);

            ax5 = GGML_F16x_VEC_LOAD(x + i + 4 * ggml_f16_epr, 4);
            ay5 = GGML_F16x_VEC_LOAD(y + i + 4 * ggml_f16_epr, 4);
            sum1 = GGML_F16x_VEC_FMA(sum1, ax5, ay5);

            ax6 = GGML_F16x_VEC_LOAD(x + i + 5 * ggml_f16_epr, 5);
            ay6 = GGML_F16x_VEC_LOAD(y + i + 5 * ggml_f16_epr, 5);
            sum2 = GGML_F16x_VEC_FMA(sum2, ax6, ay6);

            ax7 = GGML_F16x_VEC_LOAD(x + i + 6 * ggml_f16_epr, 6);
            ay7 = GGML_F16x_VEC_LOAD(y + i + 6 * ggml_f16_epr, 6);
            sum3 = GGML_F16x_VEC_FMA(sum3, ax7, ay7);

            ax8 = GGML_F16x_VEC_LOAD(x + i + 7 * ggml_f16_epr, 7);
            ay8 = GGML_F16x_VEC_LOAD(y + i + 7 * ggml_f16_epr, 7);
            sum4 = GGML_F16x_VEC_FMA(sum4, ax8, ay8);
        }

        const int np2 = (n & ~(ggml_f16_epr - 1)); // round down to multiple of 8
        for (int k = np; k < np2; k += ggml_f16_epr) {
            svfloat16_t rx = GGML_F16x_VEC_LOAD(x + k, 0);
            svfloat16_t ry = GGML_F16x_VEC_LOAD(y + k, 0);
            sum1 = GGML_F16x_VEC_FMA(sum1, rx, ry);
        }

        if (np2 < n) {
            svbool_t pg = svwhilelt_b16(np2, n);
            svfloat16_t hx = svld1_f16(pg, (const __fp16 *)(x + np2));
            svfloat16_t hy = svld1_f16(pg, (const __fp16 *)(y + np2));

            sum1 = svmad_f16_x(pg, hx, hy, sum1);
        }
        GGML_F16x_VEC_REDUCE(sumf, sum1, sum2, sum3, sum4);
    #elif defined(__riscv_v_intrinsic)
        #if defined(__riscv_zvfh)
            int vl = __riscv_vsetvlmax_e32m2();
            vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            vfloat32m2_t vsum;
            vfloat16m1_t ax;
            vfloat16m1_t ay;
            vsum = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vmv_v_x_u32m2(0, vl));
            for (int i = 0; i < n; i += vl) {
                vl = __riscv_vsetvl_e16m1(n - i);
                ax = __riscv_vle16_v_f16m1_tu(ax, (const _Float16 *)&x[i], vl);
                ay = __riscv_vle16_v_f16m1_tu(ay, (const _Float16 *)&y[i], vl);
                vsum = __riscv_vfwmacc_vv_f32m2_tu(vsum, ax, ay, vl);
            }
            vl = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t ac0 = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(vsum, 0), __riscv_vget_v_f32m2_f32m1(vsum, 1), vl);
            vs = __riscv_vfredusum_vs_f32m1_f32m1(ac0, vs, vl);
            sumf += __riscv_vfmv_f_s_f32m1_f32(vs);
        #else
            for (int i = 0; i < n; ++i) {
                sumf += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[i])*GGML_CPU_FP16_TO_FP32(y[i]));
            }
        #endif // __riscv_zvfh
    #else
        const int np = (n & ~(GGML_F16_STEP - 1));

        GGML_F16_VEC sum[GGML_F16_ARR] = { GGML_F16_VEC_ZERO };

        GGML_F16_VEC ax[GGML_F16_ARR];
        GGML_F16_VEC ay[GGML_F16_ARR];

        for (int i = 0; i < np; i += GGML_F16_STEP) {
            for (int j = 0; j < GGML_F16_ARR; j++) {
                ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
                ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);

                sum[j] = GGML_F16_VEC_FMA(sum[j], ax[j], ay[j]);
            }
        }

        // reduce sum0..sum3 to sum0
        GGML_F16_VEC_REDUCE(sumf, sum);

        // leftovers
        for (int i = np; i < n; ++i) {
            sumf += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[i])*GGML_CPU_FP16_TO_FP32(y[i]));
        }
        // if you hit this, you are likely running outside the FP range
        assert(!isnan(sumf) && !isinf(sumf));
    #endif
#else
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[i])*GGML_CPU_FP16_TO_FP32(y[i]));
    }
#endif // GGML_SIMD

    *s = sumf;
}

void ggml_vec_silu_f32(const int n, float * y, const float * x) {
    int i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, ggml_v_silu(_mm512_loadu_ps(x + i)));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, ggml_v_silu(_mm256_loadu_ps(x + i)));
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, ggml_v_silu(_mm_loadu_ps(x + i)));
    }
#elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    const int vlen = svcntw();
    for (; i < n; i += vlen) {
        const svbool_t pg = svwhilelt_b32_s32(i, n);
        svst1_f32(pg, y + i, ggml_v_silu(pg, svld1_f32(pg, x + i)));
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, ggml_v_silu(vld1q_f32(x + i)));
    }
#endif
    for (; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]);
    }
}

void ggml_vec_swiglu_f32(const int n, float * y, const float * x, const float * g) {
    int i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, _mm512_mul_ps(ggml_v_silu(_mm512_loadu_ps(x + i)), _mm512_loadu_ps(g + i)));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_mul_ps(ggml_v_silu(_mm256_loadu_ps(x + i)), _mm256_loadu_ps(g + i)));
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, _mm_mul_ps(ggml_v_silu(_mm_loadu_ps(x + i)), _mm_loadu_ps(g + i)));
    }
#elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    const int vlen = svcntw();
    for (; i < n; i += vlen) {
        const svbool_t pg = svwhilelt_b32_s32(i, n);
        svst1_f32(pg, y + i, svmul_f32_x(pg, ggml_v_silu(pg, svld1_f32(pg, x + i)), svld1_f32(pg, g + i)));
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, vmulq_f32(ggml_v_silu(vld1q_f32(x + i)), vld1q_f32(g + i)));
    }
#elif defined(__riscv_v_intrinsic)
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat32m2_t vg = __riscv_vle32_v_f32m2(&g[i], vl);
        vfloat32m2_t vy = __riscv_vfmul_vv_f32m2(ggml_v_silu_m2(vx, vl), vg, vl);
        __riscv_vse32_v_f32m2(&y[i], vy, vl);
    }
#endif
    for (; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]) * g[i];
    }
}

ggml_float ggml_vec_cvar_f32(const int n, float * y, const float * x, const float mean) {
    int i = 0;
    ggml_float sum = 0;
// TODO: optimize to process the remaining elements in groups using the smaller vector sizes from AVX2 and SSE
// ref: https://github.com/ggml-org/llama.cpp/pull/15953#pullrequestreview-3310928344
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        __m512 val = _mm512_sub_ps(_mm512_loadu_ps(x + i),
                                   _mm512_set1_ps(mean));
        _mm512_storeu_ps(y + i, val);
        sum += (ggml_float)_mm512_reduce_add_ps(_mm512_mul_ps(val, val));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        __m256 val = _mm256_sub_ps(_mm256_loadu_ps(x + i),
                                   _mm256_set1_ps(mean));
        _mm256_storeu_ps(y + i, val);
        val = _mm256_mul_ps(val,val);
        __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1),
                                 _mm256_castps256_ps128(val));
        val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
        val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
        sum += (ggml_float)_mm_cvtss_f32(val2);
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        __m128 val = _mm_sub_ps(_mm_loadu_ps(x + i),
                                _mm_set1_ps(mean));
        _mm_storeu_ps(y + i, val);
        val = _mm_mul_ps(val, val);
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        val = _mm_add_ps(val, _mm_movehl_ps(val, val));
        val = _mm_add_ss(val, _mm_movehdup_ps(val));
#else
        __m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
        val = _mm_add_ps(val, tmp);
        tmp = _mm_movehl_ps(tmp, val);
        val = _mm_add_ss(val, tmp);
#endif  // __AVX__ || __AVX2__ || __AVX512F__
        sum += (ggml_float)_mm_cvtss_f32(val);
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        float32x4_t val = vsubq_f32(vld1q_f32(x + i),
                                    vdupq_n_f32(mean));
        vst1q_f32(y + i, val);
        val = vmulq_f32(val, val);
        sum += (ggml_float)vaddvq_f32(val);
    }
#elif defined(__VXE__) || defined(__VXE2__)
    for (; i + 3 < n; i += 4) {
        float32x4_t val = vec_sub(vec_xl(0, x + i), vec_splats(mean));
        vec_xst(val, 0, y + i);
        val = vec_mul(val, val);
        sum += (ggml_float)vec_hsum_f32x4(val);
    }
#endif
    for (; i < n; ++i) {
        float val = x[i] - mean;
        y[i] = val;
        val *= val;
        sum += (ggml_float)val;
    }
    return sum/n;
}

ggml_float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max) {
    int i = 0;
    ggml_float sum = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        __m512 val = ggml_v_expf(_mm512_sub_ps(_mm512_loadu_ps(x + i),
                                               _mm512_set1_ps(max)));
        _mm512_storeu_ps(y + i, val);
        sum += (ggml_float)_mm512_reduce_add_ps(val);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        __m256 val = ggml_v_expf(_mm256_sub_ps(_mm256_loadu_ps(x + i),
                                               _mm256_set1_ps(max)));
        _mm256_storeu_ps(y + i, val);
        __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1),
                                 _mm256_castps256_ps128(val));
        val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
        val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
        sum += (ggml_float)_mm_cvtss_f32(val2);
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        __m128 val = ggml_v_expf(_mm_sub_ps(_mm_loadu_ps(x + i),
                                            _mm_set1_ps(max)));
        _mm_storeu_ps(y + i, val);
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        val = _mm_add_ps(val, _mm_movehl_ps(val, val));
        val = _mm_add_ss(val, _mm_movehdup_ps(val));
#else
        __m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
        val = _mm_add_ps(val, tmp);
        tmp = _mm_movehl_ps(tmp, val);
        val = _mm_add_ss(val, tmp);
#endif
        sum += (ggml_float)_mm_cvtss_f32(val);
    }
#elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    const int vlen = svcntw();
    for (; i < n; i += vlen) {
        const svbool_t pg = svwhilelt_b32_s32(i, n);
        svfloat32_t val = ggml_v_expf(pg, svsub_f32_x(pg, svld1_f32(pg, x + i),
                                                svdup_n_f32_x(pg, max)));
        svst1_f32(pg, y + i, val);
        sum += (ggml_float)svaddv_f32(pg, val);
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        float32x4_t val = ggml_v_expf(vsubq_f32(vld1q_f32(x + i),
                                                vdupq_n_f32(max)));
        vst1q_f32(y + i, val);
        sum += (ggml_float)vaddvq_f32(val);
    }
#elif defined(__riscv_v_intrinsic)
    vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0, 1);
    for (int avl; i < n; i += avl) {
        avl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t val = ggml_v_expf_m2(__riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(&x[i], avl), max, avl), avl);
        __riscv_vse32_v_f32m2(&y[i], val, avl);
        vsum = __riscv_vfwredusum_vs_f32m2_f64m1(val, vsum, avl);
    }
    return (ggml_float)__riscv_vfmv_f_s_f64m1_f64(vsum);
#endif
    for (; i < n; ++i) {
        float val = expf(x[i] - max);
        sum += (ggml_float)val;
        y[i] = val;
    }
    return sum;
}

ggml_float ggml_vec_log_soft_max_f32(const int n, float * y, const float * x, float max) {
    // log(soft_max) = log(soft_max_i / soft_max_sum) = log(soft_max_i) - log(soft_max_sum) = (logit_i - max) - log(soft_max_i)

    int i = 0;
    ggml_float sum = 0;
    for (; i < n; ++i) {
        float val = x[i] - max;
        y[i] = val;
        sum += (ggml_float)expf(val);
    }
    return sum = (ggml_float)logf(sum);
}

// =============================
// FP8(E4M3) block-quant simulate
// =============================

extern "C" {

static inline float ggml__pow2_i8(const int8_t k) {
    // pow2 by bit manipulation is possible, but this is only used in conversion
    return ldexpf(1.0f, (int)k);
}

// E4M3 max finite: (2^7) * (1 + 7/8) = 240
static constexpr float GGML_FP8_E4M3_MAX_FINITE = 240.0f;
// min subnormal: 2^(1-bias) * 2^-3 = 2^-9
static constexpr float GGML_FP8_E4M3_MIN_SUB = 1.0f / 512.0f;

// Choose k so that max(|x|) / 2^k <= 240.
// k stored as int8, meaning scale = 2^k.
static inline int8_t ggml__fp8_choose_k_e4m3(const float * in, int len) {
    float amax = 0.0f;
    for (int i = 0; i < len; ++i) {
        float a = fabsf(in[i]);
        if (a > amax) amax = a;
    }
    if (amax == 0.0f) {
        return (int8_t)0;
    }
    // want: amax / 2^k <= 240  -> k >= log2(amax/240)
    float ratio = amax / GGML_FP8_E4M3_MAX_FINITE;
    int k = (int)ceilf(log2f(ratio));
    if (k < -128) k = -128;
    if (k > 127)  k = 127;
    return (int8_t)k;
}

// Quantize a single value (already scaled to fp8 domain) then dequantize back to float.
// Implements RNE + subnormals + saturate.
static inline float ggml__fp8_e4m3_quant_dequant_scalar(float v) {
    // sign
    float av = fabsf(v);
    if (av == 0.0f) {
        return copysignf(0.0f, v);
    }
    // saturate
    if (av >= GGML_FP8_E4M3_MAX_FINITE) {
        return copysignf(GGML_FP8_E4M3_MAX_FINITE, v);
    }
    // flush below min subnormal
    if (av < GGML_FP8_E4M3_MIN_SUB) {
        return copysignf(0.0f, v);
    }

    // Normalize: v = m * 2^e, with m in [1,2)
    int e;
    float m = frexpf(av, &e); // av = m * 2^e, m in [0.5,1)
    // Convert to [1,2):
    m *= 2.0f;
    e -= 1;

    // In E4M3: exponent field encodes e with bias=7, normal exponent range: -6..7
    const int e_min = -6;
    const int e_max =  7;

    if (e < e_min) {
        // subnormal: value = (mantissa / 8) * 2^-6, mantissa in [1..7]
        // So mantissa = round_RNE( av / 2^-6 * 8 ) = round_RNE( av * 512 )
        float scaled = av * 512.0f;
        int q = (int)lrintf(scaled); // RNE
        if (q < 1) q = 1;
        if (q > 7) q = 7;
        float out = (float)q / 512.0f;
        return copysignf(out, v);
    }

    // normal number
    if (e > e_max) {
        return copysignf(GGML_FP8_E4M3_MAX_FINITE, v);
    }

    // mantissa fractional bits: 3. m is in [1,2)
    float frac = m - 1.0f; // in [0,1)
    float scaled = frac * 8.0f;
    int mant = (int)lrintf(scaled); // RNE
    // Handle carry: frac rounds to 1.0 -> mant=8 -> increment exponent
    if (mant == 8) {
        mant = 0;
        e += 1;
        if (e > e_max) {
            return copysignf(GGML_FP8_E4M3_MAX_FINITE, v);
        }
    }

    float out = ldexpf(1.0f + ((float)mant) / 8.0f, e);
    // clamp again for safety
    if (out > GGML_FP8_E4M3_MAX_FINITE) out = GGML_FP8_E4M3_MAX_FINITE;
    return copysignf(out, v);
}

void ggml_sim_fp8e4m3_block_quant_dequant_f32(
        const float * in,
        float       * out,
        int           n,
        int           block,
        int8_t      * scales_out) {
    if (block <= 0) block = 16;
    int nb = (n + block - 1) / block;
    for (int b = 0; b < nb; ++b) {
        const int off = b * block;
        const int len = (off + block <= n) ? block : (n - off);
        int8_t k = ggml__fp8_choose_k_e4m3(in + off, len);
        if (scales_out) scales_out[b] = k;
        const float scale = ggml__pow2_i8(k);
        const float inv   = 1.0f / scale;

        for (int i = 0; i < len; ++i) {
            float v = in[off + i] * inv;          // map into fp8 domain
            float q = ggml__fp8_e4m3_quant_dequant_scalar(v);
            out[off + i] = q * scale;             // dequant back
        }
    }
}

void ggml_sim_fp8e4m3_block_quant_dequant_f32_to_bf16(
        const float     * in,
        ggml_bf16_t     * out,
        int               n,
        int               block,
        int8_t          * scales_out) {
    // Reuse a small scratch buffer per block to avoid allocating n floats.
    if (block <= 0) block = 16;
    int nb = (n + block - 1) / block;
    for (int b = 0; b < nb; ++b) {
        const int off = b * block;
        const int len = (off + block <= n) ? block : (n - off);
        int8_t k = ggml__fp8_choose_k_e4m3(in + off, len);
        if (scales_out) scales_out[b] = k;
        const float scale = ggml__pow2_i8(k);
        const float inv   = 1.0f / scale;
        for (int i = 0; i < len; ++i) {
            float v = in[off + i] * inv;
            float q = ggml__fp8_e4m3_quant_dequant_scalar(v);
            float r = q * scale;
            out[off + i] = GGML_FP32_TO_BF16(r);
        }
    }
}

} // extern "C"


#include "vec.h"

// ---------------------------------------------------------------------------
// FP8/FP9 block-quant simulation implementation
//
// 约定：
//  - Format由编译期宏选择：
//      F8(E4M3) / F8(E3M4) / F8(E3M4-no-subnorm)
//      F8(E2M5) / F8(E2M5-no-subnorm) / F9(E4M4-like)
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

static inline int ggml_sim_mantissa_bits(void) {
#if GGML_SIM_FP_FORMAT == GGML_SIM_FP_FORMAT_F9
    return 4;
#else
    if (GGML_SIM_FP8_LAYOUT == GGML_SIM_FP8_LAYOUT_E2M5 ||
        GGML_SIM_FP8_LAYOUT == GGML_SIM_FP8_LAYOUT_E2M5_NO_SUBNORM) {
        return 5;
    }
    return (GGML_SIM_FP8_LAYOUT == GGML_SIM_FP8_LAYOUT_E3M4 ||
            GGML_SIM_FP8_LAYOUT == GGML_SIM_FP8_LAYOUT_E3M4_NO_SUBNORM) ? 4 : 3;
#endif
}

static inline int ggml_sim_exponent_bits(void) {
#if GGML_SIM_FP_FORMAT == GGML_SIM_FP_FORMAT_F9
    return 4;
#else
    if (GGML_SIM_FP8_LAYOUT == GGML_SIM_FP8_LAYOUT_E2M5 ||
        GGML_SIM_FP8_LAYOUT == GGML_SIM_FP8_LAYOUT_E2M5_NO_SUBNORM) {
        return 2;
    }
    return (GGML_SIM_FP8_LAYOUT == GGML_SIM_FP8_LAYOUT_E3M4 ||
            GGML_SIM_FP8_LAYOUT == GGML_SIM_FP8_LAYOUT_E3M4_NO_SUBNORM) ? 3 : 4;
#endif
}

static inline bool ggml_sim_support_subnormals(void) {
#if GGML_SIM_FP_FORMAT == GGML_SIM_FP_FORMAT_F9
    return true;
#else
    return GGML_SIM_FP8_LAYOUT != GGML_SIM_FP8_LAYOUT_E3M4_NO_SUBNORM &&
           GGML_SIM_FP8_LAYOUT != GGML_SIM_FP8_LAYOUT_E2M5_NO_SUBNORM;
#endif
}

static inline int ggml_sim_exponent_bias(void) {
    int bias = (1 << (ggml_sim_exponent_bits() - 1)) - 1;
#if GGML_SIM_FP_FORMAT == GGML_SIM_FP_FORMAT_F8
    if (GGML_SIM_FP8_LAYOUT == GGML_SIM_FP8_LAYOUT_E3M4_NO_SUBNORM ||
        GGML_SIM_FP8_LAYOUT == GGML_SIM_FP8_LAYOUT_E2M5_NO_SUBNORM) {
        // Pure-normal mode requested by experiment:
        // reassign former subnormal exponent budget to normal numbers.
        // For E3M4 this yields bias=7 (old subnormal -6 maps to normal -7 with hidden bit=1).
        // For E2M5 this yields bias=6 with the same remapping rule.
        bias += ggml_sim_mantissa_bits();
    }
#endif
    return bias;
}

static inline int ggml_sim_min_norm_exp_unbiased(void) {
    // With subnormals: smallest normal is exp-field=1.
    // Without subnormals: exp-field=0 is also normal.
    return ggml_sim_support_subnormals() ? (1 - ggml_sim_exponent_bias()) : (-ggml_sim_exponent_bias());
}

static inline int ggml_sim_max_exp_field(void) {
    return (1 << ggml_sim_exponent_bits()) - 1;
}

static inline int ggml_sim_max_norm_exp_unbiased(void) {
    return ggml_sim_max_exp_field() - ggml_sim_exponent_bias();
}

static inline const char * ggml_sim_format_name(void) {
#if GGML_SIM_FP_FORMAT == GGML_SIM_FP_FORMAT_F9
    return "F9(E4M4-like)";
#else
    if (GGML_SIM_FP8_LAYOUT == GGML_SIM_FP8_LAYOUT_E3M4_NO_SUBNORM) {
        return "F8(E3M4-no-subnorm)";
    }
    if (GGML_SIM_FP8_LAYOUT == GGML_SIM_FP8_LAYOUT_E2M5_NO_SUBNORM) {
        return "F8(E2M5-no-subnorm)";
    }
    if (GGML_SIM_FP8_LAYOUT == GGML_SIM_FP8_LAYOUT_E2M5) {
        return "F8(E2M5)";
    }
    return GGML_SIM_FP8_LAYOUT == GGML_SIM_FP8_LAYOUT_E3M4 ? "F8(E3M4)" : "F8(E4M3)";
#endif
}

static inline float ggml_fp8e4m3_max_finite(void) {
    // max_finite = (2 - 2^-mant_bits) * 2^(max_normal_exp_unbiased)
    const int mant_bits = ggml_sim_mantissa_bits();
    const float sig_max = 2.0f - ldexpf(1.0f, -mant_bits);
    return ldexpf(sig_max, ggml_sim_max_norm_exp_unbiased());
}

static inline float ggml_fp8e4m3_handle_non_finite(float x) {
    if (isnan(x)) {
        return 0.0f;
    }
    const float max_f = ggml_fp8e4m3_max_finite();
    return x > 0.0f ? max_f : -max_f;
}

static inline float ggml_fp8e4m3_min_subnormal(void) {
    // exp=0, mant=1 => 2^(1-bias) * 2^(-mant_bits)
    // If subnormals are disabled, return min normal so subnormal bucket becomes empty.
    if (!ggml_sim_support_subnormals()) {
        return ldexpf(1.0f, ggml_sim_min_norm_exp_unbiased());
    }
    return ldexpf(1.0f, ggml_sim_min_norm_exp_unbiased() - ggml_sim_mantissa_bits());
}

static inline float ggml_fp8e4m3_quant_dequant_one(float x) {
    // Applies: saturate, subnormal support, RNE. No NaN/Inf handling.
    if (!isfinite(x)) {
        return ggml_fp8e4m3_handle_non_finite(x);
    }

    if (x == 0.0f) {
        return 0.0f;
    }
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fabsf(x);

    // underflow to 0 below half of min subnormal (RNE midpoint between 0 and min_sub)
    const float min_sub = ggml_fp8e4m3_min_subnormal();
    if (ax < min_sub * 0.5f) {
        return 0.0f;
    }

    // saturate overflow
    const float max_f = ggml_fp8e4m3_max_finite();
    if (ax > max_f) {
        return sign * max_f;
    }

    const int mant_bits = ggml_sim_mantissa_bits();
    const int mant_levels = 1 << mant_bits;

    const int exp_bias = ggml_sim_exponent_bias();
    const int max_exp_field = ggml_sim_max_exp_field();
    const int min_exp_field = ggml_sim_support_subnormals() ? 1 : 0;
    const int min_norm_exp = ggml_sim_min_norm_exp_unbiased();

    // normalized threshold: exp==1, mant==0 => 1.0 * 2^(1-bias)
    const float min_norm = ldexpf(1.0f, min_norm_exp);
    if (ax >= min_norm) {
        int e;
        float m = frexpf(ax, &e); // ax = m*2^e, m in [0.5,1)
        // normalize to [1,2)
        m *= 2.0f;
        e -= 1;
        // exp field = e + bias
        int ef = e + exp_bias;
        GGML_ASSERT(ef >= min_exp_field && "ef is below minimum normal exponent field");
        if (ef > max_exp_field) {
            return sign * max_f;
        }
        // mantissa quantization for frac=(m-1) in [0,1)
        float frac = m - 1.0f;
        int q = ggml_rne_int(frac * (float)mant_levels); // 0..mant_levels
        if (q == mant_levels) {
            // carry into exponent
            q = 0;
            ef += 1;
            if (ef > max_exp_field) {
                return sign * max_f;
            }
        }
        const float dq = 1.0f + (float)q / (float)mant_levels;
        const int de = ef - exp_bias;
        return sign * ldexpf(dq, de);
    } else {
        if (!ggml_sim_support_subnormals()) {
            // In pure-normal mode, values below min_norm round to either 0 or min_norm.
            return sign * min_norm;
        }
        // subnormal: exp field=0, value = 2^(1-bias) * (mant / 2^mant_bits)
        // so mant = round(ax / 2^(1-bias) * 2^mant_bits)
        const float scale = ldexpf(1.0f, -min_norm_exp) * (float)mant_levels;
        int mant = ggml_rne_int(ax * scale);
        if (mant <= 0) {
            return 0.0f;
        }
        if (mant > (mant_levels - 1)) {
            // rounding carry from subnormal into normalized range:
            // exp=1, mant=0 => min normal = 2^(1-bias)
            return sign * min_norm;
        }
        return sign * ldexpf((float)mant, min_norm_exp - mant_bits);
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
    // ---- exponent-alignment approach ----
    // Extract the unbiased exponent of amax (i.e. floor(log2(|amax|))):
    //   amax = m * 2^exp_max,  1 <= m < 2
    //
    // E*M* (no NaN/Inf) max exponent depends on selected format.
    // max_finite is derived from mantissa bits.
    //
    // We align exp_max to the FP8 max exponent:
    //   k = exp_max - fp8_max_exp
    //
    // After scaling by 2^(-k), amax falls into [2^Emax, 2^(Emax+1)).
    // Values in (max_finite, 2^(Emax+1)) may saturate to max_finite - this is a small, bounded
    // clipping that trades negligible saturation for better average precision
    // (avoids wasting a full power-of-2 headroom as the old ceil-ratio method
    // would).
    //
    // When amax < 2^Emax this yields a NEGATIVE k, meaning we multiply the
    // values by 2^|k| (scale up) before quantising, improving precision.
    const int fp8_max_exp = ggml_sim_max_norm_exp_unbiased();
    int exp_max = ilogbf(amax);
    int k = exp_max - fp8_max_exp;
    if (k < -128) k = -128;
    if (k > 127)  k = 127;
    return (int8_t)k;
}

// ---------------------------------------------------------------------------
// BF16 exact scale: scale = bf16(amax / max_finite)
// Returns the scale factor as a float (already rounded to BF16 precision).
// When amax == 0, returns 0 (the block is all-zero, inv should not be used).
// ---------------------------------------------------------------------------
static inline float ggml_choose_scale_for_block_bf16(const float * in, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(in[i]);
        if (ax > amax) amax = ax;
    }
    if (amax == 0.0f) {
        return 0.0f;
    }
    const float max_f = ggml_fp8e4m3_max_finite();
    float scale = amax / max_f;
    // Round to BF16 precision
    ggml_bf16_t scale_bf16 = GGML_FP32_TO_BF16(scale);
    return GGML_BF16_TO_FP32(scale_bf16);
}

static inline int ggml_fp8_scale_type_for_src(int src_id) {
    // src0/src1 use input scale type, src2 (matmul output QDQ) uses output scale type.
    return src_id == 2 ? GGML_SIM_FP8E4M3_SCALE_TYPE_OUT : GGML_SIM_FP8E4M3_SCALE_TYPE_IN;
}

extern "C" void ggml_sim_fp8e4m3_block_quant_dequant_f32(
        const float * in,
        float       * out,
        int           n,
        int           block,
        void        * scales_out,
        int           src_id,
        const char  * layer_name);

extern "C" void ggml_sim_fp8e4m3_block_quant_dequant_f32_to_bf16(
        const float     * in,
        ggml_bf16_t     * out,
        int               n,
        int               block,
        void            * scales_out,
        int               src_id,
        const char      * layer_name);

extern "C" void ggml_fp8_sim_stats_reset(void);
extern "C" void ggml_fp8_sim_stats_report(const char * report_file);

// ---------------------------------------------------------------------------
// Stats infrastructure
// ---------------------------------------------------------------------------
#include <mutex>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cinttypes>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

struct FP8SimSrcStats {
    int64_t total_elements;
    int64_t total_blocks;
    double  sum_sq_input;       // signal power: sum(x_i^2)
    double  sum_sq_error;       // noise power:  sum((x_i - q_i)^2)
    double  sum_abs_error;      // for MAE
    double  max_abs_error;
    double  sum_abs_rel_error;  // sum(|err|/|x|) for nonzero x
    int64_t rel_error_count;    // count of nonzero x (denominator for mean rel err)
    double  max_rel_error;
    int64_t overflow_count;     // |scaled value| > max_finite => saturated
    int64_t underflow_count;    // quantized=0 but original!=0
    int64_t subnormal_count;    // in subnormal range after quant
    int64_t zero_count;         // original was exactly 0
    int64_t normal_count;       // normal FP8 range
    int64_t scale_hist[256];    // k from -128..127, index = k+128
    int64_t sum_row_len;        // sum of n (row length) across function calls
    int64_t call_count;         // number of quant function calls
};

static FP8SimSrcStats g_fp8_stats[3]; // [0]=src0/weights, [1]=src1/activations, [2]=output QDQ
static std::unordered_map<std::string, FP8SimSrcStats> g_fp8_layer_stats[3]; // per-layer stats for src0, src1, output
static std::mutex     g_fp8_stats_mtx;
static bool           g_fp8_atexit_registered = false;
static std::atomic<int64_t> g_fp8_call_counter{0}; // global call counter for sampling
static std::atomic<int> g_fp8_sample_rate{-1}; // -1 = uninitialized; 0 = disabled; N = collect every Nth call

static int fp8_get_sample_rate(void) {
    int rate = g_fp8_sample_rate.load(std::memory_order_relaxed);
    if (rate >= 0) return rate;
    const char * env = getenv("FP8_SIM_STATS_SAMPLE");
    if (env) {
        rate = atoi(env);
        if (rate < 0) rate = 0;
    } else {
        rate = 100; // default: collect 1 in 100
    }
    // Benign race: multiple threads may store the same value
    g_fp8_sample_rate.store(rate, std::memory_order_relaxed);
    return rate;
}

static inline int fp8_src_slot(int src_id) {
    if (src_id >= 0 && src_id <= 2) {
        return src_id;
    }
    return src_id & 1;
}

static void fp8_stats_atexit_handler(void) {
    ggml_fp8_sim_stats_report("fp8_sim_analysis.log");
}

// Accumulate per-element stats for one block into a LOCAL accumulator (no lock).
// inv_scale = reciprocal of the block scale factor (maps original values into FP8 range).
// For int8 pow2 mode: inv_scale = 2^(-k).
// For bf16 exact mode: inv_scale = max_finite / amax.
// k_for_hist = integer exponent for scale_hist binning (floor(log2(scale_factor))).
static inline void fp8_accumulate_block_stats_local(
        FP8SimSrcStats & local,
        const float * original,
        const float * dequant,
        int           len,
        float         inv_scale,
        int           k_for_hist)
{
    const float max_f    = ggml_fp8e4m3_max_finite();
    const float min_sub  = ggml_fp8e4m3_min_subnormal();
    const float min_norm = ldexpf(1.0f, ggml_sim_min_norm_exp_unbiased());
    const float inv      = inv_scale;

    local.total_blocks += 1;
    {
        int bin = k_for_hist + 128;
        if (bin < 0)   bin = 0;
        if (bin > 255) bin = 255;
        local.scale_hist[bin] += 1;
    }

    for (int j = 0; j < len; ++j) {
        const float x  = original[j];
        const float qx = dequant[j];
        const float err = qx - x;

        local.sum_sq_input += (double)x * (double)x;
        local.sum_sq_error += (double)err * (double)err;

        const double ae = fabs((double)err);
        local.sum_abs_error += ae;
        if (ae > local.max_abs_error) local.max_abs_error = ae;

        if (x != 0.0f) {
            const double re = ae / fabs((double)x);
            local.sum_abs_rel_error += re;
            local.rel_error_count += 1;
            if (re > local.max_rel_error) local.max_rel_error = re;
        }

        if (x == 0.0f) {
            local.zero_count++;
        } else {
            const float ax_scaled = fabsf(x * inv);
            if (ax_scaled > max_f) {
                local.overflow_count++;
            } else if (qx == 0.0f) {
                local.underflow_count++;
            } else if (ax_scaled < min_norm && ax_scaled >= min_sub) {
                local.subnormal_count++;
            } else {
                local.normal_count++;
            }
        }
    }
}

// Merge a local accumulator into global + per-layer stats (ONE lock per call).
static void fp8_merge_stats(
        const FP8SimSrcStats & local,
        int           n,
        int           src_id,
        const char  * layer_name)
{
    std::lock_guard<std::mutex> lock(g_fp8_stats_mtx);
    const int sid = fp8_src_slot(src_id);
    FP8SimSrcStats & g = g_fp8_stats[sid];
    g.total_elements    += local.total_elements;
    g.total_blocks      += local.total_blocks;
    g.sum_sq_input      += local.sum_sq_input;
    g.sum_sq_error      += local.sum_sq_error;
    g.sum_abs_error     += local.sum_abs_error;
    if (local.max_abs_error > g.max_abs_error) g.max_abs_error = local.max_abs_error;
    g.sum_abs_rel_error += local.sum_abs_rel_error;
    g.rel_error_count   += local.rel_error_count;
    if (local.max_rel_error > g.max_rel_error) g.max_rel_error = local.max_rel_error;
    g.overflow_count    += local.overflow_count;
    g.underflow_count   += local.underflow_count;
    g.subnormal_count   += local.subnormal_count;
    g.zero_count        += local.zero_count;
    g.normal_count      += local.normal_count;
    for (int i = 0; i < 256; ++i) g.scale_hist[i] += local.scale_hist[i];
    g.sum_row_len += n;
    g.call_count++;

    if (layer_name && layer_name[0] != '\0') {
        std::string lkey(layer_name);
        FP8SimSrcStats & ls = g_fp8_layer_stats[sid][lkey];
        ls.total_elements    += local.total_elements;
        ls.total_blocks      += local.total_blocks;
        ls.sum_sq_input      += local.sum_sq_input;
        ls.sum_sq_error      += local.sum_sq_error;
        ls.sum_abs_error     += local.sum_abs_error;
        if (local.max_abs_error > ls.max_abs_error) ls.max_abs_error = local.max_abs_error;
        ls.sum_abs_rel_error += local.sum_abs_rel_error;
        ls.rel_error_count   += local.rel_error_count;
        if (local.max_rel_error > ls.max_rel_error) ls.max_rel_error = local.max_rel_error;
        ls.overflow_count    += local.overflow_count;
        ls.underflow_count   += local.underflow_count;
        ls.subnormal_count   += local.subnormal_count;
        ls.zero_count        += local.zero_count;
        ls.normal_count      += local.normal_count;
        for (int i = 0; i < 256; ++i) ls.scale_hist[i] += local.scale_hist[i];
        ls.sum_row_len += n;
        ls.call_count++;
    }

    if (!g_fp8_atexit_registered) {
        g_fp8_atexit_registered = true;
        atexit(fp8_stats_atexit_handler);
    }
}

// ---------------------------------------------------------------------------
// Modified quant/dequant with stats collection
// ---------------------------------------------------------------------------

extern "C" void ggml_sim_fp8e4m3_block_quant_dequant_f32(
        const float * in,
        float       * out,
        int           n,
        int           block,
        void        * scales_out,
        int           src_id,
        const char  * layer_name) {
    if (block <= 0) {
        block = 16;
    }

    // Decide whether to collect stats this call (sampling)
    const int sample_rate = fp8_get_sample_rate();
    const int64_t call_id = g_fp8_call_counter.fetch_add(1, std::memory_order_relaxed);
    bool collect = (sample_rate > 0) && (call_id % sample_rate == 0);

    FP8SimSrcStats local = {};
    local.total_elements = n;
    const int scale_type = ggml_fp8_scale_type_for_src(src_id);

    for (int i = 0; i < n; i += block) {
        const int len = (i + block <= n) ? block : (n - i);

        float inv;
        float mul;
        int k_hist;

        if (scale_type == GGML_SIM_FP8E4M3_SCALE_TYPE_INT8_POW2) {
            // --- int8 power-of-2 scale ---
            const int8_t k = ggml_choose_k_for_block(in + i, len);
            inv = ldexpf(1.0f, -k);
            mul = ldexpf(1.0f,  k);
            k_hist = (int)k;
            if (scales_out) {
                ((int8_t *)scales_out)[i / block] = k;
            }
        } else if (scale_type == GGML_SIM_FP8E4M3_SCALE_TYPE_BF16_EXACT) {
            // --- bf16 exact scale ---
            const float scale_f = ggml_choose_scale_for_block_bf16(in + i, len);
            inv = (scale_f > 0.0f) ? (1.0f / scale_f) : 0.0f;
            mul = scale_f;
            k_hist = (scale_f > 0.0f) ? (int)floorf(log2f(scale_f)) : 0;
            if (scales_out) {
                ((ggml_bf16_t *)scales_out)[i / block] = GGML_FP32_TO_BF16(scale_f);
            }
        } else {
            GGML_ABORT("invalid FP8 scale type: %d", scale_type);
        }

        for (int j = 0; j < len; ++j) {
            const float q = ggml_fp8e4m3_quant_dequant_one(in[i + j] * inv);
            out[i + j] = q * mul;
        }
        if (collect) {
            fp8_accumulate_block_stats_local(local, in + i, out + i, len, inv, k_hist);
        }
    }

    if (collect) {
        fp8_merge_stats(local, n, src_id, layer_name);
    }
}

extern "C" void ggml_sim_fp8e4m3_block_quant_dequant_f32_to_bf16(
        const float     * in,
        ggml_bf16_t     * out,
        int               n,
        int               block,
        void            * scales_out,
        int               src_id,
        const char      * layer_name) {
    if (block <= 0) {
        block = 16;
    }

    const int sample_rate = fp8_get_sample_rate();
    const int64_t call_id = g_fp8_call_counter.fetch_add(1, std::memory_order_relaxed);
    bool collect = (sample_rate > 0) && (call_id % sample_rate == 0);

    // Only allocate temp buffer when collecting stats (need FP32 dequant for error calc)
    float tmp_dq[4096];
    float * dq_buf = nullptr;
    if (collect) {
        dq_buf = (n <= 4096) ? tmp_dq : (float *)malloc((size_t)n * sizeof(float));
        if (dq_buf == nullptr) {
            collect = false;
        }
    }

    FP8SimSrcStats local = {};
    local.total_elements = n;
    const int scale_type = ggml_fp8_scale_type_for_src(src_id);

    for (int i = 0; i < n; i += block) {
        const int len = (i + block <= n) ? block : (n - i);

        float inv;
        float mul;
        int k_hist;

        if (scale_type == GGML_SIM_FP8E4M3_SCALE_TYPE_INT8_POW2) {
            // --- int8 power-of-2 scale ---
            const int8_t k = ggml_choose_k_for_block(in + i, len);
            inv = ldexpf(1.0f, -k);
            mul = ldexpf(1.0f,  k);
            k_hist = (int)k;
            if (scales_out) {
                ((int8_t *)scales_out)[i / block] = k;
            }
        } else if (scale_type == GGML_SIM_FP8E4M3_SCALE_TYPE_BF16_EXACT) {
            // --- bf16 exact scale ---
            const float scale_f = ggml_choose_scale_for_block_bf16(in + i, len);
            inv = (scale_f > 0.0f) ? (1.0f / scale_f) : 0.0f;
            mul = scale_f;
            k_hist = (scale_f > 0.0f) ? (int)floorf(log2f(scale_f)) : 0;
            if (scales_out) {
                ((ggml_bf16_t *)scales_out)[i / block] = GGML_FP32_TO_BF16(scale_f);
            }
        } else {
            GGML_ABORT("invalid FP8 scale type: %d", scale_type);
        }

        for (int j = 0; j < len; ++j) {
            const float q = ggml_fp8e4m3_quant_dequant_one(in[i + j] * inv);
            const float val = q * mul;
            out[i + j] = GGML_FP32_TO_BF16(val);
            if (collect) {
                dq_buf[i + j] = val;
            }
        }
        if (collect) {
            fp8_accumulate_block_stats_local(local, in + i, dq_buf + i, len, inv, k_hist);
        }
    }

    if (collect) {
        if (dq_buf && dq_buf != tmp_dq) {
            free(dq_buf);
        }
        fp8_merge_stats(local, n, src_id, layer_name);
    }
}

// ---------------------------------------------------------------------------
// Stats API
// ---------------------------------------------------------------------------

extern "C" void ggml_fp8_sim_stats_reset(void) {
    std::lock_guard<std::mutex> lock(g_fp8_stats_mtx);
    memset(g_fp8_stats, 0, sizeof(g_fp8_stats));
    g_fp8_layer_stats[0].clear();
    g_fp8_layer_stats[1].clear();
    g_fp8_layer_stats[2].clear();
}

// Helper: write one src stats section
static void fp8_write_src_section(FILE * f, int src_id, const char * name, const FP8SimSrcStats & s) {
    if (s.total_elements == 0) {
        fprintf(f, "\n  %s: (no data collected)\n", name);
        return;
    }

    const double n_elem = (double)s.total_elements;
    const double sig_power = s.sum_sq_input / n_elem;
    const double noise_power = s.sum_sq_error / n_elem;
    const double sqnr_db = (noise_power > 0.0) ? 10.0 * log10(sig_power / noise_power) : 999.0;
    const double mae = s.sum_abs_error / n_elem;
    const double rmse = sqrt(s.sum_sq_error / n_elem);
    const double mean_rel_err_pct = (s.rel_error_count > 0) ? 100.0 * (s.sum_abs_rel_error / (double)s.rel_error_count) : 0.0;
    const double max_rel_err_pct = 100.0 * s.max_rel_error;

    const int64_t classified = s.overflow_count + s.underflow_count + s.subnormal_count + s.zero_count + s.normal_count;
    auto pct = [](int64_t count, int64_t total) -> double {
        return total > 0 ? 100.0 * (double)count / (double)total : 0.0;
    };

    fprintf(f, "\n  %s Quantization Error Analysis\n", name);
    fprintf(f, "  %.*s\n", 60, "────────────────────────────────────────────────────────────");

    fprintf(f, "    Total elements processed  : %" PRId64 "\n", s.total_elements);
    fprintf(f, "    Total blocks processed    : %" PRId64 "\n", s.total_blocks);
    fprintf(f, "\n");
    fprintf(f, "    Signal power (mean x^2)   : %.6e\n", sig_power);
    fprintf(f, "    Noise power  (mean err^2) : %.6e\n", noise_power);
    fprintf(f, "    *** SQNR                  : %.2f dB ***\n", sqnr_db);
    fprintf(f, "\n");
    fprintf(f, "    Mean absolute error (MAE) : %.6e\n", mae);
    fprintf(f, "    RMS error (RMSE)          : %.6e\n", rmse);
    fprintf(f, "    Max absolute error        : %.6e\n", s.max_abs_error);
    fprintf(f, "    Mean relative error       : %.4f%%\n", mean_rel_err_pct);
    fprintf(f, "    Max relative error        : %.4f%%\n", max_rel_err_pct);
    fprintf(f, "\n");

    fprintf(f, "    Element classification (total classified: %" PRId64 "):\n", classified);
    fprintf(f, "      Normal format range   : %12" PRId64 " (%.4f%%)\n", s.normal_count,    pct(s.normal_count,    classified));
    fprintf(f, "      Subnormal (denorm)    : %12" PRId64 " (%.4f%%)\n", s.subnormal_count, pct(s.subnormal_count, classified));
    fprintf(f, "      Underflow (->0)       : %12" PRId64 " (%.4f%%)\n", s.underflow_count, pct(s.underflow_count, classified));
    fprintf(f, "      Overflow (saturated)  : %12" PRId64 " (%.4f%%)\n", s.overflow_count,  pct(s.overflow_count,  classified));
    fprintf(f, "      Zero input            : %12" PRId64 " (%.4f%%)\n", s.zero_count,      pct(s.zero_count,      classified));
    fprintf(f, "\n");

    // Scale histogram – show nonzero bins grouped
    const int scale_type = ggml_fp8_scale_type_for_src(src_id);
    if (scale_type == GGML_SIM_FP8E4M3_SCALE_TYPE_INT8_POW2) {
        fprintf(f, "    Scale exponent (k=log2, int8 pow2 mode) distribution:\n");
    } else {
        fprintf(f, "    Scale exponent (k=floor(log2(scale)), bf16 exact mode) distribution:\n");
    }
    // Find range of nonzero bins
    int kmin = 255, kmax = 0;
    int64_t total_blocks = 0;
    for (int i = 0; i < 256; ++i) {
        if (s.scale_hist[i] > 0) {
            if (i < kmin) kmin = i;
            if (i > kmax) kmax = i;
            total_blocks += s.scale_hist[i];
        }
    }
    if (total_blocks > 0) {
        for (int i = kmin; i <= kmax; ++i) {
            if (s.scale_hist[i] > 0) {
                int k_val = i - 128;
                double frac = (double)s.scale_hist[i] / (double)total_blocks;
                int bar_len = (int)(frac * 50.0);
                if (bar_len < 1 && s.scale_hist[i] > 0) bar_len = 1;
                char bar[52];
                for (int b = 0; b < bar_len && b < 50; ++b) bar[b] = '#';
                bar[bar_len < 50 ? bar_len : 50] = '\0';
                fprintf(f, "      k=%4d : %-50s %6.2f%% (%" PRId64 " blocks)\n",
                        k_val, bar, frac * 100.0, s.scale_hist[i]);
            }
        }
    }
}

extern "C" void ggml_fp8_sim_stats_report(const char * report_file) {
    std::lock_guard<std::mutex> lock(g_fp8_stats_mtx);

    // Skip if no data
    if (g_fp8_stats[0].total_elements == 0 && g_fp8_stats[1].total_elements == 0 && g_fp8_stats[2].total_elements == 0) {
        return;
    }

    // We write to two outputs: stderr and optionally a file
    FILE * outputs[2] = { stderr, nullptr };
    int nout = 1;
    if (report_file) {
        FILE * ff = fopen(report_file, "w");
        if (ff) {
            outputs[nout++] = ff;
        } else {
            fprintf(stderr, "[FP8-SIM] WARNING: cannot open report file '%s'\n", report_file);
        }
    }

    for (int oi = 0; oi < nout; ++oi) {
        FILE * f = outputs[oi];

        fprintf(f, "\n");
        fprintf(f, "================================================================================\n");
        fprintf(f, "  %s + Block Scale Quantization Simulation Analysis Report\n", ggml_sim_format_name());
        fprintf(f, "================================================================================\n");

        // Configuration
        fprintf(f, "\n  Configuration\n");
        fprintf(f, "  %.*s\n", 60, "────────────────────────────────────────────────────────────");
        const int exp_bits = ggml_sim_exponent_bits();
        const int exp_bias = ggml_sim_exponent_bias();
        const int min_sub_exp = ggml_sim_min_norm_exp_unbiased() - ggml_sim_mantissa_bits();
        fprintf(f, "    Simulated format     : %s\n", ggml_sim_format_name());
        fprintf(f, "    Exponent bits        : %d\n", exp_bits);
        fprintf(f, "    Mantissa bits        : %d\n", ggml_sim_mantissa_bits());
        fprintf(f, "    Exponent bias        : %d\n", exp_bias);
        fprintf(f, "    Subnormal support    : %s\n", ggml_sim_support_subnormals() ? "ON" : "OFF (pure-normal)");
        fprintf(f, "    Max finite value     : %.1f\n", ggml_fp8e4m3_max_finite());
        if (ggml_sim_support_subnormals()) {
            fprintf(f, "    Min subnormal value  : 2^(%d) = %.6f\n", min_sub_exp, ggml_fp8e4m3_min_subnormal());
        } else {
            fprintf(f, "    Min subnormal value  : N/A (disabled)\n");
        }
        fprintf(f, "    Block size           : %d\n", GGML_SIM_FP8E4M3_BLOCK);
    fprintf(f, "    Scale type for src0/src1      : %s\n",
        GGML_SIM_FP8E4M3_SCALE_TYPE_IN == GGML_SIM_FP8E4M3_SCALE_TYPE_INT8_POW2
        ? "int8 power-of-2 (scale = 2^k)"
        : "bf16 exact (scale = amax / max_finite)");
    fprintf(f, "    Scale type for output QDQ     : %s\n",
        GGML_SIM_FP8E4M3_SCALE_TYPE_OUT == GGML_SIM_FP8E4M3_SCALE_TYPE_INT8_POW2
        ? "int8 power-of-2 (scale = 2^k)"
        : "bf16 exact (scale = amax / max_finite)");
        fprintf(f, "    Rounding             : RNE (round to nearest even)\n");
        fprintf(f, "    Applied to src0 (weights)     : %s\n", GGML_SIM_FP8E4M3_APPLY_SRC0 ? "YES" : "NO");
        fprintf(f, "    Applied to src1 (activations) : %s\n", GGML_SIM_FP8E4M3_APPLY_SRC1 ? "YES" : "NO");
        const bool apply_out_qdq =
            (GGML_SIM_MATMUL_OUT_MODE == GGML_SIM_MATMUL_OUT_MODE_FP8E4M3) && GGML_SIM_FP8E4M3;
        fprintf(f, "    Applied to output QDQ         : %s\n", apply_out_qdq ? "YES" : "NO");

        // Per-src sections
        fp8_write_src_section(f, 0, "Weights (src0)", g_fp8_stats[0]);
        fp8_write_src_section(f, 1, "Activations (src1)", g_fp8_stats[1]);
        fp8_write_src_section(f, 2, "Matmul output QDQ (src2)", g_fp8_stats[2]);

        // =================================================================
        // Per-Layer Breakdown (sorted by SQNR ascending = worst first)
        // =================================================================
        for (int sid = 0; sid < 3; ++sid) {
            const auto & lmap = g_fp8_layer_stats[sid];
            if (lmap.empty()) continue;

            const char * src_label = sid == 0 ? "Weights (src0)" : (sid == 1 ? "Activations (src1)" : "Matmul output QDQ (src2)");
            fprintf(f, "\n  Per-Layer Breakdown: %s  (%d unique tensors)\n", src_label, (int)lmap.size());
            fprintf(f, "  %.*s\n", 60, "────────────────────────────────────────────────────────────");

            // Build sorted vector by SQNR (ascending = worst first)
            struct LayerEntry {
                std::string name;
                double sqnr;
                double rmse;
                double mae;
                int64_t overflow;
                int64_t underflow;
                int64_t total_elements;
                double noise_frac; // fraction of total noise from this layer
            };

            double total_noise_sq = 0.0;
            for (const auto & kv : lmap) {
                total_noise_sq += kv.second.sum_sq_error;
            }

            std::vector<LayerEntry> entries;
            entries.reserve(lmap.size());
            for (const auto & kv : lmap) {
                const FP8SimSrcStats & ls = kv.second;
                double ne = (double)ls.total_elements;
                double sp = ls.sum_sq_input / ne;
                double np = ls.sum_sq_error / ne;
                double sqnr = (np > 0.0) ? 10.0 * log10(sp / np) : 999.0;
                double rmse = sqrt(ls.sum_sq_error / ne);
                double mae  = ls.sum_abs_error / ne;
                double nfrac = (total_noise_sq > 0.0) ? 100.0 * ls.sum_sq_error / total_noise_sq : 0.0;
                entries.push_back({kv.first, sqnr, rmse, mae, ls.overflow_count, ls.underflow_count, ls.total_elements, nfrac});
            }

            std::sort(entries.begin(), entries.end(), [](const LayerEntry & a, const LayerEntry & b) {
                return a.sqnr < b.sqnr; // worst first
            });

            // Table header
            fprintf(f, "    %-45s %8s %10s %10s %8s %8s %8s\n",
                    "Tensor", "SQNR(dB)", "RMSE", "MAE", "Overflow", "Undflow", "Noise%%");
            fprintf(f, "    %-45s %8s %10s %10s %8s %8s %8s\n",
                    "─────────────────────────────────────────────",
                    "────────", "──────────", "──────────",
                    "────────", "────────", "────────");

            for (const auto & e : entries) {
                // Truncate long names
                std::string dname = e.name;
                if (dname.size() > 45) {
                    dname = "..." + dname.substr(dname.size() - 42);
                }
                fprintf(f, "    %-45s %8.2f %10.2e %10.2e %8" PRId64 " %8" PRId64 " %7.2f%%\n",
                        dname.c_str(), e.sqnr, e.rmse, e.mae, e.overflow, e.underflow, e.noise_frac);
            }

            // Highlight worst 5
            int nworst = (int)entries.size() < 5 ? (int)entries.size() : 5;
            fprintf(f, "\n    >>> Top %d WORST layers (lowest SQNR):\n", nworst);
            for (int wi = 0; wi < nworst; ++wi) {
                const auto & e = entries[wi];
                const char * sev;
                if (e.sqnr > 50.0)      sev = "OK";
                else if (e.sqnr > 40.0) sev = "Minor";
                else if (e.sqnr > 30.0) sev = "Moderate";
                else if (e.sqnr > 20.0) sev = "SIGNIFICANT";
                else                     sev = "!! SEVERE !!";
                fprintf(f, "        #%d  %-40s  SQNR=%.2f dB  [%s]  noise_share=%.1f%%\n",
                        wi + 1, e.name.c_str(), e.sqnr, sev, e.noise_frac);
            }
            fprintf(f, "\n");
        }

        // =================================================================
        // Closed-Loop PPL Analysis
        // =================================================================

        // --- Compute core metrics ---
        double sqnr_src0 = 999.0, sqnr_src1 = 999.0, sqnr_src2 = 999.0;
        double eps_w = 0.0, eps_a = 0.0, eps_o = 0.0; // noise-to-signal power ratios (linear)
        if (g_fp8_stats[0].total_elements > 0 && g_fp8_stats[0].sum_sq_error > 0) {
            double sp0 = g_fp8_stats[0].sum_sq_input / (double)g_fp8_stats[0].total_elements;
            double np0 = g_fp8_stats[0].sum_sq_error / (double)g_fp8_stats[0].total_elements;
            sqnr_src0 = 10.0 * log10(sp0 / np0);
            eps_w = np0 / sp0;
        }
        if (g_fp8_stats[1].total_elements > 0 && g_fp8_stats[1].sum_sq_error > 0) {
            double sp1 = g_fp8_stats[1].sum_sq_input / (double)g_fp8_stats[1].total_elements;
            double np1 = g_fp8_stats[1].sum_sq_error / (double)g_fp8_stats[1].total_elements;
            sqnr_src1 = 10.0 * log10(sp1 / np1);
            eps_a = np1 / sp1;
        }
        if (g_fp8_stats[2].total_elements > 0 && g_fp8_stats[2].sum_sq_error > 0) {
            double sp2 = g_fp8_stats[2].sum_sq_input / (double)g_fp8_stats[2].total_elements;
            double np2 = g_fp8_stats[2].sum_sq_error / (double)g_fp8_stats[2].total_elements;
            sqnr_src2 = 10.0 * log10(sp2 / np2);
            eps_o = np2 / sp2;
        }

        // Detect matmul inner dimension K from src0 row length
        int64_t K = 0;
        if (g_fp8_stats[0].call_count > 0) {
            K = g_fp8_stats[0].sum_row_len / g_fp8_stats[0].call_count;
        }

        // Detect number of layers from per-layer stats
        // Count unique "blk.N" prefixes
        int n_layers = 0;
        {
            std::unordered_map<int, bool> seen_blk;
            for (const auto & kv : g_fp8_layer_stats[0]) {
                // parse "blk.N." prefix
                if (kv.first.substr(0, 4) == "blk.") {
                    size_t dot2 = kv.first.find('.', 4);
                    if (dot2 != std::string::npos) {
                        int blk_id = atoi(kv.first.c_str() + 4);
                        seen_blk[blk_id] = true;
                    }
                }
            }
            n_layers = (int)seen_blk.size();
        }
        if (n_layers == 0) n_layers = 32; // fallback

        // Count matmuls per layer (unique tensor names per block)
        int matmuls_per_layer = 0;
        {
            for (const auto & kv : g_fp8_layer_stats[0]) {
                if (kv.first.substr(0, 4) == "blk." && kv.first.find("blk.0.") == 0) {
                    matmuls_per_layer++;
                }
            }
        }
        if (matmuls_per_layer == 0) matmuls_per_layer = 7; // fallback

        // Combined matmul/output SQNR: input-side quantization + output-side QDQ
        double eps_mm_in = eps_w + eps_a;           // input-side quantization noise contribution
        double eps_mm = eps_mm_in + eps_o;          // total observed output-side noise ratio
        double sqnr_mm = (eps_mm > 0.0) ? -10.0 * log10(eps_mm) : 999.0;

        // Error budget
        int64_t tot_of = g_fp8_stats[0].overflow_count + g_fp8_stats[1].overflow_count + g_fp8_stats[2].overflow_count;
        int64_t tot_uf = g_fp8_stats[0].underflow_count + g_fp8_stats[1].underflow_count + g_fp8_stats[2].underflow_count;
        int64_t tot_sub = g_fp8_stats[0].subnormal_count + g_fp8_stats[1].subnormal_count + g_fp8_stats[2].subnormal_count;
        int64_t tot_el = g_fp8_stats[0].total_elements + g_fp8_stats[1].total_elements + g_fp8_stats[2].total_elements;

        // ---------------------------------------------------------------
        // Step 1: Measured Quantization Error Summary
        // ---------------------------------------------------------------
        fprintf(f, "\n  Step 1: Measured Quantization Error Summary\n");
        fprintf(f, "  %.*s\n", 60, "────────────────────────────────────────────────────────────");
        fprintf(f, "    Weight per-element SQNR      : %.2f dB  (noise/signal = %.4f%%)\n",
                sqnr_src0, eps_w * 100.0);
        fprintf(f, "    Activation per-element SQNR  : %.2f dB  (noise/signal = %.4f%%)\n",
                sqnr_src1, eps_a * 100.0);
        fprintf(f, "    Output QDQ per-element SQNR  : %.2f dB  (noise/signal = %.4f%%)\n",
            sqnr_src2, eps_o * 100.0);
        fprintf(f, "\n");
        fprintf(f, "    Input-side matmul noise ratio: eps_w + eps_a = %.4f%%\n", eps_mm_in * 100.0);
        fprintf(f, "    Total output noise ratio     : eps_w + eps_a + eps_out = %.4f%%\n", eps_mm * 100.0);
        fprintf(f, "    Combined output SQNR         : %.2f dB\n", sqnr_mm);
        fprintf(f, "\n");
        fprintf(f, "    Detected matmul inner dimension K = %" PRId64 " (from src0 avg row length)\n", K);
        fprintf(f, "    Detected model depth L = %d layers, %d weight matmuls/layer\n", n_layers, matmuls_per_layer);
        fprintf(f, "\n");
        fprintf(f, "    Overflow  : %" PRId64 " / %" PRId64 " (%.6f%%)\n",
                tot_of, tot_el, tot_el > 0 ? 100.0 * (double)tot_of / (double)tot_el : 0.0);
        fprintf(f, "    Underflow : %" PRId64 " / %" PRId64 " (%.6f%%)\n",
                tot_uf, tot_el, tot_el > 0 ? 100.0 * (double)tot_uf / (double)tot_el : 0.0);
        fprintf(f, "    Subnormal : %" PRId64 " / %" PRId64 " (%.6f%%)\n",
                tot_sub, tot_el, tot_el > 0 ? 100.0 * (double)tot_sub / (double)tot_el : 0.0);
        fprintf(f, "\n");
        fprintf(f, "    Physical meaning:\n");
        fprintf(f, "      eps_w = %.4f%% means each weight has ~%.2f%% RMS relative error.\n",
                eps_w * 100.0, sqrt(eps_w) * 100.0);
        fprintf(f, "      eps_a = %.4f%% means each activation has ~%.2f%% RMS relative error.\n",
                eps_a * 100.0, sqrt(eps_a) * 100.0);
        fprintf(f, "      eps_o = %.4f%% means output QDQ adds ~%.2f%% RMS relative error.\n",
            eps_o * 100.0, sqrt(eps_o) * 100.0);
        fprintf(f, "\n");

        // ---------------------------------------------------------------
        // Step 2: From Per-Element Error to Matmul Output Error
        // ---------------------------------------------------------------
        fprintf(f, "  Step 2: From Per-Element Error to Matmul Output Error\n");
        fprintf(f, "  %.*s\n", 60, "────────────────────────────────────────────────────────────");
        fprintf(f, "    For Y = X @ W^T (matmul with K=%"  PRId64 " inner dimension):\n", K);
        fprintf(f, "\n");
        fprintf(f, "    Each output element Y[i,j] = sum_{k=1}^{K} X[i,k] * W[j,k]\n");
        fprintf(f, "    Error:  dY[i,j] = sum_k (eX[i,k]*W[j,k] + X[i,k]*eW[j,k])\n");
        fprintf(f, "\n");
        fprintf(f, "    SQNR of final stored output (after output QDQ):\n");
        fprintf(f, "      SQNR_Y ~= 1 / (eps_w + eps_a + eps_out)\n");
        fprintf(f, "             = 1 / (%.6f + %.6f + %.6f)\n", eps_w, eps_a, eps_o);
        fprintf(f, "             = %.2f  (%.2f dB)\n", 1.0/eps_mm, sqnr_mm);
        fprintf(f, "\n");
        fprintf(f, "    KEY INSIGHT: input-side quantization sets the base SQNR floor,\n");
        fprintf(f, "    and output QDQ contributes an additional additive noise term eps_out.\n");
        fprintf(f, "    The inner dimension K does NOT improve SQNR\n");
        fprintf(f, "    (both signal and noise scale with K, so the ratio is preserved).\n");
        fprintf(f, "    However, K matters for ABSOLUTE error magnitude.\n");
        fprintf(f, "\n");

        // ---------------------------------------------------------------
        // Step 3: Error Propagation Through Transformer Layers
        // ---------------------------------------------------------------
        fprintf(f, "  Step 3: Error Propagation Through %d Transformer Layers\n", n_layers);
        fprintf(f, "  %.*s\n", 60, "────────────────────────────────────────────────────────────");
        fprintf(f, "\n");
        fprintf(f, "    A Transformer layer has residual connections:\n");
        fprintf(f, "      h_{l+1} = LayerNorm( h_l + Attn(h_l) )\n");
        fprintf(f, "      h_{l+2} = LayerNorm( h_{l+1} + FFN(h_{l+1}) )\n");
        fprintf(f, "\n");
        fprintf(f, "    Three mechanisms DAMPEN error propagation:\n");
        fprintf(f, "\n");
        fprintf(f, "    (a) Residual connections (additive, not multiplicative):\n");
        fprintf(f, "        Errors add to the residual stream, they do NOT multiply.\n");
        fprintf(f, "        Naive worst-case: noise ratio grows as num_injections * eps_mm\n");
        fprintf(f, "\n");
        fprintf(f, "    (b) Sub-layer output fraction:\n");
        fprintf(f, "        Each sub-layer output is typically gamma ~= 10-30%% of the\n");
        fprintf(f, "        residual norm. So effective noise injection per sub-layer is\n");
        fprintf(f, "        gamma^2 * eps_mm, not eps_mm. (gamma ~= 0.2 typical)\n");
        fprintf(f, "\n");
        fprintf(f, "    (c) LayerNorm renormalization:\n");
        fprintf(f, "        LayerNorm rescales to unit variance after each sub-layer,\n");
        fprintf(f, "        preventing noise magnitude from growing with residual norm.\n");
        fprintf(f, "        This provides additional dampening by factor ~sqrt(L).\n");
        fprintf(f, "\n");

        // Compute corrected noise accumulation
        const double gamma = 0.2;   // sub-layer output / residual norm ratio
        const int N_inject = 2 * n_layers; // attn_out + ffn_out per layer
        const double noise_naive = (double)N_inject * eps_mm;  // worst case (no dampening)
        const double noise_residual = (double)N_inject * gamma * gamma * eps_mm; // with sub-layer scaling
        const double noise_with_ln = noise_residual / sqrt((double)n_layers); // with LayerNorm dampening

        fprintf(f, "    Quantitative estimate (L=%d, gamma=%.1f):\n", n_layers, gamma);
        fprintf(f, "      Direct residual injections          : %d (2 per layer)\n", N_inject);
        fprintf(f, "      Per-matmul total noise ratio        : %.6f (= eps_w + eps_a + eps_out)\n", eps_mm);
        fprintf(f, "\n");
        fprintf(f, "      Naive (no dampening):  %.4f%%  = %d x %.6f\n",
                noise_naive * 100.0, N_inject, eps_mm);
        fprintf(f, "      + Sub-layer scaling:   %.4f%%  = %d x %.1f^2 x %.6f\n",
                noise_residual * 100.0, N_inject, gamma, eps_mm);
        fprintf(f, "      + LayerNorm dampening: %.4f%%  = above / sqrt(%d)\n",
                noise_with_ln * 100.0, n_layers);
        fprintf(f, "\n");

        double sqnr_residual = (noise_with_ln > 0) ? -10.0 * log10(noise_with_ln) : 999.0;
        fprintf(f, "      Effective residual stream SQNR: %.2f dB\n", sqnr_residual);
        fprintf(f, "\n");

        // ---------------------------------------------------------------
        // Step 4: From Residual Noise to PPL
        // ---------------------------------------------------------------
        fprintf(f, "  Step 4: From Residual Noise to PPL\n");
        fprintf(f, "  %.*s\n", 60, "────────────────────────────────────────────────────────────");
        fprintf(f, "    PPL = exp(CE). For small perturbations:\n");
        fprintf(f, "      DPPL/PPL ~ exp(DCE) - 1 ~ DCE  (for small DCE)\n");
        fprintf(f, "\n");
        fprintf(f, "    The cross-entropy perturbation DCE depends on noise at the\n");
        fprintf(f, "    final logits, which comes from:\n");
        fprintf(f, "      (1) Accumulated residual noise amplified by LM head weights\n");
        fprintf(f, "      (2) LM head weight quantization noise itself\n");
        fprintf(f, "\n");
        fprintf(f, "    For the simplified model (DCE ~ effective_noise_ratio):\n");
        fprintf(f, "      Predicted DPPL/PPL ~ %.4f%% to %.4f%%\n",
                noise_with_ln * 100.0, noise_residual * 100.0);
        fprintf(f, "      (range: with LayerNorm dampening ... without)\n");
        fprintf(f, "\n");

        // Read baseline PPL from env var
        const char * env_ppl = getenv("FP8_SIM_BASELINE_PPL");
        if (env_ppl) {
            double ppl_base = atof(env_ppl);
            if (ppl_base > 0) {
                double pred_low = ppl_base * (1.0 + noise_with_ln);
                double pred_high = ppl_base * (1.0 + noise_residual);
                fprintf(f, "    Baseline PPL (from FP8_SIM_BASELINE_PPL): %.4f\n", ppl_base);
                fprintf(f, "    Predicted FP8 PPL range: %.4f to %.4f\n", pred_low, pred_high);
                fprintf(f, "    Predicted DPPL: %.4f to %.4f  (%.3f%% to %.3f%%)\n",
                        pred_low - ppl_base, pred_high - ppl_base,
                        noise_with_ln * 100.0, noise_residual * 100.0);
                fprintf(f, "\n");
                fprintf(f, "    Compare with your measured PPL to validate this model.\n");
                fprintf(f, "    If measured delta is within the predicted range, the\n");
                fprintf(f, "    quantization error fully explains the PPL increase.\n");
            }
        } else {
            fprintf(f, "    TIP: Set environment variable FP8_SIM_BASELINE_PPL=<your_f16_ppl>\n");
            fprintf(f, "    before running to get exact predicted PPL range.\n");
            fprintf(f, "    Example:  set FP8_SIM_BASELINE_PPL=7.52\n");
        }
        fprintf(f, "\n");

        // ---------------------------------------------------------------
        // Step 5: Dominant Error Source & Error Budget
        // ---------------------------------------------------------------
        fprintf(f, "  Step 5: Dominant Error Source & Error Budget\n");
        fprintf(f, "  %.*s\n", 60, "────────────────────────────────────────────────────────────");

        double total_noise = g_fp8_stats[0].sum_sq_error + g_fp8_stats[1].sum_sq_error + g_fp8_stats[2].sum_sq_error;
        double src0_frac = (total_noise > 0) ? 100.0 * g_fp8_stats[0].sum_sq_error / total_noise : 0.0;
        double src1_frac = (total_noise > 0) ? 100.0 * g_fp8_stats[1].sum_sq_error / total_noise : 0.0;
        double src2_frac = (total_noise > 0) ? 100.0 * g_fp8_stats[2].sum_sq_error / total_noise : 0.0;

        fprintf(f, "    Absolute noise power breakdown:\n");
        fprintf(f, "      Weight quant noise (sum err^2)    : %.6e  (%.1f%% of total)\n",
                g_fp8_stats[0].sum_sq_error, src0_frac);
        fprintf(f, "      Activation quant noise (sum err^2): %.6e  (%.1f%% of total)\n",
                g_fp8_stats[1].sum_sq_error, src1_frac);
        fprintf(f, "      Output QDQ noise (sum err^2)      : %.6e  (%.1f%% of total)\n",
            g_fp8_stats[2].sum_sq_error, src2_frac);
        fprintf(f, "\n");
        fprintf(f, "    Relative noise (SQNR, lower = worse):\n");
        fprintf(f, "      Weight SQNR     : %.2f dB  (eps_w = %.6f)\n", sqnr_src0, eps_w);
        fprintf(f, "      Activation SQNR : %.2f dB  (eps_a = %.6f)\n", sqnr_src1, eps_a);
        fprintf(f, "      Output QDQ SQNR : %.2f dB  (eps_o = %.6f)\n", sqnr_src2, eps_o);
        fprintf(f, "      Contribution to combined eps_mm:\n");
        fprintf(f, "        eps_w / eps_mm = %.1f%%  (weight contribution to matmul noise)\n",
                eps_mm > 0 ? eps_w / eps_mm * 100.0 : 0.0);
        fprintf(f, "        eps_a / eps_mm = %.1f%%  (activation contribution to matmul noise)\n",
                eps_mm > 0 ? eps_a / eps_mm * 100.0 : 0.0);
        fprintf(f, "        eps_o / eps_mm = %.1f%%  (output QDQ contribution to final noise)\n",
            eps_mm > 0 ? eps_o / eps_mm * 100.0 : 0.0);
        fprintf(f, "\n");

        if (eps_w >= eps_a && eps_w >= eps_o) {
            fprintf(f, "    --> Weights are the SQNR bottleneck (%.2f dB is the lowest source-equivalent SQNR).\n", sqnr_src0);
            fprintf(f, "        To reduce PPL: improve weight quantization (better scaling,\n");
            fprintf(f, "        larger block, or higher precision for sensitive layers).\n");
        } else if (eps_a >= eps_w && eps_a >= eps_o) {
            fprintf(f, "    --> Activations are the SQNR bottleneck (%.2f dB is the lowest source-equivalent SQNR).\n", sqnr_src1);
            fprintf(f, "        To reduce PPL: improve activation quantization or disable\n");
            fprintf(f, "        FP8 for activations (-DGGML_SIM_FP8E4M3_APPLY_SRC1=0).\n");
        } else {
            fprintf(f, "    --> Output QDQ is the bottleneck (%.2f dB).\n", sqnr_src2);
            fprintf(f, "        To reduce PPL: keep output in higher precision for sensitive\n");
            fprintf(f, "        ops or tune output scaling/placement of QDQ.\n");
        }
        fprintf(f, "\n");

        // Flag specific per-layer concerns
        fprintf(f, "    Per-layer bottleneck analysis:\n");
        for (int sid = 0; sid < 3; ++sid) {
            const auto & lmap = g_fp8_layer_stats[sid];
            const char * slbl = sid == 0 ? "src0" : (sid == 1 ? "src1" : "src2");
            int severe_count = 0;
            for (const auto & kv : lmap) {
                const FP8SimSrcStats & ls = kv.second;
                if (ls.total_elements == 0 || ls.sum_sq_input == 0) continue;
                double ne = (double)ls.total_elements;
                double lsqnr = 10.0 * log10((ls.sum_sq_input / ne) / (ls.sum_sq_error / ne));
                if (lsqnr < 25.0) severe_count++;
            }
            fprintf(f, "      %s: %d tensors with SQNR < 25 dB (significant/severe)\n",
                    slbl, severe_count);
        }
        fprintf(f, "\n");

        // ---------------------------------------------------------------
        // Step 6: Closed-Loop Summary & Conclusion
        // ---------------------------------------------------------------
        fprintf(f, "  Step 6: Closed-Loop Summary\n");
        fprintf(f, "  %.*s\n", 60, "────────────────────────────────────────────────────────────");
        fprintf(f, "\n");
        fprintf(f, "    Quantization Error Chain:\n");
        fprintf(f, "\n");
        fprintf(f, "    %s block=%d    Per-element SQNR    Matmul output SQNR\n", ggml_sim_format_name(), GGML_SIM_FP8E4M3_BLOCK);
        fprintf(f, "    ─────────────────  ─> ─────────────────  ─> ────────────────────\n");
        fprintf(f, "    Weight  : %.2f dB ──┐\n", sqnr_src0);
        fprintf(f, "    Activation: %.2f dB ─┼──> Input-side: %.2f dB\n", sqnr_src1, (eps_mm_in > 0) ? -10.0 * log10(eps_mm_in) : 999.0);
        fprintf(f, "    OutputQDQ: %.2f dB ──┘    Final output: %.2f dB\n", sqnr_src2, sqnr_mm);
        fprintf(f, "\n");
        fprintf(f, "    Matmul SQNR    %d layers         Residual SQNR\n", n_layers);
        fprintf(f, "    ───────────  ─> ─────────────  ─> ──────────────\n");
        fprintf(f, "    %.2f dB      residual+LN       %.2f dB (effective at output)\n",
                sqnr_mm, sqnr_residual);
        fprintf(f, "\n");
        fprintf(f, "    Residual SQNR   LM-head logits     Cross-Entropy     PPL\n");
        fprintf(f, "    ──────────────  ─> ────────────  ─> ─────────────  ─> ─────\n");
        fprintf(f, "    %.2f dB         + LM head noise     DCE ~ %.4f%%     DPPL/PPL ~ %.4f%%-%.4f%%\n",
                sqnr_residual, noise_with_ln * 100.0, noise_with_ln * 100.0, noise_residual * 100.0);
        fprintf(f, "\n");
        fprintf(f, "    ============================================================\n");
        fprintf(f, "    PREDICTED PPL INCREASE: %.3f%% to %.3f%%\n",
                noise_with_ln * 100.0, noise_residual * 100.0);
        fprintf(f, "    ============================================================\n");
        fprintf(f, "\n");

        // Guidance for before/after comparison
        fprintf(f, "  Guidance for Interpreting Results\n");
        fprintf(f, "  %.*s\n", 60, "────────────────────────────────────────────────────────────");
        fprintf(f, "\n");
        fprintf(f, "    If MEASURED DPPL/PPL falls within the predicted range:\n");
        fprintf(f, "      -> The %s quantization error FULLY explains the PPL increase.\n", ggml_sim_format_name());
        fprintf(f, "      -> No hidden issues; the model is well-characterized.\n");
        fprintf(f, "\n");
        fprintf(f, "    If MEASURED is HIGHER than predicted:\n");
        fprintf(f, "      -> Check per-layer breakdown for outlier layers (SQNR < 20 dB)\n");
        fprintf(f, "      -> Sensitive layers (LM head, embedding) may amplify error\n");
        fprintf(f, "      -> Consider mixed-precision: keep sensitive layers in F16\n");
        fprintf(f, "\n");
        fprintf(f, "    If MEASURED is LOWER than predicted:\n");
        fprintf(f, "      -> Model is robust to this quantization noise level\n");
        fprintf(f, "      -> %s quantization is viable for this model\n", ggml_sim_format_name());
        fprintf(f, "\n");
        fprintf(f, "    For before/after comparison (e.g., scaling fix):\n");
        fprintf(f, "      -> SQNR improvement of X dB reduces noise power by 10^(X/10)\n");
        fprintf(f, "      -> Expected PPL improvement scales proportionally with noise power\n");
        fprintf(f, "      -> Example: +15 dB SQNR -> noise reduced ~32x -> PPL delta reduced ~32x\n");
        fprintf(f, "\n");
        fprintf(f, "================================================================================\n");
        fprintf(f, "\n");
    }

    // Close file output
    if (nout > 1 && outputs[1]) {
        fclose(outputs[1]);
        fprintf(stderr, "[FP8-SIM] Analysis report written to: %s\n", report_file);
    }
}

#endif // GGML_SIM_FP8E4M3

#include <cassert>

#if GGML_REDUCTION_PROD_PROFILE
#include <atomic>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

struct ggml_reduction_prod_sample {
    int64_t reduction_id;
    int64_t n;
    double sum;
    double sum_abs;
    double sum_sq;
    double max_abs;
    double cancel_ratio;
    double neff;
    double neff_ratio;
    double frac_lt_2p_8;
    double frac_lt_2p_10;
    double frac_lt_2p_12;
    int64_t block_terms;
    int64_t block_dropped;
    double block_drop_ratio;
};

struct ggml_reduction_prod_global_stats {
    int64_t reductions;
    int64_t total_products;
    double sum_signed_products;
    double sum_abs_products;
    double sum_sq_products;
    double max_abs_product;
    double sum_cancel_ratio;
    double sum_neff_ratio;
    double sum_frac_lt_2p_8;
    double sum_frac_lt_2p_10;
    double sum_frac_lt_2p_12;
    int64_t total_block_terms;
    int64_t total_block_dropped;
    double sum_block_drop_ratio;
    int64_t sampled_kept;
    int64_t sampled_dropped;
    int64_t hist[GGML_REDUCTION_PROD_PROFILE_BINS];
};

static std::mutex g_reduction_prod_mtx;
static std::atomic<int64_t> g_reduction_prod_counter{0};
static std::atomic<int64_t> g_reduction_prod_all_block_terms{0};
static bool g_reduction_prod_atexit_registered = false;
static ggml_reduction_prod_global_stats g_reduction_prod_stats;
static std::vector<ggml_reduction_prod_sample> g_reduction_prod_samples;

static inline bool ggml_reduction_prod_should_sample(int64_t * rid_out) {
    const int sample_rate = GGML_REDUCTION_PROD_PROFILE_SAMPLE_RATE > 0 ? GGML_REDUCTION_PROD_PROFILE_SAMPLE_RATE : 1;
    const int64_t rid = g_reduction_prod_counter.fetch_add(1, std::memory_order_relaxed) + 1;
    if (rid_out) {
        *rid_out = rid;
    }
    return (rid % sample_rate) == 0;
}

static inline int ggml_reduction_prod_hist_bin(const float abs_p) {
    if (!(abs_p > 0.0f) || !std::isfinite(abs_p)) {
        return 0;
    }
    const float min_lg = (float) GGML_REDUCTION_PROD_PROFILE_HIST_MIN_LOG2;
    const float max_lg = (float) GGML_REDUCTION_PROD_PROFILE_HIST_MAX_LOG2;
    const float w = (max_lg - min_lg) / (float) GGML_REDUCTION_PROD_PROFILE_BINS;
    const float lg = std::log2(abs_p);
    int b = (int) std::floor((lg - min_lg) / w);
    if (b < 0) {
        b = 0;
    }
    if (b >= GGML_REDUCTION_PROD_PROFILE_BINS) {
        b = GGML_REDUCTION_PROD_PROFILE_BINS - 1;
    }
    return b;
}

static inline float ggml_reduction_prod_mul_bf16(
        const ggml_bf16_t * GGML_RESTRICT x,
        const ggml_bf16_t * GGML_RESTRICT y,
        const int i,
        const bool trunc_x,
        const bool trunc_y) {
    ggml_bf16_t xb = trunc_x ? ggml_bf16_rna_trunc4(x[i]) : x[i];
    ggml_bf16_t yb = trunc_y ? ggml_bf16_rna_trunc4(y[i]) : y[i];
    return GGML_BF16_TO_FP32(xb) * GGML_BF16_TO_FP32(yb);
}

static void ggml_reduction_prod_dump_atexit(void) {
    char path_summary[512];
    char path_hist[512];
    char path_samples[512];
    snprintf(path_summary, sizeof(path_summary), "%s_summary.log", GGML_REDUCTION_PROD_PROFILE_PREFIX);
    snprintf(path_hist, sizeof(path_hist), "%s_global_hist.csv", GGML_REDUCTION_PROD_PROFILE_PREFIX);
    snprintf(path_samples, sizeof(path_samples), "%s_samples.csv", GGML_REDUCTION_PROD_PROFILE_PREFIX);

    std::lock_guard<std::mutex> lock(g_reduction_prod_mtx);
    if (g_reduction_prod_stats.reductions <= 0) {
        return;
    }

    FILE * fs = fopen(path_summary, "w");
    if (fs) {
        const double nred = (double) g_reduction_prod_stats.reductions;
        const int64_t all_reductions = g_reduction_prod_counter.load(std::memory_order_relaxed);
        const int64_t all_block_terms = g_reduction_prod_all_block_terms.load(std::memory_order_relaxed);
        const double avg_n = (double) g_reduction_prod_stats.total_products / nred;
        const double avg_cancel = g_reduction_prod_stats.sum_cancel_ratio / nred;
        const double avg_neff_ratio = g_reduction_prod_stats.sum_neff_ratio / nred;
        const double avg_frac_lt_2p_8 = g_reduction_prod_stats.sum_frac_lt_2p_8 / nred;
        const double avg_frac_lt_2p_10 = g_reduction_prod_stats.sum_frac_lt_2p_10 / nred;
        const double avg_frac_lt_2p_12 = g_reduction_prod_stats.sum_frac_lt_2p_12 / nred;
        const double sampled_avg_block_drop_ratio = g_reduction_prod_stats.sum_block_drop_ratio / nred;
        const double sampled_global_block_drop_ratio = g_reduction_prod_stats.total_block_terms > 0
            ? (double) g_reduction_prod_stats.total_block_dropped / (double) g_reduction_prod_stats.total_block_terms
            : 0.0;
        const double estimated_global_block_dropped = sampled_global_block_drop_ratio * (double) all_block_terms;
        const double estimated_global_block_drop_ratio = sampled_global_block_drop_ratio;
        const double mean_abs_p = g_reduction_prod_stats.total_products > 0
            ? g_reduction_prod_stats.sum_abs_products / (double) g_reduction_prod_stats.total_products
            : 0.0;
        const double rms_p = g_reduction_prod_stats.total_products > 0
            ? std::sqrt(g_reduction_prod_stats.sum_sq_products / (double) g_reduction_prod_stats.total_products)
            : 0.0;

        fprintf(fs, "Reduction Product Profile Summary\n");
        fprintf(fs, "================================\n");
        fprintf(fs, "reductions            : %" PRId64 "\n", g_reduction_prod_stats.reductions);
        fprintf(fs, "all_reductions        : %" PRId64 "\n", all_reductions);
        fprintf(fs, "total_products        : %" PRId64 "\n", g_reduction_prod_stats.total_products);
        fprintf(fs, "avg_n_per_reduction   : %.3f\n", avg_n);
        fprintf(fs, "mean_abs_product      : %.6e\n", mean_abs_p);
        fprintf(fs, "rms_product           : %.6e\n", rms_p);
        fprintf(fs, "max_abs_product       : %.6e\n", g_reduction_prod_stats.max_abs_product);
        fprintf(fs, "block_size            : %d\n", GGML_SIM_FP8E4M3_BLOCK);
        fprintf(fs, "block_drop_log2_n     : %d\n", GGML_REDUCTION_PROD_BLOCK_DROP_LOG2_N);
        fprintf(fs, "avg_cancel_ratio      : %.6f\n", avg_cancel);
        fprintf(fs, "avg_neff_ratio        : %.6f\n", avg_neff_ratio);
        fprintf(fs, "avg_frac_lt_2^-8      : %.6f\n", avg_frac_lt_2p_8);
        fprintf(fs, "avg_frac_lt_2^-10     : %.6f\n", avg_frac_lt_2p_10);
        fprintf(fs, "avg_frac_lt_2^-12     : %.6f\n", avg_frac_lt_2p_12);
        fprintf(fs, "sampled_block_terms   : %" PRId64 "\n", g_reduction_prod_stats.total_block_terms);
        fprintf(fs, "sampled_block_dropped : %" PRId64 "\n", g_reduction_prod_stats.total_block_dropped);
        fprintf(fs, "sampled_global_block_drop_ratio: %.6f\n", sampled_global_block_drop_ratio);
        fprintf(fs, "sampled_avg_block_drop_ratio   : %.6f\n", sampled_avg_block_drop_ratio);
        fprintf(fs, "all_block_terms       : %" PRId64 "\n", all_block_terms);
        fprintf(fs, "estimated_global_block_dropped : %.0f\n", estimated_global_block_dropped);
        fprintf(fs, "estimated_global_block_drop_ratio: %.6f\n", estimated_global_block_drop_ratio);
        fprintf(fs, "sampled_kept          : %" PRId64 "\n", g_reduction_prod_stats.sampled_kept);
        fprintf(fs, "sampled_dropped       : %" PRId64 "\n", g_reduction_prod_stats.sampled_dropped);
        fclose(fs);
    }

    FILE * fh = fopen(path_hist, "w");
    if (fh) {
        const double min_lg = (double) GGML_REDUCTION_PROD_PROFILE_HIST_MIN_LOG2;
        const double max_lg = (double) GGML_REDUCTION_PROD_PROFILE_HIST_MAX_LOG2;
        const double w = (max_lg - min_lg) / (double) GGML_REDUCTION_PROD_PROFILE_BINS;
        fprintf(fh, "bin,log2_lo,log2_hi,count\n");
        for (int b = 0; b < GGML_REDUCTION_PROD_PROFILE_BINS; ++b) {
            const double lo = min_lg + w * (double) b;
            const double hi = lo + w;
            fprintf(fh, "%d,%.6f,%.6f,%" PRId64 "\n", b, lo, hi, g_reduction_prod_stats.hist[b]);
        }
        fclose(fh);
    }

    FILE * fp = fopen(path_samples, "w");
    if (fp) {
        fprintf(fp, "reduction_id,n,sum,sum_abs,sum_sq,max_abs,cancel_ratio,neff,neff_ratio,frac_lt_2p_8,frac_lt_2p_10,frac_lt_2p_12,block_terms,block_dropped,block_drop_ratio\n");
        for (const auto & s : g_reduction_prod_samples) {
            fprintf(fp,
                "%" PRId64 ",%" PRId64 ",%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%" PRId64 ",%" PRId64 ",%.9e\n",
                s.reduction_id,
                s.n,
                s.sum,
                s.sum_abs,
                s.sum_sq,
                s.max_abs,
                s.cancel_ratio,
                s.neff,
                s.neff_ratio,
                s.frac_lt_2p_8,
                s.frac_lt_2p_10,
                s.frac_lt_2p_12,
                s.block_terms,
                s.block_dropped,
                s.block_drop_ratio);
        }
        fclose(fp);
    }
}

static ggml_float ggml_reduction_prod_profile_run_bf16(
        const ggml_bf16_t * GGML_RESTRICT x,
        const ggml_bf16_t * GGML_RESTRICT y,
    const int64_t rid,
        const int n,
        const bool trunc_x,
        const bool trunc_y) {
    int64_t hist_local[GGML_REDUCTION_PROD_PROFILE_BINS];
    memset(hist_local, 0, sizeof(hist_local));

    double sum = 0.0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double max_abs = 0.0;

    for (int i = 0; i < n; ++i) {
        const float p = ggml_reduction_prod_mul_bf16(x, y, i, trunc_x, trunc_y);
        const double pd = (double) p;
        const double ap = std::fabs(pd);
        sum += pd;
        sum_abs += ap;
        sum_sq += pd * pd;
        if (ap > max_abs) {
            max_abs = ap;
        }
        hist_local[ggml_reduction_prod_hist_bin((float) ap)]++;
    }

    double frac_lt_2p_8 = 0.0;
    double frac_lt_2p_10 = 0.0;
    double frac_lt_2p_12 = 0.0;
    if (n > 0 && max_abs > 0.0) {
        int64_t c8 = 0;
        int64_t c10 = 0;
        int64_t c12 = 0;
        const double th8 = max_abs * std::ldexp(1.0, -8);
        const double th10 = max_abs * std::ldexp(1.0, -10);
        const double th12 = max_abs * std::ldexp(1.0, -12);
        for (int i = 0; i < n; ++i) {
            const float p = ggml_reduction_prod_mul_bf16(x, y, i, trunc_x, trunc_y);
            const double ap = std::fabs((double) p);
            if (ap < th8) {
                c8++;
            }
            if (ap < th10) {
                c10++;
            }
            if (ap < th12) {
                c12++;
            }
        }
        frac_lt_2p_8 = (double) c8 / (double) n;
        frac_lt_2p_10 = (double) c10 / (double) n;
        frac_lt_2p_12 = (double) c12 / (double) n;
    }

    int64_t block_terms = 0;
    int64_t block_dropped = 0;
    const int block_size = GGML_SIM_FP8E4M3_BLOCK > 0 ? GGML_SIM_FP8E4M3_BLOCK : 16;
    const int block_drop_n = GGML_REDUCTION_PROD_BLOCK_DROP_LOG2_N;
    if (n > 0 && block_size > 0 && block_drop_n >= 0) {
        double running_sum = 0.0;
        for (int i = 0; i < n; i += block_size) {
            const int len = i + block_size <= n ? block_size : (n - i);
            double block_dot = 0.0;
            for (int j = 0; j < len; ++j) {
                const float p = ggml_reduction_prod_mul_bf16(x, y, i + j, trunc_x, trunc_y);
                block_dot += (double) p;
            }

            block_terms++;
            const double th = std::fabs(running_sum) * std::ldexp(1.0, -block_drop_n);
            if (std::fabs(block_dot) < th) {
                block_dropped++;
                continue;
            }
            running_sum += block_dot;
        }
    }
    const double block_drop_ratio = block_terms > 0 ? (double) block_dropped / (double) block_terms : 0.0;

    const double cancel_ratio = sum_abs > 0.0 ? std::fabs(sum) / sum_abs : 0.0;
    const double neff = sum_sq > 0.0 ? (sum_abs * sum_abs) / sum_sq : 0.0;
    const double neff_ratio = n > 0 ? neff / (double) n : 0.0;

    {
        std::lock_guard<std::mutex> lock(g_reduction_prod_mtx);

        g_reduction_prod_stats.reductions++;
        g_reduction_prod_stats.total_products += n;
        g_reduction_prod_stats.sum_signed_products += sum;
        g_reduction_prod_stats.sum_abs_products += sum_abs;
        g_reduction_prod_stats.sum_sq_products += sum_sq;
        if (max_abs > g_reduction_prod_stats.max_abs_product) {
            g_reduction_prod_stats.max_abs_product = max_abs;
        }
        g_reduction_prod_stats.sum_cancel_ratio += cancel_ratio;
        g_reduction_prod_stats.sum_neff_ratio += neff_ratio;
        g_reduction_prod_stats.sum_frac_lt_2p_8 += frac_lt_2p_8;
        g_reduction_prod_stats.sum_frac_lt_2p_10 += frac_lt_2p_10;
        g_reduction_prod_stats.sum_frac_lt_2p_12 += frac_lt_2p_12;
        g_reduction_prod_stats.total_block_terms += block_terms;
        g_reduction_prod_stats.total_block_dropped += block_dropped;
        g_reduction_prod_stats.sum_block_drop_ratio += block_drop_ratio;

        for (int b = 0; b < GGML_REDUCTION_PROD_PROFILE_BINS; ++b) {
            g_reduction_prod_stats.hist[b] += hist_local[b];
        }

        if ((int64_t) g_reduction_prod_samples.size() < (int64_t) GGML_REDUCTION_PROD_PROFILE_MAX_SAMPLES) {
            g_reduction_prod_samples.push_back({
                rid,
                n,
                sum,
                sum_abs,
                sum_sq,
                max_abs,
                cancel_ratio,
                neff,
                neff_ratio,
                frac_lt_2p_8,
                frac_lt_2p_10,
                frac_lt_2p_12,
                block_terms,
                block_dropped,
                block_drop_ratio,
            });
            g_reduction_prod_stats.sampled_kept++;
        } else {
            g_reduction_prod_stats.sampled_dropped++;
        }

        if (!g_reduction_prod_atexit_registered) {
            g_reduction_prod_atexit_registered = true;
            atexit(ggml_reduction_prod_dump_atexit);
        }
    }

    return (ggml_float) sum;
}
#endif // GGML_REDUCTION_PROD_PROFILE

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

#if GGML_REDUCTION_PROD_PROFILE
    const int block_size_all = GGML_SIM_FP8E4M3_BLOCK > 0 ? GGML_SIM_FP8E4M3_BLOCK : 16;
    const int64_t block_terms_all = (n + block_size_all - 1) / block_size_all;
    g_reduction_prod_all_block_terms.fetch_add(block_terms_all, std::memory_order_relaxed);

    int64_t rid = 0;
    if (ggml_reduction_prod_should_sample(&rid)) {
        *s = (float) ggml_reduction_prod_profile_run_bf16(x, y, rid, n, false, false);
        return;
    }
#endif

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
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by,
    int nrc) {

    const ggml_bf16_t * GGML_RESTRICT x = (const ggml_bf16_t *) vx;
    const ggml_bf16_t * GGML_RESTRICT y = (const ggml_bf16_t *) vy;

    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_REDUCTION_PROD_PROFILE
    const int block_size_all = GGML_SIM_FP8E4M3_BLOCK > 0 ? GGML_SIM_FP8E4M3_BLOCK : 16;
    const int64_t block_terms_all = (n + block_size_all - 1) / block_size_all;
    g_reduction_prod_all_block_terms.fetch_add(block_terms_all, std::memory_order_relaxed);

    int64_t rid = 0;
    if (ggml_reduction_prod_should_sample(&rid)) {
        *s = (float) ggml_reduction_prod_profile_run_bf16(x, y, rid, n, true, GGML_MULMAT_TRUNC4_SRC1);
        return;
    }
#endif

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


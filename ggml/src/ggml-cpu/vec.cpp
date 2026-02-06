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
    // NOTE(按工程约定): 你的工程里 FP8(E4M3) 的最大有限值为 480（无 NaN/Inf）。
    // 这里将“溢出饱和阈值”设为 480。
    // （常见标准实现里 e4m3 的 max finite 可能是 240，但这里以工程定义为准。）
    return 480.0f;
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
        if (ef > 15) {
            return sign * max_f;
        }
        // mantissa bits: 3, represent fractional part of m in [1,2): frac=(m-1) in [0,1)
        float frac = m - 1.0f;
        int q = ggml_rne_int(frac * 8.0f); // 0..8
        if (q == 8) {
            // carry into exponent
            q = 0;
            ef += 1;
            if (ef > 15) {
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
    // need ax / 2^k <= max_finite (工程约定为 480)
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
        int8_t      * scales_out,
        int           src_id,
        const char  * layer_name);

extern "C" void ggml_sim_fp8e4m3_block_quant_dequant_f32_to_bf16(
        const float     * in,
        ggml_bf16_t     * out,
        int               n,
        int               block,
        int8_t          * scales_out,
        int               src_id,
        const char      * layer_name);

extern "C" void ggml_fp8_sim_stats_reset(void);
extern "C" void ggml_fp8_sim_stats_report(const char * report_file);

// ---------------------------------------------------------------------------
// Stats infrastructure
// ---------------------------------------------------------------------------
#include <mutex>
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
};

static FP8SimSrcStats g_fp8_stats[2]; // [0]=src0/weights, [1]=src1/activations
static std::unordered_map<std::string, FP8SimSrcStats> g_fp8_layer_stats[2]; // per-layer stats for src0, src1
static std::mutex     g_fp8_stats_mtx;
static bool           g_fp8_atexit_registered = false;

static void fp8_stats_atexit_handler(void) {
    ggml_fp8_sim_stats_report("fp8_sim_analysis.log");
}

// Accumulate per-element stats into thread-local buffer, then merge under lock.
// This is called from the quant/dequant functions below.
static void fp8_accumulate_block_stats(
        const float * original,  // original FP32 values
        const float * dequant,   // after quant-dequant roundtrip (FP32)
        int           len,
        int8_t        k,         // scale exponent
        int           src_id,
        const char  * layer_name)
{
    // Classify each element and compute errors
    const float max_f   = ggml_fp8e4m3_max_finite();
    const float min_sub = ggml_fp8e4m3_min_subnormal();
    const float min_norm = ldexpf(1.0f, -6);

    FP8SimSrcStats local = {};
    local.total_elements = len;
    local.total_blocks   = 1;
    local.scale_hist[(int)k + 128] = 1;

    const float inv = ldexpf(1.0f, -k);

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

        // Classify
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

    // Merge into global stats under lock
    {
        std::lock_guard<std::mutex> lock(g_fp8_stats_mtx);
        FP8SimSrcStats & g = g_fp8_stats[src_id & 1];
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

        // Per-layer accumulation
        if (layer_name && layer_name[0] != '\0') {
            // Extract layer key: use tensor name directly (e.g. "blk.5.attn_q.weight")
            std::string lkey(layer_name);
            FP8SimSrcStats & ls = g_fp8_layer_stats[src_id & 1][lkey];
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
        }

        // Register atexit handler on first call
        if (!g_fp8_atexit_registered) {
            g_fp8_atexit_registered = true;
            atexit(fp8_stats_atexit_handler);
        }
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
        int8_t      * scales_out,
        int           src_id,
        const char  * layer_name) {
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
            out[i + j] = q * mul;
        }
        // Collect stats for this block
        fp8_accumulate_block_stats(in + i, out + i, len, k, src_id, layer_name);
    }
}

extern "C" void ggml_sim_fp8e4m3_block_quant_dequant_f32_to_bf16(
        const float     * in,
        ggml_bf16_t     * out,
        int               n,
        int               block,
        int8_t          * scales_out,
        int               src_id,
        const char      * layer_name) {
    if (block <= 0) {
        block = 16;
    }
    // We need a temporary FP32 buffer to compute stats (since output is BF16)
    float tmp_dq[4096];  // stack buffer for moderate sizes
    float * dq_buf = (n <= 4096) ? tmp_dq : (float *)malloc((size_t)n * sizeof(float));

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
            const float val = q * mul;
            dq_buf[i + j] = val;
            out[i + j] = GGML_FP32_TO_BF16(val);
        }
        // Collect stats for this block
        fp8_accumulate_block_stats(in + i, dq_buf + i, len, k, src_id, layer_name);
    }

    if (dq_buf != tmp_dq) {
        free(dq_buf);
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
}

// Helper: write one src stats section
static void fp8_write_src_section(FILE * f, const char * name, const FP8SimSrcStats & s) {
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
    fprintf(f, "      Normal FP8 range      : %12" PRId64 " (%.4f%%)\n", s.normal_count,    pct(s.normal_count,    classified));
    fprintf(f, "      Subnormal (denorm)    : %12" PRId64 " (%.4f%%)\n", s.subnormal_count, pct(s.subnormal_count, classified));
    fprintf(f, "      Underflow (->0)       : %12" PRId64 " (%.4f%%)\n", s.underflow_count, pct(s.underflow_count, classified));
    fprintf(f, "      Overflow (saturated)  : %12" PRId64 " (%.4f%%)\n", s.overflow_count,  pct(s.overflow_count,  classified));
    fprintf(f, "      Zero input            : %12" PRId64 " (%.4f%%)\n", s.zero_count,      pct(s.zero_count,      classified));
    fprintf(f, "\n");

    // Scale histogram – show nonzero bins grouped
    fprintf(f, "    Scale exponent (k=log2) distribution:\n");
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
    if (g_fp8_stats[0].total_elements == 0 && g_fp8_stats[1].total_elements == 0) {
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
        fprintf(f, "  FP8(E4M3) + Block Scale Quantization Simulation Analysis Report\n");
        fprintf(f, "================================================================================\n");

        // Configuration
        fprintf(f, "\n  Configuration\n");
        fprintf(f, "  %.*s\n", 60, "────────────────────────────────────────────────────────────");
        fprintf(f, "    FP8 format           : E4M3 (1 sign, 4 exp, 3 mantissa, bias=7)\n");
        fprintf(f, "    Max finite value     : 480.0\n");
        fprintf(f, "    Min subnormal value  : 2^-9 = %.6f\n", ldexpf(1.0f, -9));
        fprintf(f, "    Block size           : %d\n", GGML_SIM_FP8E4M3_BLOCK);
        fprintf(f, "    Rounding             : RNE (round to nearest even)\n");
        fprintf(f, "    Applied to src0 (weights)     : %s\n", GGML_SIM_FP8E4M3_APPLY_SRC0 ? "YES" : "NO");
        fprintf(f, "    Applied to src1 (activations) : %s\n", GGML_SIM_FP8E4M3_APPLY_SRC1 ? "YES" : "NO");

        // Per-src sections
        fp8_write_src_section(f, "Weights (src0)", g_fp8_stats[0]);
        fp8_write_src_section(f, "Activations (src1)", g_fp8_stats[1]);

        // =================================================================
        // Per-Layer Breakdown (sorted by SQNR ascending = worst first)
        // =================================================================
        for (int sid = 0; sid < 2; ++sid) {
            const auto & lmap = g_fp8_layer_stats[sid];
            if (lmap.empty()) continue;

            const char * src_label = (sid == 0) ? "Weights (src0)" : "Activations (src1)";
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

        // Combined analysis
        fprintf(f, "\n  Combined Impact Analysis\n");
        fprintf(f, "  %.*s\n", 60, "────────────────────────────────────────────────────────────");

        double sqnr_src0 = 999.0, sqnr_src1 = 999.0;
        if (g_fp8_stats[0].total_elements > 0 && g_fp8_stats[0].sum_sq_error > 0) {
            double sp0 = g_fp8_stats[0].sum_sq_input / (double)g_fp8_stats[0].total_elements;
            double np0 = g_fp8_stats[0].sum_sq_error / (double)g_fp8_stats[0].total_elements;
            sqnr_src0 = 10.0 * log10(sp0 / np0);
        }
        if (g_fp8_stats[1].total_elements > 0 && g_fp8_stats[1].sum_sq_error > 0) {
            double sp1 = g_fp8_stats[1].sum_sq_input / (double)g_fp8_stats[1].total_elements;
            double np1 = g_fp8_stats[1].sum_sq_error / (double)g_fp8_stats[1].total_elements;
            sqnr_src1 = 10.0 * log10(sp1 / np1);
        }

        double sqnr_min = (sqnr_src0 < sqnr_src1) ? sqnr_src0 : sqnr_src1;

        fprintf(f, "    SQNR src0 (weights)     : %.2f dB\n", sqnr_src0);
        fprintf(f, "    SQNR src1 (activations) : %.2f dB\n", sqnr_src1);
        fprintf(f, "    Bottleneck SQNR         : %.2f dB (min of above)\n", sqnr_min);
        fprintf(f, "\n");

        // Error budget
        int64_t tot_of = g_fp8_stats[0].overflow_count + g_fp8_stats[1].overflow_count;
        int64_t tot_uf = g_fp8_stats[0].underflow_count + g_fp8_stats[1].underflow_count;
        int64_t tot_el = g_fp8_stats[0].total_elements + g_fp8_stats[1].total_elements;
        fprintf(f, "    Total overflow (saturated)  : %" PRId64 " / %" PRId64 " (%.6f%%)\n",
                tot_of, tot_el, tot_el > 0 ? 100.0 * (double)tot_of / (double)tot_el : 0.0);
        fprintf(f, "    Total underflow (flushed)   : %" PRId64 " / %" PRId64 " (%.6f%%)\n",
                tot_uf, tot_el, tot_el > 0 ? 100.0 * (double)tot_uf / (double)tot_el : 0.0);
        fprintf(f, "\n");

        // PPL impact estimation
        fprintf(f, "  PPL Impact Estimation\n");
        fprintf(f, "  %.*s\n", 60, "────────────────────────────────────────────────────────────");
        fprintf(f, "    The SQNR (Signal-to-Quantization-Noise Ratio) directly\n");
        fprintf(f, "    predicts the magnitude of PPL degradation:\n\n");
        fprintf(f, "    SQNR Range       Expected PPL Impact     Severity\n");
        fprintf(f, "    ──────────       ───────────────────     ────────\n");
        fprintf(f, "     > 50 dB         < 0.5%% increase         Negligible\n");
        fprintf(f, "     40-50 dB        0.5-2%% increase         Minor\n");
        fprintf(f, "     30-40 dB        2-10%% increase          Moderate\n");
        fprintf(f, "     20-30 dB        10-50%% increase         Significant\n");
        fprintf(f, "     < 20 dB         > 50%% increase          Severe\n");
        fprintf(f, "\n");

        // Mark which range we're in
        const char * severity;
        const char * ppl_range;
        if (sqnr_min > 50.0)      { severity = "Negligible";  ppl_range = "< 0.5%"; }
        else if (sqnr_min > 40.0) { severity = "Minor";       ppl_range = "0.5-2%"; }
        else if (sqnr_min > 30.0) { severity = "Moderate";    ppl_range = "2-10%"; }
        else if (sqnr_min > 20.0) { severity = "Significant"; ppl_range = "10-50%"; }
        else                      { severity = "Severe";       ppl_range = "> 50%"; }

        fprintf(f, "    >>> Your bottleneck SQNR = %.2f dB\n", sqnr_min);
        fprintf(f, "    >>> Expected PPL increase : %s\n", ppl_range);
        fprintf(f, "    >>> Severity              : %s\n", severity);
        fprintf(f, "\n");

        // Layer propagation model
        fprintf(f, "  Error Propagation Model\n");
        fprintf(f, "  %.*s\n", 60, "────────────────────────────────────────────────────────────");
        fprintf(f, "    For a Transformer with L layers, quantization noise compounds\n");
        fprintf(f, "    roughly as sqrt(2*L) (independent noise per matmul, two per layer).\n\n");
        fprintf(f, "    Effective output SQNR ~= bottleneck_SQNR - 10*log10(2*L)\n\n");

        int layers[] = {12, 24, 32, 40, 48, 64, 80};
        fprintf(f, "    Layers (L)   10*log10(2L)   Effective SQNR   Expected PPL impact\n");
        fprintf(f, "    ──────────   ────────────   ──────────────   ───────────────────\n");
        for (int li = 0; li < 7; ++li) {
            int L = layers[li];
            double penalty = 10.0 * log10(2.0 * L);
            double eff = sqnr_min - penalty;
            const char * eimp;
            if (eff > 50.0)      eimp = "< 0.5%";
            else if (eff > 40.0) eimp = "0.5-2%";
            else if (eff > 30.0) eimp = "2-10%";
            else if (eff > 20.0) eimp = "10-50%";
            else                 eimp = "> 50%";
            fprintf(f, "    %5d        %8.2f       %10.2f dB     %s\n", L, penalty, eff, eimp);
        }
        fprintf(f, "\n");

        // Dominant error source
        fprintf(f, "  Dominant Error Source\n");
        fprintf(f, "  %.*s\n", 60, "────────────────────────────────────────────────────────────");
        double total_noise = g_fp8_stats[0].sum_sq_error + g_fp8_stats[1].sum_sq_error;
        if (total_noise > 0) {
            double src0_frac = 100.0 * g_fp8_stats[0].sum_sq_error / total_noise;
            double src1_frac = 100.0 * g_fp8_stats[1].sum_sq_error / total_noise;
            fprintf(f, "    Weight quant noise contribution    : %.1f%%\n", src0_frac);
            fprintf(f, "    Activation quant noise contribution: %.1f%%\n", src1_frac);
            if (src0_frac > src1_frac) {
                fprintf(f, "    --> Weights are the dominant error source.\n");
                fprintf(f, "        To reduce PPL: increase weight precision or block size.\n");
            } else {
                fprintf(f, "    --> Activations are the dominant error source.\n");
                fprintf(f, "        To reduce PPL: increase activation precision or block size.\n");
            }
        }
        fprintf(f, "\n");

        // Conclusion
        fprintf(f, "  Conclusion\n");
        fprintf(f, "  %.*s\n", 60, "────────────────────────────────────────────────────────────");
        fprintf(f, "    The FP8(E4M3) block quantization (block=%d) introduces:\n", GGML_SIM_FP8E4M3_BLOCK);
        fprintf(f, "      - %.2f dB SQNR for weights\n", sqnr_src0);
        fprintf(f, "      - %.2f dB SQNR for activations\n", sqnr_src1);
        fprintf(f, "      - Overflow is %s: %" PRId64 " elements\n",
                tot_of == 0 ? "absent" : (100.0 * tot_of / (double)tot_el < 0.01 ? "negligible" : "present"), tot_of);
        fprintf(f, "      - Underflow is %s: %" PRId64 " elements\n",
                tot_uf == 0 ? "absent" : (100.0 * tot_uf / (double)tot_el < 0.01 ? "negligible" : "present"), tot_uf);
        fprintf(f, "\n");
        fprintf(f, "    If measured PPL increased from A to B, check:\n");
        fprintf(f, "      Delta(PPL)/A should be consistent with the '%s' severity\n", severity);
        fprintf(f, "      predicted by SQNR=%.2f dB (expected %s increase).\n", sqnr_min, ppl_range);
        fprintf(f, "\n");
        fprintf(f, "    If the observed increase is LARGER than expected:\n");
        fprintf(f, "      - Check the Per-Layer Breakdown above for layers with SQNR < 25 dB\n");
        fprintf(f, "      - Check overflow/underflow counts above\n");
        fprintf(f, "      - Sensitive layers (LM head, embedding, final norm) may dominate PPL\n");
        fprintf(f, "        even if their global noise share is small\n");
        fprintf(f, "      - Consider disabling FP8 for src0 or src1 independently\n");
        fprintf(f, "        (-DGGML_SIM_FP8E4M3_APPLY_SRC0=0 or _SRC1=0)\n");
        fprintf(f, "\n");
        fprintf(f, "    If the observed increase is SMALLER than expected:\n");
        fprintf(f, "      - The model may be robust to this level of quantization noise\n");
        fprintf(f, "      - FP8 quantization is viable for this model\n");
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


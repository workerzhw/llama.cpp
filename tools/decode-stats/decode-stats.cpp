#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kDefaultDecodeRounds = 8;
constexpr size_t kHistogramBins = 128;
constexpr double kSoftmaxLogMin = -12.0;
constexpr double kSoftmaxLogMax = 0.0;
constexpr double kSoftmaxEpsilon = 1e-12;
constexpr float kSwiGLUGateTruncationThreshold = -4.0f;

enum class tensor_kind : size_t {
    softmax = 0,
    swiglu = 1,
    swiglu_gate = 2,
    swiglu_silu = 3,
    ffn_out_pre_res = 4,
};

constexpr size_t kTensorKindCount = 5;

constexpr std::array<const char *, kTensorKindCount> kTensorPrefixes = {
    "kq_soft_max-",
    "ffn_swiglu-",
    "ffn_silu_input-",
    "ffn_silu-",
    "ffn_out_pre_res-",
};

constexpr std::array<const char *, kTensorKindCount> kTensorSlugs = {
    "softmax",
    "swiglu",
    "swiglu_gate",
    "swiglu_silu",
    "ffn_out_pre_res",
};

constexpr std::array<const char *, kTensorKindCount> kTensorTitles = {
    "Attention Softmax",
    "SwiGLU Output",
    "SwiGLU Gate (SiLU Input)",
    "SwiGLU SiLU Output",
    "FFN Output (Pre-Residual)",
};

constexpr std::array<const char *, kTensorKindCount> kTensorColors = {
    "#0f766e",
    "#2563eb",
    "#dc2626",
    "#be185d",
    "#b45309",
};

constexpr std::array<int, 5> kSoftmaxTopKCandidates = {1, 4, 8, 16, 32};
constexpr int kSoftmaxLayerSpreadK = 8;

struct tensor_capture_meta {
    tensor_kind kind;
    std::string name;
    int layer = -1;
    ggml_type type = GGML_TYPE_F32;
    std::array<int64_t, GGML_MAX_DIMS> ne{};
    size_t elements = 0;
};

struct round_capture {
    llama_token token = LLAMA_TOKEN_NULL;
    std::string token_piece;
    std::array<std::vector<float>, kTensorKindCount> values;
    std::vector<tensor_capture_meta> metas;
};

struct callback_data {
    int current_round = -1;
    int active_token_count = 0;
    bool capture_enabled = false;
    std::vector<uint8_t> scratch;
    std::vector<round_capture> prefill_rounds;
    std::vector<round_capture> decode_rounds;
    std::vector<round_capture> * active_rounds = nullptr;
};

struct histogram_summary {
    size_t n_values = 0;
    size_t n_tensors = 0;
    float min = 0.0f;
    float max = 0.0f;
    double mean = 0.0;
    double abs_mean = 0.0;
    double rms = 0.0;
    double max_abs = 0.0;
    double stddev = 0.0;
    float p01 = 0.0f;
    float p05 = 0.0f;
    float p50 = 0.0f;
    float p95 = 0.0f;
    float p99 = 0.0f;
    double positive_ratio = 0.0;
    double negative_ratio = 0.0;
    double zero_ratio = 0.0;
    double plot_min = 0.0;
    double plot_max = 0.0;
    bool plot_log = false;
    size_t clipped_low = 0;
    size_t clipped_high = 0;
    std::vector<uint64_t> bins;
};

struct layer_summary {
    int layer = -1;
    std::string tensor_name;
    histogram_summary stats;
    double primary_ratio = 0.0;
};

struct threshold_bar {
    std::string label;
    double fraction = 0.0;
};

struct topk_point {
    int k = 0;
    double element_share = 0.0;
    double mean_mass = 0.0;
    float p50_mass = 0.0f;
    float p95_mass = 0.0f;
    double mean_cutoff = 0.0;
    float p50_cutoff = 0.0f;
    float p95_cutoff = 0.0f;
};

struct topk_summary {
    int row_width = 0;
    size_t n_rows = 0;
    std::vector<topk_point> points;
};

struct layer_topk_summary {
    int layer = -1;
    std::string tensor_name;
    topk_summary stats;
};

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m models/Qwen/Qwen3-1.7B-Base-f16.gguf -p \"Hello\" -n 8 -o decode-stats\n", argv[0]);
    printf("\nThis tool captures prompt prefill token slices and autoregressive decode rounds,\n");
    printf("forces Flash Attention off, records kq_soft_max / ffn_swiglu / swiglu_gate / swiglu_silu / ffn_out_pre_res\n");
    printf("via cb_eval, and writes SVG dashboards plus prefill/decode overview plots.\n\n");
}

static bool preprocess_args(int argc, char ** argv, common_params & params, std::vector<char *> & filtered_argv) {
    filtered_argv.clear();
    filtered_argv.reserve(static_cast<size_t>(argc));
    filtered_argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];

        if (arg == "-o" || arg == "--output" || arg == "--output-file") {
            if (i + 1 >= argc) {
                LOG_ERR("%s: missing value for %s\n", __func__, argv[i]);
                return false;
            }
            params.out_file = argv[++i];
            continue;
        }

        constexpr std::string_view kShortOutputEq = "-o=";
        constexpr std::string_view kOutputEq = "--output=";
        constexpr std::string_view kOutputFileEq = "--output-file=";

        if (arg.size() > kShortOutputEq.size() && arg.substr(0, kShortOutputEq.size()) == kShortOutputEq) {
            params.out_file = std::string(arg.substr(kShortOutputEq.size()));
            continue;
        }

        if (arg.size() > kOutputEq.size() && arg.substr(0, kOutputEq.size()) == kOutputEq) {
            params.out_file = std::string(arg.substr(kOutputEq.size()));
            continue;
        }

        if (arg.size() > kOutputFileEq.size() && arg.substr(0, kOutputFileEq.size()) == kOutputFileEq) {
            params.out_file = std::string(arg.substr(kOutputFileEq.size()));
            continue;
        }

        filtered_argv.push_back(argv[i]);
    }

    return true;
}

static constexpr size_t kind_index(tensor_kind kind) {
    return static_cast<size_t>(kind);
}

static std::string_view kind_slug(tensor_kind kind) {
    return kTensorSlugs[kind_index(kind)];
}

static std::string_view kind_title(tensor_kind kind) {
    return kTensorTitles[kind_index(kind)];
}

static std::string_view kind_color(tensor_kind kind) {
    return kTensorColors[kind_index(kind)];
}

static bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

static bool uses_one_sided_truncation(tensor_kind kind) {
    return kind == tensor_kind::swiglu_gate;
}

static bool has_source_prefix(const ggml_tensor * tensor, std::string_view prefix) {
    for (size_t index = 0; index < GGML_MAX_SRC; ++index) {
        const ggml_tensor * src = tensor->src[index];
        if (src != nullptr && starts_with(src->name, prefix)) {
            return true;
        }
    }

    return false;
}

static bool is_ffn_out_pre_res_tensor(const ggml_tensor * tensor) {
    const std::string_view name(tensor->name != nullptr ? tensor->name : "");

    if (starts_with(name, "ffn_out_pre_res-")) {
        return true;
    }

    if (!starts_with(name, "ffn_out-")) {
        return false;
    }

    return tensor->op != GGML_OP_ADD || !has_source_prefix(tensor, "ffn_inp-");
}

static std::optional<tensor_kind> classify_tensor(const ggml_tensor * tensor) {
    const std::string_view name(tensor->name != nullptr ? tensor->name : "");

    if (starts_with(name, kTensorPrefixes[kind_index(tensor_kind::softmax)])) {
        return tensor_kind::softmax;
    }

    if (starts_with(name, kTensorPrefixes[kind_index(tensor_kind::swiglu)])) {
        return tensor_kind::swiglu;
    }

    if (starts_with(name, kTensorPrefixes[kind_index(tensor_kind::swiglu_gate)])) {
        return tensor_kind::swiglu_gate;
    }

    if (starts_with(name, kTensorPrefixes[kind_index(tensor_kind::swiglu_silu)])) {
        return tensor_kind::swiglu_silu;
    }

    if (is_ffn_out_pre_res_tensor(tensor)) {
        return tensor_kind::ffn_out_pre_res;
    }

    return std::nullopt;
}

static int parse_layer_id(const std::string & name) {
    const size_t pos = name.rfind('-');
    if (pos == std::string::npos || pos + 1 >= name.size()) {
        return -1;
    }

    try {
        return std::stoi(name.substr(pos + 1));
    } catch (...) {
        return -1;
    }
}

static inline float bf16_to_fp32(ggml_bf16_t value) {
    union {
        float f;
        uint32_t i;
    } result;
    result.i = static_cast<uint32_t>(value.bits) << 16;
    return result.f;
}

static float silu_transform(float value) {
    if (value >= 0.0f) {
        const double exp_neg = std::exp(-static_cast<double>(value));
        return static_cast<float>(static_cast<double>(value) / (1.0 + exp_neg));
    }

    const double exp_pos = std::exp(static_cast<double>(value));
    return static_cast<float>(static_cast<double>(value) * exp_pos / (1.0 + exp_pos));
}

static float read_scalar(const uint8_t * data, ggml_type type, size_t offset) {
    switch (type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float *>(data + offset);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(data + offset));
        case GGML_TYPE_BF16:
            return bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(data + offset));
        case GGML_TYPE_I32:
            return static_cast<float>(*reinterpret_cast<const int32_t *>(data + offset));
        case GGML_TYPE_I16:
            return static_cast<float>(*reinterpret_cast<const int16_t *>(data + offset));
        case GGML_TYPE_I8:
            return static_cast<float>(*reinterpret_cast<const int8_t *>(data + offset));
        case GGML_TYPE_I64:
            return static_cast<float>(*reinterpret_cast<const int64_t *>(data + offset));
        default:
            GGML_ABORT("unsupported tensor type for decode stats");
    }
}

static void append_tensor_values_impl(
    std::vector<float> & dst,
    const ggml_tensor * tensor,
    const uint8_t * data,
    std::optional<int64_t> token_index,
    float (*transform)(float) = nullptr) {
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    const auto push_value = [&](size_t offset) {
        float value = read_scalar(data, tensor->type, offset);
        if (transform != nullptr) {
            value = transform(value);
        }
        dst.push_back(value);
    };

    if (token_index.has_value()) {
        GGML_ASSERT(token_index.value() >= 0 && token_index.value() < tensor->ne[1]);

        const size_t elements = ggml_nelements(tensor) / static_cast<size_t>(tensor->ne[1]);
        dst.reserve(dst.size() + elements);

        for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                    const size_t offset =
                        static_cast<size_t>(i3) * tensor->nb[3] +
                        static_cast<size_t>(i2) * tensor->nb[2] +
                        static_cast<size_t>(token_index.value()) * tensor->nb[1] +
                        static_cast<size_t>(i0) * tensor->nb[0];
                    push_value(offset);
                }
            }
        }

        return;
    }

    dst.reserve(dst.size() + ggml_nelements(tensor));

    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                    const size_t offset =
                        static_cast<size_t>(i3) * tensor->nb[3] +
                        static_cast<size_t>(i2) * tensor->nb[2] +
                        static_cast<size_t>(i1) * tensor->nb[1] +
                        static_cast<size_t>(i0) * tensor->nb[0];
                    push_value(offset);
                }
            }
        }
    }
}

static void append_tensor_values(std::vector<float> & dst, const ggml_tensor * tensor, const uint8_t * data) {
    append_tensor_values_impl(dst, tensor, data, std::nullopt);
}

static void append_tensor_values_token_slice(std::vector<float> & dst, const ggml_tensor * tensor, const uint8_t * data, int64_t token_index) {
    append_tensor_values_impl(dst, tensor, data, token_index);
}

static tensor_capture_meta make_tensor_capture_meta(const ggml_tensor * tensor, tensor_kind kind, bool single_token_slice, std::string_view name_override = {}) {
    tensor_capture_meta meta;
    meta.kind = kind;
    meta.name = name_override.empty() ? std::string(tensor->name != nullptr ? tensor->name : "") : std::string(name_override);
    meta.layer = parse_layer_id(meta.name);
    meta.type = tensor->type;
    meta.elements = ggml_nelements(tensor);
    for (size_t dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        meta.ne[dim] = tensor->ne[dim];
    }

    if (single_token_slice) {
        GGML_ASSERT(meta.ne[1] > 0);
        meta.elements /= static_cast<size_t>(meta.ne[1]);
        meta.ne[1] = 1;
    }

    return meta;
}

static void capture_tensor_into_round(
    round_capture & round,
    const ggml_tensor * tensor,
    tensor_kind kind,
    const uint8_t * data,
    std::optional<int64_t> token_index = std::nullopt,
    std::string_view name_override = {},
    float (*transform)(float) = nullptr) {
    const bool single_token_slice = token_index.has_value();
    round.metas.push_back(make_tensor_capture_meta(tensor, kind, single_token_slice, name_override));

    if (single_token_slice) {
        append_tensor_values_impl(round.values[kind_index(kind)], tensor, data, token_index, transform);
    } else {
        append_tensor_values_impl(round.values[kind_index(kind)], tensor, data, std::nullopt, transform);
    }
}

static std::string swiglu_gate_capture_name(const ggml_tensor * tensor) {
    const int layer = parse_layer_id(tensor->name != nullptr ? tensor->name : "");
    if (layer >= 0) {
        return "ffn_silu_input-" + std::to_string(layer);
    }

    return "ffn_silu_input";
}

static std::string swiglu_silu_capture_name(const ggml_tensor * tensor) {
    const int layer = parse_layer_id(tensor->name != nullptr ? tensor->name : "");
    if (layer >= 0) {
        return "ffn_silu_branch-" + std::to_string(layer);
    }

    return "ffn_silu_branch";
}

static bool capture_decode_tensors(struct ggml_tensor * tensor, bool ask, void * user_data) {
    auto * cb_data = static_cast<callback_data *>(user_data);
    const auto kind = classify_tensor(tensor);

    if (ask) {
        return cb_data->capture_enabled && cb_data->active_rounds != nullptr && cb_data->current_round >= 0 && cb_data->active_token_count > 0 && kind.has_value();
    }

    if (!cb_data->capture_enabled || cb_data->active_rounds == nullptr || cb_data->current_round < 0 || cb_data->active_token_count <= 0 || !kind.has_value()) {
        return true;
    }

    const size_t nbytes = ggml_nbytes(tensor);
    cb_data->scratch.resize(nbytes);
    ggml_backend_tensor_get(tensor, cb_data->scratch.data(), 0, nbytes);

    if (cb_data->active_token_count == 1) {
        auto & round = cb_data->active_rounds->at(static_cast<size_t>(cb_data->current_round));
        capture_tensor_into_round(round, tensor, *kind, cb_data->scratch.data());

        if (*kind == tensor_kind::swiglu && tensor->src[0] != nullptr) {
            ggml_tensor * gate_source = tensor->src[0];
            const std::string gate_name = swiglu_gate_capture_name(tensor);
            const std::string silu_name = swiglu_silu_capture_name(tensor);
            cb_data->scratch.resize(ggml_nbytes(gate_source));
            ggml_backend_tensor_get(gate_source, cb_data->scratch.data(), 0, ggml_nbytes(gate_source));
            capture_tensor_into_round(round, gate_source, tensor_kind::swiglu_gate, cb_data->scratch.data(), std::nullopt, gate_name);
            capture_tensor_into_round(round, gate_source, tensor_kind::swiglu_silu, cb_data->scratch.data(), std::nullopt, silu_name, silu_transform);
        }

        return true;
    }

    GGML_ASSERT(tensor->ne[1] == cb_data->active_token_count && "prefill token split expects token axis in ne[1]");
    for (int token_offset = 0; token_offset < cb_data->active_token_count; ++token_offset) {
        auto & round = cb_data->active_rounds->at(static_cast<size_t>(cb_data->current_round + token_offset));
        capture_tensor_into_round(round, tensor, *kind, cb_data->scratch.data(), token_offset);
    }

    if (*kind == tensor_kind::swiglu && tensor->src[0] != nullptr) {
        ggml_tensor * gate_source = tensor->src[0];
        GGML_ASSERT(gate_source->ne[1] == cb_data->active_token_count && "SwiGLU gate source token axis must match active token count");

        const std::string gate_name = swiglu_gate_capture_name(tensor);
        const std::string silu_name = swiglu_silu_capture_name(tensor);
        cb_data->scratch.resize(ggml_nbytes(gate_source));
        ggml_backend_tensor_get(gate_source, cb_data->scratch.data(), 0, ggml_nbytes(gate_source));
        for (int token_offset = 0; token_offset < cb_data->active_token_count; ++token_offset) {
            auto & round = cb_data->active_rounds->at(static_cast<size_t>(cb_data->current_round + token_offset));
            capture_tensor_into_round(round, gate_source, tensor_kind::swiglu_gate, cb_data->scratch.data(), token_offset, gate_name);
            capture_tensor_into_round(round, gate_source, tensor_kind::swiglu_silu, cb_data->scratch.data(), token_offset, silu_name, silu_transform);
        }
    }

    return true;
}

static float quantile_sorted(const std::vector<float> & sorted, double q) {
    if (sorted.empty()) {
        return 0.0f;
    }

    const double pos = q * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return static_cast<float>(sorted[lo] * (1.0 - frac) + sorted[hi] * frac);
}

static size_t count_kind_tensors(const round_capture & round, tensor_kind kind) {
    return static_cast<size_t>(std::count_if(round.metas.begin(), round.metas.end(), [&](const tensor_capture_meta & meta) {
        return meta.kind == kind;
    }));
}

static histogram_summary summarize_range(const std::vector<float> & values, size_t begin, size_t count, tensor_kind kind) {
    histogram_summary summary;
    summary.n_values = count;
    summary.bins.assign(kHistogramBins, 0);

    if (count == 0) {
        return summary;
    }

    GGML_ASSERT(begin + count <= values.size());

    std::vector<float> sorted(values.begin() + static_cast<std::ptrdiff_t>(begin), values.begin() + static_cast<std::ptrdiff_t>(begin + count));
    std::sort(sorted.begin(), sorted.end());

    summary.min = sorted.front();
    summary.max = sorted.back();

    double sum = 0.0;
    double abs_sum = 0.0;
    double sum_sq = 0.0;
    size_t positive_count = 0;
    size_t negative_count = 0;
    size_t zero_count = 0;
    for (size_t index = begin; index < begin + count; ++index) {
        const float value = values[index];
        sum += value;
        abs_sum += std::abs(static_cast<double>(value));
        sum_sq += static_cast<double>(value) * static_cast<double>(value);
        summary.max_abs = std::max(summary.max_abs, std::abs(static_cast<double>(value)));
        if (value > 0.0f) {
            positive_count++;
        } else if (value < 0.0f) {
            negative_count++;
        } else {
            zero_count++;
        }
    }

    summary.mean = sum / static_cast<double>(count);
    summary.abs_mean = abs_sum / static_cast<double>(count);
    summary.rms = std::sqrt(sum_sq / static_cast<double>(count));
    summary.positive_ratio = static_cast<double>(positive_count) / static_cast<double>(count);
    summary.negative_ratio = static_cast<double>(negative_count) / static_cast<double>(count);
    summary.zero_ratio = static_cast<double>(zero_count) / static_cast<double>(count);

    const double variance = std::max(0.0, sum_sq / static_cast<double>(count) - summary.mean * summary.mean);
    summary.stddev = std::sqrt(variance);
    summary.p01 = quantile_sorted(sorted, 0.01);
    summary.p05 = quantile_sorted(sorted, 0.05);
    summary.p50 = quantile_sorted(sorted, 0.50);
    summary.p95 = quantile_sorted(sorted, 0.95);
    summary.p99 = quantile_sorted(sorted, 0.99);

    if (kind == tensor_kind::softmax) {
        summary.plot_log = true;
        summary.plot_min = kSoftmaxLogMin;
        summary.plot_max = kSoftmaxLogMax;
    } else {
        summary.plot_min = summary.p01;
        summary.plot_max = summary.p99;
        if (kind == tensor_kind::swiglu_gate) {
            summary.plot_min = std::min(summary.plot_min, static_cast<double>(kSwiGLUGateTruncationThreshold));
        }
        if (!(summary.plot_max > summary.plot_min)) {
            summary.plot_min = summary.min;
            summary.plot_max = summary.max;
        }
        if (!(summary.plot_max > summary.plot_min)) {
            const double pad = std::max(1e-5, std::abs(summary.plot_min) * 0.1 + 1e-3);
            summary.plot_min -= pad;
            summary.plot_max += pad;
        }
    }

    const double width = summary.plot_max - summary.plot_min;
    for (size_t index = begin; index < begin + count; ++index) {
        const float value = values[index];
        const double plotted = kind == tensor_kind::softmax
            ? std::log10(std::max(static_cast<double>(value), kSoftmaxEpsilon))
            : static_cast<double>(value);

        size_t bin_index = 0;
        if (plotted < summary.plot_min) {
            summary.clipped_low++;
            bin_index = 0;
        } else if (plotted > summary.plot_max) {
            summary.clipped_high++;
            bin_index = summary.bins.size() - 1;
        } else if (width <= 0.0) {
            bin_index = summary.bins.size() / 2;
        } else {
            const double normalized = (plotted - summary.plot_min) / width;
            bin_index = std::min(summary.bins.size() - 1, static_cast<size_t>(normalized * summary.bins.size()));
        }
        summary.bins[bin_index]++;
    }

    return summary;
}

static histogram_summary summarize_values(const round_capture & round, tensor_kind kind) {
    const auto & values = round.values[kind_index(kind)];

    histogram_summary summary = summarize_range(values, 0, values.size(), kind);
    summary.n_tensors = count_kind_tensors(round, kind);
    return summary;
}

static double compute_fraction(const std::vector<float> & values, size_t begin, size_t count, const std::function<bool(float)> & predicate) {
    if (count == 0) {
        return 0.0;
    }

    size_t matched = 0;
    for (size_t index = begin; index < begin + count; ++index) {
        if (predicate(values[index])) {
            matched++;
        }
    }

    return static_cast<double>(matched) / static_cast<double>(count);
}

static std::vector<threshold_bar> summarize_threshold_bars(const std::vector<float> & values, size_t begin, size_t count, tensor_kind kind) {
    std::vector<threshold_bar> bars;
    bars.reserve(5);

    if (kind == tensor_kind::softmax) {
        bars.push_back({"p <= 1e-9", compute_fraction(values, begin, count, [](float value) { return value <= 1e-9f; })});
        bars.push_back({"p <= 1e-6", compute_fraction(values, begin, count, [](float value) { return value <= 1e-6f; })});
        bars.push_back({"p <= 1e-3", compute_fraction(values, begin, count, [](float value) { return value <= 1e-3f; })});
        bars.push_back({"p >= 1e-2", compute_fraction(values, begin, count, [](float value) { return value >= 1e-2f; })});
        bars.push_back({"p >= 1e-1", compute_fraction(values, begin, count, [](float value) { return value >= 1e-1f; })});
        return bars;
    }

    if (kind == tensor_kind::swiglu_gate) {
        bars.push_back({"x <= -8", compute_fraction(values, begin, count, [](float value) { return value <= -8.0f; })});
        bars.push_back({"x <= -6", compute_fraction(values, begin, count, [](float value) { return value <= -6.0f; })});
        bars.push_back({"x <= -4 (truncate)", compute_fraction(values, begin, count, [](float value) { return value <= kSwiGLUGateTruncationThreshold; })});
        bars.push_back({"x <= -2", compute_fraction(values, begin, count, [](float value) { return value <= -2.0f; })});
        bars.push_back({"x <= -1", compute_fraction(values, begin, count, [](float value) { return value <= -1.0f; })});
        return bars;
    }

    bars.push_back({"|x| <= 1e-4", compute_fraction(values, begin, count, [](float value) { return std::abs(value) <= 1e-4f; })});
    bars.push_back({"|x| <= 1e-3", compute_fraction(values, begin, count, [](float value) { return std::abs(value) <= 1e-3f; })});
    bars.push_back({"|x| <= 1e-2", compute_fraction(values, begin, count, [](float value) { return std::abs(value) <= 1e-2f; })});
    bars.push_back({"|x| <= 1e-1", compute_fraction(values, begin, count, [](float value) { return std::abs(value) <= 1e-1f; })});
    bars.push_back({"|x| >= 1", compute_fraction(values, begin, count, [](float value) { return std::abs(value) >= 1.0f; })});
    return bars;
}

static std::vector<threshold_bar> summarize_threshold_bars(const round_capture & round, tensor_kind kind) {
    const auto & values = round.values[kind_index(kind)];
    return summarize_threshold_bars(values, 0, values.size(), kind);
}

static double primary_focus_ratio(const std::vector<float> & values, size_t begin, size_t count, tensor_kind kind) {
    if (kind == tensor_kind::softmax) {
        return compute_fraction(values, begin, count, [](float value) { return value <= 1e-6f; });
    }

    if (kind == tensor_kind::swiglu_gate) {
        return compute_fraction(values, begin, count, [](float value) { return value <= kSwiGLUGateTruncationThreshold; });
    }

    return compute_fraction(values, begin, count, [](float value) { return std::abs(value) <= 1e-2f; });
}

static std::string primary_focus_label(tensor_kind kind) {
    if (kind == tensor_kind::softmax) {
        return "near-zero share (p <= 1e-6)";
    }

    if (kind == tensor_kind::swiglu_gate) {
        return "truncation share (x <= -4)";
    }

    return "near-zero share (|x| <= 1e-2)";
}

static std::vector<layer_summary> summarize_layers(const round_capture & round, tensor_kind kind) {
    const auto & values = round.values[kind_index(kind)];
    std::vector<layer_summary> layers;
    layers.reserve(count_kind_tensors(round, kind));

    size_t offset = 0;
    for (const auto & meta : round.metas) {
        if (meta.kind != kind) {
            continue;
        }

        layer_summary layer;
        layer.layer = meta.layer;
        layer.tensor_name = meta.name;
        layer.stats = summarize_range(values, offset, meta.elements, kind);
        layer.stats.n_tensors = 1;
        layer.primary_ratio = primary_focus_ratio(values, offset, meta.elements, kind);
        layers.push_back(std::move(layer));

        offset += meta.elements;
    }

    std::sort(layers.begin(), layers.end(), [](const layer_summary & lhs, const layer_summary & rhs) {
        return lhs.layer < rhs.layer;
    });
    return layers;
}

static std::vector<int> valid_softmax_topks(int row_width) {
    std::vector<int> ks;
    for (const int candidate : kSoftmaxTopKCandidates) {
        if (candidate <= row_width) {
            ks.push_back(candidate);
        }
    }
    return ks;
}

static topk_summary summarize_topk_rows(const std::vector<float> & values, size_t begin, size_t count, int row_width) {
    topk_summary summary;
    summary.row_width = row_width;

    if (row_width <= 0 || count == 0 || count % static_cast<size_t>(row_width) != 0) {
        return summary;
    }

    const std::vector<int> ks = valid_softmax_topks(row_width);
    if (ks.empty()) {
        return summary;
    }

    summary.n_rows = count / static_cast<size_t>(row_width);
    summary.points.resize(ks.size());

    std::vector<double> element_share_sums(ks.size(), 0.0);
    std::vector<double> mean_mass_sums(ks.size(), 0.0);
    std::vector<double> mean_cutoff_sums(ks.size(), 0.0);
    std::vector<std::vector<float>> mass_samples(ks.size());
    std::vector<std::vector<float>> cutoff_samples(ks.size());

    for (size_t index = 0; index < ks.size(); ++index) {
        summary.points[index].k = ks[index];
        mass_samples[index].reserve(summary.n_rows);
        cutoff_samples[index].reserve(summary.n_rows);
    }

    const int max_k = ks.back();
    std::vector<float> row(static_cast<size_t>(row_width));

    for (size_t row_index = 0; row_index < summary.n_rows; ++row_index) {
        const size_t row_begin = begin + row_index * static_cast<size_t>(row_width);
        std::copy(values.begin() + static_cast<std::ptrdiff_t>(row_begin), values.begin() + static_cast<std::ptrdiff_t>(row_begin + row.size()), row.begin());

        if (max_k < row_width) {
            std::partial_sort(row.begin(), row.begin() + max_k, row.end(), std::greater<float>());
        } else {
            std::sort(row.begin(), row.end(), std::greater<float>());
        }

        double cumulative = 0.0;
        size_t k_index = 0;
        for (int position = 0; position < max_k; ++position) {
            cumulative += row[static_cast<size_t>(position)];
            while (k_index < ks.size() && ks[k_index] == position + 1) {
                const float cutoff = row[static_cast<size_t>(position)];
                element_share_sums[k_index] += static_cast<double>(ks[k_index]) / static_cast<double>(row_width);
                mean_mass_sums[k_index] += cumulative;
                mean_cutoff_sums[k_index] += cutoff;
                mass_samples[k_index].push_back(static_cast<float>(cumulative));
                cutoff_samples[k_index].push_back(cutoff);
                k_index++;
            }
        }
    }

    for (size_t index = 0; index < ks.size(); ++index) {
        auto & point = summary.points[index];
        const double row_count = static_cast<double>(summary.n_rows);
        point.element_share = element_share_sums[index] / row_count;
        point.mean_mass = mean_mass_sums[index] / row_count;
        point.mean_cutoff = mean_cutoff_sums[index] / row_count;

        std::sort(mass_samples[index].begin(), mass_samples[index].end());
        std::sort(cutoff_samples[index].begin(), cutoff_samples[index].end());
        point.p50_mass = quantile_sorted(mass_samples[index], 0.50);
        point.p95_mass = quantile_sorted(mass_samples[index], 0.95);
        point.p50_cutoff = quantile_sorted(cutoff_samples[index], 0.50);
        point.p95_cutoff = quantile_sorted(cutoff_samples[index], 0.95);
    }

    return summary;
}

static topk_summary summarize_softmax_topk(const round_capture & round) {
    const auto & values = round.values[kind_index(tensor_kind::softmax)];
    size_t offset = 0;
    int min_row_width = std::numeric_limits<int>::max();
    size_t total_rows = 0;

    for (const auto & meta : round.metas) {
        if (meta.kind != tensor_kind::softmax) {
            continue;
        }
        min_row_width = std::min(min_row_width, static_cast<int>(meta.ne[0]));
        total_rows += meta.elements / static_cast<size_t>(meta.ne[0]);
    }

    if (min_row_width == std::numeric_limits<int>::max() || total_rows == 0) {
        return {};
    }

    std::vector<int> ks = valid_softmax_topks(min_row_width);
    topk_summary summary;
    summary.row_width = min_row_width;
    summary.n_rows = total_rows;
    summary.points.resize(ks.size());

    std::vector<double> element_share_sums(ks.size(), 0.0);
    std::vector<double> mean_mass_sums(ks.size(), 0.0);
    std::vector<double> mean_cutoff_sums(ks.size(), 0.0);
    std::vector<std::vector<float>> mass_samples(ks.size());
    std::vector<std::vector<float>> cutoff_samples(ks.size());
    for (size_t index = 0; index < ks.size(); ++index) {
        summary.points[index].k = ks[index];
        mass_samples[index].reserve(total_rows);
        cutoff_samples[index].reserve(total_rows);
    }

    for (const auto & meta : round.metas) {
        if (meta.kind != tensor_kind::softmax) {
            continue;
        }

        const int row_width = static_cast<int>(meta.ne[0]);
        const topk_summary tensor_summary = summarize_topk_rows(values, offset, meta.elements, row_width);
        const size_t row_count = tensor_summary.n_rows;
        for (size_t index = 0; index < summary.points.size(); ++index) {
            const auto & src = tensor_summary.points[index];
            element_share_sums[index] += src.element_share * static_cast<double>(row_count);
            mean_mass_sums[index] += src.mean_mass * static_cast<double>(row_count);
            mean_cutoff_sums[index] += src.mean_cutoff * static_cast<double>(row_count);
        }

        std::vector<float> row(static_cast<size_t>(row_width));
        const int max_k = ks.back();
        for (size_t row_index = 0; row_index < row_count; ++row_index) {
            const size_t row_begin = offset + row_index * static_cast<size_t>(row_width);
            std::copy(values.begin() + static_cast<std::ptrdiff_t>(row_begin), values.begin() + static_cast<std::ptrdiff_t>(row_begin + row.size()), row.begin());
            if (max_k < row_width) {
                std::partial_sort(row.begin(), row.begin() + max_k, row.end(), std::greater<float>());
            } else {
                std::sort(row.begin(), row.end(), std::greater<float>());
            }

            double cumulative = 0.0;
            size_t k_index = 0;
            for (int position = 0; position < max_k; ++position) {
                cumulative += row[static_cast<size_t>(position)];
                while (k_index < ks.size() && ks[k_index] == position + 1) {
                    mass_samples[k_index].push_back(static_cast<float>(cumulative));
                    cutoff_samples[k_index].push_back(row[static_cast<size_t>(position)]);
                    k_index++;
                }
            }
        }

        offset += meta.elements;
    }

    const double total_row_count = static_cast<double>(summary.n_rows);
    for (size_t index = 0; index < summary.points.size(); ++index) {
        auto & point = summary.points[index];
        point.element_share = element_share_sums[index] / total_row_count;
        point.mean_mass = mean_mass_sums[index] / total_row_count;
        point.mean_cutoff = mean_cutoff_sums[index] / total_row_count;
        std::sort(mass_samples[index].begin(), mass_samples[index].end());
        std::sort(cutoff_samples[index].begin(), cutoff_samples[index].end());
        point.p50_mass = quantile_sorted(mass_samples[index], 0.50);
        point.p95_mass = quantile_sorted(mass_samples[index], 0.95);
        point.p50_cutoff = quantile_sorted(cutoff_samples[index], 0.50);
        point.p95_cutoff = quantile_sorted(cutoff_samples[index], 0.95);
    }

    return summary;
}

static std::vector<layer_topk_summary> summarize_softmax_topk_layers(const round_capture & round) {
    const auto & values = round.values[kind_index(tensor_kind::softmax)];
    std::vector<layer_topk_summary> layers;
    layers.reserve(count_kind_tensors(round, tensor_kind::softmax));

    size_t offset = 0;
    for (const auto & meta : round.metas) {
        if (meta.kind != tensor_kind::softmax) {
            continue;
        }

        layer_topk_summary layer;
        layer.layer = meta.layer;
        layer.tensor_name = meta.name;
        layer.stats = summarize_topk_rows(values, offset, meta.elements, static_cast<int>(meta.ne[0]));
        layers.push_back(std::move(layer));

        offset += meta.elements;
    }

    std::sort(layers.begin(), layers.end(), [](const layer_topk_summary & lhs, const layer_topk_summary & rhs) {
        return lhs.layer < rhs.layer;
    });
    return layers;
}

static const topk_point * find_topk_point(const topk_summary & summary, int k) {
    for (const auto & point : summary.points) {
        if (point.k == k) {
            return &point;
        }
    }
    return nullptr;
}

static std::string tensor_shape_label(const round_capture & round, tensor_kind kind) {
    for (const auto & meta : round.metas) {
        if (meta.kind != kind) {
            continue;
        }

        std::ostringstream oss;
        oss << ggml_type_name(meta.type) << " [" << meta.ne[0];
        for (size_t dim = 1; dim < GGML_MAX_DIMS; ++dim) {
            if (meta.ne[dim] <= 1) {
                continue;
            }
            oss << " x " << meta.ne[dim];
        }
        oss << "]";
        return oss.str();
    }

    return "unknown";
}

static std::string format_double(double value, int precision = 6) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

static std::string format_percent(double value, int precision = 1) {
    return format_double(value * 100.0, precision) + "%";
}

static std::string format_scientific(double value, int precision = 1) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(precision) << value;
    return oss.str();
}

static std::string format_softmax_probability_tick(double log10_p) {
    return format_scientific(std::pow(10.0, log10_p), 1);
}

static std::string csv_escape(const std::string & value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }

    std::string escaped = "\"";
    for (const char ch : value) {
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped += ch;
        }
    }
    escaped += '"';
    return escaped;
}

static void append_hex_escape(std::string & out, unsigned char value) {
    static constexpr char kHex[] = "0123456789ABCDEF";
    out += '\\';
    out += 'x';
    out += kHex[(value >> 4) & 0x0F];
    out += kHex[value & 0x0F];
}

static std::string sanitize_utf8_text(const std::string & value) {
    std::string sanitized;
    sanitized.reserve(value.size());

    for (size_t index = 0; index < value.size();) {
        const unsigned char lead = static_cast<unsigned char>(value[index]);

        if (lead < 0x20) {
            switch (lead) {
                case '\n': sanitized += "\\n"; break;
                case '\r': sanitized += "\\r"; break;
                case '\t': sanitized += "\\t"; break;
                default: append_hex_escape(sanitized, lead); break;
            }
            ++index;
            continue;
        }

        if (lead < 0x80) {
            sanitized += static_cast<char>(lead);
            ++index;
            continue;
        }

        size_t sequence_length = 0;
        if (lead >= 0xC2 && lead <= 0xDF) {
            sequence_length = 2;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            sequence_length = 3;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            sequence_length = 4;
        } else {
            append_hex_escape(sanitized, lead);
            ++index;
            continue;
        }

        if (index + sequence_length > value.size()) {
            append_hex_escape(sanitized, lead);
            ++index;
            continue;
        }

        auto is_continuation = [&](size_t offset) {
            return (static_cast<unsigned char>(value[index + offset]) & 0xC0) == 0x80;
        };

        bool valid = true;
        for (size_t offset = 1; offset < sequence_length; ++offset) {
            if (!is_continuation(offset)) {
                valid = false;
                break;
            }
        }

        if (valid && sequence_length == 3) {
            const unsigned char second = static_cast<unsigned char>(value[index + 1]);
            if ((lead == 0xE0 && second < 0xA0) || (lead == 0xED && second >= 0xA0)) {
                valid = false;
            }
        }

        if (valid && sequence_length == 4) {
            const unsigned char second = static_cast<unsigned char>(value[index + 1]);
            if ((lead == 0xF0 && second < 0x90) || (lead == 0xF4 && second > 0x8F)) {
                valid = false;
            }
        }

        if (!valid) {
            append_hex_escape(sanitized, lead);
            ++index;
            continue;
        }

        sanitized.append(value, index, sequence_length);
        index += sequence_length;
    }

    return sanitized;
}

static std::string html_escape(const std::string & value) {
    const std::string sanitized = sanitize_utf8_text(value);
    std::string escaped;
    escaped.reserve(sanitized.size());
    for (const char ch : sanitized) {
        switch (ch) {
            case '&': escaped += "&amp;"; break;
            case '<': escaped += "&lt;"; break;
            case '>': escaped += "&gt;"; break;
            case '"': escaped += "&quot;"; break;
            default: escaped += ch; break;
        }
    }
    return escaped;
}

static std::string read_text_file(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("failed to open text file: " + path.string());
    }

    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

static std::string percent_encode_uri_component(const std::string & value) {
    static constexpr char kHex[] = "0123456789ABCDEF";

    std::string encoded;
    encoded.reserve(value.size() * 3);

    for (const unsigned char ch : value) {
        const bool is_unreserved =
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= 'a' && ch <= 'z') ||
            (ch >= '0' && ch <= '9') ||
            ch == '-' || ch == '_' || ch == '.' || ch == '~';

        if (is_unreserved) {
            encoded += static_cast<char>(ch);
            continue;
        }

        encoded += '%';
        encoded += kHex[(ch >> 4) & 0x0F];
        encoded += kHex[ch & 0x0F];
    }

    return encoded;
}

static std::string svg_data_uri(const fs::path & path) {
    return "data:image/svg+xml;charset=utf-8," + percent_encode_uri_component(read_text_file(path));
}

static double plotted_value(float value, tensor_kind kind) {
    return kind == tensor_kind::softmax
        ? std::log10(std::max(static_cast<double>(value), kSoftmaxEpsilon))
        : static_cast<double>(value);
}

static double clamp01(double value) {
    return std::max(0.0, std::min(1.0, value));
}

static std::string svg_safe(const std::string & value) {
    return html_escape(value);
}

static double layer_magnitude(const layer_summary & layer, tensor_kind kind) {
    if (kind == tensor_kind::softmax) {
        return layer.stats.p99;
    }

    return std::max(std::abs(static_cast<double>(layer.stats.p01)), std::abs(static_cast<double>(layer.stats.p99)));
}

static std::optional<double> reference_plot_value(const round_capture & round, tensor_kind kind) {
    if (kind == tensor_kind::softmax) {
        for (const auto & meta : round.metas) {
            if (meta.kind == kind && meta.ne[0] > 0) {
                return std::log10(1.0 / static_cast<double>(meta.ne[0]));
            }
        }
        return std::nullopt;
    }

    if (kind == tensor_kind::swiglu_gate) {
        return static_cast<double>(kSwiGLUGateTruncationThreshold);
    }

    return 0.0;
}

static std::string reference_label(const round_capture & round, tensor_kind kind) {
    if (kind == tensor_kind::softmax) {
        for (const auto & meta : round.metas) {
            if (meta.kind == kind && meta.ne[0] > 0) {
                std::ostringstream oss;
                oss << "uniform 1/" << meta.ne[0];
                return oss.str();
            }
        }
        return "uniform";
    }

    if (kind == tensor_kind::swiglu_gate) {
        return "cut -4";
    }

    return "zero";
}

static std::pair<double, double> primary_threshold_bounds(tensor_kind kind) {
    if (kind == tensor_kind::softmax) {
        return {0.0, 1e-6};
    }

    if (kind == tensor_kind::swiglu_gate) {
        return {kSwiGLUGateTruncationThreshold, kSwiGLUGateTruncationThreshold};
    }

    return {-1e-2, 1e-2};
}

static std::string primary_threshold_window_label(tensor_kind kind) {
    if (kind == tensor_kind::softmax) {
        return "truncate p <= 1e-6";
    }

    if (kind == tensor_kind::swiglu_gate) {
        return "truncate x <= -4";
    }

    return "truncate |x| <= 1e-2";
}

static void write_histogram_svg(const fs::path & path, const histogram_summary & summary, const round_capture & round, tensor_kind kind, int round_number) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open SVG output: " + path.string());
    }

    const std::vector<layer_summary> layers = summarize_layers(round, kind);
    const std::vector<threshold_bar> threshold_bars = summarize_threshold_bars(round, kind);
    const topk_summary softmax_topk = kind == tensor_kind::softmax ? summarize_softmax_topk(round) : topk_summary{};
    const std::vector<layer_topk_summary> softmax_topk_layers = kind == tensor_kind::softmax ? summarize_softmax_topk_layers(round) : std::vector<layer_topk_summary>{};

    constexpr double width = 1280.0;
    constexpr double height = 860.0;
    constexpr double card_radius = 18.0;
    constexpr double outer = 28.0;
    const double histogram_x = outer;
    const double histogram_y = 118.0;
    const double histogram_w = 794.0;
    const double histogram_h = 406.0;
    const double stats_x = histogram_x + histogram_w + 22.0;
    const double stats_y = histogram_y;
    const double stats_w = width - stats_x - outer;
    const double stats_h = 184.0;
    const double threshold_x = stats_x;
    const double threshold_y = stats_y + stats_h + 18.0;
    const double threshold_w = stats_w;
    const double threshold_h = histogram_y + histogram_h - threshold_y;
    const double layer_x = outer;
    const double layer_y = histogram_y + histogram_h + 24.0;
    const double layer_w = width - outer * 2.0;
    const double layer_h = 252.0;
    const double plot_x = histogram_x + 48.0;
    const double plot_y = histogram_y + 50.0;
    const double plot_w = histogram_w - 84.0;
    const double plot_h = histogram_h - 102.0;

    const uint64_t max_bin = summary.bins.empty() ? 0 : *std::max_element(summary.bins.begin(), summary.bins.end());
    const double bar_width = plot_w / static_cast<double>(summary.bins.size());
    const double plot_range = summary.plot_max - summary.plot_min;

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    out << "  <defs>\n";
    out << "    <linearGradient id=\"bg\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\">\n";
    out << "      <stop offset=\"0%\" stop-color=\"#fffef7\"/>\n";
    out << "      <stop offset=\"100%\" stop-color=\"#f5f7fb\"/>\n";
    out << "    </linearGradient>\n";
    out << "    <linearGradient id=\"hist\" x1=\"0\" y1=\"0\" x2=\"0\" y2=\"1\">\n";
    out << "      <stop offset=\"0%\" stop-color=\"" << kind_color(kind) << "\" stop-opacity=\"0.95\"/>\n";
    out << "      <stop offset=\"100%\" stop-color=\"" << kind_color(kind) << "\" stop-opacity=\"0.35\"/>\n";
    out << "    </linearGradient>\n";
    out << "    <filter id=\"shadow\" x=\"-10%\" y=\"-10%\" width=\"120%\" height=\"120%\">\n";
    out << "      <feDropShadow dx=\"0\" dy=\"12\" stdDeviation=\"14\" flood-color=\"#0f172a\" flood-opacity=\"0.08\"/>\n";
    out << "    </filter>\n";
    out << "  </defs>\n";

    out << "  <rect width=\"100%\" height=\"100%\" fill=\"url(#bg)\"/>\n";
    out << "  <text x=\"" << outer << "\" y=\"42\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"28\" font-weight=\"700\" fill=\"#0f172a\">"
        << svg_safe(std::string(kind_title(kind))) << "</text>\n";
    out << "  <text x=\"" << outer << "\" y=\"68\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"14\" fill=\"#334155\">round "
        << round_number << "  |  token " << round.token << "  |  piece \"" << svg_safe(round.token_piece) << "\"</text>\n";
    out << "  <text x=\"" << outer << "\" y=\"90\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"13\" fill=\"#475569\">"
        << summary.n_tensors << " tensors  |  " << summary.n_values << " values  |  shape " << svg_safe(tensor_shape_label(round, kind))
        << "  |  plot " << (summary.plot_log ? "log10" : "linear") << "</text>\n";

    auto draw_card = [&](double x, double y, double w, double h) {
        out << "  <rect x=\"" << x << "\" y=\"" << y << "\" rx=\"" << card_radius << "\" ry=\"" << card_radius
            << "\" width=\"" << w << "\" height=\"" << h << "\" fill=\"#ffffff\" stroke=\"#dbe4ee\" stroke-width=\"1\" filter=\"url(#shadow)\"/>\n";
    };

    draw_card(histogram_x, histogram_y, histogram_w, histogram_h);
    draw_card(stats_x, stats_y, stats_w, stats_h);
    draw_card(threshold_x, threshold_y, threshold_w, threshold_h);
    draw_card(layer_x, layer_y, layer_w, layer_h);

    out << "  <text x=\"" << (histogram_x + 24.0) << "\" y=\"" << (histogram_y + 30.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"16\" font-weight=\"700\" fill=\"#111827\">Distribution</text>\n";
    out << "  <text x=\"" << (histogram_x + 24.0) << "\" y=\"" << (histogram_y + 48.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#64748b\">histogram + cumulative curve + percentile markers</text>\n";

    out << "  <rect x=\"" << plot_x << "\" y=\"" << plot_y << "\" width=\"" << plot_w << "\" height=\"" << plot_h << "\" fill=\"#f8fafc\" stroke=\"#e2e8f0\" stroke-width=\"1\" rx=\"12\" ry=\"12\"/>\n";

    for (int tick = 0; tick <= 4; ++tick) {
        const double y = plot_y + plot_h - plot_h * static_cast<double>(tick) / 4.0;
        const uint64_t tick_value = max_bin == 0 ? 0 : static_cast<uint64_t>(std::llround(static_cast<double>(max_bin) * tick / 4.0));
        out << "  <line x1=\"" << plot_x << "\" y1=\"" << y << "\" x2=\"" << (plot_x + plot_w)
            << "\" y2=\"" << y << "\" stroke=\"#e5edf5\" stroke-width=\"1\"/>\n";
        out << "  <text x=\"" << (plot_x - 10.0) << "\" y=\"" << (y + 4.0) << "\" text-anchor=\"end\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#64748b\">"
            << tick_value << "</text>\n";
    }

    const std::array<std::pair<std::string, double>, 3> quantile_lines = {{
        {"p01", plotted_value(summary.p01, kind)},
        {"p50", plotted_value(summary.p50, kind)},
        {"p99", plotted_value(summary.p99, kind)},
    }};

    for (const auto & item : quantile_lines) {
        if (plot_range <= 0.0) {
            continue;
        }
        const double normalized = clamp01((item.second - summary.plot_min) / plot_range);
        const double x = plot_x + normalized * plot_w;
        out << "  <line x1=\"" << x << "\" y1=\"" << plot_y << "\" x2=\"" << x << "\" y2=\"" << (plot_y + plot_h)
            << "\" stroke=\"#0f172a\" stroke-width=\"1.5\" stroke-dasharray=\"6 5\" opacity=\"0.55\"/>\n";
        out << "  <text x=\"" << x << "\" y=\"" << (plot_y - 8.0) << "\" text-anchor=\"middle\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#0f172a\">"
            << item.first << "</text>\n";
    }

    if (const auto ref = reference_plot_value(round, kind); ref.has_value() && plot_range > 0.0) {
        const double normalized = clamp01((ref.value() - summary.plot_min) / plot_range);
        const double x = plot_x + normalized * plot_w;
        out << "  <line x1=\"" << x << "\" y1=\"" << plot_y << "\" x2=\"" << x << "\" y2=\"" << (plot_y + plot_h)
            << "\" stroke=\"#dc2626\" stroke-width=\"1.75\" stroke-dasharray=\"2 6\" opacity=\"0.8\"/>\n";
        out << "  <text x=\"" << x << "\" y=\"" << (plot_y + 14.0) << "\" text-anchor=\"middle\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#dc2626\">"
            << svg_safe(reference_label(round, kind)) << "</text>\n";
    }

    if (max_bin > 0) {
        uint64_t cumulative = 0;
        std::ostringstream cdf_path;
        for (size_t index = 0; index < summary.bins.size(); ++index) {
            const double x = plot_x + (static_cast<double>(index) + 0.5) * bar_width;
            const double normalized = static_cast<double>(summary.bins[index]) / static_cast<double>(max_bin);
            const double bar_height = normalized * plot_h;
            const double y = plot_y + plot_h - bar_height;
            cumulative += summary.bins[index];
            const double cdf_ratio = static_cast<double>(cumulative) / static_cast<double>(summary.n_values);
            const double cdf_y = plot_y + plot_h - cdf_ratio * plot_h;

            out << "  <rect x=\"" << (plot_x + bar_width * static_cast<double>(index)) << "\" y=\"" << y
                << "\" width=\"" << std::max(1.0, bar_width - 0.8) << "\" height=\"" << bar_height
                << "\" fill=\"url(#hist)\"/>\n";

            cdf_path << (index == 0 ? "M " : " L ") << x << ' ' << cdf_y;
        }
        out << "  <path d=\"" << cdf_path.str() << "\" fill=\"none\" stroke=\"#0f172a\" stroke-width=\"2.5\" opacity=\"0.8\"/>\n";
        out << "  <text x=\"" << (plot_x + plot_w - 8.0) << "\" y=\"" << (plot_y + 14.0) << "\" text-anchor=\"end\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#0f172a\">CDF</text>\n";
    }

    out << "  <line x1=\"" << plot_x << "\" y1=\"" << (plot_y + plot_h) << "\" x2=\"" << (plot_x + plot_w)
        << "\" y2=\"" << (plot_y + plot_h) << "\" stroke=\"#334155\" stroke-width=\"1.2\"/>\n";
    out << "  <line x1=\"" << plot_x << "\" y1=\"" << plot_y << "\" x2=\"" << plot_x
        << "\" y2=\"" << (plot_y + plot_h) << "\" stroke=\"#334155\" stroke-width=\"1.2\"/>\n";

    const double axis_label_y = plot_y + plot_h + 22.0;
    out << "  <text x=\"" << plot_x << "\" y=\"" << axis_label_y << "\" text-anchor=\"start\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#64748b\">"
        << format_double(summary.plot_min, 4) << "</text>\n";
    out << "  <text x=\"" << (plot_x + plot_w / 2.0) << "\" y=\"" << axis_label_y << "\" text-anchor=\"middle\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#64748b\">"
        << format_double((summary.plot_min + summary.plot_max) * 0.5, 4) << "</text>\n";
    out << "  <text x=\"" << (plot_x + plot_w) << "\" y=\"" << axis_label_y << "\" text-anchor=\"end\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#64748b\">"
        << format_double(summary.plot_max, 4) << "</text>\n";
    out << "  <text x=\"" << (plot_x + plot_w / 2.0) << "\" y=\"" << (axis_label_y + 18.0) << "\" text-anchor=\"middle\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#111827\">"
        << (summary.plot_log ? "log10(value)" : "value") << "</text>\n";

    out << "  <text x=\"" << (stats_x + 22.0) << "\" y=\"" << (stats_y + 30.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"16\" font-weight=\"700\" fill=\"#111827\">Evidence card</text>\n";

    const double stat_left = stats_x + 22.0;
    const double stat_right = stats_x + stats_w / 2.0 + 8.0;
    const double stat_y0 = stats_y + 62.0;
    const double stat_step = 20.0;
    out << "  <text x=\"" << stat_left << "\" y=\"" << stat_y0 << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#334155\">mean</text>\n";
    out << "  <text x=\"" << (stat_left + 120.0) << "\" y=\"" << stat_y0 << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" font-weight=\"700\" fill=\"#0f172a\">" << format_double(summary.mean, 6) << "</text>\n";
    out << "  <text x=\"" << stat_left << "\" y=\"" << (stat_y0 + stat_step) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#334155\">std</text>\n";
    out << "  <text x=\"" << (stat_left + 120.0) << "\" y=\"" << (stat_y0 + stat_step) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" font-weight=\"700\" fill=\"#0f172a\">" << format_double(summary.stddev, 6) << "</text>\n";
    out << "  <text x=\"" << stat_left << "\" y=\"" << (stat_y0 + stat_step * 2.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#334155\">abs_mean</text>\n";
    out << "  <text x=\"" << (stat_left + 120.0) << "\" y=\"" << (stat_y0 + stat_step * 2.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" font-weight=\"700\" fill=\"#0f172a\">" << format_double(summary.abs_mean, 6) << "</text>\n";
    out << "  <text x=\"" << stat_left << "\" y=\"" << (stat_y0 + stat_step * 3.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#334155\">rms</text>\n";
    out << "  <text x=\"" << (stat_left + 120.0) << "\" y=\"" << (stat_y0 + stat_step * 3.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" font-weight=\"700\" fill=\"#0f172a\">" << format_double(summary.rms, 6) << "</text>\n";
    out << "  <text x=\"" << stat_left << "\" y=\"" << (stat_y0 + stat_step * 4.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#334155\">max_abs</text>\n";
    out << "  <text x=\"" << (stat_left + 120.0) << "\" y=\"" << (stat_y0 + stat_step * 4.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" font-weight=\"700\" fill=\"#0f172a\">" << format_double(summary.max_abs, 6) << "</text>\n";

    out << "  <text x=\"" << stat_right << "\" y=\"" << stat_y0 << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#334155\">p01 / p50 / p99</text>\n";
    out << "  <text x=\"" << stat_right << "\" y=\"" << (stat_y0 + stat_step) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" font-weight=\"700\" fill=\"#0f172a\">"
        << format_double(summary.p01, 5) << " / " << format_double(summary.p50, 5) << " / " << format_double(summary.p99, 5) << "</text>\n";
    out << "  <text x=\"" << stat_right << "\" y=\"" << (stat_y0 + stat_step * 2.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#334155\">sign split</text>\n";
    out << "  <text x=\"" << stat_right << "\" y=\"" << (stat_y0 + stat_step * 3.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" font-weight=\"700\" fill=\"#0f172a\">+"
        << format_percent(summary.positive_ratio, 1) << "  /  -" << format_percent(summary.negative_ratio, 1)
        << "  /  0 " << format_percent(summary.zero_ratio, 1) << "</text>\n";
    out << "  <text x=\"" << stat_right << "\" y=\"" << (stat_y0 + stat_step * 4.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#334155\">tail clipping</text>\n";
    out << "  <text x=\"" << stat_right << "\" y=\"" << (stat_y0 + stat_step * 5.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" font-weight=\"700\" fill=\"#0f172a\">low "
        << summary.clipped_low << "  |  high " << summary.clipped_high << "</text>\n";

    if (kind == tensor_kind::softmax) {
        out << "  <text x=\"" << (threshold_x + 22.0) << "\" y=\"" << (threshold_y + 30.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"16\" font-weight=\"700\" fill=\"#111827\">Top-k retention</text>\n";
        out << "  <text x=\"" << (threshold_x + 22.0) << "\" y=\"" << (threshold_y + 48.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#64748b\">per row of length " << softmax_topk.row_width << ": keep only top-k probabilities and measure retained mass</text>\n";

        const double bar_x = threshold_x + 22.0;
        const double bar_w = threshold_w - 44.0;
        const double bar_y0 = threshold_y + 78.0;
        const double bar_step = 28.0;
        for (size_t index = 0; index < softmax_topk.points.size(); ++index) {
            const auto & point = softmax_topk.points[index];
            const double y = bar_y0 + bar_step * static_cast<double>(index);
            out << "  <text x=\"" << bar_x << "\" y=\"" << (y - 8.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#334155\">top-" << point.k
                << "  keep " << format_percent(point.element_share, 2) << " elems</text>\n";
            out << "  <rect x=\"" << bar_x << "\" y=\"" << y << "\" width=\"" << bar_w << "\" height=\"10\" rx=\"5\" ry=\"5\" fill=\"#e5edf5\"/>\n";
            out << "  <rect x=\"" << bar_x << "\" y=\"" << y << "\" width=\"" << (bar_w * clamp01(point.mean_mass)) << "\" height=\"10\" rx=\"5\" ry=\"5\" fill=\"" << kind_color(kind) << "\" opacity=\"0.88\"/>\n";
            out << "  <text x=\"" << (bar_x + bar_w) << "\" y=\"" << (y - 8.0) << "\" text-anchor=\"end\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" font-weight=\"700\" fill=\"#0f172a\">retain "
                << format_percent(point.mean_mass, 2) << "</text>\n";
            out << "  <text x=\"" << bar_x << "\" y=\"" << (y + 24.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#64748b\">p50 mass="
                << format_percent(point.p50_mass, 2) << "  |  p95 mass=" << format_percent(point.p95_mass, 2)
                << "  |  cutoff~" << format_double(point.p50_cutoff, 6) << "</text>\n";
        }
    } else {
        out << "  <text x=\"" << (threshold_x + 22.0) << "\" y=\"" << (threshold_y + 30.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"16\" font-weight=\"700\" fill=\"#111827\">Threshold view</text>\n";
        out << "  <text x=\"" << (threshold_x + 22.0) << "\" y=\"" << (threshold_y + 48.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#64748b\">"
            << (kind == tensor_kind::swiglu_gate ? "gate uses one-sided negative-tail truncation; bars show how much falls below each cutoff" : "use these bars to discuss sparsity candidates before drawing conclusions")
            << "</text>\n";

        const double progress_x = threshold_x + 22.0;
        const double progress_w = threshold_w - 44.0;
        const double progress_y0 = threshold_y + 82.0;
        const double progress_step = 30.0;
        for (size_t index = 0; index < threshold_bars.size(); ++index) {
            const double y = progress_y0 + progress_step * static_cast<double>(index);
            const double fill_w = progress_w * clamp01(threshold_bars[index].fraction);
            out << "  <text x=\"" << progress_x << "\" y=\"" << (y - 8.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#334155\">"
                << svg_safe(threshold_bars[index].label) << "</text>\n";
            out << "  <rect x=\"" << progress_x << "\" y=\"" << y << "\" width=\"" << progress_w << "\" height=\"10\" rx=\"5\" ry=\"5\" fill=\"#e5edf5\"/>\n";
            out << "  <rect x=\"" << progress_x << "\" y=\"" << y << "\" width=\"" << fill_w << "\" height=\"10\" rx=\"5\" ry=\"5\" fill=\"" << kind_color(kind) << "\" opacity=\"0.88\"/>\n";
            out << "  <text x=\"" << (progress_x + progress_w) << "\" y=\"" << (y - 8.0) << "\" text-anchor=\"end\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" font-weight=\"700\" fill=\"#0f172a\">"
                << format_percent(threshold_bars[index].fraction, 2) << "</text>\n";
        }
    }

    out << "  <text x=\"" << (layer_x + 22.0) << "\" y=\"" << (layer_y + 30.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"16\" font-weight=\"700\" fill=\"#111827\">Layer profile</text>\n";
    if (kind == tensor_kind::softmax) {
        out << "  <text x=\"" << (layer_x + 22.0) << "\" y=\"" << (layer_y + 48.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#64748b\">bars = top-" << kSoftmaxLayerSpreadK
            << " retained mass  |  line = top-" << kSoftmaxLayerSpreadK << " cutoff by layer</text>\n";
    } else {
        out << "  <text x=\"" << (layer_x + 22.0) << "\" y=\"" << (layer_y + 48.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#64748b\">bars = "
            << svg_safe(primary_focus_label(kind)) << "  |  line = percentile magnitude by layer</text>\n";
    }

    const double layer_plot_x = layer_x + 56.0;
    const double layer_plot_y = layer_y + 68.0;
    const double layer_plot_w = layer_w - 88.0;
    const double layer_plot_h = 148.0;
    out << "  <rect x=\"" << layer_plot_x << "\" y=\"" << layer_plot_y << "\" width=\"" << layer_plot_w << "\" height=\"" << layer_plot_h << "\" fill=\"#f8fafc\" stroke=\"#e2e8f0\" stroke-width=\"1\" rx=\"12\" ry=\"12\"/>\n";

    for (int tick = 0; tick <= 4; ++tick) {
        const double y = layer_plot_y + layer_plot_h - layer_plot_h * static_cast<double>(tick) / 4.0;
        out << "  <line x1=\"" << layer_plot_x << "\" y1=\"" << y << "\" x2=\"" << (layer_plot_x + layer_plot_w)
            << "\" y2=\"" << y << "\" stroke=\"#e5edf5\" stroke-width=\"1\"/>\n";
        out << "  <text x=\"" << (layer_plot_x - 10.0) << "\" y=\"" << (y + 4.0) << "\" text-anchor=\"end\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#64748b\">"
            << format_percent(static_cast<double>(tick) / 4.0, 0) << "</text>\n";
    }

    if (kind == tensor_kind::softmax) {
        double max_cutoff = 0.0;
        std::ostringstream layer_line;
        for (const auto & layer : softmax_topk_layers) {
            const topk_point * point = find_topk_point(layer.stats, kSoftmaxLayerSpreadK);
            if (point == nullptr && !layer.stats.points.empty()) {
                point = &layer.stats.points.back();
            }
            if (point != nullptr) {
                max_cutoff = std::max(max_cutoff, std::max(point->mean_cutoff, static_cast<double>(point->p95_cutoff)));
            }
        }
        if (max_cutoff <= 0.0) {
            max_cutoff = 1.0;
        }

        for (size_t index = 0; index < softmax_topk_layers.size(); ++index) {
            const auto & layer = softmax_topk_layers[index];
            const topk_point * point = find_topk_point(layer.stats, kSoftmaxLayerSpreadK);
            if (point == nullptr && !layer.stats.points.empty()) {
                point = &layer.stats.points.back();
            }
            if (point == nullptr) {
                continue;
            }

            const double step = layer_plot_w / static_cast<double>(std::max<size_t>(1, softmax_topk_layers.size()));
            const double bar_x = layer_plot_x + step * static_cast<double>(index) + 6.0;
            const double bar_w = std::max(8.0, step - 12.0);
            const double ratio_h = clamp01(point->mean_mass) * layer_plot_h;
            const double bar_y = layer_plot_y + layer_plot_h - ratio_h;
            const double cutoff_y = layer_plot_y + layer_plot_h - clamp01(point->p50_cutoff / max_cutoff) * layer_plot_h;
            const double cx = bar_x + bar_w * 0.5;

            out << "  <rect x=\"" << bar_x << "\" y=\"" << bar_y << "\" width=\"" << bar_w << "\" height=\"" << ratio_h
                << "\" rx=\"4\" ry=\"4\" fill=\"" << kind_color(kind) << "\" opacity=\"0.62\"/>\n";
            out << "  <text x=\"" << cx << "\" y=\"" << (layer_plot_y + layer_plot_h + 18.0) << "\" text-anchor=\"middle\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"10\" fill=\"#475569\">"
                << layer.layer << "</text>\n";

            layer_line << (index == 0 ? "M " : " L ") << cx << ' ' << cutoff_y;
        }
        if (!softmax_topk_layers.empty()) {
            out << "  <path d=\"" << layer_line.str() << "\" fill=\"none\" stroke=\"#0f172a\" stroke-width=\"2.5\"/>\n";
            for (size_t index = 0; index < softmax_topk_layers.size(); ++index) {
                const auto & layer = softmax_topk_layers[index];
                const topk_point * point = find_topk_point(layer.stats, kSoftmaxLayerSpreadK);
                if (point == nullptr && !layer.stats.points.empty()) {
                    point = &layer.stats.points.back();
                }
                if (point == nullptr) {
                    continue;
                }
                const double step = layer_plot_w / static_cast<double>(softmax_topk_layers.size());
                const double cx = layer_plot_x + step * static_cast<double>(index) + step * 0.5;
                const double cy = layer_plot_y + layer_plot_h - clamp01(point->p50_cutoff / max_cutoff) * layer_plot_h;
                out << "  <circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"3.5\" fill=\"#0f172a\"/>\n";
            }
        }
        out << "  <text x=\"" << (layer_plot_x + layer_plot_w) << "\" y=\"" << (layer_plot_y - 10.0) << "\" text-anchor=\"end\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#0f172a\">cutoff max = "
            << format_double(max_cutoff, 6) << "</text>\n";
    } else {
        double max_layer_magnitude = 0.0;
        for (const auto & layer : layers) {
            max_layer_magnitude = std::max(max_layer_magnitude, layer_magnitude(layer, kind));
        }
        if (max_layer_magnitude <= 0.0) {
            max_layer_magnitude = 1.0;
        }

        std::ostringstream layer_line;
        for (size_t index = 0; index < layers.size(); ++index) {
            const double step = layer_plot_w / static_cast<double>(std::max<size_t>(1, layers.size()));
            const double bar_x = layer_plot_x + step * static_cast<double>(index) + 6.0;
            const double bar_w = std::max(8.0, step - 12.0);
            const double ratio_h = clamp01(layers[index].primary_ratio) * layer_plot_h;
            const double bar_y = layer_plot_y + layer_plot_h - ratio_h;
            const double magnitude_y = layer_plot_y + layer_plot_h - layer_magnitude(layers[index], kind) / max_layer_magnitude * layer_plot_h;
            const double cx = bar_x + bar_w * 0.5;

            out << "  <rect x=\"" << bar_x << "\" y=\"" << bar_y << "\" width=\"" << bar_w << "\" height=\"" << ratio_h
                << "\" rx=\"4\" ry=\"4\" fill=\"" << kind_color(kind) << "\" opacity=\"0.62\"/>\n";
            out << "  <text x=\"" << cx << "\" y=\"" << (layer_plot_y + layer_plot_h + 18.0) << "\" text-anchor=\"middle\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"10\" fill=\"#475569\">"
                << layers[index].layer << "</text>\n";

            layer_line << (index == 0 ? "M " : " L ") << cx << ' ' << magnitude_y;
        }
        if (!layers.empty()) {
            out << "  <path d=\"" << layer_line.str() << "\" fill=\"none\" stroke=\"#0f172a\" stroke-width=\"2.5\"/>\n";
            for (size_t index = 0; index < layers.size(); ++index) {
                const double step = layer_plot_w / static_cast<double>(layers.size());
                const double cx = layer_plot_x + step * static_cast<double>(index) + step * 0.5;
                const double cy = layer_plot_y + layer_plot_h - layer_magnitude(layers[index], kind) / max_layer_magnitude * layer_plot_h;
                out << "  <circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"3.5\" fill=\"#0f172a\"/>\n";
            }
        }
        out << "  <text x=\"" << (layer_plot_x + layer_plot_w) << "\" y=\"" << (layer_plot_y - 10.0) << "\" text-anchor=\"end\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#0f172a\">line max = "
            << format_double(max_layer_magnitude, 4) << "</text>\n";
    }

    out << "  <text x=\"" << (layer_x + 24.0) << "\" y=\"" << (height - 20.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#475569\">"
        << "min=" << format_double(summary.min, 6) << "  max=" << format_double(summary.max, 6)
        << "  p05=" << format_double(summary.p05, 6) << "  p95=" << format_double(summary.p95, 6)
        << "  clipped=(" << summary.clipped_low << ", " << summary.clipped_high << ")</text>\n";

    out << "</svg>\n";
}

static void write_capture_csv(const fs::path & path, const std::vector<round_capture> & rounds) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open capture CSV: " + path.string());
    }

    out << "round,token_id,token_piece,tensor_kind,tensor_name,layer,type,ne0,ne1,ne2,ne3,elements\n";
    for (size_t round_index = 0; round_index < rounds.size(); ++round_index) {
        const auto & round = rounds[round_index];
        for (const auto & meta : round.metas) {
            out << (round_index + 1) << ','
                << round.token << ','
                << csv_escape(round.token_piece) << ','
                << kind_slug(meta.kind) << ','
                << csv_escape(meta.name) << ','
                << meta.layer << ','
                << ggml_type_name(meta.type) << ','
                << meta.ne[0] << ','
                << meta.ne[1] << ','
                << meta.ne[2] << ','
                << meta.ne[3] << ','
                << meta.elements << '\n';
        }
    }
}

static void write_layer_summary_csv(
    const fs::path & path,
    const std::vector<round_capture> & rounds,
    const std::vector<std::array<std::vector<layer_summary>, kTensorKindCount>> & layer_summaries) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open layer summary CSV: " + path.string());
    }

    out << "round,token_id,token_piece,tensor_kind,layer,tensor_name,value_count,min,max,mean,abs_mean,rms,max_abs,std,p01,p50,p99,positive_ratio,negative_ratio,zero_ratio,primary_ratio\n";
    for (size_t round_index = 0; round_index < rounds.size(); ++round_index) {
        const auto & round = rounds[round_index];
        for (size_t kind_index_value = 0; kind_index_value < kTensorKindCount; ++kind_index_value) {
            const auto kind = static_cast<tensor_kind>(kind_index_value);
            for (const auto & layer : layer_summaries[round_index][kind_index_value]) {
                out << (round_index + 1) << ','
                    << round.token << ','
                    << csv_escape(round.token_piece) << ','
                    << kind_slug(kind) << ','
                    << layer.layer << ','
                    << csv_escape(layer.tensor_name) << ','
                    << layer.stats.n_values << ','
                    << format_double(layer.stats.min, 8) << ','
                    << format_double(layer.stats.max, 8) << ','
                    << format_double(layer.stats.mean, 8) << ','
                    << format_double(layer.stats.abs_mean, 8) << ','
                    << format_double(layer.stats.rms, 8) << ','
                    << format_double(layer.stats.max_abs, 8) << ','
                    << format_double(layer.stats.stddev, 8) << ','
                    << format_double(layer.stats.p01, 8) << ','
                    << format_double(layer.stats.p50, 8) << ','
                    << format_double(layer.stats.p99, 8) << ','
                    << format_double(layer.stats.positive_ratio, 8) << ','
                    << format_double(layer.stats.negative_ratio, 8) << ','
                    << format_double(layer.stats.zero_ratio, 8) << ','
                    << format_double(layer.primary_ratio, 8) << '\n';
            }
        }
    }
}

static void write_summary_csv(const fs::path & path, const std::vector<round_capture> & rounds, const std::vector<std::array<histogram_summary, kTensorKindCount>> & summaries) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open summary CSV: " + path.string());
    }

    out << "round,token_id,token_piece,tensor_kind,tensor_count,value_count,min,max,mean,std,p01,p50,p99,plot_min,plot_max,plot_scale\n";
    for (size_t round_index = 0; round_index < rounds.size(); ++round_index) {
        const auto & round = rounds[round_index];
        for (size_t kind_index_value = 0; kind_index_value < kTensorKindCount; ++kind_index_value) {
            const auto kind = static_cast<tensor_kind>(kind_index_value);
            const auto & summary = summaries[round_index][kind_index_value];
            out << (round_index + 1) << ','
                << round.token << ','
                << csv_escape(round.token_piece) << ','
                << kind_slug(kind) << ','
                << summary.n_tensors << ','
                << summary.n_values << ','
                << format_double(summary.min, 8) << ','
                << format_double(summary.max, 8) << ','
                << format_double(summary.mean, 8) << ','
                << format_double(summary.stddev, 8) << ','
                << format_double(summary.p01, 8) << ','
                << format_double(summary.p50, 8) << ','
                << format_double(summary.p99, 8) << ','
                << format_double(summary.plot_min, 8) << ','
                << format_double(summary.plot_max, 8) << ','
                << (summary.plot_log ? "log10" : "linear") << '\n';
        }
    }
}

static void build_stage_summaries(
    const std::vector<round_capture> & rounds,
    std::vector<std::array<histogram_summary, kTensorKindCount>> & summaries,
    std::vector<std::array<std::vector<layer_summary>, kTensorKindCount>> & layer_summaries) {
    summaries.assign(rounds.size(), {});
    layer_summaries.assign(rounds.size(), {});

    for (size_t round_index = 0; round_index < rounds.size(); ++round_index) {
        for (size_t kind_index_value = 0; kind_index_value < kTensorKindCount; ++kind_index_value) {
            const auto kind = static_cast<tensor_kind>(kind_index_value);
            summaries[round_index][kind_index_value] = summarize_values(rounds[round_index], kind);
            layer_summaries[round_index][kind_index_value] = summarize_layers(rounds[round_index], kind);
        }
    }
}

static void write_kind_overview_svg(
    const fs::path & path,
    const std::vector<round_capture> & rounds,
    const std::vector<std::array<histogram_summary, kTensorKindCount>> & summaries,
    const std::vector<std::array<std::vector<layer_summary>, kTensorKindCount>> & layer_summaries,
    tensor_kind kind,
    std::string_view stage_title,
    std::string_view round_prefix) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open overview SVG: " + path.string());
    }

    constexpr double width = 1360.0;
    constexpr double height = 680.0;
    constexpr double outer = 28.0;
    constexpr double card_radius = 18.0;
    const double left_x = outer;
    const double left_y = 102.0;
    const double left_w = 780.0;
    const double left_h = 520.0;
    const double right_x = left_x + left_w + 22.0;
    const double right_y = left_y;
    const double right_w = width - right_x - outer;
    const double right_h = left_h;
    std::vector<topk_summary> softmax_topk_rounds;
    std::vector<std::vector<layer_topk_summary>> softmax_topk_layer_rounds;
    int softmax_rep_k = kSoftmaxLayerSpreadK;

    if (kind == tensor_kind::softmax) {
        softmax_topk_rounds.reserve(rounds.size());
        softmax_topk_layer_rounds.reserve(rounds.size());
    }

    double full_min = std::numeric_limits<double>::infinity();
    double full_max = -std::numeric_limits<double>::infinity();
    double magnitude_max = 0.0;
    double ratio_max = 0.0;
    double softmax_spread_min = 1.0;
    double softmax_spread_max = 0.0;
    for (size_t round_index = 0; round_index < rounds.size(); ++round_index) {
        const auto & summary = summaries[round_index][kind_index(kind)];
        if (kind == tensor_kind::softmax) {
            softmax_topk_rounds.push_back(summarize_softmax_topk(rounds[round_index]));
            softmax_topk_layer_rounds.push_back(summarize_softmax_topk_layers(rounds[round_index]));

            const topk_summary & topk = softmax_topk_rounds.back();
            const topk_point * total_point = find_topk_point(topk, softmax_rep_k);
            if (total_point == nullptr && !topk.points.empty()) {
                total_point = &topk.points.back();
            }
            if (total_point != nullptr) {
                softmax_spread_min = std::min(softmax_spread_min, total_point->mean_mass);
                softmax_spread_max = std::max(softmax_spread_max, total_point->mean_mass);
            }

            for (const auto & layer : softmax_topk_layer_rounds.back()) {
                const topk_point * point = find_topk_point(layer.stats, softmax_rep_k);
                if (point == nullptr && !layer.stats.points.empty()) {
                    point = &layer.stats.points.back();
                }
                if (point == nullptr) {
                    continue;
                }
                softmax_spread_min = std::min(softmax_spread_min, point->mean_mass);
                softmax_spread_max = std::max(softmax_spread_max, point->mean_mass);
            }
        }
        full_min = std::min(full_min, plotted_value(summary.min, kind));
        full_max = std::max(full_max, plotted_value(summary.max, kind));
        magnitude_max = std::max(magnitude_max, summary.max_abs);

        if (kind != tensor_kind::softmax) {
            const double total_ratio = primary_focus_ratio(rounds[round_index].values[kind_index(kind)], 0, rounds[round_index].values[kind_index(kind)].size(), kind);
            ratio_max = std::max(ratio_max, total_ratio);
            for (const auto & layer : layer_summaries[round_index][kind_index(kind)]) {
                ratio_max = std::max(ratio_max, layer.primary_ratio);
            }
        }
    }

    if (kind == tensor_kind::softmax) {
        for (const auto & summary : softmax_topk_rounds) {
            if (find_topk_point(summary, softmax_rep_k) != nullptr) {
                continue;
            }
            if (!summary.points.empty()) {
                softmax_rep_k = std::min(softmax_rep_k, summary.points.back().k);
            }
        }

        if (!(softmax_spread_max > softmax_spread_min)) {
            softmax_spread_min = std::max(0.0, softmax_spread_min - 0.02);
            softmax_spread_max = std::min(1.0, softmax_spread_max + 0.02);
        } else {
            const double pad = std::max(0.01, (softmax_spread_max - softmax_spread_min) * 0.12);
            softmax_spread_min = std::max(0.0, softmax_spread_min - pad);
            softmax_spread_max = std::min(1.0, softmax_spread_max + pad);
        }
    }

    if (!(full_max > full_min)) {
        full_min -= 1.0;
        full_max += 1.0;
    }

    if (kind != tensor_kind::softmax) {
        const double symmetric = std::max(std::abs(full_min), std::abs(full_max));
        full_min = -symmetric;
        full_max = symmetric;
    }

    const auto [threshold_lo_raw, threshold_hi_raw] = primary_threshold_bounds(kind);
    const double threshold_lo = plotted_value(static_cast<float>(threshold_lo_raw), kind);
    const double threshold_hi = plotted_value(static_cast<float>(threshold_hi_raw), kind);

    if (kind != tensor_kind::softmax) {
        full_min = std::min(full_min, threshold_lo);
        full_max = std::max(full_max, threshold_hi);
    }

    const double value_span = full_max - full_min;
    if (!(value_span > 0.0)) {
        full_min -= 1.0;
        full_max += 1.0;
    }
    if (magnitude_max <= 0.0) {
        magnitude_max = 1.0;
    }

    ratio_max = std::max(ratio_max, 0.05);
    ratio_max = std::ceil(ratio_max * 20.0) / 20.0;

    const auto scale_y = [&](double value, double origin_y, double plot_h) {
        return origin_y + plot_h - clamp01((value - full_min) / (full_max - full_min)) * plot_h;
    };
    const auto scale_x = [&](double value, double origin_x, double plot_w) {
        return origin_x + clamp01((value - full_min) / (full_max - full_min)) * plot_w;
    };
    const auto scale_ratio_x = [&](double value, double origin_x, double plot_w) {
        return origin_x + clamp01(value / ratio_max) * plot_w;
    };

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    out << "  <defs>\n";
    out << "    <linearGradient id=\"bg\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\">\n";
    out << "      <stop offset=\"0%\" stop-color=\"#fffef7\"/>\n";
    out << "      <stop offset=\"100%\" stop-color=\"#f5f7fb\"/>\n";
    out << "    </linearGradient>\n";
    out << "    <filter id=\"shadow\" x=\"-10%\" y=\"-10%\" width=\"120%\" height=\"120%\">\n";
    out << "      <feDropShadow dx=\"0\" dy=\"12\" stdDeviation=\"14\" flood-color=\"#0f172a\" flood-opacity=\"0.08\"/>\n";
    out << "    </filter>\n";
    out << "  </defs>\n";
    out << "  <rect width=\"100%\" height=\"100%\" fill=\"url(#bg)\"/>\n";
    out << "  <text x=\"" << outer << "\" y=\"42\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"28\" font-weight=\"700\" fill=\"#0f172a\">"
        << svg_safe(std::string(kind_title(kind))) << " across " << svg_safe(std::string(stage_title)) << "</text>\n";
    if (kind == tensor_kind::softmax) {
        out << "  <text x=\"" << outer << "\" y=\"68\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"13\" fill=\"#475569\">left: full probability range + top-k cutoff marker  |  right: retained mass by k and layer spread</text>\n";
    } else {
        out << "  <text x=\"" << outer << "\" y=\"68\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"13\" fill=\"#475569\">"
            << (kind == tensor_kind::swiglu_gate ? "left: percentile band by round  |  right: one-sided truncation share and layer spread by round" : "left: percentile band by round  |  right: threshold share and layer spread by round")
            << "</text>\n";
    }

    auto draw_card = [&](double x, double y, double w, double h) {
        out << "  <rect x=\"" << x << "\" y=\"" << y << "\" rx=\"" << card_radius << "\" ry=\"" << card_radius
            << "\" width=\"" << w << "\" height=\"" << h << "\" fill=\"#ffffff\" stroke=\"#dbe4ee\" stroke-width=\"1\" filter=\"url(#shadow)\"/>\n";
    };
    draw_card(left_x, left_y, left_w, left_h);
    draw_card(right_x, right_y, right_w, right_h);

    const double left_plot_x = left_x + 64.0;
    const double left_plot_y = left_y + 70.0;
    const double left_plot_w = left_w - 108.0;
    const double left_plot_h = left_h - 162.0;
    out << "  <text x=\"" << (left_x + 22.0) << "\" y=\"" << (left_y + 30.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"16\" font-weight=\"700\" fill=\"#111827\">Full range by round</text>\n";
    if (kind == tensor_kind::softmax) {
        out << "  <text x=\"" << (left_x + 22.0) << "\" y=\"" << (left_y + 48.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#64748b\">grey whisker = min..max  |  thick band = p01..p99  |  dash = p50  |  dot = mean  |  green tick = top-" << softmax_rep_k << " cutoff  |  y-axis = log10(p)</text>\n";
    } else {
        out << "  <text x=\"" << (left_x + 22.0) << "\" y=\"" << (left_y + 48.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#64748b\">grey whisker = min..max  |  thick band = p01..p99  |  inner band = p05..p95  |  dash = p50  |  dot = mean</text>\n";
    }
    out << "  <rect x=\"" << left_plot_x << "\" y=\"" << left_plot_y << "\" width=\"" << left_plot_w << "\" height=\"" << left_plot_h << "\" fill=\"#f8fafc\" stroke=\"#e2e8f0\" stroke-width=\"1\" rx=\"12\" ry=\"12\"/>\n";

    if (kind == tensor_kind::softmax) {
        out << "  <text x=\"" << (left_plot_x - 4.0) << "\" y=\"" << (left_plot_y - 12.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#475569\">y = log10(p), tick label = p</text>\n";
    }

    if (kind != tensor_kind::softmax) {
        const double threshold_y_hi = scale_y(threshold_hi, left_plot_y, left_plot_h);
        const double threshold_y_lo = scale_y(uses_one_sided_truncation(kind) ? full_min : threshold_lo, left_plot_y, left_plot_h);
        const double threshold_band_y = std::min(threshold_y_hi, threshold_y_lo);
        const double threshold_band_h = std::abs(threshold_y_hi - threshold_y_lo);
        out << "  <rect x=\"" << left_plot_x << "\" y=\"" << threshold_band_y << "\" width=\"" << left_plot_w << "\" height=\"" << std::max(2.0, threshold_band_h)
            << "\" fill=\"#dc2626\" fill-opacity=\"0.10\"/>\n";
        out << "  <text x=\"" << (left_plot_x + 8.0) << "\" y=\"" << (threshold_band_y - 8.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#b91c1c\">"
            << svg_safe(primary_threshold_window_label(kind)) << "</text>\n";
    }

    if (kind != tensor_kind::softmax && full_min < 0.0 && full_max > 0.0) {
        const double zero_y = scale_y(0.0, left_plot_y, left_plot_h);
        out << "  <line x1=\"" << left_plot_x << "\" y1=\"" << zero_y << "\" x2=\"" << (left_plot_x + left_plot_w)
            << "\" y2=\"" << zero_y << "\" stroke=\"#dc2626\" stroke-width=\"1.5\" stroke-dasharray=\"6 6\" opacity=\"0.65\"/>\n";
        out << "  <text x=\"" << (left_plot_x + left_plot_w - 6.0) << "\" y=\"" << (zero_y - 6.0) << "\" text-anchor=\"end\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#b91c1c\">zero</text>\n";
    }

    for (int tick = 0; tick <= 5; ++tick) {
        const double value = full_min + (full_max - full_min) * static_cast<double>(tick) / 5.0;
        const double y = scale_y(value, left_plot_y, left_plot_h);
        out << "  <line x1=\"" << left_plot_x << "\" y1=\"" << y << "\" x2=\"" << (left_plot_x + left_plot_w)
            << "\" y2=\"" << y << "\" stroke=\"#e5edf5\" stroke-width=\"1\"/>\n";
        out << "  <text x=\"" << (left_plot_x - 10.0) << "\" y=\"" << (y + 4.0) << "\" text-anchor=\"end\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#64748b\">"
            << (kind == tensor_kind::softmax ? format_softmax_probability_tick(value) : format_double(value, 4)) << "</text>\n";
    }

    const double round_step = left_plot_w / static_cast<double>(std::max<size_t>(1, rounds.size()));
    for (size_t round_index = 0; round_index < rounds.size(); ++round_index) {
        const auto & summary = summaries[round_index][kind_index(kind)];
        const double cx = left_plot_x + round_step * static_cast<double>(round_index) + round_step * 0.5;
        const double whisker_top = scale_y(plotted_value(summary.max, kind), left_plot_y, left_plot_h);
        const double whisker_bottom = scale_y(plotted_value(summary.min, kind), left_plot_y, left_plot_h);
        const double p01_y = scale_y(plotted_value(summary.p01, kind), left_plot_y, left_plot_h);
        const double p99_y = scale_y(plotted_value(summary.p99, kind), left_plot_y, left_plot_h);
        const double p05_y = scale_y(plotted_value(summary.p05, kind), left_plot_y, left_plot_h);
        const double p95_y = scale_y(plotted_value(summary.p95, kind), left_plot_y, left_plot_h);
        const double p50_y = scale_y(plotted_value(summary.p50, kind), left_plot_y, left_plot_h);
        const double mean_y = scale_y(plotted_value(static_cast<float>(summary.mean), kind), left_plot_y, left_plot_h);

        out << "  <line x1=\"" << cx << "\" y1=\"" << whisker_top << "\" x2=\"" << cx << "\" y2=\"" << whisker_bottom << "\" stroke=\"#64748b\" stroke-width=\"2\" opacity=\"0.7\"/>\n";
        out << "  <line x1=\"" << cx << "\" y1=\"" << p01_y << "\" x2=\"" << cx << "\" y2=\"" << p99_y << "\" stroke=\"" << kind_color(kind) << "\" stroke-width=\"12\" stroke-linecap=\"round\" opacity=\"0.28\"/>\n";
        if (kind != tensor_kind::softmax) {
            out << "  <line x1=\"" << cx << "\" y1=\"" << p05_y << "\" x2=\"" << cx << "\" y2=\"" << p95_y << "\" stroke=\"" << kind_color(kind) << "\" stroke-width=\"8\" stroke-linecap=\"round\" opacity=\"0.72\"/>\n";
        }
        out << "  <line x1=\"" << (cx - 10.0) << "\" y1=\"" << p50_y << "\" x2=\"" << (cx + 10.0) << "\" y2=\"" << p50_y << "\" stroke=\"#0f172a\" stroke-width=\"2\"/>\n";
        out << "  <circle cx=\"" << cx << "\" cy=\"" << mean_y << "\" r=\"4\" fill=\"#0f172a\"/>\n";
        if (kind == tensor_kind::softmax) {
            const topk_point * point = find_topk_point(softmax_topk_rounds[round_index], softmax_rep_k);
            if (point == nullptr && !softmax_topk_rounds[round_index].points.empty()) {
                point = &softmax_topk_rounds[round_index].points.back();
            }
            if (point != nullptr) {
                const double cutoff_y = scale_y(plotted_value(point->p50_cutoff, kind), left_plot_y, left_plot_h);
                out << "  <line x1=\"" << (cx - 13.0) << "\" y1=\"" << cutoff_y << "\" x2=\"" << (cx + 13.0) << "\" y2=\"" << cutoff_y << "\" stroke=\"#15803d\" stroke-width=\"3\" stroke-linecap=\"round\"/>\n";
            }
        }
        out << "  <text x=\"" << cx << "\" y=\"" << (left_plot_y + left_plot_h + 20.0) << "\" text-anchor=\"middle\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#475569\">"
            << svg_safe(std::string(round_prefix)) << (round_index + 1) << "</text>\n";
        out << "  <text x=\"" << cx << "\" y=\"" << (left_plot_y + left_plot_h + 38.0) << "\" text-anchor=\"middle\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"10\" fill=\"#64748b\">"
            << svg_safe(rounds[round_index].token_piece) << "</text>\n";
        out << "  <text x=\"" << cx << "\" y=\"" << (left_plot_y + left_plot_h + 56.0) << "\" text-anchor=\"middle\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"10\" fill=\"#0f172a\">max|x| "
            << format_double(summary.max_abs, 2) << "</text>\n";
    }

    out << "  <text x=\"" << (left_plot_x + 8.0) << "\" y=\"" << (left_y + left_h - 20.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#64748b\">"
        << (kind == tensor_kind::softmax ? "full axis log10(p): [" : "full axis: [")
        << format_double(full_min, 4) << ", " << format_double(full_max, 4) << "]</text>\n";

    if (kind == tensor_kind::softmax) {
        const double topk_x = right_x + 30.0;
        const double topk_y = right_y + 78.0;
        const double topk_w = right_w - 60.0;
        const double topk_h = 232.0;
        const double spread_title_y = right_y + 352.0;
        const double spread_plot_x = right_x + 90.0;
        const double spread_plot_y = spread_title_y + 22.0;
        const double spread_plot_w = right_w - 192.0;
        const double spread_plot_h = 148.0;
        const double spread_pct_x = spread_plot_x + spread_plot_w + 12.0;

        out << "  <text x=\"" << (right_x + 22.0) << "\" y=\"" << (right_y + 30.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"16\" font-weight=\"700\" fill=\"#111827\">Top-k retained probability mass by round</text>\n";
        out << "  <text x=\"" << (right_x + 22.0) << "\" y=\"" << (right_y + 48.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#64748b\">retained probability mass = sum of the kept top-k probabilities in one softmax row</text>\n";
        out << "  <rect x=\"" << topk_x << "\" y=\"" << topk_y << "\" width=\"" << topk_w << "\" height=\"" << topk_h << "\" fill=\"#f8fafc\" stroke=\"#e2e8f0\" stroke-width=\"1\" rx=\"12\" ry=\"12\"/>\n";

        std::vector<int> topk_columns;
        for (const auto & summary : softmax_topk_rounds) {
            if (!summary.points.empty()) {
                for (const auto & point : summary.points) {
                    topk_columns.push_back(point.k);
                }
                break;
            }
        }
        if (topk_columns.empty()) {
            topk_columns.push_back(softmax_rep_k);
        }

        const double matrix_label_w = 62.0;
        const double matrix_head_h = 34.0;
        const double matrix_x = topk_x + matrix_label_w;
        const double matrix_y = topk_y + matrix_head_h;
        const double matrix_w = topk_w - matrix_label_w - 10.0;
        const double matrix_h = topk_h - matrix_head_h - 16.0;
        const double matrix_col_w = matrix_w / static_cast<double>(std::max<size_t>(1, topk_columns.size()));
        const double matrix_row_h = matrix_h / static_cast<double>(std::max<size_t>(1, rounds.size()));

        for (size_t column = 0; column < topk_columns.size(); ++column) {
            const double x = matrix_x + matrix_col_w * static_cast<double>(column);
            out << "  <text x=\"" << (x + matrix_col_w * 0.5) << "\" y=\"" << (topk_y + 16.0) << "\" text-anchor=\"middle\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" font-weight=\"700\" fill=\"#0f172a\">top-" << topk_columns[column] << "</text>\n";
            const topk_point * header_point = nullptr;
            for (const auto & summary : softmax_topk_rounds) {
                header_point = find_topk_point(summary, topk_columns[column]);
                if (header_point != nullptr) {
                    break;
                }
            }
            if (header_point != nullptr) {
                out << "  <text x=\"" << (x + matrix_col_w * 0.5) << "\" y=\"" << (topk_y + 29.0) << "\" text-anchor=\"middle\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"10\" fill=\"#64748b\">keep "
                    << format_percent(header_point->element_share, 1) << "</text>\n";
            }
        }

        for (size_t round_index = 0; round_index < rounds.size(); ++round_index) {
            const double row_y = matrix_y + matrix_row_h * static_cast<double>(round_index);
            const double cy = row_y + matrix_row_h * 0.5;
            out << "  <text x=\"" << (matrix_x - 8.0) << "\" y=\"" << (cy + 4.0) << "\" text-anchor=\"end\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#334155\">"
                << svg_safe(std::string(round_prefix)) << (round_index + 1) << "</text>\n";
            for (size_t column = 0; column < topk_columns.size(); ++column) {
                const double x = matrix_x + matrix_col_w * static_cast<double>(column);
                const topk_point * point = find_topk_point(softmax_topk_rounds[round_index], topk_columns[column]);
                out << "  <rect x=\"" << (x + 4.0) << "\" y=\"" << (row_y + 4.0) << "\" width=\"" << (matrix_col_w - 8.0) << "\" height=\"" << (matrix_row_h - 8.0)
                    << "\" rx=\"8\" ry=\"8\" fill=\"#eef3f8\"/>\n";
                if (point != nullptr) {
                    const double fill_h = (matrix_row_h - 8.0) * clamp01(point->mean_mass);
                    out << "  <rect x=\"" << (x + 4.0) << "\" y=\"" << (row_y + 4.0 + (matrix_row_h - 8.0 - fill_h)) << "\" width=\"" << (matrix_col_w - 8.0) << "\" height=\"" << fill_h
                        << "\" rx=\"8\" ry=\"8\" fill=\"" << kind_color(kind) << "\" opacity=\"0.78\"/>\n";
                    out << "  <text x=\"" << (x + matrix_col_w * 0.5) << "\" y=\"" << (row_y + matrix_row_h * 0.5 + 4.0) << "\" text-anchor=\"middle\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" font-weight=\"700\" fill=\"#0f172a\">"
                        << format_percent(point->mean_mass, 0) << "</text>\n";
                }
            }
        }

        out << "  <text x=\"" << (right_x + 22.0) << "\" y=\"" << spread_title_y << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"16\" font-weight=\"700\" fill=\"#111827\">Layer spread for top-" << softmax_rep_k << "</text>\n";
        out << "  <text x=\"" << (right_x + 22.0) << "\" y=\"" << (spread_title_y + 18.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#64748b\">x-axis zooms to actual used range  |  whisker = layer min..max retained probability mass  |  dot = round aggregate</text>\n";
        out << "  <rect x=\"" << spread_plot_x << "\" y=\"" << spread_plot_y << "\" width=\"" << spread_plot_w << "\" height=\"" << spread_plot_h << "\" fill=\"#f8fafc\" stroke=\"#e2e8f0\" stroke-width=\"1\" rx=\"12\" ry=\"12\"/>\n";

        const auto scale_softmax_spread_x = [&](double value) {
            return spread_plot_x + clamp01((value - softmax_spread_min) / (softmax_spread_max - softmax_spread_min)) * spread_plot_w;
        };

        for (int tick = 0; tick <= 4; ++tick) {
            const double ratio = softmax_spread_min + (softmax_spread_max - softmax_spread_min) * static_cast<double>(tick) / 4.0;
            const double x = scale_softmax_spread_x(ratio);
            out << "  <line x1=\"" << x << "\" y1=\"" << spread_plot_y << "\" x2=\"" << x << "\" y2=\"" << (spread_plot_y + spread_plot_h)
                << "\" stroke=\"#e5edf5\" stroke-width=\"1\"/>\n";
            out << "  <text x=\"" << x << "\" y=\"" << (spread_plot_y + spread_plot_h + 18.0) << "\" text-anchor=\"middle\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#64748b\">"
                << format_percent(ratio, 0) << "</text>\n";
        }

        const double spread_row_step = spread_plot_h / static_cast<double>(std::max<size_t>(1, rounds.size()));
        for (size_t round_index = 0; round_index < rounds.size(); ++round_index) {
            double layer_min = std::numeric_limits<double>::infinity();
            double layer_max = -std::numeric_limits<double>::infinity();
            for (const auto & layer : softmax_topk_layer_rounds[round_index]) {
                const topk_point * point = find_topk_point(layer.stats, softmax_rep_k);
                if (point == nullptr && !layer.stats.points.empty()) {
                    point = &layer.stats.points.back();
                }
                if (point == nullptr) {
                    continue;
                }
                layer_min = std::min(layer_min, point->mean_mass);
                layer_max = std::max(layer_max, point->mean_mass);
            }
            if (!std::isfinite(layer_min)) {
                layer_min = 0.0;
                layer_max = 0.0;
            }

            const topk_point * total_point = find_topk_point(softmax_topk_rounds[round_index], softmax_rep_k);
            if (total_point == nullptr && !softmax_topk_rounds[round_index].points.empty()) {
                total_point = &softmax_topk_rounds[round_index].points.back();
            }

            const double cy = spread_plot_y + spread_row_step * static_cast<double>(round_index) + spread_row_step * 0.5;
            const double x_lo = scale_softmax_spread_x(layer_min);
            const double x_hi = scale_softmax_spread_x(layer_max);
            const double x_total = scale_softmax_spread_x(total_point != nullptr ? total_point->mean_mass : softmax_spread_min);

            out << "  <text x=\"" << (spread_plot_x - 10.0) << "\" y=\"" << (cy + 4.0) << "\" text-anchor=\"end\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#334155\">"
                << svg_safe(std::string(round_prefix)) << (round_index + 1) << "</text>\n";
            out << "  <line x1=\"" << x_lo << "\" y1=\"" << cy << "\" x2=\"" << x_hi << "\" y2=\"" << cy << "\" stroke=\"#64748b\" stroke-width=\"3\" opacity=\"0.72\"/>\n";
            out << "  <circle cx=\"" << x_total << "\" cy=\"" << cy << "\" r=\"4\" fill=\"" << kind_color(kind) << "\"/>\n";
            out << "  <text x=\"" << spread_pct_x << "\" y=\"" << (cy + 4.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" font-weight=\"700\" fill=\"#0f172a\">"
                << (total_point != nullptr ? ("agg " + format_percent(total_point->mean_mass, 2) + "  |  cutoff p50=" + format_double(total_point->p50_cutoff, 6) + "  p95=" + format_double(total_point->p95_cutoff, 6)) : std::string("n/a"))
                << "</text>\n";
        }

        out << "  <text x=\"" << (right_x + 22.0) << "\" y=\"" << (right_y + right_h - 24.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#475569\">full axis log10(p) ["
            << format_double(full_min, 4) << ", " << format_double(full_max, 4) << "]  |  green tick on left = top-" << softmax_rep_k
            << " cutoff position  |  retained probability mass = sum(top-k probabilities in one row)</text>\n";

        out << "</svg>\n";
        return;
    }

    const double range_top_x = right_x + 64.0;
    const double range_top_y = right_y + 78.0;
    const double range_label_w = 72.0;
    const double range_pct_w = 92.0;
    const double range_plot_x = range_top_x + range_label_w;
    const double range_plot_w = right_w - 64.0 - range_label_w - range_pct_w - 20.0;
    const double range_plot_h = 228.0;
    const double range_pct_x = range_plot_x + range_plot_w + 12.0;

    out << "  <text x=\"" << (right_x + 22.0) << "\" y=\"" << (right_y + 30.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"16\" font-weight=\"700\" fill=\"#111827\">Truncation window on full scale</text>\n";
    out << "  <text x=\"" << (right_x + 22.0) << "\" y=\"" << (right_y + 48.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#64748b\">line = full min..max  |  thick segment = p01..p99  |  red band = truncation window  |  right text = hit ratio</text>\n";
    out << "  <rect x=\"" << range_plot_x << "\" y=\"" << range_top_y << "\" width=\"" << range_plot_w << "\" height=\"" << range_plot_h << "\" fill=\"#f8fafc\" stroke=\"#e2e8f0\" stroke-width=\"1\" rx=\"12\" ry=\"12\"/>\n";

    const double threshold_x_lo = scale_x(uses_one_sided_truncation(kind) ? full_min : threshold_lo, range_plot_x, range_plot_w);
    const double threshold_x_hi = scale_x(threshold_hi, range_plot_x, range_plot_w);
    out << "  <rect x=\"" << std::min(threshold_x_lo, threshold_x_hi) << "\" y=\"" << range_top_y << "\" width=\"" << std::max(2.0, std::abs(threshold_x_hi - threshold_x_lo))
        << "\" height=\"" << range_plot_h << "\" fill=\"#dc2626\" fill-opacity=\"0.10\"/>\n";
    out << "  <text x=\"" << std::min(threshold_x_lo, threshold_x_hi) + 4.0 << "\" y=\"" << (range_top_y - 8.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#b91c1c\">"
        << svg_safe(primary_threshold_window_label(kind)) << "</text>\n";

    if (kind != tensor_kind::softmax && full_min < 0.0 && full_max > 0.0) {
        const double zero_x = scale_x(0.0, range_plot_x, range_plot_w);
        out << "  <line x1=\"" << zero_x << "\" y1=\"" << range_top_y << "\" x2=\"" << zero_x << "\" y2=\"" << (range_top_y + range_plot_h)
            << "\" stroke=\"#dc2626\" stroke-width=\"1.25\" stroke-dasharray=\"5 6\" opacity=\"0.65\"/>\n";
    }

    const double range_row_step = range_plot_h / static_cast<double>(std::max<size_t>(1, rounds.size()));
    for (size_t round_index = 0; round_index < rounds.size(); ++round_index) {
        const auto & summary = summaries[round_index][kind_index(kind)];
        const double cy = range_top_y + range_row_step * static_cast<double>(round_index) + range_row_step * 0.5;
        const double x_min = scale_x(plotted_value(summary.min, kind), range_plot_x, range_plot_w);
        const double x_max = scale_x(plotted_value(summary.max, kind), range_plot_x, range_plot_w);
        const double x_p01 = scale_x(plotted_value(summary.p01, kind), range_plot_x, range_plot_w);
        const double x_p99 = scale_x(plotted_value(summary.p99, kind), range_plot_x, range_plot_w);
        const double x_mean = scale_x(plotted_value(static_cast<float>(summary.mean), kind), range_plot_x, range_plot_w);
        const double ratio = primary_focus_ratio(rounds[round_index].values[kind_index(kind)], 0, rounds[round_index].values[kind_index(kind)].size(), kind);

        out << "  <text x=\"" << (range_top_x + range_label_w - 8.0) << "\" y=\"" << (cy + 4.0) << "\" text-anchor=\"end\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#334155\">"
            << svg_safe(std::string(round_prefix)) << (round_index + 1) << " " << svg_safe(rounds[round_index].token_piece) << "</text>\n";
        out << "  <line x1=\"" << x_min << "\" y1=\"" << cy << "\" x2=\"" << x_max << "\" y2=\"" << cy << "\" stroke=\"#64748b\" stroke-width=\"2\" opacity=\"0.8\"/>\n";
        out << "  <line x1=\"" << x_p01 << "\" y1=\"" << cy << "\" x2=\"" << x_p99 << "\" y2=\"" << cy << "\" stroke=\"" << kind_color(kind) << "\" stroke-width=\"8\" stroke-linecap=\"round\" opacity=\"0.82\"/>\n";
        out << "  <circle cx=\"" << x_mean << "\" cy=\"" << cy << "\" r=\"3.5\" fill=\"#0f172a\"/>\n";
        out << "  <text x=\"" << range_pct_x << "\" y=\"" << (cy + 4.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" font-weight=\"700\" fill=\"#0f172a\">"
            << format_percent(ratio, 2) << "</text>\n";
    }

    out << "  <text x=\"" << range_plot_x << "\" y=\"" << (range_top_y + range_plot_h + 18.0) << "\" text-anchor=\"start\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#64748b\">"
        << format_double(full_min, 4) << "</text>\n";
    if (kind != tensor_kind::softmax && full_min < 0.0 && full_max > 0.0) {
        const double zero_x = scale_x(0.0, range_plot_x, range_plot_w);
        out << "  <text x=\"" << zero_x << "\" y=\"" << (range_top_y + range_plot_h + 18.0) << "\" text-anchor=\"middle\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#64748b\">0</text>\n";
    }
    out << "  <text x=\"" << (range_plot_x + range_plot_w) << "\" y=\"" << (range_top_y + range_plot_h + 18.0) << "\" text-anchor=\"end\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#64748b\">"
        << format_double(full_max, 4) << "</text>\n";

    const double spread_title_y = right_y + 352.0;
    const double spread_plot_x = right_x + 90.0;
    const double spread_plot_y = spread_title_y + 22.0;
    const double spread_plot_w = right_w - 192.0;
    const double spread_plot_h = 148.0;
    const double spread_pct_x = spread_plot_x + spread_plot_w + 12.0;

    out << "  <text x=\"" << (right_x + 22.0) << "\" y=\"" << spread_title_y << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"16\" font-weight=\"700\" fill=\"#111827\">Layer spread of truncation ratio</text>\n";
    out << "  <text x=\"" << (right_x + 22.0) << "\" y=\"" << (spread_title_y + 18.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"12\" fill=\"#64748b\">whisker = layer min..max  |  dot = all-layer aggregate  |  right text = exact percentage</text>\n";
    out << "  <rect x=\"" << spread_plot_x << "\" y=\"" << spread_plot_y << "\" width=\"" << spread_plot_w << "\" height=\"" << spread_plot_h << "\" fill=\"#f8fafc\" stroke=\"#e2e8f0\" stroke-width=\"1\" rx=\"12\" ry=\"12\"/>\n";

    for (int tick = 0; tick <= 4; ++tick) {
        const double ratio = ratio_max * static_cast<double>(tick) / 4.0;
        const double x = scale_ratio_x(ratio, spread_plot_x, spread_plot_w);
        out << "  <line x1=\"" << x << "\" y1=\"" << spread_plot_y << "\" x2=\"" << x << "\" y2=\"" << (spread_plot_y + spread_plot_h)
            << "\" stroke=\"#e5edf5\" stroke-width=\"1\"/>\n";
        out << "  <text x=\"" << x << "\" y=\"" << (spread_plot_y + spread_plot_h + 18.0) << "\" text-anchor=\"middle\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#64748b\">"
            << format_percent(ratio, 0) << "</text>\n";
    }

    const double spread_row_step = spread_plot_h / static_cast<double>(std::max<size_t>(1, rounds.size()));
    for (size_t round_index = 0; round_index < rounds.size(); ++round_index) {
        double layer_min = std::numeric_limits<double>::infinity();
        double layer_max = -std::numeric_limits<double>::infinity();
        for (const auto & layer : layer_summaries[round_index][kind_index(kind)]) {
            layer_min = std::min(layer_min, layer.primary_ratio);
            layer_max = std::max(layer_max, layer.primary_ratio);
        }
        if (!std::isfinite(layer_min)) {
            layer_min = 0.0;
            layer_max = 0.0;
        }

        const double total_ratio = primary_focus_ratio(rounds[round_index].values[kind_index(kind)], 0, rounds[round_index].values[kind_index(kind)].size(), kind);
        const double cy = spread_plot_y + spread_row_step * static_cast<double>(round_index) + spread_row_step * 0.5;
        const double x_lo = scale_ratio_x(layer_min, spread_plot_x, spread_plot_w);
        const double x_hi = scale_ratio_x(layer_max, spread_plot_x, spread_plot_w);
        const double x_total = scale_ratio_x(total_ratio, spread_plot_x, spread_plot_w);

        out << "  <text x=\"" << (spread_plot_x - 10.0) << "\" y=\"" << (cy + 4.0) << "\" text-anchor=\"end\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#334155\">"
            << svg_safe(std::string(round_prefix)) << (round_index + 1) << "</text>\n";
        out << "  <line x1=\"" << x_lo << "\" y1=\"" << cy << "\" x2=\"" << x_hi << "\" y2=\"" << cy << "\" stroke=\"#64748b\" stroke-width=\"3\" opacity=\"0.72\"/>\n";
        out << "  <circle cx=\"" << x_total << "\" cy=\"" << cy << "\" r=\"4\" fill=\"" << kind_color(kind) << "\"/>\n";
        out << "  <text x=\"" << spread_pct_x << "\" y=\"" << (cy + 4.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" font-weight=\"700\" fill=\"#0f172a\">"
            << format_percent(total_ratio, 2) << "</text>\n";
    }

    const std::string footer_threshold_note = kind == tensor_kind::swiglu_gate
        ? ("truncate x <= " + format_double(kSwiGLUGateTruncationThreshold, 6))
        : ("threshold window [" + format_double(threshold_lo_raw, 6) + ", " + format_double(threshold_hi_raw, 6) + "]");
    out << "  <text x=\"" << (right_x + 22.0) << "\" y=\"" << (right_y + right_h - 24.0) << "\" font-family=\"IBM Plex Mono, JetBrains Mono, monospace\" font-size=\"11\" fill=\"#475569\">full axis ["
        << format_double(full_min, 4) << ", " << format_double(full_max, 4) << "]  |  "
        << svg_safe(footer_threshold_note)
        << "  |  peak max_abs=" << format_double(magnitude_max, 4) << "</text>\n";

    out << "</svg>\n";
}

static void write_index_html(
    const fs::path & path,
    const fs::path & plot_dir,
    const common_params & params,
    const std::vector<round_capture> & prefill_rounds,
    const std::array<fs::path, kTensorKindCount> & prefill_overview_paths,
    const std::vector<round_capture> & decode_rounds,
    const std::vector<std::array<fs::path, kTensorKindCount>> & plot_paths,
    const std::array<fs::path, kTensorKindCount> & overview_paths) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open index HTML: " + path.string());
    }

    out << "<!doctype html>\n<html lang=\"en\">\n<head>\n";
    out << "  <meta charset=\"utf-8\">\n";
    out << "  <title>Prefill + Decode Stats</title>\n";
    out << "  <style>\n";
    out << "    :root { --bg: #fcfbf5; --ink: #0f172a; --muted: #475569; --card: #ffffff; --line: #dbe4ee; --softmax: #0f766e; --swiglu: #2563eb; --gate: #dc2626; --silu: #be185d; --ffn: #b45309; }\n";
    out << "    body { font-family: 'IBM Plex Mono', 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 24px; color: var(--ink); background: radial-gradient(circle at top left, #fffef7, var(--bg) 48%, #f2f5fb 100%); }\n";
    out << "    h1, h2 { margin: 0 0 12px; }\n";
    out << "    .meta, .panel { background: var(--card); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06); }\n";
    out << "    .meta { margin-bottom: 24px; padding: 18px 20px; white-space: pre-wrap; line-height: 1.6; }\n";
    out << "    .overview-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; margin-bottom: 24px; }\n";
    out << "    .overview-grid .panel, .round-grid .panel { padding: 14px; }\n";
    out << "    .overview-grid img, .round-grid img { width: 100%; height: auto; display: block; border-radius: 12px; }\n";
    out << "    .legend { display: flex; gap: 14px; flex-wrap: wrap; margin: 10px 0 24px; color: var(--muted); font-size: 13px; }\n";
    out << "    .swatch { display: inline-flex; align-items: center; gap: 8px; }\n";
    out << "    .swatch i { width: 12px; height: 12px; border-radius: 999px; display: inline-block; }\n";
    out << "    .round-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 18px; margin-bottom: 24px; }\n";
    out << "    .round-title { margin: 0 0 12px; font-size: 16px; }\n";
    out << "    .caption { margin: 0 0 10px; color: var(--muted); font-size: 12px; line-height: 1.5; }\n";
    out << "  </style>\n</head>\n<body>\n";
    out << "  <h1>Prefill + Decode Stats</h1>\n";
    out << "  <div class=\"meta\">\n";
    out << "model: " << html_escape(params.model.path) << "\n";
    out << "prompt: " << html_escape(params.prompt) << "\n";
    out << "prefill_tokens: " << prefill_rounds.size() << "\n";
    out << "decode_rounds: " << decode_rounds.size() << "\n";
    out << "flash_attn: forced_off\n";
    out << "plot_dir: " << html_escape(plot_dir.string()) << "\n";
    out << "  </div>\n";
    out << "  <div class=\"legend\">\n";
    out << "    <span class=\"swatch\"><i style=\"background: var(--softmax)\"></i>Softmax</span>\n";
    out << "    <span class=\"swatch\"><i style=\"background: var(--swiglu)\"></i>SwiGLU</span>\n";
    out << "    <span class=\"swatch\"><i style=\"background: var(--gate)\"></i>SwiGLU gate / SiLU input</span>\n";
    out << "    <span class=\"swatch\"><i style=\"background: var(--silu)\"></i>SwiGLU SiLU branch</span>\n";
    out << "    <span class=\"swatch\"><i style=\"background: var(--ffn)\"></i>FFN output</span>\n";
    out << "    <span>Prefill gets overview-only charts; decode keeps overview plus per-round dashboards. Softmax uses top-k evidence; all FFN-side tensors use threshold evidence.</span>\n";
    out << "  </div>\n";
    out << "  <h2>Prefill Overview</h2>\n";
    out << "  <div class=\"overview-grid\">\n";
    for (size_t kind_index_value = 0; kind_index_value < kTensorKindCount; ++kind_index_value) {
        const auto kind = static_cast<tensor_kind>(kind_index_value);
        const std::string embedded_svg = svg_data_uri(prefill_overview_paths[kind_index_value]);
        out << "    <div class=\"panel\">\n";
        out << "      <p class=\"round-title\">" << html_escape(std::string(kind_title(kind))) << " prefill-token overview</p>\n";
        out << "      <img src=\"" << embedded_svg << "\" alt=\"prefill overview\">\n";
        out << "    </div>\n";
    }
    out << "  </div>\n";
    out << "  <h2>Decode Overview</h2>\n";
    out << "  <div class=\"overview-grid\">\n";
    for (size_t kind_index_value = 0; kind_index_value < kTensorKindCount; ++kind_index_value) {
        const auto kind = static_cast<tensor_kind>(kind_index_value);
        const std::string embedded_svg = svg_data_uri(overview_paths[kind_index_value]);
        out << "    <div class=\"panel\">\n";
        out << "      <p class=\"round-title\">" << html_escape(std::string(kind_title(kind))) << " decode overview</p>\n";
        out << "      <img src=\"" << embedded_svg << "\" alt=\"overview\">\n";
        out << "    </div>\n";
    }
    out << "  </div>\n";
    out << "  <h2>Per Round</h2>\n";
    for (size_t round_index = 0; round_index < decode_rounds.size(); ++round_index) {
        const auto & round = decode_rounds[round_index];
        out << "  <div class=\"meta\">round " << (round_index + 1) << "  |  token=" << round.token << "  |  piece=\"" << html_escape(round.token_piece) << "\"</div>\n";
        out << "  <div class=\"round-grid\">\n";
        for (size_t kind_index_value = 0; kind_index_value < kTensorKindCount; ++kind_index_value) {
            const auto kind = static_cast<tensor_kind>(kind_index_value);
            const std::string embedded_svg = svg_data_uri(plot_paths[round_index][kind_index_value]);
            out << "    <div class=\"panel\">\n";
            out << "      <p class=\"round-title\">" << html_escape(std::string(kind_title(kind))) << "</p>\n";
            out << "      <p class=\"caption\">This panel is meant for discussion, not immediate pruning decisions. Softmax uses top-k evidence; all FFN-side tensors use threshold evidence.</p>\n";
            out << "      <img src=\"" << embedded_svg << "\" alt=\"" << html_escape(std::string(kind_title(kind))) << "\">\n";
            out << "    </div>\n";
        }
        out << "  </div>\n";
    }
    out << "</body>\n</html>\n";
}

static void validate_round_capture(const round_capture & round, int round_number) {
    for (size_t kind_index_value = 0; kind_index_value < kTensorKindCount; ++kind_index_value) {
        const auto kind = static_cast<tensor_kind>(kind_index_value);
        if (round.values[kind_index_value].empty()) {
            std::ostringstream oss;
            oss << "round " << round_number << " did not capture any tensors for " << kind_slug(kind)
                << "; make sure the model uses these nodes and Flash Attention is disabled";
            throw std::runtime_error(oss.str());
        }
    }
}

static void decode_prompt(llama_context * ctx, const std::vector<llama_token> & tokens, int32_t n_batch, callback_data & cb_data) {
    (void) n_batch;
    cb_data.prefill_rounds.assign(tokens.size(), {});
    for (size_t index = 0; index < tokens.size(); ++index) {
        cb_data.prefill_rounds[index].token = tokens[index];
        cb_data.prefill_rounds[index].token_piece = common_token_to_piece(ctx, tokens[index], true);
    }
    cb_data.active_rounds = &cb_data.prefill_rounds;

    for (size_t index = 0; index < tokens.size(); ++index) {
        llama_token token = tokens[index];

        cb_data.current_round = static_cast<int>(index);
        cb_data.active_token_count = 1;
        cb_data.capture_enabled = true;
        if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0) {
            cb_data.capture_enabled = false;
            cb_data.current_round = -1;
            cb_data.active_token_count = 0;
            cb_data.active_rounds = nullptr;
            throw std::runtime_error("llama_decode() failed during prompt prefill");
        }
        cb_data.capture_enabled = false;
        cb_data.current_round = -1;
        cb_data.active_token_count = 0;

        validate_round_capture(cb_data.prefill_rounds.at(index), static_cast<int>(index + 1));
    }

    cb_data.active_rounds = nullptr;
}

} // namespace

int main(int argc, char ** argv) {
    common_params params;
    params.out_file = "decode-stats";
    params.n_predict = kDefaultDecodeRounds;

    std::vector<char *> filtered_argv;
    if (!preprocess_args(argc, argv, params, filtered_argv)) {
        print_usage(argc, argv);
        return 1;
    }

    if (!common_params_parse(static_cast<int>(filtered_argv.size()), filtered_argv.data(), params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    if (params.prompt.empty()) {
        params.prompt = "Hello";
    }

    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.warmup = false;

    callback_data cb_data;
    params.cb_eval = capture_decode_tensors;
    params.cb_eval_user_data = &cb_data;

    const int decode_rounds = params.n_predict > 0 ? params.n_predict : kDefaultDecodeRounds;
    cb_data.decode_rounds.resize(static_cast<size_t>(decode_rounds));

    if (params.n_predict <= 0) {
        LOG_INF("%s: n_predict <= 0, forcing decode rounds to %d\n", __func__, decode_rounds);
    }

    const fs::path out_dir(params.out_file.empty() ? "decode-stats" : params.out_file);
    const fs::path plot_dir = out_dir / "plots";

    try {
        fs::create_directories(plot_dir);
    } catch (const std::exception & exc) {
        LOG_ERR("%s: failed to create output directory '%s': %s\n", __func__, out_dir.string().c_str(), exc.what());
        return 1;
    }

    print_build_info();
    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    const auto llama_init = common_init_from_params(params);
    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to initialize model/context\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const auto prompt_tokens = common_tokenize(ctx, params.prompt, add_bos);
    if (prompt_tokens.empty()) {
        LOG_ERR("%s: prompt tokenized to zero tokens\n", __func__);
        return 1;
    }

    LOG_INF("%s: forcing flash attention off for explicit kq_soft_max capture\n", __func__);
    LOG_INF("%s: prompt tokens = %zu, decode rounds = %d, output dir = %s\n", __func__, prompt_tokens.size(), decode_rounds, out_dir.string().c_str());

    try {
        decode_prompt(ctx, prompt_tokens, params.n_batch, cb_data);
    } catch (const std::exception & exc) {
        LOG_ERR("%s: %s\n", __func__, exc.what());
        return 1;
    }

    LOG_INF("%s: captured %zu prefill token(s)\n", __func__, cb_data.prefill_rounds.size());

    common_sampler * sampler = common_sampler_init(model, params.sampling);
    if (sampler == nullptr) {
        LOG_ERR("%s: failed to initialize sampler\n", __func__);
        return 1;
    }

    for (const auto token : prompt_tokens) {
        common_sampler_accept(sampler, token, false);
    }

    std::string generated_text;

    for (int round = 0; round < decode_rounds; ++round) {
        llama_token token = common_sampler_sample(sampler, ctx, -1);
        common_sampler_accept(sampler, token, true);

        auto & round_capture = cb_data.decode_rounds.at(static_cast<size_t>(round));
        round_capture.token = token;
        round_capture.token_piece = common_token_to_piece(ctx, token, true);
        generated_text += round_capture.token_piece;

        cb_data.current_round = round;
        cb_data.active_token_count = 1;
        cb_data.capture_enabled = true;
        cb_data.active_rounds = &cb_data.decode_rounds;
        if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0) {
            LOG_ERR("%s: llama_decode() failed during decode round %d\n", __func__, round + 1);
            cb_data.capture_enabled = false;
            cb_data.current_round = -1;
            cb_data.active_token_count = 0;
            cb_data.active_rounds = nullptr;
            common_sampler_free(sampler);
            return 1;
        }
        cb_data.capture_enabled = false;
        cb_data.current_round = -1;
        cb_data.active_token_count = 0;
        cb_data.active_rounds = nullptr;

        try {
            validate_round_capture(round_capture, round + 1);
        } catch (const std::exception & exc) {
            LOG_ERR("%s: %s\n", __func__, exc.what());
            common_sampler_free(sampler);
            return 1;
        }

        LOG_INF("%s: round %d token %d '%s' | softmax=%zu values, swiglu=%zu values, swiglu_gate=%zu values, swiglu_silu=%zu values, ffn_out=%zu values\n",
            __func__, round + 1, token, round_capture.token_piece.c_str(),
            round_capture.values[kind_index(tensor_kind::softmax)].size(),
            round_capture.values[kind_index(tensor_kind::swiglu)].size(),
            round_capture.values[kind_index(tensor_kind::swiglu_gate)].size(),
            round_capture.values[kind_index(tensor_kind::swiglu_silu)].size(),
            round_capture.values[kind_index(tensor_kind::ffn_out_pre_res)].size());
    }

    std::vector<std::array<histogram_summary, kTensorKindCount>> prefill_summaries;
    std::vector<std::array<std::vector<layer_summary>, kTensorKindCount>> prefill_layer_summaries;
    std::array<fs::path, kTensorKindCount> prefill_overview_paths;
    std::vector<std::array<histogram_summary, kTensorKindCount>> summaries;
    std::vector<std::array<std::vector<layer_summary>, kTensorKindCount>> layer_summaries;
    std::vector<std::array<fs::path, kTensorKindCount>> plot_paths(cb_data.decode_rounds.size());
    std::array<fs::path, kTensorKindCount> overview_paths;

    try {
        build_stage_summaries(cb_data.prefill_rounds, prefill_summaries, prefill_layer_summaries);
        build_stage_summaries(cb_data.decode_rounds, summaries, layer_summaries);

        for (size_t round_index = 0; round_index < cb_data.decode_rounds.size(); ++round_index) {
            for (size_t kind_index_value = 0; kind_index_value < kTensorKindCount; ++kind_index_value) {
                const auto kind = static_cast<tensor_kind>(kind_index_value);
                const fs::path plot_path = plot_dir / ("round_" + (round_index + 1 < 10 ? std::string("0") : std::string()) + std::to_string(round_index + 1) + "_" + std::string(kind_slug(kind)) + ".svg");
                plot_paths[round_index][kind_index_value] = plot_path;
                write_histogram_svg(plot_path, summaries[round_index][kind_index_value], cb_data.decode_rounds[round_index], kind, static_cast<int>(round_index + 1));
            }
        }

        for (size_t kind_index_value = 0; kind_index_value < kTensorKindCount; ++kind_index_value) {
            const auto kind = static_cast<tensor_kind>(kind_index_value);
            prefill_overview_paths[kind_index_value] = plot_dir / ("prefill_overview_" + std::string(kind_slug(kind)) + ".svg");
            write_kind_overview_svg(prefill_overview_paths[kind_index_value], cb_data.prefill_rounds, prefill_summaries, prefill_layer_summaries, kind, "prefill tokens", "t");
            overview_paths[kind_index_value] = plot_dir / ("overview_" + std::string(kind_slug(kind)) + ".svg");
            write_kind_overview_svg(overview_paths[kind_index_value], cb_data.decode_rounds, summaries, layer_summaries, kind, "decode rounds", "r");
        }

        write_capture_csv(out_dir / "prefill_captures.csv", cb_data.prefill_rounds);
        write_summary_csv(out_dir / "prefill_summary.csv", cb_data.prefill_rounds, prefill_summaries);
        write_layer_summary_csv(out_dir / "prefill_layer_summary.csv", cb_data.prefill_rounds, prefill_layer_summaries);
        write_capture_csv(out_dir / "captures.csv", cb_data.decode_rounds);
        write_summary_csv(out_dir / "summary.csv", cb_data.decode_rounds, summaries);
        write_layer_summary_csv(out_dir / "layer_summary.csv", cb_data.decode_rounds, layer_summaries);
        write_index_html(out_dir / "index.html", plot_dir, params, cb_data.prefill_rounds, prefill_overview_paths, cb_data.decode_rounds, plot_paths, overview_paths);

        std::ofstream generated(out_dir / "generated.txt");
        generated << generated_text;
    } catch (const std::exception & exc) {
        LOG_ERR("%s: failed to write output artifacts: %s\n", __func__, exc.what());
        common_sampler_free(sampler);
        return 1;
    }

    LOG_INF("%s: wrote prefill + decode stats to %s\n", __func__, out_dir.string().c_str());
    LOG_INF("%s: open %s\n", __func__, (out_dir / "index.html").string().c_str());

    common_sampler_free(sampler);
    llama_perf_context_print(ctx);
    llama_backend_free();

    return 0;
}
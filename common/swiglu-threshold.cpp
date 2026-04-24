#include "swiglu-threshold.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

constexpr size_t kStageCount = 3;

enum class swiglu_threshold_histogram_kind {
    abs_nonzero = 0,
    negative_tail_magnitude,
};

struct swiglu_threshold_counters {
    uint64_t tensor_count = 0;
    uint64_t total_values = 0;
    uint64_t original_zero_values = 0;
    uint64_t truncated_nonzero_values = 0;
    uint64_t final_zero_values = 0;
};

struct swiglu_threshold_collect_histogram {
    swiglu_threshold_histogram_kind kind = swiglu_threshold_histogram_kind::abs_nonzero;
    double log10_min = 0.0;
    double log10_max = 0.0;
    std::vector<uint64_t> bins;
    uint64_t underflow_count = 0;
    uint64_t overflow_count = 0;
    float min_abs_nonzero = std::numeric_limits<float>::infinity();
    float max_abs_nonzero = 0.0f;
};

struct swiglu_threshold_stage_stats {
    swiglu_threshold_counters counters;
    swiglu_threshold_counters output_counters;
    swiglu_threshold_collect_histogram abs_histogram;
    std::vector<float> channel_max_values;
    bool has_output_counters = false;
    std::string output_kind;
};

struct swiglu_threshold_layer_stats {
    int layer = -1;
    float threshold = 0.0f;
    bool has_channel_thresholds = false;
    size_t channel_threshold_count = 0;
    float channel_threshold_min = 0.0f;
    float channel_threshold_max = 0.0f;
    float channel_threshold_mean = 0.0f;
    std::array<swiglu_threshold_stage_stats, kStageCount> stage_stats{};
};

struct common_swiglu_threshold_runtime {
    common_activation_threshold_kind target_kind = common_activation_threshold_kind::swiglu;
    std::string config_path;
    std::string report_path;
    std::string collect_prefix;
    std::shared_ptr<common_swiglu_threshold_runtime> secondary_runtime;
    std::map<int, float> thresholds;
    std::map<int, std::vector<float>> channel_thresholds;
    std::map<int, swiglu_threshold_layer_stats> layer_stats;
    std::array<swiglu_threshold_counters, kStageCount> totals{};
    std::array<swiglu_threshold_counters, kStageCount> output_totals{};
    std::array<std::string, kStageCount> output_kind_names{};
    std::vector<uint8_t> scratch;
    std::vector<uint8_t> aux_scratch;
    int collect_bins = 0;
    double collect_log10_min = 0.0;
    double collect_log10_max = 0.0;
    common_swiglu_threshold_stage current_stage = common_swiglu_threshold_stage::inactive;
    int active_tokens = 0;
    bool apply_enabled = false;
    bool collect_enabled = false;
    bool warned_unsupported_type = false;
};

namespace {

constexpr std::string_view kSwiGLUPrefix = "ffn_swiglu-";
constexpr std::string_view kSiluPrefix = "ffn_silu-";

static std::string trim_copy(const std::string & value) {
    const size_t start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }

    const size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

static bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

static const char * target_kind_name(common_activation_threshold_kind kind) {
    switch (kind) {
        case common_activation_threshold_kind::swiglu:
            return "swiglu";
        case common_activation_threshold_kind::silu:
            return "silu";
        case common_activation_threshold_kind::silu_input:
            return "silu_input";
    }

    return "unknown";
}

static bool threshold_channel_requested(const std::string & config_path, const std::string & collect_prefix) {
    return !config_path.empty() || !collect_prefix.empty();
}

static bool is_supported_dual_output_pair(
    common_activation_threshold_kind primary_kind,
    common_activation_threshold_kind secondary_kind) {
    return (primary_kind == common_activation_threshold_kind::swiglu && secondary_kind == common_activation_threshold_kind::silu) ||
        (primary_kind == common_activation_threshold_kind::silu && secondary_kind == common_activation_threshold_kind::swiglu);
}

static int threshold_apply_priority(common_activation_threshold_kind kind) {
    switch (kind) {
        case common_activation_threshold_kind::silu:
            return 0;
        case common_activation_threshold_kind::swiglu:
            return 1;
        case common_activation_threshold_kind::silu_input:
            return 2;
    }

    return 0;
}

static bool parse_target_kind_text(const std::string & text, common_activation_threshold_kind & value) {
    if (text == "swiglu") {
        value = common_activation_threshold_kind::swiglu;
        return true;
    }

    if (text == "silu") {
        value = common_activation_threshold_kind::silu;
        return true;
    }

    if (text == "silu_input" || text == "silu-input") {
        value = common_activation_threshold_kind::silu_input;
        return true;
    }

    return false;
}

static swiglu_threshold_histogram_kind histogram_kind_for_target_kind(common_activation_threshold_kind kind) {
    switch (kind) {
        case common_activation_threshold_kind::swiglu:
        case common_activation_threshold_kind::silu:
            return swiglu_threshold_histogram_kind::abs_nonzero;
        case common_activation_threshold_kind::silu_input:
            return swiglu_threshold_histogram_kind::negative_tail_magnitude;
    }

    return swiglu_threshold_histogram_kind::abs_nonzero;
}

static const char * histogram_kind_name(swiglu_threshold_histogram_kind kind) {
    switch (kind) {
        case swiglu_threshold_histogram_kind::abs_nonzero:
            return "abs_nonzero";
        case swiglu_threshold_histogram_kind::negative_tail_magnitude:
            return "negative_tail_magnitude";
    }

    return "unknown";
}

static int parse_layer_id(std::string_view name) {
    const size_t pos = name.rfind('-');
    if (pos == std::string_view::npos || pos + 1 >= name.size()) {
        return -1;
    }

    try {
        return std::stoi(std::string(name.substr(pos + 1)));
    } catch (...) {
        return -1;
    }
}

static size_t stage_index(common_swiglu_threshold_stage stage) {
    switch (stage) {
        case common_swiglu_threshold_stage::prefill:
            return 0;
        case common_swiglu_threshold_stage::decode:
            return 1;
        case common_swiglu_threshold_stage::ppl:
            return 2;
        case common_swiglu_threshold_stage::inactive:
            GGML_ABORT("inactive stage has no counter index");
    }

    GGML_ABORT("invalid SwiGLU threshold stage");
}

static const char * stage_name(common_swiglu_threshold_stage stage) {
    switch (stage) {
        case common_swiglu_threshold_stage::prefill:
            return "prefill";
        case common_swiglu_threshold_stage::decode:
            return "decode";
        case common_swiglu_threshold_stage::ppl:
            return "ppl";
        case common_swiglu_threshold_stage::inactive:
            return "inactive";
    }

    return "unknown";
}

static bool read_scalar(const uint8_t * data, ggml_type type, size_t offset, float & value) {
    switch (type) {
        case GGML_TYPE_F32:
            value = *reinterpret_cast<const float *>(data + offset);
            return true;
        case GGML_TYPE_F16:
            value = ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(data + offset));
            return true;
        case GGML_TYPE_BF16:
            value = ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(data + offset));
            return true;
        default:
            return false;
    }
}

static bool write_scalar(uint8_t * data, ggml_type type, size_t offset, float value) {
    switch (type) {
        case GGML_TYPE_F32:
            *reinterpret_cast<float *>(data + offset) = value;
            return true;
        case GGML_TYPE_F16:
            *reinterpret_cast<ggml_fp16_t *>(data + offset) = ggml_fp32_to_fp16(value);
            return true;
        case GGML_TYPE_BF16:
            *reinterpret_cast<ggml_bf16_t *>(data + offset) = ggml_fp32_to_bf16(value);
            return true;
        default:
            return false;
    }
}

static double safe_ratio(uint64_t numerator, uint64_t denominator) {
    return denominator == 0 ? 0.0 : static_cast<double>(numerator) / static_cast<double>(denominator);
}

static void accumulate_counters(
    swiglu_threshold_counters & counters,
    uint64_t tensor_count,
    uint64_t total_values,
    uint64_t original_zero_values,
    uint64_t truncated_nonzero_values,
    uint64_t final_zero_values) {
    counters.tensor_count += tensor_count;
    counters.total_values += total_values;
    counters.original_zero_values += original_zero_values;
    counters.truncated_nonzero_values += truncated_nonzero_values;
    counters.final_zero_values += final_zero_values;
}

static void note_output_kind(std::string & current_kind, const char * next_kind) {
    if (next_kind == nullptr || next_kind[0] == '\0') {
        return;
    }

    if (current_kind.empty()) {
        current_kind = next_kind;
        return;
    }

    if (current_kind != next_kind) {
        current_kind = "mixed";
    }
}

static std::string format_double(double value, int precision = 8) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed, std::ios::floatfield);
    oss.precision(precision);
    oss << value;
    return oss.str();
}

static std::string csv_escape(const std::string & value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }

    std::string escaped = "\"";
    for (const char ch : value) {
        if (ch == '\"') {
            escaped += "\"\"";
        } else {
            escaped += ch;
        }
    }
    escaped += '\"';
    return escaped;
}

static bool parse_int_text(const std::string & text, int minimum_value, int & value) {
    char * end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == nullptr || *end != '\0' || parsed < minimum_value || parsed > std::numeric_limits<int>::max()) {
        return false;
    }

    value = static_cast<int>(parsed);
    return true;
}

static bool parse_float_text(const std::string & text, float & value) {
    char * end = nullptr;
    const float parsed = std::strtof(text.c_str(), &end);
    if (end == nullptr || *end != '\0') {
        return false;
    }

    value = parsed;
    return true;
}

static float silu_transform(float value) {
    if (value >= 0.0f) {
        const double exp_neg = std::exp(-static_cast<double>(value));
        return static_cast<float>(static_cast<double>(value) / (1.0 + exp_neg));
    }

    const double exp_pos = std::exp(static_cast<double>(value));
    return static_cast<float>(static_cast<double>(value) * exp_pos / (1.0 + exp_pos));
}

static void init_histogram(swiglu_threshold_collect_histogram & histogram, const common_swiglu_threshold_runtime & runtime) {
    histogram.kind = histogram_kind_for_target_kind(runtime.target_kind);
    histogram.log10_min = runtime.collect_log10_min;
    histogram.log10_max = runtime.collect_log10_max;
    histogram.bins.assign(static_cast<size_t>(runtime.collect_bins), 0);
    histogram.underflow_count = 0;
    histogram.overflow_count = 0;
    histogram.min_abs_nonzero = std::numeric_limits<float>::infinity();
    histogram.max_abs_nonzero = 0.0f;
}

static void record_histogram_sample(swiglu_threshold_collect_histogram & histogram, float tracked_value) {
    if (!(tracked_value > 0.0f)) {
        return;
    }

    histogram.min_abs_nonzero = std::min(histogram.min_abs_nonzero, tracked_value);
    histogram.max_abs_nonzero = std::max(histogram.max_abs_nonzero, tracked_value);

    const double log_value = std::log10(static_cast<double>(tracked_value));
    if (log_value < histogram.log10_min) {
        histogram.underflow_count += 1;
        return;
    }

    if (log_value >= histogram.log10_max) {
        histogram.overflow_count += 1;
        return;
    }

    const double normalized = (log_value - histogram.log10_min) / (histogram.log10_max - histogram.log10_min);
    const size_t index = std::min(histogram.bins.size() - 1, static_cast<size_t>(normalized * histogram.bins.size()));
    histogram.bins[index] += 1;
}

static bool tracked_value_for_target_kind(common_activation_threshold_kind kind, float value, float & tracked_value) {
    switch (kind) {
        case common_activation_threshold_kind::swiglu:
        case common_activation_threshold_kind::silu:
            if (value == 0.0f) {
                return false;
            }
            tracked_value = std::abs(value);
            return tracked_value > 0.0f;
        case common_activation_threshold_kind::silu_input:
            if (!(value < 0.0f)) {
                return false;
            }
            tracked_value = -value;
            return tracked_value > 0.0f;
    }

    return false;
}

static bool should_truncate_value(common_activation_threshold_kind kind, float value, float threshold) {
    if (threshold <= 0.0f || value == 0.0f) {
        return false;
    }

    switch (kind) {
        case common_activation_threshold_kind::swiglu:
        case common_activation_threshold_kind::silu:
            return std::abs(value) <= threshold;
        case common_activation_threshold_kind::silu_input:
            return value <= -threshold;
    }

    return false;
}

static float threshold_for_layer(const common_swiglu_threshold_runtime & runtime, int layer) {
    const auto it = runtime.thresholds.find(layer);
    return it == runtime.thresholds.end() ? 0.0f : it->second;
}

static void describe_layer_thresholds(
    const common_swiglu_threshold_runtime & runtime,
    int layer,
    swiglu_threshold_layer_stats & layer_stats) {
    layer_stats.threshold = threshold_for_layer(runtime, layer);

    const auto channel_it = runtime.channel_thresholds.find(layer);
    if (channel_it == runtime.channel_thresholds.end()) {
        return;
    }

    float min_threshold = std::numeric_limits<float>::infinity();
    float max_threshold = 0.0f;
    double sum_threshold = 0.0;
    size_t count = 0;
    for (const float threshold : channel_it->second) {
        if (!(threshold > 0.0f)) {
            continue;
        }

        min_threshold = std::min(min_threshold, threshold);
        max_threshold = std::max(max_threshold, threshold);
        sum_threshold += threshold;
        count += 1;
    }

    if (count == 0) {
        return;
    }

    layer_stats.has_channel_thresholds = true;
    layer_stats.channel_threshold_count = count;
    layer_stats.channel_threshold_min = min_threshold;
    layer_stats.channel_threshold_max = max_threshold;
    layer_stats.channel_threshold_mean = static_cast<float>(sum_threshold / static_cast<double>(count));
    if (!(layer_stats.threshold > 0.0f)) {
        layer_stats.threshold = layer_stats.channel_threshold_mean;
    }
}

static float threshold_for_layer_channel(const common_swiglu_threshold_runtime & runtime, int layer, int channel) {
    const auto channel_it = runtime.channel_thresholds.find(layer);
    if (channel_it != runtime.channel_thresholds.end() && channel >= 0 && static_cast<size_t>(channel) < channel_it->second.size()) {
        const float threshold = channel_it->second[static_cast<size_t>(channel)];
        if (threshold > 0.0f) {
            return threshold;
        }
    }

    return threshold_for_layer(runtime, layer);
}

static bool collect_channel_max_values(const common_swiglu_threshold_runtime & runtime) {
    return runtime.collect_enabled && runtime.target_kind == common_activation_threshold_kind::silu;
}

static void ensure_channel_max_storage(swiglu_threshold_stage_stats & stage_stats, int64_t channel_count) {
    if (channel_count <= 0) {
        return;
    }

    const size_t count = static_cast<size_t>(channel_count);
    if (stage_stats.channel_max_values.size() < count) {
        stage_stats.channel_max_values.resize(count, -std::numeric_limits<float>::infinity());
    }
}

static void record_channel_max_value(swiglu_threshold_stage_stats & stage_stats, int channel, float value) {
    if (channel < 0 || static_cast<size_t>(channel) >= stage_stats.channel_max_values.size()) {
        return;
    }

    const float magnitude = std::abs(value);
    float & current = stage_stats.channel_max_values[static_cast<size_t>(channel)];
    if (!std::isfinite(current) || magnitude > current) {
        current = magnitude;
    }
}

static bool is_swiglu_tensor(const ggml_tensor * tensor) {
    const std::string_view name(tensor->name != nullptr ? tensor->name : "");
    return starts_with(name, kSwiGLUPrefix);
}

static bool is_silu_tensor(const ggml_tensor * tensor) {
    const std::string_view name(tensor->name != nullptr ? tensor->name : "");
    return starts_with(name, kSiluPrefix);
}

static swiglu_threshold_layer_stats & ensure_layer_stats(common_swiglu_threshold_runtime & runtime, int layer) {
    auto [it, inserted] = runtime.layer_stats.try_emplace(layer);
    if (inserted) {
        it->second.layer = layer;
        describe_layer_thresholds(runtime, layer, it->second);
        if (runtime.collect_enabled) {
            for (auto & stage_stats : it->second.stage_stats) {
                init_histogram(stage_stats.abs_histogram, runtime);
            }
        }
    }
    return it->second;
}

static bool is_target_tensor(const common_swiglu_threshold_runtime & runtime, const ggml_tensor * tensor) {
    switch (runtime.target_kind) {
        case common_activation_threshold_kind::swiglu:
            return is_swiglu_tensor(tensor);
        case common_activation_threshold_kind::silu:
            return is_silu_tensor(tensor) || is_swiglu_tensor(tensor);
        case common_activation_threshold_kind::silu_input:
            return is_silu_tensor(tensor) || is_swiglu_tensor(tensor);
    }

    return false;
}

static bool process_direct_tensor(common_swiglu_threshold_runtime & runtime, ggml_tensor * tensor, bool use_silu_transform) {
    const std::string_view name(tensor->name != nullptr ? tensor->name : "");
    const int layer = parse_layer_id(name);
    swiglu_threshold_layer_stats & layer_stats = ensure_layer_stats(runtime, layer);
    const common_swiglu_threshold_stage stage = runtime.current_stage;
    const size_t stage_slot = stage_index(stage);
    swiglu_threshold_stage_stats & layer_stage_stats = layer_stats.stage_stats[stage_slot];
    const bool mirror_output_counters = runtime.target_kind == common_activation_threshold_kind::swiglu && is_swiglu_tensor(tensor);
    const char * actual_output_kind = mirror_output_counters ? "swiglu" : "";

    if (collect_channel_max_values(runtime)) {
        ensure_channel_max_storage(layer_stage_stats, tensor->ne[0]);
    }

    const size_t nbytes = ggml_nbytes(tensor);
    runtime.scratch.resize(nbytes);
    ggml_backend_tensor_get(tensor, runtime.scratch.data(), 0, nbytes);

    uint64_t total_values = 0;
    uint64_t original_zero_values = 0;
    uint64_t truncated_nonzero_values = 0;
    uint64_t final_zero_values = 0;
    bool modified = false;

    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                    const size_t offset =
                        static_cast<size_t>(i3) * tensor->nb[3] +
                        static_cast<size_t>(i2) * tensor->nb[2] +
                        static_cast<size_t>(i1) * tensor->nb[1] +
                        static_cast<size_t>(i0) * tensor->nb[0];

                    float value = 0.0f;
                    if (!read_scalar(runtime.scratch.data(), tensor->type, offset, value)) {
                        if (!runtime.warned_unsupported_type) {
                            LOG_WRN("%s: unsupported %s tensor type %s; skipping truncation/stat collection\n", __func__, target_kind_name(runtime.target_kind), ggml_type_name(tensor->type));
                            runtime.warned_unsupported_type = true;
                        }
                        return true;
                    }

                    if (use_silu_transform) {
                        value = silu_transform(value);
                    }

                    const bool original_zero = value == 0.0f;
                    bool final_zero = original_zero;
                    float tracked_value = 0.0f;
                    const bool tracked = tracked_value_for_target_kind(runtime.target_kind, value, tracked_value);
                    total_values++;
                    if (original_zero) {
                        original_zero_values++;
                    }

                    if (runtime.collect_enabled && tracked) {
                        record_histogram_sample(layer_stage_stats.abs_histogram, tracked_value);
                    }
                    if (collect_channel_max_values(runtime)) {
                        record_channel_max_value(layer_stage_stats, static_cast<int>(i0), value);
                    }

                    const float threshold = threshold_for_layer_channel(runtime, layer, static_cast<int>(i0));
                    if (!original_zero && runtime.apply_enabled && should_truncate_value(runtime.target_kind, value, threshold)) {
                        truncated_nonzero_values++;
                        final_zero = true;
                        modified = true;
                        if (!write_scalar(runtime.scratch.data(), tensor->type, offset, 0.0f)) {
                            return true;
                        }
                    }

                    if (final_zero) {
                        final_zero_values++;
                    }
                }
            }
        }
    }

    accumulate_counters(runtime.totals[stage_slot], 1, total_values, original_zero_values, truncated_nonzero_values, final_zero_values);
    accumulate_counters(layer_stage_stats.counters, 1, total_values, original_zero_values, truncated_nonzero_values, final_zero_values);

    if (mirror_output_counters) {
        layer_stage_stats.has_output_counters = true;
        note_output_kind(layer_stage_stats.output_kind, actual_output_kind);
        note_output_kind(runtime.output_kind_names[stage_slot], actual_output_kind);
        accumulate_counters(runtime.output_totals[stage_slot], 1, total_values, original_zero_values, truncated_nonzero_values, final_zero_values);
        accumulate_counters(layer_stage_stats.output_counters, 1, total_values, original_zero_values, truncated_nonzero_values, final_zero_values);
    }

    if (modified) {
        ggml_backend_tensor_set(tensor, runtime.scratch.data(), 0, nbytes);
    }

    return true;
}

static bool process_derived_silu_tensor(common_swiglu_threshold_runtime & runtime, ggml_tensor * tensor) {
    GGML_ASSERT(is_swiglu_tensor(tensor));

    const char * actual_output_kind = is_swiglu_tensor(tensor) ? "swiglu" : (is_silu_tensor(tensor) ? "silu" : "");

    ggml_tensor * gate_tensor = tensor->src[0];
    if (gate_tensor == nullptr) {
        if (!runtime.warned_unsupported_type) {
            LOG_WRN("%s: missing source[0] for fused SwiGLU tensor %s; skipping SiLU threshold/stat collection\n", __func__, tensor->name);
            runtime.warned_unsupported_type = true;
        }
        return true;
    }

    const std::string_view name(tensor->name != nullptr ? tensor->name : "");
    const int layer = parse_layer_id(name);
    swiglu_threshold_layer_stats & layer_stats = ensure_layer_stats(runtime, layer);
    const common_swiglu_threshold_stage stage = runtime.current_stage;
    const size_t stage_slot = stage_index(stage);
    swiglu_threshold_stage_stats & layer_stage_stats = layer_stats.stage_stats[stage_slot];

    if (collect_channel_max_values(runtime)) {
        ensure_channel_max_storage(layer_stage_stats, tensor->ne[0]);
    }

    const size_t nbytes = ggml_nbytes(tensor);
    runtime.scratch.resize(nbytes);
    ggml_backend_tensor_get(tensor, runtime.scratch.data(), 0, nbytes);

    const size_t gate_nbytes = ggml_nbytes(gate_tensor);
    runtime.aux_scratch.resize(gate_nbytes);
    ggml_backend_tensor_get(gate_tensor, runtime.aux_scratch.data(), 0, gate_nbytes);

    uint64_t total_values = 0;
    uint64_t original_zero_values = 0;
    uint64_t truncated_nonzero_values = 0;
    uint64_t final_zero_values = 0;
    uint64_t output_original_zero_values = 0;
    uint64_t output_truncated_nonzero_values = 0;
    uint64_t output_final_zero_values = 0;
    bool modified = false;

    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                    const size_t offset =
                        static_cast<size_t>(i3) * tensor->nb[3] +
                        static_cast<size_t>(i2) * tensor->nb[2] +
                        static_cast<size_t>(i1) * tensor->nb[1] +
                        static_cast<size_t>(i0) * tensor->nb[0];
                    const size_t gate_offset =
                        static_cast<size_t>(i3) * gate_tensor->nb[3] +
                        static_cast<size_t>(i2) * gate_tensor->nb[2] +
                        static_cast<size_t>(i1) * gate_tensor->nb[1] +
                        static_cast<size_t>(i0) * gate_tensor->nb[0];

                    float gate_value = 0.0f;
                    if (!read_scalar(runtime.aux_scratch.data(), gate_tensor->type, gate_offset, gate_value)) {
                        if (!runtime.warned_unsupported_type) {
                            LOG_WRN("%s: unsupported fused SiLU source tensor type %s; skipping SiLU threshold/stat collection\n", __func__, ggml_type_name(gate_tensor->type));
                            runtime.warned_unsupported_type = true;
                        }
                        return true;
                    }

                    float output_value = 0.0f;
                    if (!read_scalar(runtime.scratch.data(), tensor->type, offset, output_value)) {
                        if (!runtime.warned_unsupported_type) {
                            LOG_WRN("%s: unsupported fused SwiGLU output tensor type %s; skipping SiLU threshold/stat collection\n", __func__, ggml_type_name(tensor->type));
                            runtime.warned_unsupported_type = true;
                        }
                        return true;
                    }

                    const float value = silu_transform(gate_value);
                    const bool original_zero = value == 0.0f;
                    const bool output_original_zero = output_value == 0.0f;
                    bool final_zero = original_zero;
                    bool output_final_zero = output_original_zero;
                    float tracked_value = 0.0f;
                    const bool tracked = tracked_value_for_target_kind(runtime.target_kind, value, tracked_value);
                    total_values++;
                    if (original_zero) {
                        original_zero_values++;
                    }

                    if (runtime.collect_enabled && tracked) {
                        record_histogram_sample(layer_stage_stats.abs_histogram, tracked_value);
                    }
                    if (collect_channel_max_values(runtime)) {
                        record_channel_max_value(layer_stage_stats, static_cast<int>(i0), value);
                    }
                    if (output_original_zero) {
                        output_original_zero_values++;
                    }

                    const float threshold = threshold_for_layer_channel(runtime, layer, static_cast<int>(i0));
                    if (!original_zero && runtime.apply_enabled && should_truncate_value(runtime.target_kind, value, threshold)) {
                        truncated_nonzero_values++;
                        if (!output_original_zero) {
                            output_truncated_nonzero_values++;
                        }
                        final_zero = true;
                        output_final_zero = true;
                        modified = true;
                        if (!write_scalar(runtime.scratch.data(), tensor->type, offset, 0.0f)) {
                            return true;
                        }
                    }

                    if (final_zero) {
                        final_zero_values++;
                    }
                    if (output_final_zero) {
                        output_final_zero_values++;
                    }
                }
            }
        }
    }

    accumulate_counters(runtime.totals[stage_slot], 1, total_values, original_zero_values, truncated_nonzero_values, final_zero_values);
    accumulate_counters(layer_stage_stats.counters, 1, total_values, original_zero_values, truncated_nonzero_values, final_zero_values);

    layer_stage_stats.has_output_counters = true;
    note_output_kind(layer_stage_stats.output_kind, actual_output_kind);
    note_output_kind(runtime.output_kind_names[stage_slot], actual_output_kind);
    accumulate_counters(runtime.output_totals[stage_slot], 1, total_values, output_original_zero_values, output_truncated_nonzero_values, output_final_zero_values);
    accumulate_counters(layer_stage_stats.output_counters, 1, total_values, output_original_zero_values, output_truncated_nonzero_values, output_final_zero_values);

    if (modified) {
        ggml_backend_tensor_set(tensor, runtime.scratch.data(), 0, nbytes);
    }

    return true;
}

static bool process_derived_silu_input_tensor(common_swiglu_threshold_runtime & runtime, ggml_tensor * tensor) {
    GGML_ASSERT(is_swiglu_tensor(tensor) || is_silu_tensor(tensor));

    ggml_tensor * gate_tensor = tensor->src[0];
    if (gate_tensor == nullptr) {
        if (!runtime.warned_unsupported_type) {
            LOG_WRN("%s: missing source[0] for %s tensor %s; skipping SiLU-input threshold/stat collection\n", __func__, tensor->name, tensor->name);
            runtime.warned_unsupported_type = true;
        }
        return true;
    }

    const char * actual_output_kind = is_swiglu_tensor(tensor) ? "swiglu" : (is_silu_tensor(tensor) ? "silu" : "");
    const std::string_view name(tensor->name != nullptr ? tensor->name : "");
    const int layer = parse_layer_id(name);
    swiglu_threshold_layer_stats & layer_stats = ensure_layer_stats(runtime, layer);
    const float threshold = layer_stats.threshold;
    const common_swiglu_threshold_stage stage = runtime.current_stage;
    const size_t stage_slot = stage_index(stage);
    swiglu_threshold_stage_stats & layer_stage_stats = layer_stats.stage_stats[stage_slot];

    const size_t nbytes = ggml_nbytes(tensor);
    runtime.scratch.resize(nbytes);
    ggml_backend_tensor_get(tensor, runtime.scratch.data(), 0, nbytes);

    const size_t gate_nbytes = ggml_nbytes(gate_tensor);
    runtime.aux_scratch.resize(gate_nbytes);
    ggml_backend_tensor_get(gate_tensor, runtime.aux_scratch.data(), 0, gate_nbytes);

    uint64_t total_values = 0;
    uint64_t original_zero_values = 0;
    uint64_t truncated_nonzero_values = 0;
    uint64_t final_zero_values = 0;
    uint64_t output_original_zero_values = 0;
    uint64_t output_truncated_nonzero_values = 0;
    uint64_t output_final_zero_values = 0;
    bool modified = false;

    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                    const size_t offset =
                        static_cast<size_t>(i3) * tensor->nb[3] +
                        static_cast<size_t>(i2) * tensor->nb[2] +
                        static_cast<size_t>(i1) * tensor->nb[1] +
                        static_cast<size_t>(i0) * tensor->nb[0];
                    const size_t gate_offset =
                        static_cast<size_t>(i3) * gate_tensor->nb[3] +
                        static_cast<size_t>(i2) * gate_tensor->nb[2] +
                        static_cast<size_t>(i1) * gate_tensor->nb[1] +
                        static_cast<size_t>(i0) * gate_tensor->nb[0];

                    float gate_value = 0.0f;
                    if (!read_scalar(runtime.aux_scratch.data(), gate_tensor->type, gate_offset, gate_value)) {
                        if (!runtime.warned_unsupported_type) {
                            LOG_WRN("%s: unsupported SiLU-input source tensor type %s; skipping SiLU-input threshold/stat collection\n", __func__, ggml_type_name(gate_tensor->type));
                            runtime.warned_unsupported_type = true;
                        }
                        return true;
                    }

                    float output_value = 0.0f;
                    if (!read_scalar(runtime.scratch.data(), tensor->type, offset, output_value)) {
                        if (!runtime.warned_unsupported_type) {
                            LOG_WRN("%s: unsupported SiLU-input output tensor type %s; skipping SiLU-input threshold/stat collection\n", __func__, ggml_type_name(tensor->type));
                            runtime.warned_unsupported_type = true;
                        }
                        return true;
                    }

                    const bool original_zero = gate_value == 0.0f;
                    const bool output_original_zero = output_value == 0.0f;
                    bool final_zero = original_zero;
                    bool output_final_zero = output_original_zero;
                    float tracked_value = 0.0f;
                    const bool tracked = tracked_value_for_target_kind(runtime.target_kind, gate_value, tracked_value);

                    total_values++;
                    if (original_zero) {
                        original_zero_values++;
                    }
                    if (runtime.collect_enabled && tracked) {
                        record_histogram_sample(layer_stage_stats.abs_histogram, tracked_value);
                    }
                    if (output_original_zero) {
                        output_original_zero_values++;
                    }

                    if (!original_zero && runtime.apply_enabled && should_truncate_value(runtime.target_kind, gate_value, threshold)) {
                        truncated_nonzero_values++;
                        if (!output_original_zero) {
                            output_truncated_nonzero_values++;
                        }
                        final_zero = true;
                        output_final_zero = true;
                        modified = true;
                        if (!write_scalar(runtime.scratch.data(), tensor->type, offset, 0.0f)) {
                            return true;
                        }
                    }

                    if (final_zero) {
                        final_zero_values++;
                    }
                    if (output_final_zero) {
                        output_final_zero_values++;
                    }
                }
            }
        }
    }

    accumulate_counters(runtime.totals[stage_slot], 1, total_values, original_zero_values, truncated_nonzero_values, final_zero_values);
    accumulate_counters(layer_stage_stats.counters, 1, total_values, original_zero_values, truncated_nonzero_values, final_zero_values);

    layer_stage_stats.has_output_counters = true;
    note_output_kind(layer_stage_stats.output_kind, actual_output_kind);
    note_output_kind(runtime.output_kind_names[stage_slot], actual_output_kind);
    accumulate_counters(runtime.output_totals[stage_slot], 1, total_values, output_original_zero_values, output_truncated_nonzero_values, output_final_zero_values);
    accumulate_counters(layer_stage_stats.output_counters, 1, total_values, output_original_zero_values, output_truncated_nonzero_values, output_final_zero_values);

    if (modified) {
        ggml_backend_tensor_set(tensor, runtime.scratch.data(), 0, nbytes);
    }

    return true;
}

static bool process_tensor(common_swiglu_threshold_runtime & runtime, ggml_tensor * tensor) {
    switch (runtime.target_kind) {
        case common_activation_threshold_kind::swiglu:
            return process_direct_tensor(runtime, tensor, false);
        case common_activation_threshold_kind::silu:
            if (is_silu_tensor(tensor)) {
                return process_direct_tensor(runtime, tensor, false);
            }
            if (is_swiglu_tensor(tensor)) {
                return process_derived_silu_tensor(runtime, tensor);
            }
            return true;
        case common_activation_threshold_kind::silu_input:
            if (is_silu_tensor(tensor) || is_swiglu_tensor(tensor)) {
                return process_derived_silu_input_tensor(runtime, tensor);
            }
            return true;
    }

    return true;
}

static bool swiglu_threshold_cb(struct ggml_tensor * tensor, bool ask, void * user_data) {
    auto * runtime = static_cast<common_swiglu_threshold_runtime *>(user_data);
    if (runtime == nullptr || runtime->current_stage == common_swiglu_threshold_stage::inactive) {
        return true;
    }

    if (ask) {
        return is_target_tensor(*runtime, tensor) ||
            (runtime->secondary_runtime != nullptr && is_target_tensor(*runtime->secondary_runtime, tensor));
    }

    if (!is_target_tensor(*runtime, tensor) &&
        (runtime->secondary_runtime == nullptr || !is_target_tensor(*runtime->secondary_runtime, tensor))) {
        return true;
    }

    common_swiglu_threshold_runtime * first = runtime;
    common_swiglu_threshold_runtime * second = runtime->secondary_runtime.get();
    if (second != nullptr && threshold_apply_priority(second->target_kind) < threshold_apply_priority(first->target_kind)) {
        std::swap(first, second);
    }

    if (first != nullptr && is_target_tensor(*first, tensor) && !process_tensor(*first, tensor)) {
        return false;
    }

    if (second != nullptr && is_target_tensor(*second, tensor) && !process_tensor(*second, tensor)) {
        return false;
    }

    return true;
}

static bool parse_config_file(
    const std::string & path,
    std::map<int, float> & thresholds,
    std::map<int, std::vector<float>> & channel_thresholds,
    std::string & error_message) {
    std::ifstream in(path);
    if (!in.is_open()) {
        error_message = "failed to open config file: " + path;
        return false;
    }

    enum class threshold_config_layout {
        unknown,
        layer_only,
        layer_channel,
    };

    threshold_config_layout layout = threshold_config_layout::unknown;
    std::string line;
    int line_number = 0;
    while (std::getline(in, line)) {
        line_number++;
        line = trim_copy(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        if (line == "layer,threshold") {
            if (layout == threshold_config_layout::layer_channel) {
                error_message = "invalid config header mix in " + path + ": both layer-only and layer-channel thresholds are present";
                return false;
            }
            layout = threshold_config_layout::layer_only;
            continue;
        }

        if (line == "layer,channel,threshold") {
            if (layout == threshold_config_layout::layer_only) {
                error_message = "invalid config header mix in " + path + ": both layer-only and layer-channel thresholds are present";
                return false;
            }
            layout = threshold_config_layout::layer_channel;
            continue;
        }

        std::vector<std::string> fields;
        size_t field_start = 0;
        while (field_start <= line.size()) {
            const size_t comma = line.find(',', field_start);
            if (comma == std::string::npos) {
                fields.push_back(trim_copy(line.substr(field_start)));
                break;
            }

            fields.push_back(trim_copy(line.substr(field_start, comma - field_start)));
            field_start = comma + 1;
        }

        if (fields.size() != 2 && fields.size() != 3) {
            error_message = "invalid config line " + std::to_string(line_number) + " in " + path + ": expected 'layer,threshold' or 'layer,channel,threshold'";
            return false;
        }

        if (fields.size() == 2 && layout == threshold_config_layout::layer_channel) {
            error_message = "invalid config line " + std::to_string(line_number) + " in " + path + ": mixed layer-only and layer-channel thresholds";
            return false;
        }

        if (fields.size() == 3 && layout == threshold_config_layout::layer_only) {
            error_message = "invalid config line " + std::to_string(line_number) + " in " + path + ": mixed layer-only and layer-channel thresholds";
            return false;
        }

        layout = fields.size() == 2 ? threshold_config_layout::layer_only : threshold_config_layout::layer_channel;

        const std::string & layer_text = fields[0];
        const std::string & threshold_text = fields.back();
        if (layer_text.empty() || threshold_text.empty()) {
            error_message = "invalid config line " + std::to_string(line_number) + " in " + path + ": empty layer or threshold";
            return false;
        }

        char * layer_end = nullptr;
        const long layer_value = std::strtol(layer_text.c_str(), &layer_end, 10);
        if (layer_end == nullptr || *layer_end != '\0' || layer_value < 0) {
            error_message = "invalid layer at line " + std::to_string(line_number) + " in " + path + ": " + layer_text;
            return false;
        }

        char * threshold_end = nullptr;
        const float threshold_value = std::strtof(threshold_text.c_str(), &threshold_end);
        if (threshold_end == nullptr || *threshold_end != '\0' || threshold_value <= 0.0f) {
            error_message = "invalid threshold at line " + std::to_string(line_number) + " in " + path + ": " + threshold_text;
            return false;
        }

        const int layer = static_cast<int>(layer_value);
        if (fields.size() == 2) {
            if (thresholds.find(layer) != thresholds.end()) {
                error_message = "duplicate layer threshold at line " + std::to_string(line_number) + " in " + path + ": " + layer_text;
                return false;
            }

            thresholds[layer] = threshold_value;
            continue;
        }

        const std::string & channel_text = fields[1];
        char * channel_end = nullptr;
        const long channel_value = std::strtol(channel_text.c_str(), &channel_end, 10);
        if (channel_end == nullptr || *channel_end != '\0' || channel_value < 0) {
            error_message = "invalid channel at line " + std::to_string(line_number) + " in " + path + ": " + channel_text;
            return false;
        }

        std::vector<float> & layer_channel_thresholds = channel_thresholds[layer];
        const size_t channel = static_cast<size_t>(channel_value);
        if (layer_channel_thresholds.size() <= channel) {
            layer_channel_thresholds.resize(channel + 1, 0.0f);
        }
        if (layer_channel_thresholds[channel] > 0.0f) {
            error_message = "duplicate channel threshold at line " + std::to_string(line_number) + " in " + path + ": layer=" + layer_text + ", channel=" + channel_text;
            return false;
        }

        layer_channel_thresholds[channel] = threshold_value;
    }

    return true;
}

static fs::path collect_summary_path(const common_swiglu_threshold_runtime & runtime) {
    return fs::path(runtime.collect_prefix + "_summary.csv");
}

static fs::path collect_hist_path(const common_swiglu_threshold_runtime & runtime) {
    return fs::path(runtime.collect_prefix + "_hist.csv");
}

static fs::path collect_channel_max_path(const common_swiglu_threshold_runtime & runtime) {
    return fs::path(runtime.collect_prefix + "_channel_max.csv");
}

static bool write_collect_outputs(const common_swiglu_threshold_runtime & runtime) {
    if (!runtime.collect_enabled || runtime.collect_prefix.empty()) {
        return true;
    }

    const fs::path summary_path = collect_summary_path(runtime);
    const fs::path hist_path = collect_hist_path(runtime);
    const bool write_channel_max = runtime.target_kind == common_activation_threshold_kind::silu;
    const fs::path channel_max_path = collect_channel_max_path(runtime);

    if (summary_path.has_parent_path()) {
        fs::create_directories(summary_path.parent_path());
    }
    if (hist_path.has_parent_path()) {
        fs::create_directories(hist_path.parent_path());
    }
    if (write_channel_max && channel_max_path.has_parent_path()) {
        fs::create_directories(channel_max_path.parent_path());
    }

    std::ofstream summary_out(summary_path);
    if (!summary_out.is_open()) {
        LOG_ERR("%s: failed to open collection summary path %s\n", __func__, summary_path.string().c_str());
        return false;
    }

    std::ofstream hist_out(hist_path);
    if (!hist_out.is_open()) {
        LOG_ERR("%s: failed to open collection hist path %s\n", __func__, hist_path.string().c_str());
        return false;
    }

    std::ofstream channel_max_out;
    if (write_channel_max) {
        channel_max_out.open(channel_max_path);
        if (!channel_max_out.is_open()) {
            LOG_ERR("%s: failed to open collection channel-max path %s\n", __func__, channel_max_path.string().c_str());
            return false;
        }
    }

    summary_out << "target_kind,stage,layer,threshold,total_values,original_zero_values,original_zero_ratio,nonzero_values,nonzero_ratio,tracked_kind,tracked_values,tracked_ratio,min_abs_nonzero,max_abs_nonzero,collect_log10_min,collect_log10_max,collect_bins,underflow_count,overflow_count,output_kind,output_total_values,output_original_zero_values,output_original_zero_ratio,output_nonzero_values,output_nonzero_ratio,output_truncated_nonzero_values,output_truncated_nonzero_ratio,output_final_zero_values,output_final_zero_ratio\n";
    hist_out << "target_kind,tracked_kind,stage,layer,bin_index,bin_lo,bin_hi,count\n";
    if (write_channel_max) {
        channel_max_out << "target_kind,stage,layer,channel,max_abs_activation\n";
    }

    const std::array<common_swiglu_threshold_stage, kStageCount> stages = {
        common_swiglu_threshold_stage::prefill,
        common_swiglu_threshold_stage::decode,
        common_swiglu_threshold_stage::ppl,
    };

    for (const auto & [layer, layer_stats] : runtime.layer_stats) {
        for (const auto stage : stages) {
            const swiglu_threshold_stage_stats & stage_stats = layer_stats.stage_stats[stage_index(stage)];
            const swiglu_threshold_counters & counters = stage_stats.counters;
            if (counters.total_values == 0) {
                continue;
            }

            const uint64_t nonzero_values = counters.total_values - counters.original_zero_values;
            const uint64_t tracked_values = stage_stats.abs_histogram.underflow_count + stage_stats.abs_histogram.overflow_count + std::accumulate(stage_stats.abs_histogram.bins.begin(), stage_stats.abs_histogram.bins.end(), uint64_t(0));
            const swiglu_threshold_counters & output_counters = stage_stats.output_counters;
            const uint64_t output_nonzero_values = output_counters.total_values >= output_counters.original_zero_values
                ? output_counters.total_values - output_counters.original_zero_values
                : 0;
            const char * output_kind = stage_stats.output_kind.empty() ? "" : stage_stats.output_kind.c_str();
            const swiglu_threshold_collect_histogram & histogram = stage_stats.abs_histogram;
            summary_out << target_kind_name(runtime.target_kind) << ','
                << stage_name(stage) << ','
                << layer << ','
                << format_double(layer_stats.threshold, 8) << ','
                << counters.total_values << ','
                << counters.original_zero_values << ','
                << format_double(safe_ratio(counters.original_zero_values, counters.total_values), 8) << ','
                << nonzero_values << ','
                << format_double(safe_ratio(nonzero_values, counters.total_values), 8) << ','
                << histogram_kind_name(histogram.kind) << ','
                << tracked_values << ','
                << format_double(safe_ratio(tracked_values, counters.total_values), 8) << ','
                << format_double(nonzero_values == 0 ? 0.0 : histogram.min_abs_nonzero, 8) << ','
                << format_double(nonzero_values == 0 ? 0.0 : histogram.max_abs_nonzero, 8) << ','
                << format_double(histogram.log10_min, 8) << ','
                << format_double(histogram.log10_max, 8) << ','
                << histogram.bins.size() << ','
                << histogram.underflow_count << ','
                << histogram.overflow_count << ','
                << output_kind << ','
                << output_counters.total_values << ','
                << output_counters.original_zero_values << ','
                << format_double(safe_ratio(output_counters.original_zero_values, output_counters.total_values), 8) << ','
                << output_nonzero_values << ','
                << format_double(safe_ratio(output_nonzero_values, output_counters.total_values), 8) << ','
                << output_counters.truncated_nonzero_values << ','
                << format_double(safe_ratio(output_counters.truncated_nonzero_values, output_counters.total_values), 8) << ','
                << output_counters.final_zero_values << ','
                << format_double(safe_ratio(output_counters.final_zero_values, output_counters.total_values), 8) << '\n';

            const double bin_width = histogram.bins.empty() ? 0.0 : (histogram.log10_max - histogram.log10_min) / static_cast<double>(histogram.bins.size());
            for (size_t index = 0; index < histogram.bins.size(); ++index) {
                const double bin_lo = histogram.log10_min + bin_width * static_cast<double>(index);
                const double bin_hi = bin_lo + bin_width;
                hist_out << target_kind_name(runtime.target_kind) << ','
                    << histogram_kind_name(histogram.kind) << ','
                    << stage_name(stage) << ','
                    << layer << ','
                    << index << ','
                    << format_double(bin_lo, 8) << ','
                    << format_double(bin_hi, 8) << ','
                    << histogram.bins[index] << '\n';
            }

            if (write_channel_max) {
                const auto & channel_max_values = stage_stats.channel_max_values;
                for (size_t channel = 0; channel < channel_max_values.size(); ++channel) {
                    if (!std::isfinite(channel_max_values[channel])) {
                        continue;
                    }

                    channel_max_out << target_kind_name(runtime.target_kind) << ','
                        << stage_name(stage) << ','
                        << layer << ','
                        << channel << ','
                        << format_double(channel_max_values[channel], 8) << '\n';
                }
            }
        }
    }

    LOG_INF("%s: wrote collection summary to %s\n", __func__, summary_path.string().c_str());
    LOG_INF("%s: wrote collection histogram to %s\n", __func__, hist_path.string().c_str());
    if (write_channel_max) {
        LOG_INF("%s: wrote collection channel maxima to %s\n", __func__, channel_max_path.string().c_str());
    }
    return true;
}

} // namespace

bool common_swiglu_threshold_preprocess_args(
    int argc,
    char ** argv,
    common_swiglu_threshold_options & options,
    std::vector<char *> & filtered_argv) {
    filtered_argv.clear();
    filtered_argv.reserve(static_cast<size_t>(argc));
    if (argc > 0) {
        filtered_argv.push_back(argv[0]);
    }

    for (int index = 1; index < argc; ++index) {
        const std::string_view arg = argv[index];
        auto consume_value = [&](std::string & dst) -> bool {
            if (index + 1 >= argc) {
                LOG_ERR("%s: missing value for %s\n", __func__, argv[index]);
                return false;
            }
            dst = argv[++index];
            return true;
        };
        auto consume_int_value = [&](int & dst, int minimum_value) -> bool {
            std::string text;
            if (!consume_value(text)) {
                return false;
            }
            if (!parse_int_text(text, minimum_value, dst)) {
                LOG_ERR("%s: invalid integer value for %s: %s\n", __func__, argv[index - 1], text.c_str());
                return false;
            }
            return true;
        };
        auto consume_float_value = [&](float & dst) -> bool {
            std::string text;
            if (!consume_value(text)) {
                return false;
            }
            if (!parse_float_text(text, dst)) {
                LOG_ERR("%s: invalid float value for %s: %s\n", __func__, argv[index - 1], text.c_str());
                return false;
            }
            return true;
        };
        auto consume_kind_value = [&](common_activation_threshold_kind & dst) -> bool {
            std::string text;
            if (!consume_value(text)) {
                return false;
            }
            if (!parse_target_kind_text(text, dst)) {
                LOG_ERR("%s: invalid activation threshold kind for %s: %s (expected swiglu, silu, or silu_input)\n", __func__, argv[index - 1], text.c_str());
                return false;
            }
            return true;
        };

        if (arg == "--swiglu-threshold-kind") {
            if (!consume_kind_value(options.target_kind)) {
                return false;
            }
            continue;
        }

        if (arg == "--swiglu-threshold-secondary-kind") {
            if (!consume_kind_value(options.secondary_target_kind)) {
                return false;
            }
            options.has_secondary_target_kind = true;
            continue;
        }

        if (arg == "--swiglu-threshold-config") {
            if (!consume_value(options.config_path)) {
                return false;
            }
            continue;
        }

        if (arg == "--swiglu-threshold-secondary-config") {
            if (!consume_value(options.secondary_config_path)) {
                return false;
            }
            continue;
        }

        if (arg == "--swiglu-threshold-collect") {
            if (!consume_value(options.collect_prefix)) {
                return false;
            }
            continue;
        }

        if (arg == "--swiglu-threshold-secondary-collect") {
            if (!consume_value(options.secondary_collect_prefix)) {
                return false;
            }
            continue;
        }

        if (arg == "--swiglu-threshold-collect-bins") {
            if (!consume_int_value(options.collect_bins, 1)) {
                return false;
            }
            continue;
        }

        if (arg == "--swiglu-threshold-collect-log-min") {
            if (!consume_float_value(options.collect_log10_min)) {
                return false;
            }
            continue;
        }

        if (arg == "--swiglu-threshold-collect-log-max") {
            if (!consume_float_value(options.collect_log10_max)) {
                return false;
            }
            continue;
        }

        if (arg == "--swiglu-threshold-report") {
            if (!consume_value(options.report_path)) {
                return false;
            }
            continue;
        }

        if (arg == "--swiglu-threshold-secondary-report") {
            if (!consume_value(options.secondary_report_path)) {
                return false;
            }
            continue;
        }

        constexpr std::string_view kKindEq = "--swiglu-threshold-kind=";
        constexpr std::string_view kSecondaryKindEq = "--swiglu-threshold-secondary-kind=";
        constexpr std::string_view kConfigEq = "--swiglu-threshold-config=";
        constexpr std::string_view kSecondaryConfigEq = "--swiglu-threshold-secondary-config=";
        constexpr std::string_view kCollectEq = "--swiglu-threshold-collect=";
        constexpr std::string_view kSecondaryCollectEq = "--swiglu-threshold-secondary-collect=";
        constexpr std::string_view kCollectBinsEq = "--swiglu-threshold-collect-bins=";
        constexpr std::string_view kCollectLogMinEq = "--swiglu-threshold-collect-log-min=";
        constexpr std::string_view kCollectLogMaxEq = "--swiglu-threshold-collect-log-max=";
        constexpr std::string_view kReportEq = "--swiglu-threshold-report=";
        constexpr std::string_view kSecondaryReportEq = "--swiglu-threshold-secondary-report=";
        if (arg.size() > kKindEq.size() && arg.substr(0, kKindEq.size()) == kKindEq) {
            const std::string text(arg.substr(kKindEq.size()));
            if (!parse_target_kind_text(text, options.target_kind)) {
                LOG_ERR("%s: invalid activation threshold kind for %s (expected swiglu, silu, or silu_input)\n", __func__, argv[index]);
                return false;
            }
            continue;
        }
        if (arg.size() > kSecondaryKindEq.size() && arg.substr(0, kSecondaryKindEq.size()) == kSecondaryKindEq) {
            const std::string text(arg.substr(kSecondaryKindEq.size()));
            if (!parse_target_kind_text(text, options.secondary_target_kind)) {
                LOG_ERR("%s: invalid activation threshold kind for %s (expected swiglu, silu, or silu_input)\n", __func__, argv[index]);
                return false;
            }
            options.has_secondary_target_kind = true;
            continue;
        }
        if (arg.size() > kConfigEq.size() && arg.substr(0, kConfigEq.size()) == kConfigEq) {
            options.config_path = std::string(arg.substr(kConfigEq.size()));
            continue;
        }
        if (arg.size() > kSecondaryConfigEq.size() && arg.substr(0, kSecondaryConfigEq.size()) == kSecondaryConfigEq) {
            options.secondary_config_path = std::string(arg.substr(kSecondaryConfigEq.size()));
            continue;
        }
        if (arg.size() > kCollectEq.size() && arg.substr(0, kCollectEq.size()) == kCollectEq) {
            options.collect_prefix = std::string(arg.substr(kCollectEq.size()));
            continue;
        }
        if (arg.size() > kSecondaryCollectEq.size() && arg.substr(0, kSecondaryCollectEq.size()) == kSecondaryCollectEq) {
            options.secondary_collect_prefix = std::string(arg.substr(kSecondaryCollectEq.size()));
            continue;
        }
        if (arg.size() > kCollectBinsEq.size() && arg.substr(0, kCollectBinsEq.size()) == kCollectBinsEq) {
            const std::string text(arg.substr(kCollectBinsEq.size()));
            if (!parse_int_text(text, 1, options.collect_bins)) {
                LOG_ERR("%s: invalid integer value for %s\n", __func__, argv[index]);
                return false;
            }
            continue;
        }
        if (arg.size() > kCollectLogMinEq.size() && arg.substr(0, kCollectLogMinEq.size()) == kCollectLogMinEq) {
            const std::string text(arg.substr(kCollectLogMinEq.size()));
            if (!parse_float_text(text, options.collect_log10_min)) {
                LOG_ERR("%s: invalid float value for %s\n", __func__, argv[index]);
                return false;
            }
            continue;
        }
        if (arg.size() > kCollectLogMaxEq.size() && arg.substr(0, kCollectLogMaxEq.size()) == kCollectLogMaxEq) {
            const std::string text(arg.substr(kCollectLogMaxEq.size()));
            if (!parse_float_text(text, options.collect_log10_max)) {
                LOG_ERR("%s: invalid float value for %s\n", __func__, argv[index]);
                return false;
            }
            continue;
        }
        if (arg.size() > kReportEq.size() && arg.substr(0, kReportEq.size()) == kReportEq) {
            options.report_path = std::string(arg.substr(kReportEq.size()));
            continue;
        }
        if (arg.size() > kSecondaryReportEq.size() && arg.substr(0, kSecondaryReportEq.size()) == kSecondaryReportEq) {
            options.secondary_report_path = std::string(arg.substr(kSecondaryReportEq.size()));
            continue;
        }

        filtered_argv.push_back(argv[index]);
    }

    return true;
}

bool common_swiglu_threshold_requested(const common_swiglu_threshold_options & options) {
    return threshold_channel_requested(options.config_path, options.collect_prefix) ||
        threshold_channel_requested(options.secondary_config_path, options.secondary_collect_prefix);
}

static std::shared_ptr<common_swiglu_threshold_runtime> common_swiglu_threshold_init_single(
    common_activation_threshold_kind target_kind,
    const std::string & config_path,
    const std::string & report_path,
    const std::string & collect_prefix,
    int collect_bins,
    float collect_log10_min,
    float collect_log10_max,
    std::string & error_message) {
    if (!threshold_channel_requested(config_path, collect_prefix)) {
        return nullptr;
    }

    auto runtime = std::make_shared<common_swiglu_threshold_runtime>();
    runtime->target_kind = target_kind;
    runtime->config_path = config_path;
    runtime->report_path = report_path;
    runtime->collect_prefix = collect_prefix;
    runtime->collect_bins = collect_bins;
    runtime->collect_log10_min = collect_log10_min;
    runtime->collect_log10_max = collect_log10_max;
    runtime->apply_enabled = !config_path.empty();
    runtime->collect_enabled = !collect_prefix.empty();

    if (runtime->collect_enabled) {
        if (runtime->collect_bins <= 0) {
            error_message = std::string("invalid ") + target_kind_name(runtime->target_kind) + " collection bin count: " + std::to_string(runtime->collect_bins);
            return nullptr;
        }

        if (!(runtime->collect_log10_max > runtime->collect_log10_min)) {
            error_message = std::string("invalid ") + target_kind_name(runtime->target_kind) + " collection log10 range: expected max > min";
            return nullptr;
        }
    }

    if (runtime->apply_enabled) {
        if (!parse_config_file(config_path, runtime->thresholds, runtime->channel_thresholds, error_message)) {
            return nullptr;
        }

        if (runtime->thresholds.empty() && runtime->channel_thresholds.empty()) {
            error_message = std::string(target_kind_name(runtime->target_kind)) + " threshold config is empty: " + config_path;
            return nullptr;
        }

        if (!runtime->channel_thresholds.empty()) {
            size_t channel_threshold_count = 0;
            for (const auto & [layer, values] : runtime->channel_thresholds) {
                GGML_UNUSED(layer);
                channel_threshold_count += std::count_if(values.begin(), values.end(), [](float value) { return value > 0.0f; });
            }

            LOG_INF("%s: loaded %zu %s channel thresholds across %zu layers from %s\n",
                __func__,
                channel_threshold_count,
                target_kind_name(runtime->target_kind),
                runtime->channel_thresholds.size(),
                runtime->config_path.c_str());
        } else {
            LOG_INF("%s: loaded %zu layer %s thresholds from %s\n", __func__, runtime->thresholds.size(), target_kind_name(runtime->target_kind), runtime->config_path.c_str());
        }
    }

    if (runtime->collect_enabled) {
        LOG_INF("%s: enabled %s collection at %s (bins=%d, log10 range=[%s, %s])\n",
            __func__,
            target_kind_name(runtime->target_kind),
            runtime->collect_prefix.c_str(),
            runtime->collect_bins,
            format_double(runtime->collect_log10_min, 4).c_str(),
            format_double(runtime->collect_log10_max, 4).c_str());
    }

    return runtime;
}

std::shared_ptr<common_swiglu_threshold_runtime> common_swiglu_threshold_init(
    const common_swiglu_threshold_options & options,
    std::string & error_message) {
    if (!common_swiglu_threshold_requested(options)) {
        return nullptr;
    }

    bool primary_requested = threshold_channel_requested(options.config_path, options.collect_prefix);
    bool secondary_requested = threshold_channel_requested(options.secondary_config_path, options.secondary_collect_prefix);

    common_activation_threshold_kind primary_kind = options.target_kind;
    std::string primary_config_path = options.config_path;
    std::string primary_report_path = options.report_path;
    std::string primary_collect_prefix = options.collect_prefix;

    if (!primary_requested && secondary_requested) {
        if (!options.has_secondary_target_kind) {
            error_message = "secondary threshold channel requires --swiglu-threshold-secondary-kind";
            return nullptr;
        }

        primary_kind = options.secondary_target_kind;
        primary_config_path = options.secondary_config_path;
        primary_report_path = options.secondary_report_path;
        primary_collect_prefix = options.secondary_collect_prefix;
        primary_requested = true;
        secondary_requested = false;
    }

    auto runtime = common_swiglu_threshold_init_single(
        primary_kind,
        primary_config_path,
        primary_report_path,
        primary_collect_prefix,
        options.collect_bins,
        options.collect_log10_min,
        options.collect_log10_max,
        error_message);
    if (!runtime) {
        return nullptr;
    }

    if (secondary_requested) {
        if (!options.has_secondary_target_kind) {
            error_message = "secondary threshold channel requires --swiglu-threshold-secondary-kind";
            return nullptr;
        }

        if (!is_supported_dual_output_pair(primary_kind, options.secondary_target_kind)) {
            error_message = "simultaneous threshold application currently supports only the swiglu+silu pair";
            return nullptr;
        }

        runtime->secondary_runtime = common_swiglu_threshold_init_single(
            options.secondary_target_kind,
            options.secondary_config_path,
            options.secondary_report_path,
            options.secondary_collect_prefix,
            options.collect_bins,
            options.collect_log10_min,
            options.collect_log10_max,
            error_message);
        if (!runtime->secondary_runtime) {
            return nullptr;
        }
    }

    return runtime;
}

bool common_swiglu_threshold_is_enabled(const common_swiglu_threshold_runtime * runtime) {
    return runtime != nullptr;
}

void common_swiglu_threshold_attach(common_params & params, common_swiglu_threshold_runtime * runtime) {
    if (runtime == nullptr) {
        return;
    }

    params.cb_eval = swiglu_threshold_cb;
    params.cb_eval_user_data = runtime;
}

void common_swiglu_threshold_begin(common_swiglu_threshold_runtime * runtime, common_swiglu_threshold_stage stage, int active_tokens) {
    if (runtime == nullptr) {
        return;
    }

    runtime->current_stage = stage;
    runtime->active_tokens = active_tokens;
    common_swiglu_threshold_begin(runtime->secondary_runtime.get(), stage, active_tokens);
}

void common_swiglu_threshold_end(common_swiglu_threshold_runtime * runtime) {
    if (runtime == nullptr) {
        return;
    }

    runtime->current_stage = common_swiglu_threshold_stage::inactive;
    runtime->active_tokens = 0;
    common_swiglu_threshold_end(runtime->secondary_runtime.get());
}

bool common_swiglu_threshold_write_report(const common_swiglu_threshold_runtime * runtime, const std::string & tool_name) {
    if (runtime == nullptr) {
        return true;
    }

    auto print_stage_summary = [&](common_swiglu_threshold_stage stage) {
        const swiglu_threshold_counters & totals = runtime->totals[stage_index(stage)];
        if (totals.total_values == 0) {
            return;
        }

        LOG_INF("%s-threshold[%s]: tensors=%llu total=%llu original_zero=%s truncated_nonzero=%s final_zero=%s\n",
            target_kind_name(runtime->target_kind),
            stage_name(stage),
            static_cast<unsigned long long>(totals.tensor_count),
            static_cast<unsigned long long>(totals.total_values),
            format_double(safe_ratio(totals.original_zero_values, totals.total_values), 6).c_str(),
            format_double(safe_ratio(totals.truncated_nonzero_values, totals.total_values), 6).c_str(),
            format_double(safe_ratio(totals.final_zero_values, totals.total_values), 6).c_str());

        const swiglu_threshold_counters & output_totals = runtime->output_totals[stage_index(stage)];
        if (output_totals.total_values > 0) {
            const char * output_kind = runtime->output_kind_names[stage_index(stage)].empty() ? "output" : runtime->output_kind_names[stage_index(stage)].c_str();
            LOG_INF("%s-threshold[%s][%s-output]: tensors=%llu total=%llu original_zero=%s truncated_nonzero=%s final_zero=%s\n",
                target_kind_name(runtime->target_kind),
                stage_name(stage),
                output_kind,
                static_cast<unsigned long long>(output_totals.tensor_count),
                static_cast<unsigned long long>(output_totals.total_values),
                format_double(safe_ratio(output_totals.original_zero_values, output_totals.total_values), 6).c_str(),
                format_double(safe_ratio(output_totals.truncated_nonzero_values, output_totals.total_values), 6).c_str(),
                format_double(safe_ratio(output_totals.final_zero_values, output_totals.total_values), 6).c_str());
        }
    };

    print_stage_summary(common_swiglu_threshold_stage::prefill);
    print_stage_summary(common_swiglu_threshold_stage::decode);
    print_stage_summary(common_swiglu_threshold_stage::ppl);

    if (!runtime->report_path.empty()) {
        try {
            const fs::path report_path(runtime->report_path);
            if (report_path.has_parent_path()) {
                fs::create_directories(report_path.parent_path());
            }

            std::ofstream out(report_path);
            if (!out.is_open()) {
                LOG_ERR("%s: failed to open report path %s\n", __func__, runtime->report_path.c_str());
                return false;
            }

            out << "tool,target_kind,config_path,stage,layer,threshold,threshold_mode,channel_threshold_count,channel_threshold_min,channel_threshold_mean,channel_threshold_max,tensor_count,total_values,original_zero_values,original_zero_ratio,truncated_nonzero_values,truncated_nonzero_ratio,final_zero_values,final_zero_ratio,output_kind,output_tensor_count,output_total_values,output_original_zero_values,output_original_zero_ratio,output_truncated_nonzero_values,output_truncated_nonzero_ratio,output_final_zero_values,output_final_zero_ratio\n";

            auto write_counter_row = [&](common_swiglu_threshold_stage stage, int layer, const swiglu_threshold_layer_stats * layer_stats, const swiglu_threshold_counters & counters, const swiglu_threshold_counters & output_counters, std::string_view output_kind) {
                if (counters.total_values == 0) {
                    return;
                }

                const float threshold = layer_stats != nullptr ? layer_stats->threshold : 0.0f;
                const char * threshold_mode = "";
                size_t channel_threshold_count = 0;
                float channel_threshold_min = 0.0f;
                float channel_threshold_mean = 0.0f;
                float channel_threshold_max = 0.0f;
                if (layer_stats != nullptr) {
                    if (layer_stats->has_channel_thresholds) {
                        threshold_mode = "channel";
                        channel_threshold_count = layer_stats->channel_threshold_count;
                        channel_threshold_min = layer_stats->channel_threshold_min;
                        channel_threshold_mean = layer_stats->channel_threshold_mean;
                        channel_threshold_max = layer_stats->channel_threshold_max;
                    } else if (threshold > 0.0f) {
                        threshold_mode = "layer";
                    }
                }

                out << csv_escape(tool_name) << ','
                    << target_kind_name(runtime->target_kind) << ','
                    << csv_escape(runtime->config_path) << ','
                    << stage_name(stage) << ','
                    << layer << ','
                    << format_double(threshold, 8) << ','
                    << threshold_mode << ','
                    << channel_threshold_count << ','
                    << format_double(channel_threshold_min, 8) << ','
                    << format_double(channel_threshold_mean, 8) << ','
                    << format_double(channel_threshold_max, 8) << ','
                    << counters.tensor_count << ','
                    << counters.total_values << ','
                    << counters.original_zero_values << ','
                    << format_double(safe_ratio(counters.original_zero_values, counters.total_values), 8) << ','
                    << counters.truncated_nonzero_values << ','
                    << format_double(safe_ratio(counters.truncated_nonzero_values, counters.total_values), 8) << ','
                    << counters.final_zero_values << ','
                    << format_double(safe_ratio(counters.final_zero_values, counters.total_values), 8) << ','
                    << output_kind << ','
                    << output_counters.tensor_count << ','
                    << output_counters.total_values << ','
                    << output_counters.original_zero_values << ','
                    << format_double(safe_ratio(output_counters.original_zero_values, output_counters.total_values), 8) << ','
                    << output_counters.truncated_nonzero_values << ','
                    << format_double(safe_ratio(output_counters.truncated_nonzero_values, output_counters.total_values), 8) << ','
                    << output_counters.final_zero_values << ','
                    << format_double(safe_ratio(output_counters.final_zero_values, output_counters.total_values), 8) << '\n';
            };

            const std::array<common_swiglu_threshold_stage, kStageCount> stages = {
                common_swiglu_threshold_stage::prefill,
                common_swiglu_threshold_stage::decode,
                common_swiglu_threshold_stage::ppl,
            };

            for (const auto stage : stages) {
                write_counter_row(stage, -1, nullptr, runtime->totals[stage_index(stage)], runtime->output_totals[stage_index(stage)], runtime->output_kind_names[stage_index(stage)]);
            }

            for (const auto & [layer, stats] : runtime->layer_stats) {
                for (const auto stage : stages) {
                    const swiglu_threshold_stage_stats & stage_stats = stats.stage_stats[stage_index(stage)];
                    write_counter_row(stage, layer, &stats, stage_stats.counters, stage_stats.output_counters, stage_stats.output_kind);
                }
            }

            LOG_INF("%s: wrote report to %s\n", __func__, runtime->report_path.c_str());
        } catch (const std::exception & exc) {
            LOG_ERR("%s: failed to write report %s: %s\n", __func__, runtime->report_path.c_str(), exc.what());
            return false;
        }
    }

    try {
        if (!write_collect_outputs(*runtime)) {
            return false;
        }
    } catch (const std::exception & exc) {
        LOG_ERR("%s: failed to write collection outputs for %s: %s\n", __func__, runtime->collect_prefix.c_str(), exc.what());
        return false;
    }

    return common_swiglu_threshold_write_report(runtime->secondary_runtime.get(), tool_name);
}
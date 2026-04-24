#pragma once

#include "common.h"

#include <memory>
#include <string>
#include <vector>

enum class common_swiglu_threshold_stage {
    inactive = 0,
    prefill,
    decode,
    ppl,
};

enum class common_activation_threshold_kind {
    swiglu = 0,
    silu,
    silu_input,
};

struct common_swiglu_threshold_options {
    common_activation_threshold_kind target_kind = common_activation_threshold_kind::swiglu;
    std::string config_path;
    std::string report_path;
    std::string collect_prefix;
    common_activation_threshold_kind secondary_target_kind = common_activation_threshold_kind::swiglu;
    std::string secondary_config_path;
    std::string secondary_report_path;
    std::string secondary_collect_prefix;
    bool has_secondary_target_kind = false;
    int collect_bins = 256;
    float collect_log10_min = -12.0f;
    float collect_log10_max = 2.0f;
};

struct common_swiglu_threshold_runtime;

bool common_swiglu_threshold_preprocess_args(
    int argc,
    char ** argv,
    common_swiglu_threshold_options & options,
    std::vector<char *> & filtered_argv);

bool common_swiglu_threshold_requested(const common_swiglu_threshold_options & options);

std::shared_ptr<common_swiglu_threshold_runtime> common_swiglu_threshold_init(
    const common_swiglu_threshold_options & options,
    std::string & error_message);

bool common_swiglu_threshold_is_enabled(const common_swiglu_threshold_runtime * runtime);

void common_swiglu_threshold_attach(common_params & params, common_swiglu_threshold_runtime * runtime);

void common_swiglu_threshold_begin(common_swiglu_threshold_runtime * runtime, common_swiglu_threshold_stage stage, int active_tokens);

void common_swiglu_threshold_end(common_swiglu_threshold_runtime * runtime);

bool common_swiglu_threshold_write_report(const common_swiglu_threshold_runtime * runtime, const std::string & tool_name);
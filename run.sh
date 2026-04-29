#!/usr/bin/env bash
set -euo pipefail

# Local experiment driver for activation-threshold and profiler workflows.
# Detailed documentation lives in docs/development/run-sh-activation-workflow.md.

readonly SCRIPT_NAME="$(basename "$0")"
readonly RUN_DOC_PATH="docs/development/run-sh-activation-workflow.md"

die() {
  echo "${SCRIPT_NAME}: $*" >&2
  exit 1
}

print_help() {
  cat <<EOF
Usage:
  bash run.sh
  RUN_KIND=swiglu-collect CASE_FILTER=Llama-3.2-1B bash run.sh
  RUN_KIND=activation-flow SWIGLU_THRESHOLD_KIND=silu CASE_FILTER=Qwen-3-1.7B SKIP_BUILD=0 bash run.sh
  RUN_KIND=activation-flow SWIGLU_THRESHOLD_KIND=swiglu+silu CASE_FILTER=Qwen-3-1.7B SKIP_BUILD=0 bash run.sh
  bash run.sh --help

Supported RUN_KIND values:
  cli
  perplexity
  decode-stats
  swiglu-collect         (aliases: collect, activation-collect)
  swiglu-generate        (aliases: generate, activation-generate)
  activation-flow        (alias: flow)

High-value environment variables:
  CASE_FILTER                 Substring match over case name / case slug.
  RUN_KIND                    Execution mode.
  SIM_Q8Q8                    Enable symmetric Q8/Q8 replay with int8 power-of-2 block scale.
  SIM_Q8Q8_SRC0_BLOCK         src0 block size for Q8/Q8 replay.
  SIM_Q8Q8_SRC1_BLOCK         src1 block size for Q8/Q8 replay.
  SIM_Q4Q6_SRC1_QMODE         src1 replay mode: 0=symmetric Q6, 1=asymmetric Q6+zp, 2=logarithmic Q6-exp.
  SIM_Q4Q6_SRC1_LOG_STEP      Logarithmic src1 exponent divisor for Q6-exp replay: exponent uses q/step.
  SWIGLU_THRESHOLD_KIND       Activation target: swiglu, silu, silu_input, or swiglu+silu.
  SWIGLU_THRESHOLD_ENABLE     Enable threshold runtime for cli/perplexity.
  SWIGLU_THRESHOLD_PROFILE    Threshold profile name or generated.
  SWIGLU_GENERATE_MODE        target-profile or channel-max.
  SWIGLU_CHANNEL_THRESHOLD_RATIO  For channel-max mode: per-channel threshold = absolute channel max * ratio.
  SKIP_BUILD                  1 = reuse existing build, 0 = rebuild.
  BUILD_DIR                   Build tree, default: build.
  FLOW_STEPS                  For activation-flow, default: collect,generate,perplexity,cli.
  FLOW_REUSE_ARTIFACTS        For activation-flow, skip completed steps when set to 1.
  FLOW_DECODE_N_PREDICT       Finite token count used by flow cli/collect when case N_PREDICT=-1.

Detailed reference:
  ${RUN_DOC_PATH}
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || "${RUN_HELP:-0}" == "1" ]]; then
  print_help
  exit 0
fi

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_MODEL="${MODEL:-models/Qwen/Qwen3-1.7B-Base-f16.gguf}"
DEFAULT_DATA="${DATA:-models/hf/wiki.test.raw}"
DEFAULT_OUT_DIR="${OUT_DIR:-kv_dump_logs}"
DEFAULT_PROMPT="${PROMPT:-你好，请简要介绍一下KV cache。}"

DEFAULT_CTX="${CTX:-4096}"
DEFAULT_THREADS="${THREADS:-$(nproc)}"
DEFAULT_N_PREDICT="${N_PREDICT:-128}"
DEFAULT_SEQ_ID="${SEQ_ID:-0}"
DEFAULT_BATCH="${BATCH:-4096}"
DEFAULT_UBATCH="${UBATCH:-4096}"
DEFAULT_STRIDE="${STRIDE:-0}"

# Compile-time simulation switches
# GGML_SIM_MATMUL_OUT_MODE:
#   0 => FP8 output QDQ (E4M3/E3M4/E2M5 selected by SIM_FP8_LAYOUT, gated by GGML_SIM_FP8E4M3)
#   1 => BF16 round-trip output simulation (F32 -> BF16 -> F32)
DEFAULT_SIM_FP8="${SIM_FP8:-1}"
DEFAULT_SIM_FP_FORMAT="${SIM_FP_FORMAT:-8}"
# FP8 sub-format (effective only when SIM_FP_FORMAT=8):
#   0=E4M3, 1=E3M4, 2=E3M4_NO_SUBNORM, 3=E2M5, 4=E2M5_NO_SUBNORM
DEFAULT_SIM_FP8_LAYOUT="${SIM_FP8_LAYOUT:-0}"
DEFAULT_SIM_FP8_APPLY_SRC0="${SIM_FP8_APPLY_SRC0:-1}"
DEFAULT_SIM_FP8_APPLY_SRC1="${SIM_FP8_APPLY_SRC1:-1}"
# Legacy single switch, kept for compatibility with existing local workflows.
DEFAULT_SIM_FP8_SCALE_TYPE="${SIM_FP8_SCALE_TYPE:-1}"
DEFAULT_SIM_FP8_SCALE_TYPE_IN="${SIM_FP8_SCALE_TYPE_IN:-${DEFAULT_SIM_FP8_SCALE_TYPE}}"
DEFAULT_SIM_FP8_SCALE_TYPE_OUT="${SIM_FP8_SCALE_TYPE_OUT:-${DEFAULT_SIM_FP8_SCALE_TYPE}}"
DEFAULT_SIM_FP8_BLOCK="${SIM_FP8_BLOCK:-16}"
DEFAULT_SIM_Q4Q6="${SIM_Q4Q6:-0}"
DEFAULT_SIM_Q4Q6_APPLY_SRC0="${SIM_Q4Q6_APPLY_SRC0:-1}"
DEFAULT_SIM_Q4Q6_APPLY_SRC1="${SIM_Q4Q6_APPLY_SRC1:-1}"
DEFAULT_SIM_Q4Q6_SRC0_BLOCK="${SIM_Q4Q6_SRC0_BLOCK:-16}"
DEFAULT_SIM_Q4Q6_SRC1_BLOCK="${SIM_Q4Q6_SRC1_BLOCK:-16}"
DEFAULT_SIM_Q4Q6_SRC1_QMODE="${SIM_Q4Q6_SRC1_QMODE:-0}"
DEFAULT_SIM_Q4Q6_SRC1_LOG_STEP="${SIM_Q4Q6_SRC1_LOG_STEP:-1}"
DEFAULT_SIM_Q8Q8="${SIM_Q8Q8:-0}"
DEFAULT_SIM_Q8Q8_APPLY_SRC0="${SIM_Q8Q8_APPLY_SRC0:-1}"
DEFAULT_SIM_Q8Q8_APPLY_SRC1="${SIM_Q8Q8_APPLY_SRC1:-1}"
DEFAULT_SIM_Q8Q8_SRC0_BLOCK="${SIM_Q8Q8_SRC0_BLOCK:-16}"
DEFAULT_SIM_Q8Q8_SRC1_BLOCK="${SIM_Q8Q8_SRC1_BLOCK:-16}"
DEFAULT_SIM_MATMUL_OUT_MODE="${SIM_MATMUL_OUT_MODE:-1}"
DEFAULT_SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE="${SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE:-0}"
DEFAULT_SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP="${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP:--3}"
DEFAULT_SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP="${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP:--1}"
DEFAULT_FP8_SIM_STATS_SAMPLE="${FP8_SIM_STATS_SAMPLE:-100}"

# Reduction-product profiler switches
DEFAULT_REDUCTION_PROD_PROFILE="${REDUCTION_PROD_PROFILE:-0}"
DEFAULT_REDUCTION_PROD_PROFILE_BINS="${REDUCTION_PROD_PROFILE_BINS:-256}"
DEFAULT_REDUCTION_PROD_PROFILE_HIST_MIN_LOG2="${REDUCTION_PROD_PROFILE_HIST_MIN_LOG2:--128}"
DEFAULT_REDUCTION_PROD_PROFILE_HIST_MAX_LOG2="${REDUCTION_PROD_PROFILE_HIST_MAX_LOG2:-128}"
DEFAULT_REDUCTION_PROD_PROFILE_SAMPLE_RATE="${REDUCTION_PROD_PROFILE_SAMPLE_RATE:-1000}"
DEFAULT_REDUCTION_PROD_BLOCK_DROP_LOG2_N="${REDUCTION_PROD_BLOCK_DROP_LOG2_N:-10}"
DEFAULT_REDUCTION_PROD_PROFILE_MAX_SAMPLES="${REDUCTION_PROD_PROFILE_MAX_SAMPLES:-2000}"
DEFAULT_REDUCTION_PROD_PROFILE_PREFIX="${REDUCTION_PROD_PROFILE_PREFIX:-reduction_prod_profile}"

DEFAULT_DUMP_DOT="${DUMP_DOT:-0}"
DEFAULT_CASE_FILTER="${CASE_FILTER:-Llama-3.2-1B}"

# Historical prefix kept for compatibility. The activation kind is controlled by
# SWIGLU_THRESHOLD_KIND, so the same knobs now serve SwiGLU, SiLU-output,
# SiLU-input gate truncation, and the combined SwiGLU+SiLU output path.
DEFAULT_SWIGLU_THRESHOLD_ENABLE="${SWIGLU_THRESHOLD_ENABLE:-1}"
DEFAULT_SWIGLU_THRESHOLD_KIND="${SWIGLU_THRESHOLD_KIND:-swiglu}"
DEFAULT_SWIGLU_THRESHOLD_PROFILE="${SWIGLU_THRESHOLD_PROFILE:-generated}"
DEFAULT_SWIGLU_THRESHOLD_CONFIG="${SWIGLU_THRESHOLD_CONFIG:-auto}"
DEFAULT_SWIGLU_COLLECT_PREFIX="${SWIGLU_COLLECT_PREFIX:-auto}"
DEFAULT_SWIGLU_COLLECT_BINS="${SWIGLU_COLLECT_BINS:-256}"
DEFAULT_SWIGLU_COLLECT_LOG10_MIN="${SWIGLU_COLLECT_LOG10_MIN:--12}"
DEFAULT_SWIGLU_COLLECT_LOG10_MAX="${SWIGLU_COLLECT_LOG10_MAX:-2}"
DEFAULT_SWIGLU_TARGET_PROFILE_KIND="${SWIGLU_TARGET_PROFILE_KIND:-minimal}"
DEFAULT_SWIGLU_TARGET_PROFILE="${SWIGLU_TARGET_PROFILE:-auto}"
DEFAULT_SWIGLU_TARGET_SCALE="${SWIGLU_TARGET_SCALE:-20.0}"
DEFAULT_SWIGLU_GENERATED_CONFIG="${SWIGLU_GENERATED_CONFIG:-auto}"
DEFAULT_SWIGLU_GENERATED_REPORT="${SWIGLU_GENERATED_REPORT:-auto}"
DEFAULT_SWIGLU_GENERATE_MODE="${SWIGLU_GENERATE_MODE:-target-profile}"
DEFAULT_SWIGLU_CHANNEL_THRESHOLD_RATIO="${SWIGLU_CHANNEL_THRESHOLD_RATIO:-0.10}"

DEFAULT_BUILD_DIR="${BUILD_DIR:-build}"
DEFAULT_SKIP_BUILD="${SKIP_BUILD:-0}"
DEFAULT_RUN_KIND="${RUN_KIND:-cli}"

# activation-flow is a one-command pipeline that runs collect -> generate ->
# perplexity -> cli by default. FLOW_DECODE_N_PREDICT is only used when the case
# itself sets N_PREDICT=-1 and the flow needs a finite decode/collect budget.
DEFAULT_FLOW_STEPS="${FLOW_STEPS:-collect,generate,perplexity,cli}"
DEFAULT_FLOW_REUSE_ARTIFACTS="${FLOW_REUSE_ARTIFACTS:-0}"
DEFAULT_FLOW_DECODE_N_PREDICT="${FLOW_DECODE_N_PREDICT:-128}"

# -----------------------------------------------------------------------------
# Case catalog
# -----------------------------------------------------------------------------

# Most local experiment cases only differ in case name and model path. Keep the
# common knobs in one place to avoid having five partially duplicated blocks.
make_standard_case_spec() {
  local case_name="$1"
  local model_path="$2"

  cat <<EOF
${case_name}
|# paths
|MODEL=${model_path}
|DATA=models/hf/wiki.test.raw
|OUT_DIR=kv_dump_logs
|PROMPT=你好，请简要介绍一下KV cache。

|# runtime
|CTX=4096
|THREADS=$(nproc)
|N_PREDICT=-1
|SEQ_ID=0
|BATCH=4096
|UBATCH=4096
|STRIDE=0

|# fp8-sim
|SIM_FP8=0
|SIM_FP_FORMAT=8
|SIM_FP8_LAYOUT=2
|SIM_FP8_APPLY_SRC0=1
|SIM_FP8_APPLY_SRC1=1
|SIM_FP8_SCALE_TYPE=0
|SIM_FP8_SCALE_TYPE_IN=0
|SIM_FP8_SCALE_TYPE_OUT=1
|SIM_FP8_BLOCK=16
|SIM_Q4Q6=1
|SIM_Q4Q6_APPLY_SRC0=1
|SIM_Q4Q6_APPLY_SRC1=1
|SIM_Q4Q6_SRC0_BLOCK=16
|SIM_Q4Q6_SRC1_BLOCK=16
|SIM_Q4Q6_SRC1_QMODE=0
|SIM_Q4Q6_SRC1_LOG_STEP=1
|SIM_Q8Q8=0
|SIM_Q8Q8_APPLY_SRC0=1
|SIM_Q8Q8_APPLY_SRC1=1
|SIM_Q8Q8_SRC0_BLOCK=16
|SIM_Q8Q8_SRC1_BLOCK=16
|SIM_MATMUL_OUT_MODE=1
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE=0
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP=-3
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP=-2
|FP8_SIM_STATS_SAMPLE=10

|# reduction-profiler
|REDUCTION_PROD_PROFILE=0
|REDUCTION_PROD_PROFILE_BINS=256
|REDUCTION_PROD_PROFILE_HIST_MIN_LOG2=-128
|REDUCTION_PROD_PROFILE_HIST_MAX_LOG2=128
|REDUCTION_PROD_PROFILE_SAMPLE_RATE=1000
|REDUCTION_PROD_BLOCK_DROP_LOG2_N=10
|REDUCTION_PROD_PROFILE_MAX_SAMPLES=10000

|# misc
|DUMP_DOT=0
EOF
}

CASE_LLAMA_3___2_1B_F8E3M4_NORMAL="$(make_standard_case_spec "Llama-3.2-1B-f8e3m4-normal" "models/hf/Llama-3.2-1B-Instruct-f16.gguf")"
CASE_QWEN_3_1___7B_F8E3M4_NORMAL="$(make_standard_case_spec "Qwen-3-1.7B-f8e3m4-normal" "models/Qwen/Qwen3-1.7B-Base-f16.gguf")"
CASE_LLAMA_3___2_3B_F8E3M4_NORMAL="$(make_standard_case_spec "Llama-3.2-3B-f8e3m4-normal" "models/hf/Llama-3___2-3B-Instruct-f16.gguf")"
CASE_LLAMA_2_7B_F8E3M4_NORMAL="$(make_standard_case_spec "Llama-2-7B-f8e3m4-normal" "models/hf/llama-2-7B-F16.gguf")"
CASE_QWEN_3_8B_F8E3M4_NORMAL="$(make_standard_case_spec "Qwen-3-8B-f8e3m4-normal" "models/Qwen/Qwen3-8B-f16.gguf")"

# Keep execution order here. Cases not actively used stay commented out but are
# still defined above for quick re-enabling.
RUN_CASES=(
  "$CASE_LLAMA_3___2_1B_F8E3M4_NORMAL"
  "$CASE_QWEN_3_1___7B_F8E3M4_NORMAL"
  "$CASE_LLAMA_3___2_3B_F8E3M4_NORMAL"
  "$CASE_LLAMA_2_7B_F8E3M4_NORMAL"
  "$CASE_QWEN_3_8B_F8E3M4_NORMAL"
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

sanitize_name() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

trim_case_field() {
  printf '%s' "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

resolve_python_bin() {
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s' "${PYTHON}"
    return 0
  fi

  if [[ -x ".venv/bin/python" ]]; then
    printf '%s' ".venv/bin/python"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  return 1
}

activation_threshold_kind_label() {
  case "$1" in
    swiglu)
      printf '%s' 'SwiGLU'
      ;;
    silu)
      printf '%s' 'SiLU'
      ;;
    silu_input)
      printf '%s' 'SiLU input'
      ;;
    swiglu_silu)
      printf '%s' 'SwiGLU + SiLU'
      ;;
    *)
      printf '%s' "$1"
      ;;
  esac
}

normalize_activation_threshold_kind() {
  case "$1" in
    swiglu|silu|silu_input)
      printf '%s' "$1"
      ;;
    swiglu_silu|swiglu+silu|silu+swiglu|dual)
      printf '%s' 'swiglu_silu'
      ;;
    silu-input)
      printf '%s' 'silu_input'
      ;;
    *)
      return 1
      ;;
  esac
}

activation_threshold_kind_is_dual_output() {
  [[ "$1" == "swiglu_silu" ]]
}

activation_threshold_runtime_primary_kind() {
  if activation_threshold_kind_is_dual_output "$1"; then
    printf '%s' 'silu'
  else
    printf '%s' "$1"
  fi
}

activation_threshold_runtime_secondary_kind() {
  if activation_threshold_kind_is_dual_output "$1"; then
    printf '%s' 'swiglu'
  else
    printf '%s' ''
  fi
}

activation_threshold_config_root() {
  printf 'tools/%s-threshold-configs' "$1"
}

activation_threshold_target_root() {
  printf 'tools/%s-threshold-targets' "$1"
}

resolve_threshold_family_context() {
  local case_slug="$1"
  local case_dir="$2"
  local family_kind="$3"
  local threshold_report_path="$4"
  local -n out_config_root="$5"
  local -n out_target_root="$6"
  local -n out_collect_prefix="$7"
  local -n out_collect_summary="$8"
  local -n out_collect_hist="$9"
  local -n out_target_profile="${10}"
  local -n out_generated_config="${11}"
  local -n out_generated_report="${12}"
  local -n out_threshold_config="${13}"
  local -n out_threshold_report="${14}"

  out_config_root="$(activation_threshold_config_root "${family_kind}")"
  out_target_root="$(activation_threshold_target_root "${family_kind}")"

  out_collect_prefix="${SWIGLU_COLLECT_PREFIX}"
  if [[ "${out_collect_prefix}" == "auto" ]]; then
    out_collect_prefix="${case_dir}/${case_slug}_${family_kind}_collect"
  fi
  out_collect_summary="${out_collect_prefix}_summary.csv"
  out_collect_hist="${out_collect_prefix}_hist.csv"

  out_target_profile="${SWIGLU_TARGET_PROFILE}"
  if [[ "${out_target_profile}" == "auto" ]]; then
    out_target_profile="${out_target_root}/${SWIGLU_TARGET_PROFILE_KIND}/${case_slug}.csv"
  fi

  out_generated_config="${SWIGLU_GENERATED_CONFIG}"
  if [[ "${out_generated_config}" == "auto" ]]; then
    out_generated_config="${case_dir}/${case_slug}_${family_kind}_threshold_generated.csv"
  fi

  out_generated_report="${SWIGLU_GENERATED_REPORT}"
  if [[ "${out_generated_report}" == "auto" ]]; then
    out_generated_report="${case_dir}/${case_slug}_${family_kind}_threshold_generated_summary.csv"
  fi

  out_threshold_config="${SWIGLU_THRESHOLD_CONFIG}"
  if [[ "${out_threshold_config}" == "auto" ]]; then
    if [[ "${SWIGLU_THRESHOLD_PROFILE}" == "generated" ]]; then
      out_threshold_config="${out_generated_config}"
    else
      out_threshold_config="${out_config_root}/${SWIGLU_THRESHOLD_PROFILE}/${case_slug}.csv"
    fi
  fi

  out_threshold_report="${threshold_report_path}"
}

normalize_run_kind() {
  case "$1" in
    cli|perplexity|decode-stats|swiglu-collect|swiglu-generate|activation-flow)
      printf '%s' "$1"
      ;;
    collect|activation-collect)
      printf '%s' 'swiglu-collect'
      ;;
    generate|activation-generate)
      printf '%s' 'swiglu-generate'
      ;;
    flow)
      printf '%s' 'activation-flow'
      ;;
    *)
      return 1
      ;;
  esac
}

is_analysis_mode() {
  case "$1" in
    cli|perplexity|decode-stats)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

build_target_for_mode() {
  case "$1" in
    cli|swiglu-collect)
      printf '%s' 'llama-cli'
      ;;
    perplexity)
      printf '%s' 'llama-perplexity'
      ;;
    decode-stats)
      printf '%s' 'llama-decode-stats'
      ;;
    *)
      printf '%s' ''
      ;;
  esac
}

expect_one_of() {
  local name="$1"
  local value="$2"
  shift 2

  local candidate
  for candidate in "$@"; do
    if [[ "${value}" == "${candidate}" ]]; then
      return 0
    fi
  done

  die "invalid ${name}=${value} (expected: $*)"
}

expect_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^-?[0-9]+$ ]] || die "invalid ${name}=${value} (expected integer)"
}

expect_positive_integer() {
  local name="$1"
  local value="$2"
  expect_integer "${name}" "${value}"
  [[ "${value}" -gt 0 ]] || die "invalid ${name}=${value} (expected positive integer)"
}

expect_non_negative_integer() {
  local name="$1"
  local value="$2"
  expect_integer "${name}" "${value}"
  [[ "${value}" -ge 0 ]] || die "invalid ${name}=${value} (expected non-negative integer)"
}

expect_ordered_integers() {
  local min_name="$1"
  local min_value="$2"
  local max_name="$3"
  local max_value="$4"
  [[ "${max_value}" -gt "${min_value}" ]] || die "invalid range: ${min_name}=${min_value}, ${max_name}=${max_value} (expected ${max_name} > ${min_name})"
}

# -----------------------------------------------------------------------------
# Mutable per-case state
# -----------------------------------------------------------------------------

SIM_FLAGS=""
FLOW_STEP_LIST=()
FINISHED_RUNS=()
USED_OUT_DIRS=()
ANALYSIS_MODES_RAN=0
CURRENT_FLOW_MODE=0
declare -A USER_ENV_OVERRIDES=()

# Resolved activation context for the currently executing mode.
RESOLVED_ACTIVATION_KIND=""
RESOLVED_ACTIVATION_LABEL=""
RESOLVED_PRIMARY_ACTIVATION_KIND=""
RESOLVED_SECONDARY_ACTIVATION_KIND=""
RESOLVED_ACTIVATION_CONFIG_ROOT=""
RESOLVED_ACTIVATION_TARGET_ROOT=""
RESOLVED_THRESHOLD_ENABLED="0"
RESOLVED_THRESHOLD_REPORT=""
RESOLVED_SECONDARY_THRESHOLD_REPORT=""
RESOLVED_COLLECT_PREFIX=""
RESOLVED_COLLECT_SUMMARY=""
RESOLVED_COLLECT_HIST=""
RESOLVED_SECONDARY_COLLECT_PREFIX=""
RESOLVED_SECONDARY_COLLECT_SUMMARY=""
RESOLVED_SECONDARY_COLLECT_HIST=""
RESOLVED_TARGET_PROFILE=""
RESOLVED_GENERATED_CONFIG=""
RESOLVED_GENERATED_REPORT=""
RESOLVED_THRESHOLD_CONFIG=""
RESOLVED_SECONDARY_ACTIVATION_CONFIG_ROOT=""
RESOLVED_SECONDARY_ACTIVATION_TARGET_ROOT=""
RESOLVED_SECONDARY_TARGET_PROFILE=""
RESOLVED_SECONDARY_GENERATED_CONFIG=""
RESOLVED_SECONDARY_GENERATED_REPORT=""
RESOLVED_SECONDARY_THRESHOLD_CONFIG=""
MODE_PYTHON_BIN=""
THRESHOLD_ARGS=()

remember_user_env_overrides() {
  local key=""
  local -a keys=(
    MODEL DATA OUT_DIR PROMPT
    CTX THREADS N_PREDICT SEQ_ID BATCH UBATCH STRIDE
    SIM_FP8 SIM_FP_FORMAT SIM_FP8_LAYOUT SIM_FP8_APPLY_SRC0 SIM_FP8_APPLY_SRC1
    SIM_FP8_SCALE_TYPE SIM_FP8_SCALE_TYPE_IN SIM_FP8_SCALE_TYPE_OUT SIM_FP8_BLOCK
    SIM_Q4Q6 SIM_Q4Q6_APPLY_SRC0 SIM_Q4Q6_APPLY_SRC1 SIM_Q4Q6_SRC0_BLOCK SIM_Q4Q6_SRC1_BLOCK SIM_Q4Q6_SRC1_QMODE SIM_Q4Q6_SRC1_LOG_STEP
    SIM_Q8Q8 SIM_Q8Q8_APPLY_SRC0 SIM_Q8Q8_APPLY_SRC1 SIM_Q8Q8_SRC0_BLOCK SIM_Q8Q8_SRC1_BLOCK
    SIM_MATMUL_OUT_MODE
    SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP
    FP8_SIM_STATS_SAMPLE
    REDUCTION_PROD_PROFILE REDUCTION_PROD_PROFILE_BINS REDUCTION_PROD_PROFILE_HIST_MIN_LOG2
    REDUCTION_PROD_PROFILE_HIST_MAX_LOG2 REDUCTION_PROD_PROFILE_SAMPLE_RATE REDUCTION_PROD_BLOCK_DROP_LOG2_N
    REDUCTION_PROD_PROFILE_MAX_SAMPLES REDUCTION_PROD_PROFILE_PREFIX
    DUMP_DOT CASE_FILTER BUILD_DIR SKIP_BUILD RUN_KIND
    SWIGLU_THRESHOLD_ENABLE SWIGLU_THRESHOLD_KIND SWIGLU_THRESHOLD_PROFILE SWIGLU_THRESHOLD_CONFIG
    SWIGLU_COLLECT_PREFIX SWIGLU_COLLECT_BINS SWIGLU_COLLECT_LOG10_MIN SWIGLU_COLLECT_LOG10_MAX
    SWIGLU_TARGET_PROFILE_KIND SWIGLU_TARGET_PROFILE SWIGLU_TARGET_SCALE
    SWIGLU_GENERATED_CONFIG SWIGLU_GENERATED_REPORT SWIGLU_GENERATE_MODE SWIGLU_CHANNEL_THRESHOLD_RATIO
    FLOW_STEPS FLOW_REUSE_ARTIFACTS FLOW_DECODE_N_PREDICT
    PYTHON
  )

  for key in "${keys[@]}"; do
    if [[ -v ${key} ]]; then
      USER_ENV_OVERRIDES["${key}"]="${!key}"
    fi
  done
}

apply_user_env_overrides() {
  local key=""
  for key in "${!USER_ENV_OVERRIDES[@]}"; do
    printf -v "${key}" '%s' "${USER_ENV_OVERRIDES[${key}]}"
  done
}

reset_case_defaults() {
  MODEL="${DEFAULT_MODEL}"
  DATA="${DEFAULT_DATA}"
  OUT_DIR="${DEFAULT_OUT_DIR}"
  PROMPT="${DEFAULT_PROMPT}"

  CTX="${DEFAULT_CTX}"
  THREADS="${DEFAULT_THREADS}"
  N_PREDICT="${DEFAULT_N_PREDICT}"
  SEQ_ID="${DEFAULT_SEQ_ID}"
  BATCH="${DEFAULT_BATCH}"
  UBATCH="${DEFAULT_UBATCH}"
  STRIDE="${DEFAULT_STRIDE}"

  SIM_FP8="${DEFAULT_SIM_FP8}"
  SIM_FP_FORMAT="${DEFAULT_SIM_FP_FORMAT}"
  SIM_FP8_LAYOUT="${DEFAULT_SIM_FP8_LAYOUT}"
  SIM_FP8_APPLY_SRC0="${DEFAULT_SIM_FP8_APPLY_SRC0}"
  SIM_FP8_APPLY_SRC1="${DEFAULT_SIM_FP8_APPLY_SRC1}"
  SIM_FP8_SCALE_TYPE="${DEFAULT_SIM_FP8_SCALE_TYPE}"
  SIM_FP8_SCALE_TYPE_IN="${DEFAULT_SIM_FP8_SCALE_TYPE_IN}"
  SIM_FP8_SCALE_TYPE_OUT="${DEFAULT_SIM_FP8_SCALE_TYPE_OUT}"
  SIM_FP8_BLOCK="${DEFAULT_SIM_FP8_BLOCK}"
  SIM_Q4Q6="${DEFAULT_SIM_Q4Q6}"
  SIM_Q4Q6_APPLY_SRC0="${DEFAULT_SIM_Q4Q6_APPLY_SRC0}"
  SIM_Q4Q6_APPLY_SRC1="${DEFAULT_SIM_Q4Q6_APPLY_SRC1}"
  SIM_Q4Q6_SRC0_BLOCK="${DEFAULT_SIM_Q4Q6_SRC0_BLOCK}"
  SIM_Q4Q6_SRC1_BLOCK="${DEFAULT_SIM_Q4Q6_SRC1_BLOCK}"
  SIM_Q4Q6_SRC1_QMODE="${DEFAULT_SIM_Q4Q6_SRC1_QMODE}"
  SIM_Q4Q6_SRC1_LOG_STEP="${DEFAULT_SIM_Q4Q6_SRC1_LOG_STEP}"
  SIM_Q8Q8="${DEFAULT_SIM_Q8Q8}"
  SIM_Q8Q8_APPLY_SRC0="${DEFAULT_SIM_Q8Q8_APPLY_SRC0}"
  SIM_Q8Q8_APPLY_SRC1="${DEFAULT_SIM_Q8Q8_APPLY_SRC1}"
  SIM_Q8Q8_SRC0_BLOCK="${DEFAULT_SIM_Q8Q8_SRC0_BLOCK}"
  SIM_Q8Q8_SRC1_BLOCK="${DEFAULT_SIM_Q8Q8_SRC1_BLOCK}"
  SIM_MATMUL_OUT_MODE="${DEFAULT_SIM_MATMUL_OUT_MODE}"
  SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE="${DEFAULT_SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE}"
  SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP="${DEFAULT_SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP}"
  SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP="${DEFAULT_SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP}"
  FP8_SIM_STATS_SAMPLE="${DEFAULT_FP8_SIM_STATS_SAMPLE}"

  REDUCTION_PROD_PROFILE="${DEFAULT_REDUCTION_PROD_PROFILE}"
  REDUCTION_PROD_PROFILE_BINS="${DEFAULT_REDUCTION_PROD_PROFILE_BINS}"
  REDUCTION_PROD_PROFILE_HIST_MIN_LOG2="${DEFAULT_REDUCTION_PROD_PROFILE_HIST_MIN_LOG2}"
  REDUCTION_PROD_PROFILE_HIST_MAX_LOG2="${DEFAULT_REDUCTION_PROD_PROFILE_HIST_MAX_LOG2}"
  REDUCTION_PROD_PROFILE_SAMPLE_RATE="${DEFAULT_REDUCTION_PROD_PROFILE_SAMPLE_RATE}"
  REDUCTION_PROD_BLOCK_DROP_LOG2_N="${DEFAULT_REDUCTION_PROD_BLOCK_DROP_LOG2_N}"
  REDUCTION_PROD_PROFILE_MAX_SAMPLES="${DEFAULT_REDUCTION_PROD_PROFILE_MAX_SAMPLES}"
  REDUCTION_PROD_PROFILE_PREFIX="${DEFAULT_REDUCTION_PROD_PROFILE_PREFIX}"

  DUMP_DOT="${DEFAULT_DUMP_DOT}"
  DOT_FILE=""

  CASE_FILTER="${DEFAULT_CASE_FILTER}"
  BUILD_DIR="${DEFAULT_BUILD_DIR}"
  SKIP_BUILD="${DEFAULT_SKIP_BUILD}"
  RUN_KIND="${DEFAULT_RUN_KIND}"

  SWIGLU_THRESHOLD_ENABLE="${DEFAULT_SWIGLU_THRESHOLD_ENABLE}"
  SWIGLU_THRESHOLD_KIND="${DEFAULT_SWIGLU_THRESHOLD_KIND}"
  SWIGLU_THRESHOLD_PROFILE="${DEFAULT_SWIGLU_THRESHOLD_PROFILE}"
  SWIGLU_THRESHOLD_CONFIG="${DEFAULT_SWIGLU_THRESHOLD_CONFIG}"
  SWIGLU_COLLECT_PREFIX="${DEFAULT_SWIGLU_COLLECT_PREFIX}"
  SWIGLU_COLLECT_BINS="${DEFAULT_SWIGLU_COLLECT_BINS}"
  SWIGLU_COLLECT_LOG10_MIN="${DEFAULT_SWIGLU_COLLECT_LOG10_MIN}"
  SWIGLU_COLLECT_LOG10_MAX="${DEFAULT_SWIGLU_COLLECT_LOG10_MAX}"
  SWIGLU_TARGET_PROFILE_KIND="${DEFAULT_SWIGLU_TARGET_PROFILE_KIND}"
  SWIGLU_TARGET_PROFILE="${DEFAULT_SWIGLU_TARGET_PROFILE}"
  SWIGLU_TARGET_SCALE="${DEFAULT_SWIGLU_TARGET_SCALE}"
  SWIGLU_GENERATED_CONFIG="${DEFAULT_SWIGLU_GENERATED_CONFIG}"
  SWIGLU_GENERATED_REPORT="${DEFAULT_SWIGLU_GENERATED_REPORT}"
  SWIGLU_GENERATE_MODE="${DEFAULT_SWIGLU_GENERATE_MODE}"
  SWIGLU_CHANNEL_THRESHOLD_RATIO="${DEFAULT_SWIGLU_CHANNEL_THRESHOLD_RATIO}"

  FLOW_STEPS="${DEFAULT_FLOW_STEPS}"
  FLOW_REUSE_ARTIFACTS="${DEFAULT_FLOW_REUSE_ARTIFACTS}"
  FLOW_DECODE_N_PREDICT="${DEFAULT_FLOW_DECODE_N_PREDICT}"
}

apply_case_overrides() {
  local case_spec="$1"
  local key=""
  local value=""
  local field=""

  case_spec="${case_spec//$'\n'/|}"
  local IFS='|'
  read -r -a case_fields <<< "${case_spec}"

  CASE_NAME="$(trim_case_field "${case_fields[0]}")"

  for field in "${case_fields[@]:1}"; do
    field="$(trim_case_field "${field}")"
    if [[ -z "${field}" || "${field:0:1}" == "#" ]]; then
      continue
    fi

    [[ "${field}" == *=* ]] || die "invalid case field for ${CASE_NAME}: ${field}"
    key="${field%%=*}"
    value="${field#*=}"
    printf -v "${key}" '%s' "${value}"
  done
}

parse_flow_steps() {
  FLOW_STEP_LIST=()

  local raw_step=""
  local normalized=""
  local IFS=','
  read -r -a raw_steps <<< "${FLOW_STEPS}"

  [[ "${#raw_steps[@]}" -gt 0 ]] || die "FLOW_STEPS is empty"

  for raw_step in "${raw_steps[@]}"; do
    raw_step="$(trim_case_field "${raw_step}")"
    [[ -n "${raw_step}" ]] || continue

    normalized="$(normalize_run_kind "${raw_step}")" || die "invalid FLOW_STEPS entry: ${raw_step}"
    [[ "${normalized}" != "activation-flow" ]] || die "FLOW_STEPS cannot contain activation-flow"
    FLOW_STEP_LIST+=("${normalized}")
  done

  [[ "${#FLOW_STEP_LIST[@]}" -gt 0 ]] || die "FLOW_STEPS does not contain any executable step"
}

validate_case() {
  RUN_KIND="$(normalize_run_kind "${RUN_KIND}")" || die "invalid RUN_KIND=${RUN_KIND}"
  SWIGLU_THRESHOLD_KIND="$(normalize_activation_threshold_kind "${SWIGLU_THRESHOLD_KIND}")" || die "invalid SWIGLU_THRESHOLD_KIND=${SWIGLU_THRESHOLD_KIND} (expected swiglu, silu, silu_input, or swiglu+silu)"

  expect_one_of "SIM_MATMUL_OUT_MODE" "${SIM_MATMUL_OUT_MODE}" 0 1
  expect_one_of "SIM_FP_FORMAT" "${SIM_FP_FORMAT}" 8 9
  expect_one_of "SIM_FP8_LAYOUT" "${SIM_FP8_LAYOUT}" 0 1 2 3 4
  expect_one_of "SIM_Q4Q6" "${SIM_Q4Q6}" 0 1
  expect_one_of "SIM_Q4Q6_APPLY_SRC0" "${SIM_Q4Q6_APPLY_SRC0}" 0 1
  expect_one_of "SIM_Q4Q6_APPLY_SRC1" "${SIM_Q4Q6_APPLY_SRC1}" 0 1
  expect_one_of "SIM_Q4Q6_SRC1_QMODE" "${SIM_Q4Q6_SRC1_QMODE}" 0 1 2
  expect_one_of "SIM_Q8Q8" "${SIM_Q8Q8}" 0 1
  expect_one_of "SIM_Q8Q8_APPLY_SRC0" "${SIM_Q8Q8_APPLY_SRC0}" 0 1
  expect_one_of "SIM_Q8Q8_APPLY_SRC1" "${SIM_Q8Q8_APPLY_SRC1}" 0 1
  expect_one_of "SIM_FP8_SCALE_TYPE_IN" "${SIM_FP8_SCALE_TYPE_IN}" 0 1
  expect_one_of "SIM_FP8_SCALE_TYPE_OUT" "${SIM_FP8_SCALE_TYPE_OUT}" 0 1
  expect_one_of "SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE" "${SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE}" 0 1
  expect_one_of "REDUCTION_PROD_PROFILE" "${REDUCTION_PROD_PROFILE}" 0 1
  expect_one_of "SWIGLU_THRESHOLD_ENABLE" "${SWIGLU_THRESHOLD_ENABLE}" 0 1
  expect_one_of "FLOW_REUSE_ARTIFACTS" "${FLOW_REUSE_ARTIFACTS}" 0 1
  expect_one_of "SWIGLU_GENERATE_MODE" "${SWIGLU_GENERATE_MODE}" target-profile channel-max

  expect_integer "SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP" "${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP}"
  expect_integer "SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP" "${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP}"
  [[ "${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP}" -le "${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP}" ]] || die "invalid small-exp erase range: min=${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP}, max=${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP} (expected min <= max)"

  expect_positive_integer "FP8_SIM_STATS_SAMPLE" "${FP8_SIM_STATS_SAMPLE}"
  expect_positive_integer "SIM_Q4Q6_SRC0_BLOCK" "${SIM_Q4Q6_SRC0_BLOCK}"
  expect_positive_integer "SIM_Q4Q6_SRC1_BLOCK" "${SIM_Q4Q6_SRC1_BLOCK}"
  expect_positive_integer "SIM_Q4Q6_SRC1_LOG_STEP" "${SIM_Q4Q6_SRC1_LOG_STEP}"
  expect_positive_integer "SIM_Q8Q8_SRC0_BLOCK" "${SIM_Q8Q8_SRC0_BLOCK}"
  expect_positive_integer "SIM_Q8Q8_SRC1_BLOCK" "${SIM_Q8Q8_SRC1_BLOCK}"
  expect_positive_integer "REDUCTION_PROD_PROFILE_BINS" "${REDUCTION_PROD_PROFILE_BINS}"
  expect_integer "REDUCTION_PROD_PROFILE_HIST_MIN_LOG2" "${REDUCTION_PROD_PROFILE_HIST_MIN_LOG2}"
  expect_integer "REDUCTION_PROD_PROFILE_HIST_MAX_LOG2" "${REDUCTION_PROD_PROFILE_HIST_MAX_LOG2}"
  expect_ordered_integers "REDUCTION_PROD_PROFILE_HIST_MIN_LOG2" "${REDUCTION_PROD_PROFILE_HIST_MIN_LOG2}" "REDUCTION_PROD_PROFILE_HIST_MAX_LOG2" "${REDUCTION_PROD_PROFILE_HIST_MAX_LOG2}"
  expect_positive_integer "REDUCTION_PROD_PROFILE_SAMPLE_RATE" "${REDUCTION_PROD_PROFILE_SAMPLE_RATE}"
  expect_non_negative_integer "REDUCTION_PROD_BLOCK_DROP_LOG2_N" "${REDUCTION_PROD_BLOCK_DROP_LOG2_N}"
  expect_positive_integer "REDUCTION_PROD_PROFILE_MAX_SAMPLES" "${REDUCTION_PROD_PROFILE_MAX_SAMPLES}"
  expect_positive_integer "FLOW_DECODE_N_PREDICT" "${FLOW_DECODE_N_PREDICT}"
  [[ -n "${REDUCTION_PROD_PROFILE_PREFIX}" ]] || die "invalid REDUCTION_PROD_PROFILE_PREFIX: empty string"

  if [[ "${SIM_Q4Q6}" == "1" && "${SIM_FP8}" == "1" ]]; then
    die "SIM_Q4Q6 and SIM_FP8 cannot be enabled together"
  fi

  if [[ "${SIM_Q8Q8}" == "1" && "${SIM_FP8}" == "1" ]]; then
    die "SIM_Q8Q8 and SIM_FP8 cannot be enabled together"
  fi

  if [[ "${SIM_Q8Q8}" == "1" && "${SIM_Q4Q6}" == "1" ]]; then
    die "SIM_Q8Q8 and SIM_Q4Q6 cannot be enabled together"
  fi

  if [[ "${SIM_Q4Q6}" == "1" && "${SIM_MATMUL_OUT_MODE}" != "1" ]]; then
    die "SIM_Q4Q6 requires SIM_MATMUL_OUT_MODE=1 (BF16 output round-trip)"
  fi

  if [[ "${SIM_Q8Q8}" == "1" && "${SIM_MATMUL_OUT_MODE}" != "1" ]]; then
    die "SIM_Q8Q8 requires SIM_MATMUL_OUT_MODE=1 (BF16 output round-trip)"
  fi

  if activation_threshold_kind_is_dual_output "${SWIGLU_THRESHOLD_KIND}"; then
    [[ "${SWIGLU_THRESHOLD_CONFIG}" == "auto" ]] || die "SWIGLU_THRESHOLD_CONFIG must stay auto for SWIGLU_THRESHOLD_KIND=swiglu+silu"
    [[ "${SWIGLU_COLLECT_PREFIX}" == "auto" ]] || die "SWIGLU_COLLECT_PREFIX must stay auto for SWIGLU_THRESHOLD_KIND=swiglu+silu"
    [[ "${SWIGLU_TARGET_PROFILE}" == "auto" ]] || die "SWIGLU_TARGET_PROFILE must stay auto for SWIGLU_THRESHOLD_KIND=swiglu+silu"
    [[ "${SWIGLU_GENERATED_CONFIG}" == "auto" ]] || die "SWIGLU_GENERATED_CONFIG must stay auto for SWIGLU_THRESHOLD_KIND=swiglu+silu"
    [[ "${SWIGLU_GENERATED_REPORT}" == "auto" ]] || die "SWIGLU_GENERATED_REPORT must stay auto for SWIGLU_THRESHOLD_KIND=swiglu+silu"
  fi

  if [[ "${SWIGLU_GENERATE_MODE}" == "channel-max" ]]; then
    [[ "${SWIGLU_THRESHOLD_KIND}" == "silu" ]] || die "SWIGLU_GENERATE_MODE=channel-max currently requires SWIGLU_THRESHOLD_KIND=silu"
    [[ "${SWIGLU_THRESHOLD_CONFIG}" == "auto" ]] || die "SWIGLU_THRESHOLD_CONFIG must stay auto for SWIGLU_GENERATE_MODE=channel-max"
    [[ "${SWIGLU_THRESHOLD_PROFILE}" == "generated" ]] || die "SWIGLU_THRESHOLD_PROFILE must be generated for SWIGLU_GENERATE_MODE=channel-max"
    awk -v ratio="${SWIGLU_CHANNEL_THRESHOLD_RATIO}" 'BEGIN { exit !(ratio >= 0 && ratio <= 1) }' || die "invalid SWIGLU_CHANNEL_THRESHOLD_RATIO=${SWIGLU_CHANNEL_THRESHOLD_RATIO} (expected 0 <= ratio <= 1)"
  fi

  if [[ "${SWIGLU_THRESHOLD_ENABLE}" == "1" && "${RUN_KIND}" == "decode-stats" ]]; then
    die "$(activation_threshold_kind_label "${SWIGLU_THRESHOLD_KIND}") threshold experiment is only supported with RUN_KIND=perplexity or RUN_KIND=cli"
  fi

  if [[ "${RUN_KIND}" == "activation-flow" ]]; then
    parse_flow_steps
  fi
}

build_case_flags() {
  SIM_FLAGS="-DGGML_SIM_FP8E4M3=${SIM_FP8} \
  -DGGML_SIM_FP_FORMAT=${SIM_FP_FORMAT} \
  -DGGML_SIM_FP8_LAYOUT=${SIM_FP8_LAYOUT} \
  -DGGML_SIM_FP8E4M3_APPLY_SRC0=${SIM_FP8_APPLY_SRC0} \
  -DGGML_SIM_FP8E4M3_APPLY_SRC1=${SIM_FP8_APPLY_SRC1} \
  -DGGML_SIM_FP8E4M3_SCALE_TYPE=${SIM_FP8_SCALE_TYPE_IN} \
  -DGGML_SIM_FP8E4M3_SCALE_TYPE_IN=${SIM_FP8_SCALE_TYPE_IN} \
  -DGGML_SIM_FP8E4M3_SCALE_TYPE_OUT=${SIM_FP8_SCALE_TYPE_OUT} \
  -DGGML_SIM_FP8E4M3_BLOCK=${SIM_FP8_BLOCK} \
  -DGGML_SIM_Q4Q6=${SIM_Q4Q6} \
  -DGGML_SIM_Q4Q6_APPLY_SRC0=${SIM_Q4Q6_APPLY_SRC0} \
  -DGGML_SIM_Q4Q6_APPLY_SRC1=${SIM_Q4Q6_APPLY_SRC1} \
  -DGGML_SIM_Q4Q6_SRC0_BLOCK=${SIM_Q4Q6_SRC0_BLOCK} \
  -DGGML_SIM_Q4Q6_SRC1_BLOCK=${SIM_Q4Q6_SRC1_BLOCK} \
  -DGGML_SIM_Q4Q6_SRC1_QMODE=${SIM_Q4Q6_SRC1_QMODE} \
  -DGGML_SIM_Q4Q6_SRC1_LOG_STEP=${SIM_Q4Q6_SRC1_LOG_STEP} \
  -DGGML_SIM_Q8Q8=${SIM_Q8Q8} \
  -DGGML_SIM_Q8Q8_APPLY_SRC0=${SIM_Q8Q8_APPLY_SRC0} \
  -DGGML_SIM_Q8Q8_APPLY_SRC1=${SIM_Q8Q8_APPLY_SRC1} \
  -DGGML_SIM_Q8Q8_SRC0_BLOCK=${SIM_Q8Q8_SRC0_BLOCK} \
  -DGGML_SIM_Q8Q8_SRC1_BLOCK=${SIM_Q8Q8_SRC1_BLOCK} \
  -DGGML_SIM_MATMUL_OUT_MODE=${SIM_MATMUL_OUT_MODE} \
  -DGGML_SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE=${SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE} \
  -DGGML_SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP=${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP} \
  -DGGML_SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP=${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP} \
  -DGGML_REDUCTION_PROD_PROFILE=${REDUCTION_PROD_PROFILE} \
  -DGGML_REDUCTION_PROD_PROFILE_BINS=${REDUCTION_PROD_PROFILE_BINS} \
  -DGGML_REDUCTION_PROD_PROFILE_HIST_MIN_LOG2=${REDUCTION_PROD_PROFILE_HIST_MIN_LOG2} \
  -DGGML_REDUCTION_PROD_PROFILE_HIST_MAX_LOG2=${REDUCTION_PROD_PROFILE_HIST_MAX_LOG2} \
  -DGGML_REDUCTION_PROD_PROFILE_SAMPLE_RATE=${REDUCTION_PROD_PROFILE_SAMPLE_RATE} \
  -DGGML_REDUCTION_PROD_BLOCK_DROP_LOG2_N=${REDUCTION_PROD_BLOCK_DROP_LOG2_N} \
  -DGGML_REDUCTION_PROD_PROFILE_MAX_SAMPLES=${REDUCTION_PROD_PROFILE_MAX_SAMPLES} \
  -DGGML_REDUCTION_PROD_PROFILE_PREFIX=\\\"${REDUCTION_PROD_PROFILE_PREFIX}\\\""
}

threshold_enabled_for_mode() {
  local mode="$1"
  case "${mode}" in
    cli|perplexity)
      if [[ "${CURRENT_FLOW_MODE}" == "1" ]]; then
        printf '%s' '1'
      else
        printf '%s' "${SWIGLU_THRESHOLD_ENABLE}"
      fi
      ;;
    *)
      printf '%s' '0'
      ;;
  esac
}

effective_n_predict_for_mode() {
  local mode="$1"
  local value="${N_PREDICT}"

  if [[ "${mode}" == "swiglu-collect" && "${value}" == "-1" ]]; then
    printf '%s' "${FLOW_DECODE_N_PREDICT}"
    return 0
  fi

  if [[ "${CURRENT_FLOW_MODE}" == "1" && "${mode}" == "cli" && "${value}" == "-1" ]]; then
    printf '%s' "${FLOW_DECODE_N_PREDICT}"
    return 0
  fi

  printf '%s' "${value}"
}

resolve_activation_context() {
  local case_slug="$1"
  local case_dir="$2"
  local mode="$3"

  RESOLVED_ACTIVATION_KIND="${SWIGLU_THRESHOLD_KIND}"
  RESOLVED_ACTIVATION_LABEL="$(activation_threshold_kind_label "${RESOLVED_ACTIVATION_KIND}")"
  RESOLVED_PRIMARY_ACTIVATION_KIND="$(activation_threshold_runtime_primary_kind "${RESOLVED_ACTIVATION_KIND}")"
  RESOLVED_SECONDARY_ACTIVATION_KIND="$(activation_threshold_runtime_secondary_kind "${RESOLVED_ACTIVATION_KIND}")"
  RESOLVED_THRESHOLD_ENABLED="$(threshold_enabled_for_mode "${mode}")"
  RESOLVED_SECONDARY_THRESHOLD_REPORT=""
  RESOLVED_SECONDARY_COLLECT_PREFIX=""
  RESOLVED_SECONDARY_COLLECT_SUMMARY=""
  RESOLVED_SECONDARY_COLLECT_HIST=""
  RESOLVED_SECONDARY_ACTIVATION_CONFIG_ROOT=""
  RESOLVED_SECONDARY_ACTIVATION_TARGET_ROOT=""
  RESOLVED_SECONDARY_TARGET_PROFILE=""
  RESOLVED_SECONDARY_GENERATED_CONFIG=""
  RESOLVED_SECONDARY_GENERATED_REPORT=""
  RESOLVED_SECONDARY_THRESHOLD_CONFIG=""

  resolve_threshold_family_context \
    "${case_slug}" \
    "${case_dir}" \
    "${RESOLVED_PRIMARY_ACTIVATION_KIND}" \
    "${case_dir}/${case_slug}_${RESOLVED_PRIMARY_ACTIVATION_KIND}_threshold_${mode}.csv" \
    RESOLVED_ACTIVATION_CONFIG_ROOT \
    RESOLVED_ACTIVATION_TARGET_ROOT \
    RESOLVED_COLLECT_PREFIX \
    RESOLVED_COLLECT_SUMMARY \
    RESOLVED_COLLECT_HIST \
    RESOLVED_TARGET_PROFILE \
    RESOLVED_GENERATED_CONFIG \
    RESOLVED_GENERATED_REPORT \
    RESOLVED_THRESHOLD_CONFIG \
    RESOLVED_THRESHOLD_REPORT

  if [[ -n "${RESOLVED_SECONDARY_ACTIVATION_KIND}" ]]; then
    resolve_threshold_family_context \
      "${case_slug}" \
      "${case_dir}" \
      "${RESOLVED_SECONDARY_ACTIVATION_KIND}" \
      "${case_dir}/${case_slug}_${RESOLVED_SECONDARY_ACTIVATION_KIND}_threshold_${mode}.csv" \
      RESOLVED_SECONDARY_ACTIVATION_CONFIG_ROOT \
      RESOLVED_SECONDARY_ACTIVATION_TARGET_ROOT \
      RESOLVED_SECONDARY_COLLECT_PREFIX \
      RESOLVED_SECONDARY_COLLECT_SUMMARY \
      RESOLVED_SECONDARY_COLLECT_HIST \
      RESOLVED_SECONDARY_TARGET_PROFILE \
      RESOLVED_SECONDARY_GENERATED_CONFIG \
      RESOLVED_SECONDARY_GENERATED_REPORT \
      RESOLVED_SECONDARY_THRESHOLD_CONFIG \
      RESOLVED_SECONDARY_THRESHOLD_REPORT
  fi

  MODE_PYTHON_BIN=""
  THRESHOLD_ARGS=()
  if [[ "${RESOLVED_THRESHOLD_ENABLED}" == "1" ]]; then
    THRESHOLD_ARGS=(
      --swiglu-threshold-kind "${RESOLVED_PRIMARY_ACTIVATION_KIND}"
      --swiglu-threshold-config "${RESOLVED_THRESHOLD_CONFIG}"
      --swiglu-threshold-report "${RESOLVED_THRESHOLD_REPORT}"
    )
    if [[ -n "${RESOLVED_SECONDARY_ACTIVATION_KIND}" ]]; then
      THRESHOLD_ARGS+=(
        --swiglu-threshold-secondary-kind "${RESOLVED_SECONDARY_ACTIVATION_KIND}"
        --swiglu-threshold-secondary-config "${RESOLVED_SECONDARY_THRESHOLD_CONFIG}"
        --swiglu-threshold-secondary-report "${RESOLVED_SECONDARY_THRESHOLD_REPORT}"
      )
    fi
  fi
}

ensure_mode_inputs() {
  local mode="$1"

  if [[ "${mode}" == "swiglu-generate" ]]; then
    [[ -f "${RESOLVED_COLLECT_SUMMARY}" && -f "${RESOLVED_COLLECT_HIST}" ]] || die "missing $(activation_threshold_kind_label "${RESOLVED_PRIMARY_ACTIVATION_KIND}") collection files for prefix: ${RESOLVED_COLLECT_PREFIX}"

    if [[ "${SWIGLU_GENERATE_MODE}" == "target-profile" ]]; then
      [[ -f "${RESOLVED_TARGET_PROFILE}" ]] || die "missing $(activation_threshold_kind_label "${RESOLVED_PRIMARY_ACTIVATION_KIND}") target profile: ${RESOLVED_TARGET_PROFILE}"
    else
      [[ -f "${RESOLVED_COLLECT_PREFIX}_channel_max.csv" ]] || die "missing $(activation_threshold_kind_label "${RESOLVED_PRIMARY_ACTIVATION_KIND}") channel-max collection file: ${RESOLVED_COLLECT_PREFIX}_channel_max.csv"
    fi

    if [[ -n "${RESOLVED_SECONDARY_ACTIVATION_KIND}" ]]; then
      [[ -f "${RESOLVED_SECONDARY_TARGET_PROFILE}" ]] || die "missing $(activation_threshold_kind_label "${RESOLVED_SECONDARY_ACTIVATION_KIND}") target profile: ${RESOLVED_SECONDARY_TARGET_PROFILE}"
      [[ -f "${RESOLVED_SECONDARY_COLLECT_SUMMARY}" && -f "${RESOLVED_SECONDARY_COLLECT_HIST}" ]] || die "missing $(activation_threshold_kind_label "${RESOLVED_SECONDARY_ACTIVATION_KIND}") collection files for prefix: ${RESOLVED_SECONDARY_COLLECT_PREFIX}"
    fi

    MODE_PYTHON_BIN="$(resolve_python_bin || true)"
    [[ -n "${MODE_PYTHON_BIN}" ]] || die "python3 not found and .venv/bin/python is unavailable"
  fi

  if [[ "${RESOLVED_THRESHOLD_ENABLED}" == "1" ]]; then
    [[ -f "${RESOLVED_THRESHOLD_CONFIG}" ]] || die "missing ${RESOLVED_ACTIVATION_LABEL} threshold config: ${RESOLVED_THRESHOLD_CONFIG}"
    if [[ -n "${RESOLVED_SECONDARY_ACTIVATION_KIND}" ]]; then
      [[ -f "${RESOLVED_SECONDARY_THRESHOLD_CONFIG}" ]] || die "missing ${RESOLVED_ACTIVATION_LABEL} secondary threshold config: ${RESOLVED_SECONDARY_THRESHOLD_CONFIG}"
    fi
  fi
}

primary_artifacts_exist_for_mode() {
  local mode="$1"
  local case_dir="$2"
  local default_log="$3"

  case "${mode}" in
    swiglu-collect)
      if [[ -n "${RESOLVED_SECONDARY_ACTIVATION_KIND}" ]]; then
        [[ -f "${RESOLVED_COLLECT_SUMMARY}" && -f "${RESOLVED_COLLECT_HIST}" && -f "${RESOLVED_SECONDARY_COLLECT_SUMMARY}" && -f "${RESOLVED_SECONDARY_COLLECT_HIST}" ]]
      elif [[ "${SWIGLU_GENERATE_MODE}" == "channel-max" && "${RESOLVED_PRIMARY_ACTIVATION_KIND}" == "silu" ]]; then
        [[ -f "${RESOLVED_COLLECT_SUMMARY}" && -f "${RESOLVED_COLLECT_HIST}" && -f "${RESOLVED_COLLECT_PREFIX}_channel_max.csv" ]]
      else
        [[ -f "${RESOLVED_COLLECT_SUMMARY}" && -f "${RESOLVED_COLLECT_HIST}" ]]
      fi
      ;;
    swiglu-generate)
      if [[ -n "${RESOLVED_SECONDARY_ACTIVATION_KIND}" ]]; then
        [[ -f "${RESOLVED_GENERATED_CONFIG}" && -f "${RESOLVED_GENERATED_REPORT}" && -f "${RESOLVED_SECONDARY_GENERATED_CONFIG}" && -f "${RESOLVED_SECONDARY_GENERATED_REPORT}" ]]
      else
        [[ -f "${RESOLVED_GENERATED_CONFIG}" && -f "${RESOLVED_GENERATED_REPORT}" ]]
      fi
      ;;
    cli|perplexity)
      if [[ "${RESOLVED_THRESHOLD_ENABLED}" == "1" ]]; then
        if [[ -n "${RESOLVED_SECONDARY_ACTIVATION_KIND}" ]]; then
          [[ -f "${RESOLVED_THRESHOLD_REPORT}" && -f "${RESOLVED_SECONDARY_THRESHOLD_REPORT}" ]]
        else
          [[ -f "${RESOLVED_THRESHOLD_REPORT}" ]]
        fi
      else
        [[ -f "${default_log}" ]]
      fi
      ;;
    decode-stats)
      [[ -d "${case_dir}/decode_stats" || -f "${default_log}" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

prepare_build_for_modes() {
  local build_dir="$1"
  local skip_build="$2"
  shift 2

  build_case_flags

  local mode=""
  local target=""
  local -a targets=()

  for mode in "$@"; do
    target="$(build_target_for_mode "${mode}")"
    [[ -n "${target}" ]] || continue

    if [[ " ${targets[*]} " != *" ${target} "* ]]; then
      targets+=("${target}")
    fi
  done

  [[ "${#targets[@]}" -gt 0 ]] || return 0

  local build_bin="./${build_dir}/bin"

  if [[ "${skip_build}" == "1" ]]; then
    for target in "${targets[@]}"; do
      [[ -x "${build_bin}/${target}" ]] || {
        echo "missing prebuilt binary: ${build_bin}/${target}" >&2
        echo "set SKIP_BUILD=0 once to build it, or point BUILD_DIR at an existing build tree" >&2
        exit 1
      }
    done
    return 0
  fi

  rm -rf "${build_dir}"

  cmake -B "${build_dir}" \
    -DCMAKE_C_FLAGS="${SIM_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${SIM_FLAGS}" \
    -DLLAMA_CURL=OFF

  cmake --build "${build_dir}" --config Release --target "${targets[@]}" -j "${THREADS}"
}

print_mode_banner() {
  local mode="$1"
  local case_dir="$2"
  local default_log="$3"
  local skip_build="$4"
  local effective_n_predict="$5"

  echo "============================================================"
  echo "Running case: ${CASE_NAME}"
  echo "run kind    : ${mode}"
  echo "case dir    : ${case_dir}"
  echo "model       : ${MODEL}"
  echo "ctx/b/ub    : ${CTX}/${BATCH}/${UBATCH}"
  echo "n_predict   : ${effective_n_predict}"
  echo "fp8/layout  : ${SIM_FP8}/${SIM_FP8_LAYOUT}"
  echo "scale in/out: ${SIM_FP8_SCALE_TYPE_IN}/${SIM_FP8_SCALE_TYPE_OUT}"

  case "${mode}" in
    swiglu-collect)
      if [[ -n "${RESOLVED_SECONDARY_ACTIVATION_KIND}" ]]; then
        echo "activation  : ${RESOLVED_ACTIVATION_KIND} collect (silu_prefix=${RESOLVED_COLLECT_PREFIX}, swiglu_prefix=${RESOLVED_SECONDARY_COLLECT_PREFIX}, bins=${SWIGLU_COLLECT_BINS}, log10=[${SWIGLU_COLLECT_LOG10_MIN}, ${SWIGLU_COLLECT_LOG10_MAX}])"
      else
        if [[ "${SWIGLU_GENERATE_MODE}" == "channel-max" && "${RESOLVED_PRIMARY_ACTIVATION_KIND}" == "silu" ]]; then
          echo "activation  : ${RESOLVED_ACTIVATION_KIND} collect (prefix=${RESOLVED_COLLECT_PREFIX}, bins=${SWIGLU_COLLECT_BINS}, log10=[${SWIGLU_COLLECT_LOG10_MIN}, ${SWIGLU_COLLECT_LOG10_MAX}], channel-max=on)"
        else
          echo "activation  : ${RESOLVED_ACTIVATION_KIND} collect (prefix=${RESOLVED_COLLECT_PREFIX}, bins=${SWIGLU_COLLECT_BINS}, log10=[${SWIGLU_COLLECT_LOG10_MIN}, ${SWIGLU_COLLECT_LOG10_MAX}])"
        fi
      fi
      ;;
    swiglu-generate)
      if [[ -n "${RESOLVED_SECONDARY_ACTIVATION_KIND}" ]]; then
        echo "activation  : ${RESOLVED_ACTIVATION_KIND} generate (profile_kind=${SWIGLU_TARGET_PROFILE_KIND}, swiglu_profile=${RESOLVED_SECONDARY_TARGET_PROFILE}, silu_profile=${RESOLVED_TARGET_PROFILE}, scale=${SWIGLU_TARGET_SCALE})"
      else
        if [[ "${SWIGLU_GENERATE_MODE}" == "channel-max" ]]; then
          echo "activation  : ${RESOLVED_ACTIVATION_KIND} generate (mode=channel-max, ratio=${SWIGLU_CHANNEL_THRESHOLD_RATIO}, collect_prefix=${RESOLVED_COLLECT_PREFIX})"
        else
          echo "activation  : ${RESOLVED_ACTIVATION_KIND} generate (profile_kind=${SWIGLU_TARGET_PROFILE_KIND}, profile=${RESOLVED_TARGET_PROFILE}, scale=${SWIGLU_TARGET_SCALE})"
        fi
      fi
      ;;
    cli|perplexity)
      if [[ "${RESOLVED_THRESHOLD_ENABLED}" == "1" ]]; then
        if [[ -n "${RESOLVED_SECONDARY_ACTIVATION_KIND}" ]]; then
          echo "activation  : ${RESOLVED_ACTIVATION_KIND} threshold on (profile=${SWIGLU_THRESHOLD_PROFILE}, silu_config=${RESOLVED_THRESHOLD_CONFIG}, swiglu_config=${RESOLVED_SECONDARY_THRESHOLD_CONFIG})"
        else
          echo "activation  : ${RESOLVED_ACTIVATION_KIND} threshold on (profile=${SWIGLU_THRESHOLD_PROFILE}, config=${RESOLVED_THRESHOLD_CONFIG})"
        fi
      else
        echo "activation  : ${RESOLVED_ACTIVATION_KIND} threshold off"
      fi
      ;;
    *)
      echo "activation  : ${RESOLVED_ACTIVATION_KIND} threshold off"
      ;;
  esac

  if [[ "${mode}" == "swiglu-generate" ]]; then
    echo "build dir   : n/a (generator only)"
  else
    echo "build dir   : ${BUILD_DIR} (skip build: ${skip_build})"
  fi
  echo "default log : ${default_log}"
}

report_mode_artifacts() {
  local mode="$1"
  local profiler_log="$2"
  local profiler_interval_hist_csv="$3"

  if is_analysis_mode "${mode}"; then
    if [[ -f fp8_sim_analysis.log ]]; then
      mv -f fp8_sim_analysis.log "${profiler_log}"
      echo "profiler log: ${profiler_log}"
    else
      echo "profiler log: not generated"
    fi

    if [[ -f fp8_sim_analysis_interval_hist.csv ]]; then
      mv -f fp8_sim_analysis_interval_hist.csv "${profiler_interval_hist_csv}"
      echo "fp8 interval csv: ${profiler_interval_hist_csv}"
    else
      echo "fp8 interval csv: not generated"
    fi
  fi

  if [[ "${RESOLVED_THRESHOLD_ENABLED}" == "1" ]]; then
    if [[ -f "${RESOLVED_THRESHOLD_REPORT}" ]]; then
      echo "${RESOLVED_PRIMARY_ACTIVATION_KIND} threshold report: ${RESOLVED_THRESHOLD_REPORT}"
    else
      echo "${RESOLVED_PRIMARY_ACTIVATION_KIND} threshold report: not generated"
    fi
    if [[ -n "${RESOLVED_SECONDARY_ACTIVATION_KIND}" ]]; then
      if [[ -f "${RESOLVED_SECONDARY_THRESHOLD_REPORT}" ]]; then
        echo "${RESOLVED_SECONDARY_ACTIVATION_KIND} threshold report: ${RESOLVED_SECONDARY_THRESHOLD_REPORT}"
      else
        echo "${RESOLVED_SECONDARY_ACTIVATION_KIND} threshold report: not generated"
      fi
    fi
  fi

  if [[ "${mode}" == "swiglu-collect" ]]; then
    if [[ -f "${RESOLVED_COLLECT_SUMMARY}" ]]; then
      echo "${RESOLVED_PRIMARY_ACTIVATION_KIND} collect summary: ${RESOLVED_COLLECT_SUMMARY}"
    else
      echo "${RESOLVED_PRIMARY_ACTIVATION_KIND} collect summary: not generated"
    fi
    if [[ -f "${RESOLVED_COLLECT_HIST}" ]]; then
      echo "${RESOLVED_PRIMARY_ACTIVATION_KIND} collect hist   : ${RESOLVED_COLLECT_HIST}"
    else
      echo "${RESOLVED_PRIMARY_ACTIVATION_KIND} collect hist   : not generated"
    fi
    if [[ -f "${RESOLVED_COLLECT_PREFIX}_channel_max.csv" ]]; then
      echo "${RESOLVED_PRIMARY_ACTIVATION_KIND} collect channel-max: ${RESOLVED_COLLECT_PREFIX}_channel_max.csv"
    fi
    if [[ -n "${RESOLVED_SECONDARY_ACTIVATION_KIND}" ]]; then
      if [[ -f "${RESOLVED_SECONDARY_COLLECT_SUMMARY}" ]]; then
        echo "${RESOLVED_SECONDARY_ACTIVATION_KIND} collect summary: ${RESOLVED_SECONDARY_COLLECT_SUMMARY}"
      else
        echo "${RESOLVED_SECONDARY_ACTIVATION_KIND} collect summary: not generated"
      fi
      if [[ -f "${RESOLVED_SECONDARY_COLLECT_HIST}" ]]; then
        echo "${RESOLVED_SECONDARY_ACTIVATION_KIND} collect hist   : ${RESOLVED_SECONDARY_COLLECT_HIST}"
      else
        echo "${RESOLVED_SECONDARY_ACTIVATION_KIND} collect hist   : not generated"
      fi
    fi
  fi

  if [[ "${mode}" == "swiglu-generate" ]]; then
    if [[ -f "${RESOLVED_GENERATED_CONFIG}" ]]; then
      echo "${RESOLVED_PRIMARY_ACTIVATION_KIND} generated config : ${RESOLVED_GENERATED_CONFIG}"
    else
      echo "${RESOLVED_PRIMARY_ACTIVATION_KIND} generated config : not generated"
    fi
    if [[ -f "${RESOLVED_GENERATED_REPORT}" ]]; then
      echo "${RESOLVED_PRIMARY_ACTIVATION_KIND} generated report : ${RESOLVED_GENERATED_REPORT}"
    else
      echo "${RESOLVED_PRIMARY_ACTIVATION_KIND} generated report : not generated"
    fi
    if [[ -n "${RESOLVED_SECONDARY_ACTIVATION_KIND}" ]]; then
      if [[ -f "${RESOLVED_SECONDARY_GENERATED_CONFIG}" ]]; then
        echo "${RESOLVED_SECONDARY_ACTIVATION_KIND} generated config : ${RESOLVED_SECONDARY_GENERATED_CONFIG}"
      else
        echo "${RESOLVED_SECONDARY_ACTIVATION_KIND} generated config : not generated"
      fi
      if [[ -f "${RESOLVED_SECONDARY_GENERATED_REPORT}" ]]; then
        echo "${RESOLVED_SECONDARY_ACTIVATION_KIND} generated report : ${RESOLVED_SECONDARY_GENERATED_REPORT}"
      else
        echo "${RESOLVED_SECONDARY_ACTIVATION_KIND} generated report : not generated"
      fi
    fi
  fi
}

postprocess_mode_outputs() {
  local mode="$1"
  local case_slug="$2"
  local case_dir="$3"
  local profiler_interval_hist_csv="$4"

  is_analysis_mode "${mode}" || return 0

  ANALYSIS_MODES_RAN=1

  local profiler_interval_hist_png="${case_dir}/${case_slug}_fp8_interval_hist.png"
  local relation_prefix="${case_dir}/${case_slug}_reduction_prod_profile"
  local relation_csv="${case_dir}/${case_slug}_block_psum_relation.csv"
  local relation_md="${case_dir}/${case_slug}_block_psum_relation.md"

  if [[ -f "${profiler_interval_hist_csv}" && -f "scripts/plot_fp8_interval_hist.py" ]]; then
    python3 scripts/plot_fp8_interval_hist.py \
      --csv "${profiler_interval_hist_csv}" \
      --out "${profiler_interval_hist_png}" || echo "fp8 interval plot: failed"
    if [[ -f "${profiler_interval_hist_png}" ]]; then
      echo "fp8 interval png: ${profiler_interval_hist_png}"
    else
      echo "fp8 interval png: not generated"
    fi
  fi

  if [[ -f "${relation_prefix}_block_samples.csv" && -f "scripts/analyze_block_psum_relation.py" ]]; then
    python3 scripts/analyze_block_psum_relation.py \
      --prefix "${relation_prefix}" \
      --thresholds 5,10,15 \
      --out-csv "${relation_csv}" \
      --out-md "${relation_md}" || echo "relation analysis: failed"
    if [[ -f "${relation_csv}" ]]; then
      echo "relation csv: ${relation_csv}"
    else
      echo "relation csv: not generated"
    fi
    if [[ -f "${relation_md}" ]]; then
      echo "relation md : ${relation_md}"
    else
      echo "relation md : not generated"
    fi
  fi

  if [[ -f "${relation_prefix}_block_samples.csv" && -f "scripts/plot_block_psum_relation.py" ]]; then
    python3 scripts/plot_block_psum_relation.py \
      --prefix "${relation_prefix}" \
      --thresholds 5,10,15 \
      --out-dir "${case_dir}" || echo "relation plots: failed"
    if [[ -f "${case_dir}/block_over_psum_hist.png" ]]; then
      echo "relation hist: ${case_dir}/block_over_psum_hist.png"
    else
      echo "relation hist: not generated"
    fi
    if [[ -f "${case_dir}/block_over_psum_percent.png" ]]; then
      echo "relation pct : ${case_dir}/block_over_psum_percent.png"
    else
      echo "relation pct : not generated"
    fi
    if [[ -f "${case_dir}/block_over_psum_cdf.png" ]]; then
      echo "relation cdf : ${case_dir}/block_over_psum_cdf.png"
    else
      echo "relation cdf : not generated"
    fi
  fi

  if [[ "${DUMP_DOT}" == "1" && -f "${DOT_FILE}" ]] && command -v dot >/dev/null 2>&1; then
    dot -Tpng "${DOT_FILE}" -o "${DOT_FILE%.dot}.png"
  fi
}

run_single_mode() {
  local case_slug="$1"
  local case_dir="$2"
  local mode="$3"
  local build_prepared="$4"

  local build_dir="${BUILD_DIR}"
  local build_bin="./${build_dir}/bin"
  local default_log="${case_dir}/${case_slug}_${mode}.log"
  local profiler_log="${case_dir}/${case_slug}_fp8_sim_analysis.log"
  local profiler_interval_hist_csv="${case_dir}/${case_slug}_fp8_interval_hist.csv"
  local decode_stats_dir="${case_dir}/decode_stats"
  local effective_n_predict="$(effective_n_predict_for_mode "${mode}")"
  local build_skip_value="${SKIP_BUILD}"

  mkdir -p "${OUT_DIR}" "${case_dir}"
  resolve_activation_context "${case_slug}" "${case_dir}" "${mode}"
  ensure_mode_inputs "${mode}"
  print_mode_banner "${mode}" "${case_dir}" "${default_log}" "${build_skip_value}" "${effective_n_predict}"

  if [[ "${build_prepared}" != "1" ]]; then
    prepare_build_for_modes "${build_dir}" "${build_skip_value}" "${mode}"
  fi

  export FP8_SIM_STATS_SAMPLE="${FP8_SIM_STATS_SAMPLE}"
  export GGML_MATMUL_DIST=0

  if [[ "${DUMP_DOT}" == "1" ]]; then
    DOT_FILE="${case_dir}/${case_slug}.dot"
    export LLAMA_DUMP_DOT="${DOT_FILE}"
  else
    unset LLAMA_DUMP_DOT || true
  fi

  rm -f fp8_sim_analysis.log fp8_sim_analysis_interval_hist.csv

  case "${mode}" in
    cli)
      "${build_bin}/llama-cli" \
        -no-cnv \
        -m "${MODEL}" \
        -p "${PROMPT}" \
        -c "${CTX}" \
        -t "${THREADS}" \
        -n "${effective_n_predict}" \
        --seq-state-out-id "${SEQ_ID}" \
        "${THRESHOLD_ARGS[@]}" \
        2>&1 | tee "${default_log}"
      ;;
    perplexity)
      "${build_bin}/llama-perplexity" \
        -m "${MODEL}" \
        -f "${DATA}" \
        -c "${CTX}" \
        -b "${BATCH}" \
        -ub "${UBATCH}" \
        -t "${THREADS}" \
        --ppl-stride "${STRIDE}" \
        "${THRESHOLD_ARGS[@]}" \
        2>&1 | tee "${default_log}"
      ;;
    decode-stats)
      "${build_bin}/llama-decode-stats" \
        -m "${MODEL}" \
        -p "${PROMPT}" \
        -c "${CTX}" \
        -t "${THREADS}" \
        -n "${effective_n_predict}" \
        -o "${decode_stats_dir}" \
        2>&1 | tee "${default_log}"
      ;;
    swiglu-collect)
      if [[ -n "${RESOLVED_SECONDARY_ACTIVATION_KIND}" ]]; then
        "${build_bin}/llama-cli" \
          -no-cnv \
          -m "${MODEL}" \
          -p "${PROMPT}" \
          -c "${CTX}" \
          -t "${THREADS}" \
          -n "${effective_n_predict}" \
          --seq-state-out-id "${SEQ_ID}" \
          --swiglu-threshold-kind "${RESOLVED_PRIMARY_ACTIVATION_KIND}" \
          --swiglu-threshold-collect "${RESOLVED_COLLECT_PREFIX}" \
          --swiglu-threshold-collect-bins "${SWIGLU_COLLECT_BINS}" \
          --swiglu-threshold-collect-log-min "${SWIGLU_COLLECT_LOG10_MIN}" \
          --swiglu-threshold-collect-log-max "${SWIGLU_COLLECT_LOG10_MAX}" \
          --swiglu-threshold-secondary-kind "${RESOLVED_SECONDARY_ACTIVATION_KIND}" \
          --swiglu-threshold-secondary-collect "${RESOLVED_SECONDARY_COLLECT_PREFIX}" \
          2>&1 | tee "${default_log}"
      else
        "${build_bin}/llama-cli" \
          -no-cnv \
          -m "${MODEL}" \
          -p "${PROMPT}" \
          -c "${CTX}" \
          -t "${THREADS}" \
          -n "${effective_n_predict}" \
          --seq-state-out-id "${SEQ_ID}" \
          --swiglu-threshold-kind "${RESOLVED_PRIMARY_ACTIVATION_KIND}" \
          --swiglu-threshold-collect "${RESOLVED_COLLECT_PREFIX}" \
          --swiglu-threshold-collect-bins "${SWIGLU_COLLECT_BINS}" \
          --swiglu-threshold-collect-log-min "${SWIGLU_COLLECT_LOG10_MIN}" \
          --swiglu-threshold-collect-log-max "${SWIGLU_COLLECT_LOG10_MAX}" \
          2>&1 | tee "${default_log}"
      fi
      ;;
    swiglu-generate)
      if [[ -n "${RESOLVED_SECONDARY_ACTIVATION_KIND}" ]]; then
        {
          "${MODE_PYTHON_BIN}" tools/swiglu-threshold-configs/generate.py \
            --collect-prefix "${RESOLVED_SECONDARY_COLLECT_PREFIX}" \
            --target-profile "${RESOLVED_SECONDARY_TARGET_PROFILE}" \
            --output "${RESOLVED_SECONDARY_GENERATED_CONFIG}" \
            --report "${RESOLVED_SECONDARY_GENERATED_REPORT}" \
            --scale "${SWIGLU_TARGET_SCALE}" \
            --threshold-kind "${RESOLVED_SECONDARY_ACTIVATION_KIND}"
          "${MODE_PYTHON_BIN}" tools/swiglu-threshold-configs/generate.py \
            --collect-prefix "${RESOLVED_COLLECT_PREFIX}" \
            --target-profile "${RESOLVED_TARGET_PROFILE}" \
            --output "${RESOLVED_GENERATED_CONFIG}" \
            --report "${RESOLVED_GENERATED_REPORT}" \
            --scale "${SWIGLU_TARGET_SCALE}" \
            --threshold-kind "${RESOLVED_PRIMARY_ACTIVATION_KIND}"
        } 2>&1 | tee "${default_log}"
      else
        if [[ "${SWIGLU_GENERATE_MODE}" == "channel-max" ]]; then
          "${MODE_PYTHON_BIN}" tools/swiglu-threshold-configs/generate.py \
            --collect-prefix "${RESOLVED_COLLECT_PREFIX}" \
            --output "${RESOLVED_GENERATED_CONFIG}" \
            --report "${RESOLVED_GENERATED_REPORT}" \
            --generate-mode channel-max \
            --channel-threshold-ratio "${SWIGLU_CHANNEL_THRESHOLD_RATIO}" \
            --threshold-kind "${RESOLVED_PRIMARY_ACTIVATION_KIND}" \
            2>&1 | tee "${default_log}"
        else
          "${MODE_PYTHON_BIN}" tools/swiglu-threshold-configs/generate.py \
            --collect-prefix "${RESOLVED_COLLECT_PREFIX}" \
            --target-profile "${RESOLVED_TARGET_PROFILE}" \
            --output "${RESOLVED_GENERATED_CONFIG}" \
            --report "${RESOLVED_GENERATED_REPORT}" \
            --scale "${SWIGLU_TARGET_SCALE}" \
            --threshold-kind "${RESOLVED_PRIMARY_ACTIVATION_KIND}" \
            2>&1 | tee "${default_log}"
        fi
      fi
      ;;
    *)
      die "unsupported mode in run_single_mode: ${mode}"
      ;;
  esac

  report_mode_artifacts "${mode}" "${profiler_log}" "${profiler_interval_hist_csv}"
  postprocess_mode_outputs "${mode}" "${case_slug}" "${case_dir}" "${profiler_interval_hist_csv}"

  FINISHED_RUNS+=("${CASE_NAME}:${mode}:${default_log}")
  USED_OUT_DIRS+=("${OUT_DIR}")
}

run_activation_flow_case() {
  local case_slug="$1"
  local case_dir="$2"
  local step=""
  local default_log=""

  echo "============================================================"
  echo "Running case: ${CASE_NAME}"
  echo "run kind    : activation-flow"
  echo "case dir    : ${case_dir}"
  echo "flow steps  : ${FLOW_STEPS}"
  echo "flow reuse  : ${FLOW_REUSE_ARTIFACTS}"
  echo "build dir   : ${BUILD_DIR} (skip build: ${SKIP_BUILD})"

  CURRENT_FLOW_MODE=1
  prepare_build_for_modes "${BUILD_DIR}" "${SKIP_BUILD}" "${FLOW_STEP_LIST[@]}"

  for step in "${FLOW_STEP_LIST[@]}"; do
    resolve_activation_context "${case_slug}" "${case_dir}" "${step}"
    default_log="${case_dir}/${case_slug}_${step}.log"

    if [[ "${FLOW_REUSE_ARTIFACTS}" == "1" ]] && primary_artifacts_exist_for_mode "${step}" "${case_dir}" "${default_log}"; then
      echo "--- step: ${step} (reuse existing artifacts) ---"
      FINISHED_RUNS+=("${CASE_NAME}:${step}:reused")
      USED_OUT_DIRS+=("${OUT_DIR}")
      continue
    fi

    echo "--- step: ${step} ---"
    run_single_mode "${case_slug}" "${case_dir}" "${step}" "1"
  done

  CURRENT_FLOW_MODE=0
}

maybe_generate_global_compare_tables() {
  [[ "${ANALYSIS_MODES_RAN}" == "1" ]] || return 0
  [[ -f "scripts/make_reduction_drop_compare_table.py" ]] || return 0

  local unique_out_dirs=""
  local out_root=""
  local compare_csv=""
  local compare_md=""

  unique_out_dirs=$(printf "%s\n" "${USED_OUT_DIRS[@]}" | awk 'NF' | sort -u)
  for out_root in ${unique_out_dirs}; do
    compare_csv="${out_root}/reduction_block_drop_compare.csv"
    compare_md="${out_root}/reduction_block_drop_compare.md"
    python3 scripts/make_reduction_drop_compare_table.py \
      --root "${out_root}" \
      --out-csv "${compare_csv}" \
      --out-md "${compare_md}" || echo "compare table: failed for ${out_root}"
    if [[ -f "${compare_csv}" ]]; then
      echo "compare csv : ${compare_csv}"
    else
      echo "compare csv : not generated"
    fi
    if [[ -f "${compare_md}" ]]; then
      echo "compare md  : ${compare_md}"
    else
      echo "compare md  : not generated"
    fi
  done
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

mkdir -p "${DEFAULT_OUT_DIR}"

[[ "${#RUN_CASES[@]}" -gt 0 ]] || die "RUN_CASES is empty"

remember_user_env_overrides

matched_cases=0

for case_spec in "${RUN_CASES[@]}"; do
  reset_case_defaults
  apply_case_overrides "${case_spec}"
  apply_user_env_overrides

  case_slug="$(sanitize_name "${CASE_NAME}")"
  [[ -n "${case_slug}" ]] || die "invalid case name: ${CASE_NAME}"

  if [[ -n "${CASE_FILTER}" && "${CASE_NAME}" != *"${CASE_FILTER}"* && "${case_slug}" != *"${CASE_FILTER}"* ]]; then
    continue
  fi

  matched_cases=$((matched_cases + 1))
  validate_case

  case_dir="${OUT_DIR}/${case_slug}"
  REDUCTION_PROD_PROFILE_PREFIX="${case_dir}/${case_slug}_reduction_prod_profile"

  if [[ "${RUN_KIND}" == "activation-flow" ]]; then
    run_activation_flow_case "${case_slug}" "${case_dir}"
  else
    run_single_mode "${case_slug}" "${case_dir}" "${RUN_KIND}" "0"
  fi
done

[[ "${matched_cases}" -gt 0 ]] || die "no cases matched CASE_FILTER=${CASE_FILTER}"

maybe_generate_global_compare_tables

echo "done."
for finished in "${FINISHED_RUNS[@]}"; do
  echo "${finished}"
done
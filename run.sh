#!/usr/bin/env bash
set -euo pipefail

DEFAULT_MODEL="${MODEL:-models/Qwen/Qwen3-1.7B-Base-f16.gguf}"
DEFAULT_DATA="${DATA:-models/hf/wiki.test.raw}"
DEFAULT_OUT_DIR="${OUT_DIR:-kv_dump_logs}"
DEFAULT_PROMPT="${PROMPT:-你好，请简要介绍一下KV cache。}"

DEFAULT_CTX="${CTX:-512}"
DEFAULT_THREADS="${THREADS:-$(nproc)}"
DEFAULT_N_PREDICT="${N_PREDICT:--1}"
DEFAULT_SEQ_ID="${SEQ_ID:-0}"
DEFAULT_BATCH="${BATCH:-2048}"
DEFAULT_UBATCH="${UBATCH:-512}"
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
# Legacy single switch (kept for compatibility)
DEFAULT_SIM_FP8_SCALE_TYPE="${SIM_FP8_SCALE_TYPE:-1}"
# New split switches
DEFAULT_SIM_FP8_SCALE_TYPE_IN="${SIM_FP8_SCALE_TYPE_IN:-${DEFAULT_SIM_FP8_SCALE_TYPE}}" 
DEFAULT_SIM_FP8_SCALE_TYPE_OUT="${SIM_FP8_SCALE_TYPE_OUT:-${DEFAULT_SIM_FP8_SCALE_TYPE}}"
DEFAULT_SIM_FP8_BLOCK="${SIM_FP8_BLOCK:-16}"
DEFAULT_SIM_MATMUL_OUT_MODE="${SIM_MATMUL_OUT_MODE:-1}"
DEFAULT_SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE="${SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE:-0}"
DEFAULT_SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP="${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP:--3}"
DEFAULT_SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP="${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP:--1}"

# Reduction-product profiler switches
# 0/1: enable online reduction-product profiling in BF16 dot kernels.
# Maps to: GGML_REDUCTION_PROD_PROFILE
DEFAULT_REDUCTION_PROD_PROFILE="${REDUCTION_PROD_PROFILE:-0}"

# Number of histogram bins for global |x*y| magnitude distribution (log2 domain).
# Maps to: GGML_REDUCTION_PROD_PROFILE_BINS
DEFAULT_REDUCTION_PROD_PROFILE_BINS="${REDUCTION_PROD_PROFILE_BINS:-256}"

# Lower bound (inclusive bin edge domain) for log2(|x*y|) histogram.
# Maps to: GGML_REDUCTION_PROD_PROFILE_HIST_MIN_LOG2
DEFAULT_REDUCTION_PROD_PROFILE_HIST_MIN_LOG2="${REDUCTION_PROD_PROFILE_HIST_MIN_LOG2:--128}" 

# Upper bound for log2(|x*y|) histogram. Must be greater than MIN_LOG2.
# Maps to: GGML_REDUCTION_PROD_PROFILE_HIST_MAX_LOG2
DEFAULT_REDUCTION_PROD_PROFILE_HIST_MAX_LOG2="${REDUCTION_PROD_PROFILE_HIST_MAX_LOG2:-128}"

# Keep 1 sampled reduction record per N reductions in samples.csv.
# Maps to: GGML_REDUCTION_PROD_PROFILE_SAMPLE_RATE
DEFAULT_REDUCTION_PROD_PROFILE_SAMPLE_RATE="${REDUCTION_PROD_PROFILE_SAMPLE_RATE:-1000}"

# Simulated block-drop threshold exponent n in:
#   |block_dot| < |running_sum| * 2^-n  => counted as dropped (stats only)
# Maps to: GGML_REDUCTION_PROD_BLOCK_DROP_LOG2_N
DEFAULT_REDUCTION_PROD_BLOCK_DROP_LOG2_N="${REDUCTION_PROD_BLOCK_DROP_LOG2_N:-10}"

# Max sampled reduction records retained in memory before dropping extras.
# Maps to: GGML_REDUCTION_PROD_PROFILE_MAX_SAMPLES
DEFAULT_REDUCTION_PROD_PROFILE_MAX_SAMPLES="${REDUCTION_PROD_PROFILE_MAX_SAMPLES:-2000}"

# Output file prefix for profiler artifacts:
#   <prefix>_summary.log
#   <prefix>_global_hist.csv
#   <prefix>_samples.csv
#   <prefix>_block_samples.csv
# Maps to: GGML_REDUCTION_PROD_PROFILE_PREFIX
DEFAULT_REDUCTION_PROD_PROFILE_PREFIX="${REDUCTION_PROD_PROFILE_PREFIX:-reduction_prod_profile}"

# Graph dump switches
DEFAULT_DUMP_DOT="${DUMP_DOT:-0}"

RUN_KIND="${RUN_KIND:-perplexity}"

# 在这里维护一组测试用例。
# 格式: "用例名|KEY=VALUE|KEY=VALUE|..."
# 这里尽量把所有 case 级配置项都显式写全，避免阅读时再去反推 DEFAULT_*。
# 例外：
#   - RUN_KIND 是全局运行模式，不是 case 级字段；
#   - REDUCTION_PROD_PROFILE_PREFIX 会在运行时按 case_dir/case_slug 自动派生。
#
# 每个 case 内部按以下顺序排版：
#   1. 模型/输入路径
#   2. 运行参数
#   3. FP8 仿真参数
#   4. reduction profiler 参数
#   5. 其他开关
# 可在 case 字符串内部使用 "|# ..." 作为分组标题；解析时会自动忽略。
#
# 当前 layout 对照：
#   - SIM_FP8_LAYOUT=0 : E4M3
#   - SIM_FP8_LAYOUT=1 : E3M4
#   - SIM_FP8_LAYOUT=2 : E3M4_NO_SUBNORM
#   - SIM_FP8_LAYOUT=3 : E2M5
#   - SIM_FP8_LAYOUT=4 : E2M5_NO_SUBNORM

CASE_LLAMA_3___2_1B_F8E3M4_NORMAL=$(cat <<EOF
Llama-3.2-1B-f8e3m4-normal
|# paths
|MODEL=models/hf/Llama-3.2-1B-Instruct-f16.gguf
|DATA=models/hf/wiki.test.raw
|OUT_DIR=kv_dump_logs
|PROMPT=你好，请简要介绍一下KV cache。

|# runtime
|CTX=512
|THREADS=$(nproc)
|N_PREDICT=-1
|SEQ_ID=0
|BATCH=2048
|UBATCH=512
|STRIDE=0

|# fp8-sim
|SIM_FP8=1
|SIM_FP_FORMAT=8
|SIM_FP8_LAYOUT=2
|SIM_FP8_APPLY_SRC0=1
|SIM_FP8_APPLY_SRC1=1
|SIM_FP8_SCALE_TYPE=0
|SIM_FP8_SCALE_TYPE_IN=0
|SIM_FP8_SCALE_TYPE_OUT=1
|SIM_FP8_BLOCK=16
|SIM_MATMUL_OUT_MODE=1
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE=1
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP=4
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP=4

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
)

CASE_QWEN_3_1___7B_F8E3M4_NORMAL=$(cat <<EOF
Qwen-3-1.7B-f8e3m4-normal
|# paths
|MODEL=models/Qwen/Qwen3-1.7B-Base-f16.gguf
|DATA=models/hf/wiki.test.raw
|OUT_DIR=kv_dump_logs
|PROMPT=你好，请简要介绍一下KV cache。

|# runtime
|CTX=512
|THREADS=$(nproc)
|N_PREDICT=-1
|SEQ_ID=0
|BATCH=2048
|UBATCH=512
|STRIDE=0

|# fp8-sim
|SIM_FP8=1
|SIM_FP_FORMAT=8
|SIM_FP8_LAYOUT=2
|SIM_FP8_APPLY_SRC0=1
|SIM_FP8_APPLY_SRC1=1
|SIM_FP8_SCALE_TYPE=0
|SIM_FP8_SCALE_TYPE_IN=0
|SIM_FP8_SCALE_TYPE_OUT=1
|SIM_FP8_BLOCK=16
|SIM_MATMUL_OUT_MODE=1
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE=1
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP=-3
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP=-3

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
)

CASE_LLAMA_3___2_3B_F8E3M4_NORMAL=$(cat <<EOF
Llama-3.2-3B-f8e3m4-normal
|# paths
|MODEL=models/hf/Llama-3___2-3B-Instruct-f16.gguf
|DATA=models/hf/wiki.test.raw
|OUT_DIR=kv_dump_logs
|PROMPT=你好，请简要介绍一下KV cache。

|# runtime
|CTX=512
|THREADS=$(nproc)
|N_PREDICT=-1
|SEQ_ID=0
|BATCH=2048
|UBATCH=512
|STRIDE=0

|# fp8-sim
|SIM_FP8=1
|SIM_FP_FORMAT=8
|SIM_FP8_LAYOUT=2
|SIM_FP8_APPLY_SRC0=1
|SIM_FP8_APPLY_SRC1=1
|SIM_FP8_SCALE_TYPE=0
|SIM_FP8_SCALE_TYPE_IN=0
|SIM_FP8_SCALE_TYPE_OUT=1
|SIM_FP8_BLOCK=16
|SIM_MATMUL_OUT_MODE=1
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE=1
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP=-3
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP=-3

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
)

CASE_LLAMA_2_7B_F8E3M4_NORMAL=$(cat <<EOF
Llama-2-7B-f8e3m4-normal
|# paths
|MODEL=models/hf/llama-2-7B-F16.gguf
|DATA=models/hf/wiki.test.raw
|OUT_DIR=kv_dump_logs
|PROMPT=你好，请简要介绍一下KV cache。

|# runtime
|CTX=512
|THREADS=$(nproc)
|N_PREDICT=-1
|SEQ_ID=0
|BATCH=2048
|UBATCH=512
|STRIDE=0

|# fp8-sim
|SIM_FP8=1
|SIM_FP_FORMAT=8
|SIM_FP8_LAYOUT=2
|SIM_FP8_APPLY_SRC0=1
|SIM_FP8_APPLY_SRC1=1
|SIM_FP8_SCALE_TYPE=0
|SIM_FP8_SCALE_TYPE_IN=0
|SIM_FP8_SCALE_TYPE_OUT=1
|SIM_FP8_BLOCK=16
|SIM_MATMUL_OUT_MODE=1
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE=1
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP=-3
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP=-3

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
)

CASE_QWEN_3_8B_F8E3M4_NORMAL=$(cat <<EOF
Qwen-3-8B-f8e3m4-normal
|# paths
|MODEL=models/Qwen/Qwen3-8B-f16.gguf
|DATA=models/hf/wiki.test.raw
|OUT_DIR=kv_dump_logs
|PROMPT=你好，请简要介绍一下KV cache。

|# runtime
|CTX=512
|THREADS=$(nproc)
|N_PREDICT=-1
|SEQ_ID=0
|BATCH=2048
|UBATCH=512
|STRIDE=0

|# fp8-sim
|SIM_FP8=1
|SIM_FP_FORMAT=8
|SIM_FP8_LAYOUT=2
|SIM_FP8_APPLY_SRC0=1
|SIM_FP8_APPLY_SRC1=1
|SIM_FP8_SCALE_TYPE=0
|SIM_FP8_SCALE_TYPE_IN=0
|SIM_FP8_SCALE_TYPE_OUT=1
|SIM_FP8_BLOCK=16
|SIM_MATMUL_OUT_MODE=1
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE=1
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP=-3
|SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP=-3

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
)

# 这里只保留执行顺序；新增/删除 case 时优先在上面定义，再在这里引用。
RUN_CASES=(
  "$CASE_LLAMA_3___2_1B_F8E3M4_NORMAL"
  "$CASE_QWEN_3_1___7B_F8E3M4_NORMAL"
  "$CASE_LLAMA_3___2_3B_F8E3M4_NORMAL"
  "$CASE_LLAMA_2_7B_F8E3M4_NORMAL"
  "$CASE_QWEN_3_8B_F8E3M4_NORMAL"
)

sanitize_name() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

trim_case_field() {
  printf '%s' "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
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
  SIM_MATMUL_OUT_MODE="${DEFAULT_SIM_MATMUL_OUT_MODE}"
  SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE="${DEFAULT_SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE}"
  SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP="${DEFAULT_SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP}"
  SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP="${DEFAULT_SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP}"

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
}

apply_case_overrides() {
  local case_spec="$1"
  case_spec="${case_spec//$'\n'/|}"
  local IFS='|'
  read -r -a case_fields <<< "${case_spec}"

  CASE_NAME="$(trim_case_field "${case_fields[0]}")"

  local field
  for field in "${case_fields[@]:1}"; do
    field="$(trim_case_field "${field}")"
    if [[ -z "${field}" || "${field:0:1}" == "#" ]]; then
      continue
    fi
    export "${field}"
  done
}

validate_case() {
  if [[ "${SIM_MATMUL_OUT_MODE}" != "0" && "${SIM_MATMUL_OUT_MODE}" != "1" ]]; then
    echo "invalid SIM_MATMUL_OUT_MODE=${SIM_MATMUL_OUT_MODE} (expected 0 or 1)" >&2
    exit 1
  fi

  if [[ "${SIM_FP_FORMAT}" != "8" && "${SIM_FP_FORMAT}" != "9" ]]; then
    echo "invalid SIM_FP_FORMAT=${SIM_FP_FORMAT} (expected 8 or 9)" >&2
    exit 1
  fi

  if [[ "${SIM_FP8_LAYOUT}" != "0" && "${SIM_FP8_LAYOUT}" != "1" && "${SIM_FP8_LAYOUT}" != "2" && "${SIM_FP8_LAYOUT}" != "3" && "${SIM_FP8_LAYOUT}" != "4" ]]; then
    echo "invalid SIM_FP8_LAYOUT=${SIM_FP8_LAYOUT} (expected 0, 1, 2, 3 or 4)" >&2
    exit 1
  fi

  if [[ "${SIM_FP8_SCALE_TYPE_IN}" != "0" && "${SIM_FP8_SCALE_TYPE_IN}" != "1" ]]; then
    echo "invalid SIM_FP8_SCALE_TYPE_IN=${SIM_FP8_SCALE_TYPE_IN} (expected 0 or 1)" >&2
    exit 1
  fi

  if [[ "${SIM_FP8_SCALE_TYPE_OUT}" != "0" && "${SIM_FP8_SCALE_TYPE_OUT}" != "1" ]]; then
    echo "invalid SIM_FP8_SCALE_TYPE_OUT=${SIM_FP8_SCALE_TYPE_OUT} (expected 0 or 1)" >&2
    exit 1
  fi

  if [[ "${SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE}" != "0" && "${SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE}" != "1" ]]; then
    echo "invalid SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE=${SIM_FP8_E3M4_NO_SUBNORM_ZERO_ENABLE} (expected 0 or 1)" >&2
    exit 1
  fi

  if ! [[ "${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP}" =~ ^-?[0-9]+$ ]]; then
    echo "invalid SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP=${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP} (expected integer)" >&2
    exit 1
  fi

  if ! [[ "${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP}" =~ ^-?[0-9]+$ ]]; then
    echo "invalid SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP=${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP} (expected integer)" >&2
    exit 1
  fi

  if [[ "${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP}" -gt "${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP}" ]]; then
    echo "invalid small-exp erase range: min=${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MIN_EXP}, max=${SIM_FP8_E3M4_NO_SUBNORM_ZERO_MAX_EXP} (expected min <= max)" >&2
    exit 1
  fi

  if [[ "${REDUCTION_PROD_PROFILE}" != "0" && "${REDUCTION_PROD_PROFILE}" != "1" ]]; then
    echo "invalid REDUCTION_PROD_PROFILE=${REDUCTION_PROD_PROFILE} (expected 0 or 1)" >&2
    exit 1
  fi

  if ! [[ "${REDUCTION_PROD_PROFILE_BINS}" =~ ^[0-9]+$ ]] || [[ "${REDUCTION_PROD_PROFILE_BINS}" -le 0 ]]; then
    echo "invalid REDUCTION_PROD_PROFILE_BINS=${REDUCTION_PROD_PROFILE_BINS} (expected positive integer)" >&2
    exit 1
  fi

  if ! [[ "${REDUCTION_PROD_PROFILE_HIST_MIN_LOG2}" =~ ^-?[0-9]+$ ]]; then
    echo "invalid REDUCTION_PROD_PROFILE_HIST_MIN_LOG2=${REDUCTION_PROD_PROFILE_HIST_MIN_LOG2} (expected integer)" >&2
    exit 1
  fi

  if ! [[ "${REDUCTION_PROD_PROFILE_HIST_MAX_LOG2}" =~ ^-?[0-9]+$ ]]; then
    echo "invalid REDUCTION_PROD_PROFILE_HIST_MAX_LOG2=${REDUCTION_PROD_PROFILE_HIST_MAX_LOG2} (expected integer)" >&2
    exit 1
  fi

  if [[ "${REDUCTION_PROD_PROFILE_HIST_MAX_LOG2}" -le "${REDUCTION_PROD_PROFILE_HIST_MIN_LOG2}" ]]; then
    echo "invalid reduction profile log2 range: min=${REDUCTION_PROD_PROFILE_HIST_MIN_LOG2}, max=${REDUCTION_PROD_PROFILE_HIST_MAX_LOG2} (expected max > min)" >&2
    exit 1
  fi

  if ! [[ "${REDUCTION_PROD_PROFILE_SAMPLE_RATE}" =~ ^[0-9]+$ ]] || [[ "${REDUCTION_PROD_PROFILE_SAMPLE_RATE}" -le 0 ]]; then
    echo "invalid REDUCTION_PROD_PROFILE_SAMPLE_RATE=${REDUCTION_PROD_PROFILE_SAMPLE_RATE} (expected positive integer)" >&2
    exit 1
  fi

  if ! [[ "${REDUCTION_PROD_BLOCK_DROP_LOG2_N}" =~ ^[0-9]+$ ]]; then
    echo "invalid REDUCTION_PROD_BLOCK_DROP_LOG2_N=${REDUCTION_PROD_BLOCK_DROP_LOG2_N} (expected non-negative integer)" >&2
    exit 1
  fi

  if ! [[ "${REDUCTION_PROD_PROFILE_MAX_SAMPLES}" =~ ^[0-9]+$ ]] || [[ "${REDUCTION_PROD_PROFILE_MAX_SAMPLES}" -le 0 ]]; then
    echo "invalid REDUCTION_PROD_PROFILE_MAX_SAMPLES=${REDUCTION_PROD_PROFILE_MAX_SAMPLES} (expected positive integer)" >&2
    exit 1
  fi

  if [[ -z "${REDUCTION_PROD_PROFILE_PREFIX}" ]]; then
    echo "invalid REDUCTION_PROD_PROFILE_PREFIX: empty string" >&2
    exit 1
  fi

  if [[ "${RUN_KIND}" != "perplexity" && "${RUN_KIND}" != "cli" ]]; then
    echo "invalid RUN_KIND=${RUN_KIND} (expected perplexity or cli)" >&2
    exit 1
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

run_case() {
  local case_slug="$1"
  local case_dir="$2"
  local default_log="$3"
  local profiler_log="$4"
  local build_target="llama-perplexity"

  build_case_flags

  if [[ "${RUN_KIND}" == "cli" ]]; then
    build_target="llama-cli"
  fi

  rm -rf build
  mkdir -p "${OUT_DIR}" "${case_dir}"

  cmake -B build \
    -DCMAKE_C_FLAGS="${SIM_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${SIM_FLAGS}" \
    -DLLAMA_CURL=OFF

  cmake --build build --config Release --target "${build_target}" -j "${THREADS}"

  export FP8_SIM_STATS_SAMPLE=100
  export GGML_MATMUL_DIST=0

  if [[ "${DUMP_DOT}" == "1" ]]; then
    DOT_FILE="${case_dir}/${case_slug}.dot"
    export LLAMA_DUMP_DOT="${DOT_FILE}"
  else
    unset LLAMA_DUMP_DOT || true
  fi

  rm -f fp8_sim_analysis.log

  if [[ "${RUN_KIND}" == "cli" ]]; then
    ./build/bin/llama-cli \
      -m "${MODEL}" \
      -p "${PROMPT}" \
      -c "${CTX}" \
      -t "${THREADS}" \
      -n "${N_PREDICT}" \
      --seq-state-out-id "${SEQ_ID}" \
      2>&1 | tee "${default_log}"
  else
    ./build/bin/llama-perplexity \
      -m "${MODEL}" \
      -f "${DATA}" \
      -c "${CTX}" \
      -b "${BATCH}" \
      -ub "${UBATCH}" \
      -t "${THREADS}" \
      --ppl-stride "${STRIDE}" \
      2>&1 | tee "${default_log}"
  fi

  if [[ -f fp8_sim_analysis.log ]]; then
    mv -f fp8_sim_analysis.log "${profiler_log}"
    echo "profiler log: ${profiler_log}"
  else
    echo "profiler log: not generated"
  fi

  if [[ "${DUMP_DOT}" == "1" && -f "${DOT_FILE}" ]] && command -v dot >/dev/null 2>&1; then
    dot -Tpng "${DOT_FILE}" -o "${DOT_FILE%.dot}.png"
  fi
}

# source ~/miniforge3/bin/activate
mkdir -p "${DEFAULT_OUT_DIR}"

if [[ "${#RUN_CASES[@]}" -eq 0 ]]; then
  echo "RUN_CASES is empty" >&2
  exit 1
fi

declare -a finished_cases=()
declare -a used_out_dirs=()

for case_spec in "${RUN_CASES[@]}"; do
  reset_case_defaults
  apply_case_overrides "${case_spec}"
  validate_case

  case_slug="$(sanitize_name "${CASE_NAME}")"
  if [[ -z "${case_slug}" ]]; then
    echo "invalid case name: ${CASE_NAME}" >&2
    exit 1
  fi

  case_dir="${OUT_DIR}/${case_slug}"
  default_log="${case_dir}/${case_slug}_${RUN_KIND}.log"
  profiler_log="${case_dir}/${case_slug}_fp8_sim_analysis.log"

  REDUCTION_PROD_PROFILE_PREFIX="${case_dir}/${case_slug}_reduction_prod_profile"

  echo "============================================================"
  echo "Running case: ${CASE_NAME}"
  echo "case dir    : ${case_dir}"
  echo "model       : ${MODEL}"
  echo "ctx/b/ub    : ${CTX}/${BATCH}/${UBATCH}"
  echo "fp8/layout  : ${SIM_FP8}/${SIM_FP8_LAYOUT}"
  echo "scale in/out: ${SIM_FP8_SCALE_TYPE_IN}/${SIM_FP8_SCALE_TYPE_OUT}"
  echo "default log : ${default_log}"

  run_case "${case_slug}" "${case_dir}" "${default_log}" "${profiler_log}"

  relation_prefix="${case_dir}/${case_slug}_reduction_prod_profile"
  if [[ -f "${relation_prefix}_block_samples.csv" ]] && [[ -f "scripts/analyze_block_psum_relation.py" ]]; then
    relation_csv="${case_dir}/${case_slug}_block_psum_relation.csv"
    relation_md="${case_dir}/${case_slug}_block_psum_relation.md"
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
  if [[ -f "${relation_prefix}_block_samples.csv" ]] && [[ -f "scripts/plot_block_psum_relation.py" ]]; then
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

  finished_cases+=("${CASE_NAME}:${default_log}")
  used_out_dirs+=("${OUT_DIR}")
done

if [[ -f "scripts/make_reduction_drop_compare_table.py" ]]; then
  # Generate concise comparison tables for each output root used in this run.
  unique_out_dirs=$(printf "%s\n" "${used_out_dirs[@]}" | awk 'NF' | sort -u)
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
fi

echo "done."
for finished in "${finished_cases[@]}"; do
  echo "${finished}"
done
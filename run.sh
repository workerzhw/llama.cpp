#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-models/hf/llama-2-7B-F16.gguf}"
DATA="${DATA:-models/hf/wiki.test.raw}"
OUT_DIR="${OUT_DIR:-kv_dump_logs}"
PROMPT="${PROMPT:-你好，请简要介绍一下KV cache。}"

CTX="${CTX:-512}"
THREADS="${THREADS:-$(nproc)}"
N_PREDICT="${N_PREDICT:--1}"
SEQ_ID="${SEQ_ID:-0}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"
STRIDE="${STRIDE:-0}"

# Compile-time simulation switches
# GGML_SIM_MATMUL_OUT_MODE:
#   0 => FP8 output QDQ (E4M3/E3M4/E2M5 selected by SIM_FP8_LAYOUT, gated by GGML_SIM_FP8E4M3)
#   1 => BF16 round-trip output simulation (F32 -> BF16 -> F32)
SIM_FP8="${SIM_FP8:-1}"
SIM_FP_FORMAT="${SIM_FP_FORMAT:-8}"
# FP8 sub-format (effective only when SIM_FP_FORMAT=8):
#   0=E4M3, 1=E3M4, 2=E3M4_NO_SUBNORM, 3=E2M5, 4=E2M5_NO_SUBNORM
SIM_FP8_LAYOUT="${SIM_FP8_LAYOUT:-3}"
SIM_FP8_APPLY_SRC0="${SIM_FP8_APPLY_SRC0:-1}"
SIM_FP8_APPLY_SRC1="${SIM_FP8_APPLY_SRC1:-1}"
# Legacy single switch (kept for compatibility)
SIM_FP8_SCALE_TYPE="${SIM_FP8_SCALE_TYPE:-0}"
# New split switches
SIM_FP8_SCALE_TYPE_IN="${SIM_FP8_SCALE_TYPE_IN:-${SIM_FP8_SCALE_TYPE}}"
SIM_FP8_SCALE_TYPE_OUT="${SIM_FP8_SCALE_TYPE_OUT:-${SIM_FP8_SCALE_TYPE}}"
SIM_FP8_BLOCK="${SIM_FP8_BLOCK:-16}"
SIM_MATMUL_OUT_MODE="${SIM_MATMUL_OUT_MODE:-1}"

# Reduction-product profiler switches
# 0/1: enable online reduction-product profiling in BF16 dot kernels.
# Maps to: GGML_REDUCTION_PROD_PROFILE
REDUCTION_PROD_PROFILE="${REDUCTION_PROD_PROFILE:-0}"

# Number of histogram bins for global |x*y| magnitude distribution (log2 domain).
# Maps to: GGML_REDUCTION_PROD_PROFILE_BINS
REDUCTION_PROD_PROFILE_BINS="${REDUCTION_PROD_PROFILE_BINS:-128}"

# Lower bound (inclusive bin edge domain) for log2(|x*y|) histogram.
# Maps to: GGML_REDUCTION_PROD_PROFILE_HIST_MIN_LOG2
REDUCTION_PROD_PROFILE_HIST_MIN_LOG2="${REDUCTION_PROD_PROFILE_HIST_MIN_LOG2:--40}"

# Upper bound for log2(|x*y|) histogram. Must be greater than MIN_LOG2.
# Maps to: GGML_REDUCTION_PROD_PROFILE_HIST_MAX_LOG2
REDUCTION_PROD_PROFILE_HIST_MAX_LOG2="${REDUCTION_PROD_PROFILE_HIST_MAX_LOG2:-40}"

# Keep 1 sampled reduction record per N reductions in samples.csv.
# Maps to: GGML_REDUCTION_PROD_PROFILE_SAMPLE_RATE
REDUCTION_PROD_PROFILE_SAMPLE_RATE="${REDUCTION_PROD_PROFILE_SAMPLE_RATE:-1000}"

# Max sampled reduction records retained in memory before dropping extras.
# Maps to: GGML_REDUCTION_PROD_PROFILE_MAX_SAMPLES
REDUCTION_PROD_PROFILE_MAX_SAMPLES="${REDUCTION_PROD_PROFILE_MAX_SAMPLES:-2000}"

# Output file prefix for profiler artifacts:
#   <prefix>_summary.log
#   <prefix>_global_hist.csv
#   <prefix>_samples.csv
# Maps to: GGML_REDUCTION_PROD_PROFILE_PREFIX
REDUCTION_PROD_PROFILE_PREFIX="${REDUCTION_PROD_PROFILE_PREFIX:-reduction_prod_profile}"

# Graph dump switches
DUMP_DOT="${DUMP_DOT:-0}"
DOT_FILE="${DOT_FILE:-${OUT_DIR}/llama.dot}"

if [[ "${SIM_MATMUL_OUT_MODE}" != "0" && "${SIM_MATMUL_OUT_MODE}" != "1" ]]; then
  echo "invalid SIM_MATMUL_OUT_MODE=${SIM_MATMUL_OUT_MODE} (expected 0 or 1)" >&2
  exit 1
fi

if [[ "${SIM_FP_FORMAT}" != "8" && "${SIM_FP_FORMAT}" != "9" ]]; then
  echo "invalid SIM_FP_FORMAT=${SIM_FP_FORMAT} (expected 8 or 9)" >&2
  exit 1
fi

if [[ "${SIM_FP8_LAYOUT}" != "0" && "${SIM_FP8_LAYOUT}" != "1" && "${SIM_FP8_LAYOUT}" != "2" && "${SIM_FP8_LAYOUT}"  != "3" && "${SIM_FP8_LAYOUT}" != "4" ]]; then
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

if ! [[ "${REDUCTION_PROD_PROFILE_MAX_SAMPLES}" =~ ^[0-9]+$ ]] || [[ "${REDUCTION_PROD_PROFILE_MAX_SAMPLES}" -le 0 ]]; then
  echo "invalid REDUCTION_PROD_PROFILE_MAX_SAMPLES=${REDUCTION_PROD_PROFILE_MAX_SAMPLES} (expected positive integer)" >&2
  exit 1
fi

if [[ -z "${REDUCTION_PROD_PROFILE_PREFIX}" ]]; then
  echo "invalid REDUCTION_PROD_PROFILE_PREFIX: empty string" >&2
  exit 1
fi

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
  -DGGML_REDUCTION_PROD_PROFILE=${REDUCTION_PROD_PROFILE} \
  -DGGML_REDUCTION_PROD_PROFILE_BINS=${REDUCTION_PROD_PROFILE_BINS} \
  -DGGML_REDUCTION_PROD_PROFILE_HIST_MIN_LOG2=${REDUCTION_PROD_PROFILE_HIST_MIN_LOG2} \
  -DGGML_REDUCTION_PROD_PROFILE_HIST_MAX_LOG2=${REDUCTION_PROD_PROFILE_HIST_MAX_LOG2} \
  -DGGML_REDUCTION_PROD_PROFILE_SAMPLE_RATE=${REDUCTION_PROD_PROFILE_SAMPLE_RATE} \
  -DGGML_REDUCTION_PROD_PROFILE_MAX_SAMPLES=${REDUCTION_PROD_PROFILE_MAX_SAMPLES} \
  -DGGML_REDUCTION_PROD_PROFILE_PREFIX=\"${REDUCTION_PROD_PROFILE_PREFIX}\""

source ~/miniforge3/bin/activate
rm -rf build
mkdir -p "${OUT_DIR}"

cmake -B build \
  -DCMAKE_C_COMPILER=x86_64-conda-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=x86_64-conda-linux-gnu-g++ \
  -DCMAKE_C_FLAGS="${SIM_FLAGS}" \
  -DCMAKE_CXX_FLAGS="${SIM_FLAGS}" \
  -DLLAMA_CURL=OFF

cmake --build build --config Release --target llama-perplexity -j "${THREADS}"

export FP8_SIM_STATS_SAMPLE=100
# Disable matmul distribution logging to avoid extra overhead and file output.
export GGML_MATMUL_DIST=0

if [[ "${DUMP_DOT}" == "1" ]]; then
  export LLAMA_DUMP_DOT="${DOT_FILE}"
fi

FP8_CLI_LOG="${OUT_DIR}/fp8_sim_analysis_cli.log"
FP8_PPL_LOG="${OUT_DIR}/fp8_sim_analysis_ppl.log"

# ./build/bin/llama-cli \
#   -m "${MODEL}" \
#   -p "${PROMPT}" \
#   -c "${CTX}" \
#   -t "${THREADS}" \
#   -n "${N_PREDICT}" \
#   --seq-state-out "${SEQ_BIN}" \
#   --seq-state-out-id "${SEQ_ID}" \
#   | tee "${RUN_LOG}"

# if [ -f fp8_sim_analysis.log ]; then
#   mv -f fp8_sim_analysis.log "${FP8_CLI_LOG}"
# fi

./build/bin/llama-perplexity \
  -m "${MODEL}" \
  -f "${DATA}" \
  -c "${CTX}" \
  -b "${BATCH}" \
  -ub "${UBATCH}" \
  -t "${THREADS}" \
  --ppl-stride "${STRIDE}"

if [ -f fp8_sim_analysis.log ]; then
  mv -f fp8_sim_analysis.log "${FP8_PPL_LOG}"
fi

if [[ "${DUMP_DOT}" == "1" && -f "${DOT_FILE}" ]]; then
  if command -v dot >/dev/null 2>&1; then
    DOT_PNG="${DOT_FILE%.dot}.png"
    dot -Tpng "${DOT_FILE}" -o "${DOT_PNG}"
  fi
fi

echo "done."
echo "fp8 ppl log  : ${FP8_PPL_LOG}"
if [[ "${DUMP_DOT}" == "1" ]]; then
  echo "dot graph    : ${DOT_FILE}"
  if command -v dot >/dev/null 2>&1; then
    echo "dot png      : ${DOT_FILE%.dot}.png"
  fi
fi
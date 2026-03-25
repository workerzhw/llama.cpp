#!/usr/bin/env bash
set -euo pipefail

DEFAULT_MODEL="${MODEL:-models/hf/unsloth-Llama-3.2-1B-Instruct-f16.gguf}"
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
# Maps to: GGML_REDUCTION_PROD_PROFILE_PREFIX
DEFAULT_REDUCTION_PROD_PROFILE_PREFIX="${REDUCTION_PROD_PROFILE_PREFIX:-reduction_prod_profile}"

# Graph dump switches
DEFAULT_DUMP_DOT="${DUMP_DOT:-0}"

RUN_KIND="${RUN_KIND:-perplexity}"

# 在这里维护一组测试用例。
# 格式: "用例名|KEY=VALUE|KEY=VALUE|..."
# 未覆盖的参数会继承上面的 DEFAULT_* 值。
RUN_CASES=(
  "llama-3.2-1B-f8e3m4|MODEL=models/hf/unsloth-Llama-3.2-1B-Instruct-f16.gguf|DATA=models/hf/wiki.test.raw|OUT_DIR=kv_dump_logs|PROMPT=你好，请简要介绍一下KV cache。|CTX=512|N_PREDICT=-1|SEQ_ID=0|BATCH=2048|UBATCH=512|STRIDE=0|SIM_FP8=1|SIM_FP_FORMAT=8|SIM_FP8_LAYOUT=1|SIM_FP8_APPLY_SRC0=1|SIM_FP8_APPLY_SRC1=1|SIM_FP8_SCALE_TYPE=0|SIM_FP8_SCALE_TYPE_IN=0|SIM_FP8_SCALE_TYPE_OUT=1|SIM_FP8_BLOCK=16|SIM_MATMUL_OUT_MODE=1|REDUCTION_PROD_PROFILE=0|REDUCTION_PROD_PROFILE_BINS=256|REDUCTION_PROD_PROFILE_HIST_MIN_LOG2=-128|REDUCTION_PROD_PROFILE_HIST_MAX_LOG2=128|REDUCTION_PROD_PROFILE_SAMPLE_RATE=1000|REDUCTION_PROD_PROFILE_MAX_SAMPLES=2000|DUMP_DOT=0"
  "llama-3.2-1B-f8e3m4-normal|MODEL=models/hf/unsloth-Llama-3.2-1B-Instruct-f16.gguf|DATA=models/hf/wiki.test.raw|OUT_DIR=kv_dump_logs|PROMPT=你好，请简要介绍一下KV cache。|CTX=512|N_PREDICT=-1|SEQ_ID=0|BATCH=2048|UBATCH=512|STRIDE=0|SIM_FP8=1|SIM_FP_FORMAT=8|SIM_FP8_LAYOUT=2|SIM_FP8_APPLY_SRC0=1|SIM_FP8_APPLY_SRC1=1|SIM_FP8_SCALE_TYPE=0|SIM_FP8_SCALE_TYPE_IN=0|SIM_FP8_SCALE_TYPE_OUT=1|SIM_FP8_BLOCK=16|SIM_MATMUL_OUT_MODE=1|REDUCTION_PROD_PROFILE=0|REDUCTION_PROD_PROFILE_BINS=256|REDUCTION_PROD_PROFILE_HIST_MIN_LOG2=-128|REDUCTION_PROD_PROFILE_HIST_MAX_LOG2=128|REDUCTION_PROD_PROFILE_SAMPLE_RATE=1000|REDUCTION_PROD_PROFILE_MAX_SAMPLES=2000|DUMP_DOT=0"
)

sanitize_name() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
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
  local IFS='|'
  read -r -a case_fields <<< "${case_spec}"

  CASE_NAME="${case_fields[0]}"

  local field
  for field in "${case_fields[@]:1}"; do
    if [[ -n "${field}" ]]; then
      export "${field}"
    fi
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

  build_case_flags

  rm -rf build
  mkdir -p "${OUT_DIR}" "${case_dir}"

  cmake -B build \
    -DCMAKE_C_COMPILER=x86_64-conda-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=x86_64-conda-linux-gnu-g++ \
    -DCMAKE_C_FLAGS="${SIM_FLAGS}" \
    -DCMAKE_CXX_FLAGS="${SIM_FLAGS}" \
    -DLLAMA_CURL=OFF

  cmake --build build --config Release --target llama-perplexity -j "${THREADS}"

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
  fi

  if [[ "${DUMP_DOT}" == "1" && -f "${DOT_FILE}" ]] && command -v dot >/dev/null 2>&1; then
    dot -Tpng "${DOT_FILE}" -o "${DOT_FILE%.dot}.png"
  fi
}

source ~/miniforge3/bin/activate
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
      --out-md "${compare_md}" || true
    echo "compare csv : ${compare_csv}"
    echo "compare md  : ${compare_md}"
  done
fi

echo "done."
for finished in "${finished_cases[@]}"; do
  echo "${finished}"
done
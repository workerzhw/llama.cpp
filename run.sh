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
#   0 => FP8(E4M3) output QDQ (existing behavior, gated by GGML_SIM_FP8E4M3)
#   1 => BF16 round-trip output simulation (F32 -> BF16 -> F32)
SIM_FP8="${SIM_FP8:-1}"
SIM_FP_FORMAT="${SIM_FP_FORMAT:-8}"
SIM_FP8_APPLY_SRC0="${SIM_FP8_APPLY_SRC0:-1}"
SIM_FP8_APPLY_SRC1="${SIM_FP8_APPLY_SRC1:-1}"
# Legacy single switch (kept for compatibility)
SIM_FP8_SCALE_TYPE="${SIM_FP8_SCALE_TYPE:-0}"
# New split switches
SIM_FP8_SCALE_TYPE_IN="${SIM_FP8_SCALE_TYPE_IN:-${SIM_FP8_SCALE_TYPE}}"
SIM_FP8_SCALE_TYPE_OUT="${SIM_FP8_SCALE_TYPE_OUT:-${SIM_FP8_SCALE_TYPE}}"
SIM_FP8_BLOCK="${SIM_FP8_BLOCK:-16}"
SIM_MATMUL_OUT_MODE="${SIM_MATMUL_OUT_MODE:-1}"

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

if [[ "${SIM_FP8_SCALE_TYPE_IN}" != "0" && "${SIM_FP8_SCALE_TYPE_IN}" != "1" ]]; then
  echo "invalid SIM_FP8_SCALE_TYPE_IN=${SIM_FP8_SCALE_TYPE_IN} (expected 0 or 1)" >&2
  exit 1
fi

if [[ "${SIM_FP8_SCALE_TYPE_OUT}" != "0" && "${SIM_FP8_SCALE_TYPE_OUT}" != "1" ]]; then
  echo "invalid SIM_FP8_SCALE_TYPE_OUT=${SIM_FP8_SCALE_TYPE_OUT} (expected 0 or 1)" >&2
  exit 1
fi

SIM_FLAGS="-DGGML_SIM_FP8E4M3=${SIM_FP8} \
  -DGGML_SIM_FP_FORMAT=${SIM_FP_FORMAT} \
  -DGGML_SIM_FP8E4M3_APPLY_SRC0=${SIM_FP8_APPLY_SRC0} \
  -DGGML_SIM_FP8E4M3_APPLY_SRC1=${SIM_FP8_APPLY_SRC1} \
  -DGGML_SIM_FP8E4M3_SCALE_TYPE=${SIM_FP8_SCALE_TYPE_IN} \
  -DGGML_SIM_FP8E4M3_SCALE_TYPE_IN=${SIM_FP8_SCALE_TYPE_IN} \
  -DGGML_SIM_FP8E4M3_SCALE_TYPE_OUT=${SIM_FP8_SCALE_TYPE_OUT} \
  -DGGML_SIM_FP8E4M3_BLOCK=${SIM_FP8_BLOCK} \
  -DGGML_SIM_MATMUL_OUT_MODE=${SIM_MATMUL_OUT_MODE}"

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
export GGML_MATMUL_DIST=1
export GGML_MATMUL_DIST_SAMPLE=100
export GGML_MATMUL_DIST_FILE="${OUT_DIR}/matmul.log"

if [[ "${DUMP_DOT}" == "1" ]]; then
  export LLAMA_DUMP_DOT="${DOT_FILE}"
fi

SEQ_BIN="${OUT_DIR}/kv_seq_${SEQ_ID}.bin"
SEQ_TXT="${OUT_DIR}/kv_seq_${SEQ_ID}.txt"
RUN_LOG="${OUT_DIR}/run.log"
PPL_LOG="${OUT_DIR}/ppl.log"
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
  --ppl-stride "${STRIDE}" \
  --seq-state-out "${SEQ_BIN}" \
  --seq-state-out-id "${SEQ_ID}" \
  | tee "${PPL_LOG}"

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
echo "seq state bin: ${SEQ_BIN}"
echo "ppl log      : ${PPL_LOG}"
echo "matmul log   : ${OUT_DIR}/matmul.log"
echo "fp8 ppl log  : ${FP8_PPL_LOG}"
if [[ "${DUMP_DOT}" == "1" ]]; then
  echo "dot graph    : ${DOT_FILE}"
  if command -v dot >/dev/null 2>&1; then
    echo "dot png      : ${DOT_FILE%.dot}.png"
  fi
fi
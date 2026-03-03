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

source ~/miniforge3/bin/activate
rm -rf build
mkdir -p "${OUT_DIR}"

cmake -B build \
  -DCMAKE_C_COMPILER=x86_64-conda-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=x86_64-conda-linux-gnu-g++ \
  -DCMAKE_C_FLAGS="-DGGML_SIM_FP8E4M3=1 -DGGML_SIM_FP8E4M3_APPLY_SRC0=1 -DGGML_SIM_FP8E4M3_APPLY_SRC1=1 -DGGML_SIM_FP8E4M3_SCALE_TYPE=0 -DGGML_SIM_FP8E4M3_BLOCK=128" \
  -DCMAKE_CXX_FLAGS="-DGGML_SIM_FP8E4M3=1 -DGGML_SIM_FP8E4M3_APPLY_SRC0=1 -DGGML_SIM_FP8E4M3_APPLY_SRC1=1 -DGGML_SIM_FP8E4M3_SCALE_TYPE=0 -DGGML_SIM_FP8E4M3_BLOCK=128" \
  -DLLAMA_CURL=OFF

cmake --build build --config Release --target llama-perplexity -j "${THREADS}"

export FP8_SIM_STATS_SAMPLE=100
export GGML_MATMUL_DIST=1
export GGML_MATMUL_DIST_SAMPLE=100
export GGML_MATMUL_DIST_FILE="${OUT_DIR}/matmul.log"

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

echo "done."
echo "seq state bin: ${SEQ_BIN}"
echo "ppl log      : ${PPL_LOG}"
echo "matmul log   : ${OUT_DIR}/matmul.log"
echo "fp8 ppl log  : ${FP8_PPL_LOG}"
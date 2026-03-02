#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-models/hf/llama-2-7B-F16.gguf}"
DATA="${DATA:-models/hf/wiki.test.raw}"
OUT_DIR="${OUT_DIR:-kv_dump_logs}"
PROMPT="${PROMPT:-你好，请简要介绍一下KV cache。}"

CTX="${CTX:-4096}"
THREADS="${THREADS:-$(nproc)}"
N_PREDICT="${N_PREDICT:-64}"
SEQ_ID="${SEQ_ID:-0}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"
STRIDE="${STRIDE:-512}"

rm -rf build
mkdir -p "${OUT_DIR}"

cmake -B build \
  -DCMAKE_C_FLAGS="-DGGML_SIM_FP8E4M3=1 -DGGML_SIM_FP8E4M3_APPLY_SRC0=1 -DGGML_SIM_FP8E4M3_APPLY_SRC1=1" \
  -DCMAKE_CXX_FLAGS="-DGGML_SIM_FP8E4M3=1 -DGGML_SIM_FP8E4M3_APPLY_SRC0=1 -DGGML_SIM_FP8E4M3_APPLY_SRC1=1" \
  -DLLAMA_CURL=OFF

cmake --build build --config Release --target llama-cli llama-perplexity -j "${THREADS}"

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

./build/bin/llama-cli \
  -m "${MODEL}" \
  -p "${PROMPT}" \
  -c "${CTX}" \
  -t "${THREADS}" \
  -n "${N_PREDICT}" \
  --seq-state-out "${SEQ_BIN}" \
  --seq-state-out-id "${SEQ_ID}" \
  | tee "${RUN_LOG}"

if [ -f fp8_sim_analysis.log ]; then
  mv -f fp8_sim_analysis.log "${FP8_CLI_LOG}"
fi

./build/bin/llama-perplexity \
  -m "${MODEL}" \
  -f "${DATA}" \
  -c "${CTX}" \
  -b "${BATCH}" \
  -ub "${UBATCH}" \
  -t "${THREADS}" \
  --ppl-stride "${STRIDE}" \
  | tee "${PPL_LOG}"

if [ -f fp8_sim_analysis.log ]; then
  mv -f fp8_sim_analysis.log "${FP8_PPL_LOG}"
fi

if command -v xxd >/dev/null 2>&1; then
  {
    echo "=== KV SEQ STATE REPORT ==="
    echo "timestamp      : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "model          : ${MODEL}"
    echo "seq_id         : ${SEQ_ID}"
    echo "ctx            : ${CTX}"
    echo "threads        : ${THREADS}"
    echo "n_predict      : ${N_PREDICT}"
    echo "run_log        : ${RUN_LOG}"
    echo "matmul_log     : ${OUT_DIR}/matmul.log"
    echo "bin_file       : ${SEQ_BIN}"
    echo "bin_size_bytes : $(wc -c < "${SEQ_BIN}")"
    if command -v sha256sum >/dev/null 2>&1; then
      echo "sha256         : $(sha256sum "${SEQ_BIN}" | awk '{print $1}')"
    fi
    echo
    echo "=== HEXDUMP (FULL, 16 BYTES PER LINE) ==="
    xxd -g 1 -c 16 "${SEQ_BIN}"
  } > "${SEQ_TXT}"
else
  {
    echo "=== KV SEQ STATE REPORT ==="
    echo "timestamp      : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "model          : ${MODEL}"
    echo "seq_id         : ${SEQ_ID}"
    echo "ctx            : ${CTX}"
    echo "threads        : ${THREADS}"
    echo "n_predict      : ${N_PREDICT}"
    echo "run_log        : ${RUN_LOG}"
    echo "matmul_log     : ${OUT_DIR}/matmul.log"
    echo "bin_file       : ${SEQ_BIN}"
    echo "bin_size_bytes : $(wc -c < "${SEQ_BIN}")"
    if command -v sha256sum >/dev/null 2>&1; then
      echo "sha256         : $(sha256sum "${SEQ_BIN}" | awk '{print $1}')"
    fi
    echo
    echo "=== HEXDUMP (FULL) ==="
    od -An -tx1 -v "${SEQ_BIN}"
  } > "${SEQ_TXT}"
fi

echo "done."
echo "seq state bin: ${SEQ_BIN}"
echo "seq state txt: ${SEQ_TXT}"
echo "run log      : ${RUN_LOG}"
echo "ppl log      : ${PPL_LOG}"
echo "matmul log   : ${OUT_DIR}/matmul.log"
echo "fp8 cli log  : ${FP8_CLI_LOG}"
echo "fp8 ppl log  : ${FP8_PPL_LOG}"
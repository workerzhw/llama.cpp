#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

DEFAULT_DATA_FILE="${REPO_ROOT}/wikitext-2-raw/wiki.test.raw"
if [[ -f "${REPO_ROOT}/models-mnt/wikitext/wikitext-2-raw/wiki.test.raw" ]]; then
    DEFAULT_DATA_FILE="${REPO_ROOT}/models-mnt/wikitext/wikitext-2-raw/wiki.test.raw"
fi

BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
BIN_PATH="${BIN_PATH:-${BUILD_DIR}/bin/llama-perplexity}"
MODEL_PATH_ENV="${MODEL_PATH:-}"
DATA_FILE="${DATA_FILE:-${DEFAULT_DATA_FILE}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/ppl_logs}"
NGL="${NGL:-0}"
CTX_SIZE="${CTX_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-}"
CHUNKS="${CHUNKS:-}"
PPL_STRIDE="${PPL_STRIDE:-0}"
PPL_OUTPUT_TYPE="${PPL_OUTPUT_TYPE:-0}"
THREADS="${THREADS:-}"
THREADS_BATCH="${THREADS_BATCH:-}"
AUTO_BUILD="${AUTO_BUILD:-1}"
AUTO_DOWNLOAD_DATA="${AUTO_DOWNLOAD_DATA:-1}"
MODEL_PATHS=()
EXTRA_ARGS=()

detect_default_threads() {
    local detected_threads=""

    if command -v nproc >/dev/null 2>&1; then
        detected_threads="$(nproc 2>/dev/null || true)"
    elif command -v getconf >/dev/null 2>&1; then
        detected_threads="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
    fi

    if [[ "${detected_threads}" =~ ^[0-9]+$ ]] && [[ "${detected_threads}" -gt 0 ]]; then
        printf '%s\n' "${detected_threads}"
        return 0
    fi

    printf '1\n'
}

usage() {
    cat <<'EOF'
Usage: ./run.sh [options] [-- extra llama-perplexity args]

Options:
    -m, --model PATH          GGUF model path. Repeatable, runs sequentially.
    -f, --file, --data PATH   Evaluation text path. Default: repo wikitext-2 test set.
    --build-dir PATH          CMake build directory. Default: <repo>/build
    --bin PATH                llama-perplexity binary path.
    --log-dir PATH            Log output directory. Default: <repo>/ppl_logs
    --ngl N                   GPU layers. Default: 0
    -c, --ctx-size N          Context size. Default: 1024
    -b, --batch-size N        Batch size. Default: 512
    -ub, --ubatch-size N      Physical batch size. Default: llama-perplexity default
    --chunks N                Number of chunks. Default: llama-perplexity default (-1 = all)
    --ppl-stride N            PPL stride. Default: 0
    --ppl-output-type N       Output type. Default: 0
    -t, --threads N           CPU threads. Default: nproc
    -tb, --threads-batch N    CPU threads for batch/prompt processing. Default: same as --threads
    --no-build                Do not auto-build llama-perplexity when missing.
    --no-download             Do not auto-download default Wikitext-2 data.
    -h, --help                Show this help.

Examples:
  ./run.sh -m /path/to/model.gguf
  ./run.sh -m /path/to/a.gguf -m /path/to/b.gguf
  ./run.sh /path/to/a.gguf /path/to/b.gguf
  MODEL_PATH=/path/to/model.gguf NGL=99 ./run.sh
  ./run.sh -m /path/to/model.gguf -- --repeat-last-n 0
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

print_cmd() {
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            MODEL_PATHS+=("$2")
            shift 2
            ;;
        -f|--file|--data)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            DATA_FILE="$2"
            shift 2
            ;;
        --build-dir)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            BUILD_DIR="$2"
            BIN_PATH="${BUILD_DIR}/bin/llama-perplexity"
            shift 2
            ;;
        --bin)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            BIN_PATH="$2"
            shift 2
            ;;
        --log-dir)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            LOG_DIR="$2"
            shift 2
            ;;
        --ngl)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            NGL="$2"
            shift 2
            ;;
        -c|--ctx-size)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            CTX_SIZE="$2"
            shift 2
            ;;
        -b|--batch-size)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            BATCH_SIZE="$2"
            shift 2
            ;;
        -ub|--ubatch-size)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            UBATCH_SIZE="$2"
            shift 2
            ;;
        --chunks)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            CHUNKS="$2"
            shift 2
            ;;
        --ppl-stride)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            PPL_STRIDE="$2"
            shift 2
            ;;
        --ppl-output-type)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            PPL_OUTPUT_TYPE="$2"
            shift 2
            ;;
        -t|--threads)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            THREADS="$2"
            shift 2
            ;;
        -tb|--threads-batch)
            [[ $# -ge 2 ]] || die "Missing value for $1"
            THREADS_BATCH="$2"
            shift 2
            ;;
        --no-build)
            AUTO_BUILD=0
            shift
            ;;
        --no-download)
            AUTO_DOWNLOAD_DATA=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            if [[ "$1" == -* ]]; then
                die "Unknown option: $1. Use -- to pass extra llama-perplexity args."
            else
                MODEL_PATHS+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ -z "${THREADS}" ]]; then
    THREADS="$(detect_default_threads)"
fi

if [[ -z "${THREADS_BATCH}" ]]; then
    THREADS_BATCH="${THREADS}"
fi

if [[ ${#MODEL_PATHS[@]} -eq 0 && -n "${MODEL_PATH_ENV}" ]]; then
    MODEL_PATHS+=("${MODEL_PATH_ENV}")
fi

[[ ${#MODEL_PATHS[@]} -gt 0 ]] || die "Please provide at least one model path via -m/--model, positional args, or MODEL_PATH."

for model_path in "${MODEL_PATHS[@]}"; do
    [[ -f "${model_path}" ]] || die "Model file not found: ${model_path}"
done

if [[ ! -x "${BIN_PATH}" ]]; then
    if [[ "${AUTO_BUILD}" != "1" ]]; then
        die "llama-perplexity not found: ${BIN_PATH}"
    fi

    JOBS=8
    if command -v nproc >/dev/null 2>&1; then
        JOBS="$(nproc)"
    fi

    echo "Building llama-perplexity in ${BUILD_DIR} ..."
    cmake --build "${BUILD_DIR}" --target llama-perplexity -j "${JOBS}"
fi

if [[ ! -f "${DATA_FILE}" ]]; then
    if [[ "${AUTO_DOWNLOAD_DATA}" == "1" && "${DATA_FILE}" == "${REPO_ROOT}/wikitext-2-raw/wiki.test.raw" ]]; then
        echo "Downloading Wikitext-2 into ${REPO_ROOT} ..."
        (
            cd "${REPO_ROOT}"
            ./scripts/get-wikitext-2.sh
        )
    else
        die "Evaluation text not found: ${DATA_FILE}"
    fi
fi

[[ -f "${DATA_FILE}" ]] || die "Evaluation text not found after setup: ${DATA_FILE}"

mkdir -p "${LOG_DIR}"

echo "Repository root: ${REPO_ROOT}"
echo "Data: ${DATA_FILE}"
echo "Models: ${#MODEL_PATHS[@]}"
echo "Threads: ${THREADS} (batch: ${THREADS_BATCH})"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
MODEL_COUNT=${#MODEL_PATHS[@]}
MODEL_INDEX=0
FAILURES=0
SUMMARY_LINES=()

for model_path in "${MODEL_PATHS[@]}"; do
    MODEL_INDEX=$((MODEL_INDEX + 1))
    MODEL_NAME="$(basename -- "${model_path}")"
    MODEL_NAME="${MODEL_NAME%.gguf}"
    LOG_FILE="${LOG_DIR}/${MODEL_NAME}_ppl_${RUN_ID}_$(printf '%02d' "${MODEL_INDEX}").log"

    CMD=(
        "${BIN_PATH}"
        "--model" "${model_path}"
        "-f" "${DATA_FILE}"
        "-ngl" "${NGL}"
        "-c" "${CTX_SIZE}"
        "-b" "${BATCH_SIZE}"
        "--ppl-output-type" "${PPL_OUTPUT_TYPE}"
    )

    if [[ -n "${UBATCH_SIZE}" ]]; then
        CMD+=("-ub" "${UBATCH_SIZE}")
    fi

    if [[ -n "${CHUNKS}" ]]; then
        CMD+=("--chunks" "${CHUNKS}")
    fi

    if [[ "${PPL_STRIDE}" != "0" ]]; then
        CMD+=("--ppl-stride" "${PPL_STRIDE}")
    fi

    if [[ -n "${THREADS}" ]]; then
        CMD+=("-t" "${THREADS}")
    fi

    if [[ -n "${THREADS_BATCH}" ]]; then
        CMD+=("-tb" "${THREADS_BATCH}")
    fi

    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        CMD+=("${EXTRA_ARGS[@]}")
    fi

    echo
    echo "=== [${MODEL_INDEX}/${MODEL_COUNT}] ${model_path} ==="
    echo "Log: ${LOG_FILE}"
    print_cmd "${CMD[@]}"

    set +e
    "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
    STATUS=$?
    set -e

    FINAL_LINE="$(grep 'Final estimate: PPL = ' "${LOG_FILE}" | tail -n 1 || true)"
    if [[ -n "${FINAL_LINE}" ]]; then
        echo
        echo "${FINAL_LINE}"
    fi

    if [[ ${STATUS} -eq 0 ]]; then
        SUMMARY_LINES+=("[OK] ${model_path} | ${FINAL_LINE:-Final estimate not found} | ${LOG_FILE}")
    else
        FAILURES=$((FAILURES + 1))
        SUMMARY_LINES+=("[FAIL:${STATUS}] ${model_path} | ${FINAL_LINE:-Final estimate not found} | ${LOG_FILE}")
    fi

    echo "Saved log to ${LOG_FILE}"
done

echo
echo "Summary:"
for summary_line in "${SUMMARY_LINES[@]}"; do
    echo "${summary_line}"
done

if [[ ${FAILURES} -ne 0 ]]; then
    exit 1
fi
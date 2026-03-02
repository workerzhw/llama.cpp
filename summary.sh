#!/usr/bin/env bash
set -euo pipefail

DIR="${1:-ppl_kv_logs}"
OUT="${2:-${DIR}/summary.csv}"

echo "case,ppl" > "${OUT}"
for f in "${DIR}"/ppl_*.log; do
  [ -e "$f" ] || continue
  c="$(basename "$f" .log | sed 's/^ppl_//')"
  p="$(grep -E 'Final estimate: PPL =|Mean PPL\(Q\)' "$f" | tail -n1 | sed -E 's/.*PPL(\(Q\))?[^0-9]*([0-9]+\.[0-9]+).*/\2/')"
  echo "${c},${p}" >> "${OUT}"
done

echo "saved: ${OUT}"
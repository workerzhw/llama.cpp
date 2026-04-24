# SwiGLU Target Profiles

For the full `run.sh` option reference and the one-command `activation-flow`, see
`docs/development/run-sh-activation-workflow.md`.

These priors are used when `SWIGLU_THRESHOLD_KIND=swiglu`.
For the parallel SiLU-output flow, use `tools/silu-threshold-targets/` with the same `run.sh` modes and set
`SWIGLU_THRESHOLD_KIND=silu`.

These CSV files define the calibration prior used by `tools/swiglu-threshold-configs/generate.py`.

CSV format:

```text
group,layers,prefill_target,decode_target
front,0-3,0.002,0.002
mid,4-11,0.006,0.008
tail,12-15,0.004,0.005
```

Field semantics:

- `layers`: semicolon-separated layer ids or closed ranges, for example `0-3;8;10-11`
- `prefill_target`: desired added-zero ratio on SwiGLU outputs during prompt prefill; `layer 1` is the generator anchor
- `decode_target`: desired added-zero ratio on SwiGLU outputs during autoregressive decode; this is kept for reporting, not for threshold selection

Both targets are defined against total output elements, not only non-zero elements.
For `SWIGLU_THRESHOLD_KIND=swiglu`, the generator does not emit per-layer thresholds.
It takes the `prefill_target` of `layer 1`, inverts the collected `prefill` distribution of that layer,
and uses the resulting single threshold as a global threshold for every layer and every stage.
The per-layer `prefill_target` and `decode_target` values remain in the report so you can compare
how far each layer/stage lands from its local target under that shared threshold.

Minimal execution flow with `run.sh`:

```bash
# 1) collect per-layer prefill/decode SwiGLU output distributions
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=swiglu-collect \
SWIGLU_THRESHOLD_KIND=swiglu \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=1 \
N_PREDICT=128 \
bash run.sh

# 2) generate a runtime threshold table from the collected histograms
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=swiglu-generate \
SWIGLU_THRESHOLD_KIND=swiglu \
SWIGLU_TARGET_PROFILE_KIND=minimal \
SWIGLU_TARGET_SCALE=1.0 \
bash run.sh

# 3) validate quality in perplexity mode using the generated table
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=perplexity \
SWIGLU_THRESHOLD_KIND=swiglu \
SWIGLU_THRESHOLD_ENABLE=1 \
SWIGLU_THRESHOLD_PROFILE=generated \
SKIP_BUILD=1 \
bash run.sh

# 4) inspect decode-side realized added-zero ratio using the same generated table
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=cli \
SWIGLU_THRESHOLD_KIND=swiglu \
SWIGLU_THRESHOLD_ENABLE=1 \
SWIGLU_THRESHOLD_PROFILE=generated \
SKIP_BUILD=1 \
N_PREDICT=128 \
bash run.sh
```

Expected outputs under `kv_dump_logs/<case>/`:

- `<case>_swiglu_collect_summary.csv`: per-layer per-stage element counts and histogram metadata
- `<case>_swiglu_collect_hist.csv`: per-layer per-stage absolute-value histogram bins
- `<case>_swiglu_threshold_generated.csv`: generated runtime `layer,threshold` table; every listed layer shares one global threshold
- `<case>_swiglu_threshold_generated_summary.csv`: layerwise target vs realized estimate report under that shared threshold
- `<case>_swiglu_threshold_perplexity.csv`: realized PPL-stage truncation report from the runtime callback
- `<case>_swiglu_threshold_cli.csv`: realized decode-stage truncation report from the runtime callback
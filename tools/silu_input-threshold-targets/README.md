# SiLU Input Target Profiles

For the full `run.sh` option reference and the one-command `activation-flow`, see
`docs/development/run-sh-activation-workflow.md`.

These CSV files define the calibration prior used by `tools/swiglu-threshold-configs/generate.py` when the
runtime is configured with `SWIGLU_THRESHOLD_KIND=silu_input`.

This path targets the raw gate tensor before SiLU activation.
Collection is one-sided: it tracks only negative-tail magnitudes, and runtime truncation zeroes the downstream
activation when `gate_value <= -threshold`.

CSV format:

```text
group,layers,prefill_target,decode_target
front,0-3,0.002,0.002
mid,4-11,0.006,0.008
tail,12-15,0.004,0.005
```

Field semantics:

- `layers`: semicolon-separated layer ids or closed ranges, for example `0-3;8;10-11`
- `prefill_target`: desired added-zero ratio induced by one-sided gate-input truncation during prompt prefill
- `decode_target`: desired added-zero ratio induced by one-sided gate-input truncation during autoregressive decode

Both targets are defined against total output elements, not only tracked negative-tail values.
The generator inverts the collected per-layer distributions separately for `prefill` and `decode`,
then chooses the more conservative threshold so runtime still uses one threshold per layer.

Minimal execution flow with `run.sh`:

```bash
# 1) collect per-layer prefill/decode gate-input distributions before SiLU
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=swiglu-collect \
SWIGLU_THRESHOLD_KIND=silu_input \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=1 \
N_PREDICT=128 \
bash run.sh

# 2) generate a runtime threshold table from the collected histograms
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=swiglu-generate \
SWIGLU_THRESHOLD_KIND=silu_input \
SWIGLU_TARGET_PROFILE_KIND=minimal \
SWIGLU_TARGET_SCALE=1.0 \
bash run.sh

# 3) validate quality in perplexity mode using the generated table
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=perplexity \
SWIGLU_THRESHOLD_KIND=silu_input \
SWIGLU_THRESHOLD_ENABLE=1 \
SWIGLU_THRESHOLD_PROFILE=generated \
SKIP_BUILD=1 \
bash run.sh

# 4) inspect decode-side realized added-zero ratio using the same generated table
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=cli \
SWIGLU_THRESHOLD_KIND=silu_input \
SWIGLU_THRESHOLD_ENABLE=1 \
SWIGLU_THRESHOLD_PROFILE=generated \
SKIP_BUILD=1 \
N_PREDICT=128 \
bash run.sh
```

Expected outputs under `kv_dump_logs/<case>/`:

- `<case>_silu_input_collect_summary.csv`: per-layer per-stage element counts and histogram metadata
- `<case>_silu_input_collect_hist.csv`: per-layer per-stage negative-tail magnitude histogram bins
- `<case>_silu_input_threshold_generated.csv`: generated runtime `layer,threshold` table
- `<case>_silu_input_threshold_generated_summary.csv`: layerwise target vs estimated achievable sparsity report
- `<case>_silu_input_threshold_perplexity.csv`: realized PPL-stage truncation report from the runtime callback
- `<case>_silu_input_threshold_cli.csv`: realized decode-stage truncation report from the runtime callback

For `*_silu_input_threshold_perplexity.csv` and `*_silu_input_threshold_cli.csv`:

- `original_zero_*`, `truncated_nonzero_*`, `final_zero_*` describe the raw gate-input statistics that drive one-sided truncation
- `output_*` describes the realized zero-rate on the final SwiGLU output after truncating before SiLU
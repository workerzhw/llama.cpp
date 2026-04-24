# SwiGLU Threshold Profiles

For the full `run.sh` option reference and the one-command `activation-flow`, see
`docs/development/run-sh-activation-workflow.md`.

This directory stores runtime threshold tables for the SwiGLU-output experiment.
The shared runtime can also target SiLU outputs by setting `SWIGLU_THRESHOLD_KIND=silu` in `run.sh`.
SiLU-specific target priors live under `tools/silu-threshold-targets/`, while the generator script in this
directory is shared by both activation kinds.

CSV format:

```text
layer,threshold
0,1e-2
1,1e-2
15,5e-3
```

Only non-zero thresholds need to be listed. Layers omitted from the file default to `off`.

`phase1/` contains the static per-layer SwiGLU-output threshold tables for the first-stage experiment.

`generate.py` converts collected per-stage activation output histograms into a runtime threshold table.
It reads:

```text
<collect-prefix>_summary.csv
<collect-prefix>_hist.csv
```

and a group target profile from `tools/swiglu-threshold-targets/`, then emits a `layer,threshold`
CSV compatible with `--swiglu-threshold-config`.

The same generator also serves the SiLU-output path. With `SWIGLU_THRESHOLD_KIND=silu`
and `SWIGLU_GENERATE_MODE=channel-max`, it reads:

```text
<collect-prefix>_channel_max.csv
```

and emits a `layer,channel,threshold` table where each threshold is
`abs_channel_max * SWIGLU_CHANNEL_THRESHOLD_RATIO`.
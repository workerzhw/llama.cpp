# SiLU Threshold Profiles

For the full `run.sh` option reference and the one-command `activation-flow`, see
`docs/development/run-sh-activation-workflow.md`.

This directory is reserved for runtime threshold tables targeting the SiLU-output experiment.

The current calibration flow uses the shared generator at `tools/swiglu-threshold-configs/generate.py`.
Select the SiLU path from `run.sh` by setting `SWIGLU_THRESHOLD_KIND=silu`.

Generated per-case runtime tables are written under `kv_dump_logs/<case>/` as:

```text
<case>_silu_threshold_generated.csv
<case>_silu_threshold_generated_summary.csv
```

The default `target-profile` mode emits the historical `layer,threshold` format.
If you set `SWIGLU_GENERATE_MODE=channel-max`, the same generated config path instead stores:

```text
layer,channel,threshold
```

In that mode, the generator reads `<case>_silu_collect_channel_max.csv` and sets each threshold to
`abs_channel_max * SWIGLU_CHANNEL_THRESHOLD_RATIO`.

If you later want hand-authored SiLU threshold tables, place them under subdirectories here and refer to them
through `SWIGLU_THRESHOLD_PROFILE=<profile-name>`.
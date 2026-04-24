# SiLU Input Threshold Profiles

For the full `run.sh` option reference and the one-command `activation-flow`, see
`docs/development/run-sh-activation-workflow.md`.

This directory is reserved for runtime threshold tables targeting the SiLU-input experiment.
Select the gate-input path from `run.sh` by setting `SWIGLU_THRESHOLD_KIND=silu_input`.

Generated per-case runtime tables are written under `kv_dump_logs/<case>/` as:

```text
<case>_silu_input_threshold_generated.csv
<case>_silu_input_threshold_generated_summary.csv
```

For this activation kind, CSV thresholds store positive magnitudes.
At runtime they are interpreted as one-sided negative-tail cutoffs and the gate input is truncated when
`value <= -threshold` before SiLU is evaluated.

If you later want hand-authored SiLU-input threshold tables, place them under subdirectories here and refer to them
through `SWIGLU_THRESHOLD_PROFILE=<profile-name>`.
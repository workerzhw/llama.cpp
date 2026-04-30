# run.sh Activation Workflow

This document describes the local experiment driver at `run.sh`.

The script is intended for local activation-threshold experiments, decode statistics,
and reduction-product profiler runs. It is not a general project launcher.

## 1. What the script does

`run.sh` wraps five related activities:

- running `llama-cli`
- running `llama-perplexity`
- running `llama-decode-stats`
- collecting activation distributions for threshold calibration
- generating threshold tables from collected distributions

It also handles:

- case selection
- per-case build flags
- build reuse vs rebuild
- artifact path resolution
- postprocessing of profiler outputs
- optional one-command activation experiment pipelines

## 2. Quick start

Show built-in help:

```bash
bash run.sh --help
```

All command snippets below are literal shell commands.
You can paste them exactly as shown from the repository root.
Lines ending with `\` are just Bash line continuations, so you can either:

- paste the multi-line block as-is
- or remove the trailing `\` characters and run it as one long line

If you use `SKIP_BUILD=1`, the required binaries must already exist under `BUILD_DIR`.
If this is the first run for a mode or the build tree is stale, switch to `SKIP_BUILD=0` once.

Run one case in `cli` mode using the current defaults:

```bash
bash run.sh
```

Run a native perplexity measurement with all replay modes disabled.
Set `SIM_MATMUL_OUT_MODE=0` as well, otherwise the output path still does a BF16 round-trip even when `SIM_Q4Q6`, `SIM_Q8Q8`, and `SIM_FP8` are all off:
the command also pins `MODEL` explicitly, so you can replace it with the exact `gguf` you want to test.

```bash
CASE_FILTER=Qwen-3-8B \
MODEL=models/Qwen/Qwen3-8B-Q5_K_M.gguf \
RUN_KIND=perplexity \
SIM_Q4Q6=0 \
SIM_Q8Q8=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=0 \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
bash run.sh
```

Run a pure `SIM_Q4Q6` perplexity measurement without mixing in activation-threshold
truncation from the FFN sparsity experiments.
The key switch is `SWIGLU_THRESHOLD_ENABLE=0`; otherwise `perplexity` and `cli`
will also load/apply the threshold config path resolved by the script.
In the current CPU canonical wiring, `SIM_Q4Q6` replays `src0` with symmetric Q6 blocks.
`src1` also uses Q6 blocks, and can optionally switch to asymmetric Q6+zero-point with
`SIM_Q4Q6_SRC1_QMODE=1`, or to logarithmic `Q6-exp + BF16 scale` replay with
`SIM_Q4Q6_SRC1_QMODE=2`.
For the logarithmic mode, `SIM_Q4Q6_SRC1_LOG_STEP` controls the exponent divisor so the replay uses
`2^(q/step)` spacing; the default is `1`:

```bash
CASE_FILTER=Llama-2-7B \
MODEL=models/hf/llama-2-7B-Q5_K_M.gguf \
RUN_KIND=perplexity \
SIM_Q4Q6=1 \
SIM_Q4Q6_SRC1_QMODE=2 \
SIM_Q4Q6_SRC1_LOG_STEP=4 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=1 \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
bash run.sh
```

Run the same style of replay experiment with symmetric `Q8/Q8` blocks on both
weights and activations.
`SIM_Q8Q8` is a standalone mode and cannot be combined with `SIM_Q4Q6` or `SIM_FP8`.
Both `src0` and `src1` use symmetric uniform `Q8` replay, with one `int8` power-of-2
block scale per block (`scale = 2^k`).
As with `SIM_Q4Q6`, keep `SWIGLU_THRESHOLD_ENABLE=0` to avoid mixing threshold experiments
into the perplexity result, and keep `SIM_MATMUL_OUT_MODE=1` so the output path still
uses the BF16 round-trip:

```bash
CASE_FILTER=Qwen-3-8B \
MODEL=models/Qwen/Qwen3-8B-Q5_K_M.gguf \
RUN_KIND=perplexity \
SIM_Q8Q8=1 \
SIM_Q8Q8_SRC0_BLOCK=32 \
SIM_Q8Q8_SRC1_BLOCK=32 \
SIM_Q4Q6=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=1 \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
bash run.sh
```

Split `Q8/Q8` replay + FFN sparsity into two commands when you only care about `ppl`.
Use the first command to do `collect -> generate` only, so you can inspect the generated
threshold reports and decide whether the selected thresholds already achieve the sparsity
you want before paying for the final `perplexity` run.
Keep `MODEL` on the `f16` GGUF here: `SIM_Q8Q8=1` injects the runtime `Q8/Q8` replay,
and the generated `silu/swiglu` thresholds are then applied back onto that same runtime path.

Start with this first command:

```bash
CASE_FILTER=Qwen-3-8B \
MODEL=models/Qwen/Qwen3-8B-Q5_K_M.gguf \
RUN_KIND=activation-flow \
FLOW_STEPS=collect,generate \
SIM_Q8Q8=1 \
SIM_Q8Q8_SRC0_BLOCK=32 \
SIM_Q8Q8_SRC1_BLOCK=32 \
SIM_Q4Q6=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=1 \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
SWIGLU_TARGET_PROFILE_KIND=minimal \
SWIGLU_TARGET_SCALE=50.0 \
SKIP_BUILD=0 \
bash run.sh
```

This first command writes both generated threshold files:

- `kv_dump_logs/<case>/<case>_silu_threshold_generated.csv`
- `kv_dump_logs/<case>/<case>_swiglu_threshold_generated.csv`

and both generated summary reports:

- `kv_dump_logs/<case>/<case>_silu_threshold_generated_summary.csv`
- `kv_dump_logs/<case>/<case>_swiglu_threshold_generated_summary.csv`

Use those `*_threshold_generated_summary.csv` files to judge whether the chosen thresholds
already hit your target sparsity. The key columns are:

- `final_threshold`: the threshold that will actually be written into the generated config
- `prefill_final_estimated_added_zero_ratio`: estimated added-zero ratio on `prefill`
- `decode_final_estimated_added_zero_ratio`: estimated added-zero ratio on `decode`

If the estimated sparsity is still too low or too high, adjust `SWIGLU_TARGET_SCALE`
and rerun this first command until the generated report reaches the range you want.

Once you are satisfied with the generated thresholds, run the final `perplexity` step as a
second command:

```bash
CASE_FILTER=Qwen-3-8B \
MODEL=models/Qwen/Qwen3-8B-Q5_K_M.gguf \
RUN_KIND=perplexity \
SIM_Q8Q8=1 \
SIM_Q8Q8_SRC0_BLOCK=32 \
SIM_Q8Q8_SRC1_BLOCK=32 \
SIM_Q4Q6=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=1 \
SWIGLU_THRESHOLD_ENABLE=1 \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
SWIGLU_THRESHOLD_PROFILE=generated \
SKIP_BUILD=0 \
bash run.sh
```

If those generated dual-threshold files already exist and you only want to rerun the final
`ppl` apply step later, reuse the same second command.

Run a full SiLU threshold pipeline in one command.
For a first run, use `SKIP_BUILD=0` so the script can build every required binary:

```bash
CASE_FILTER=Llama-3.2-1B \
RUN_KIND=activation-flow \
SWIGLU_THRESHOLD_KIND=silu \
SKIP_BUILD=0 \
bash run.sh
```

After that first successful build, you can switch back to `SKIP_BUILD=1` for reruns.

Run a full per-channel SiLU threshold pipeline.
This variant first collects the maximum absolute SiLU output magnitude for every layer/channel,
then generates one runtime threshold per channel with `threshold = abs_channel_max * SWIGLU_CHANNEL_THRESHOLD_RATIO`:

```bash
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=activation-flow \
SWIGLU_THRESHOLD_KIND=silu \
SWIGLU_GENERATE_MODE=channel-max \
SWIGLU_CHANNEL_THRESHOLD_RATIO=0.10 \
SKIP_BUILD=0 \
bash run.sh
```

Run the same pipeline at the raw gate input before SiLU.
This `silu_input` path applies one-sided negative-tail truncation before activation:

```bash
CASE_FILTER=Llama-3.2-1B \
RUN_KIND=activation-flow \
SWIGLU_THRESHOLD_KIND=silu_input \
SKIP_BUILD=0 \
bash run.sh
```

Run the same pipeline with both SiLU-output and SwiGLU-output truncation enabled.
This combined mode collects, generates, and applies both threshold families in one flow:

```bash
CASE_FILTER=Llama-3.2-1B \
RUN_KIND=activation-flow \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
SKIP_BUILD=0 \
bash run.sh
```

Run the same workflow manually, step by step.
This path is mainly useful for debugging or inspecting intermediate artifacts.
Because single-step modes build only the target needed by that step, a safe first-run
manual workflow uses `SKIP_BUILD=0` on the binary-using steps:

```bash
CASE_FILTER=Llama-3.2-1B \
RUN_KIND=swiglu-collect \
SWIGLU_THRESHOLD_KIND=swiglu \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
N_PREDICT=128 \
bash run.sh

CASE_FILTER=Llama-3.2-1B \
RUN_KIND=swiglu-generate \
SWIGLU_THRESHOLD_KIND=swiglu \
SWIGLU_TARGET_PROFILE_KIND=minimal \
SWIGLU_TARGET_SCALE=1.0 \
bash run.sh

CASE_FILTER=Llama-3.2-1B \
RUN_KIND=perplexity \
SWIGLU_THRESHOLD_KIND=swiglu \
SWIGLU_THRESHOLD_ENABLE=1 \
SWIGLU_THRESHOLD_PROFILE=generated \
SKIP_BUILD=0 \
bash run.sh

CASE_FILTER=Llama-3.2-1B \
RUN_KIND=cli \
SWIGLU_THRESHOLD_KIND=swiglu \
SWIGLU_THRESHOLD_ENABLE=1 \
SWIGLU_THRESHOLD_PROFILE=generated \
SKIP_BUILD=0 \
N_PREDICT=128 \
bash run.sh
```

Run the same SiLU workflow with per-channel thresholds, step by step.
This mode does not use a target profile. `swiglu-collect` writes a per-stage
`_channel_max.csv`, and `swiglu-generate` turns it into a `layer,channel,threshold` table:

```bash
CASE_FILTER=Llama-3.2-1B \
RUN_KIND=swiglu-collect \
SWIGLU_THRESHOLD_KIND=silu \
SWIGLU_GENERATE_MODE=channel-max \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
N_PREDICT=128 \
bash run.sh

CASE_FILTER=Llama-3.2-1B \
RUN_KIND=swiglu-generate \
SWIGLU_THRESHOLD_KIND=silu \
SWIGLU_GENERATE_MODE=channel-max \
SWIGLU_CHANNEL_THRESHOLD_RATIO=0.10 \
bash run.sh

CASE_FILTER=Llama-3.2-1B \
RUN_KIND=perplexity \
SWIGLU_THRESHOLD_KIND=silu \
SWIGLU_THRESHOLD_ENABLE=1 \
SWIGLU_THRESHOLD_PROFILE=generated \
SKIP_BUILD=0 \
bash run.sh

CASE_FILTER=Llama-3.2-1B \
RUN_KIND=cli \
SWIGLU_THRESHOLD_KIND=silu \
SWIGLU_THRESHOLD_ENABLE=1 \
SWIGLU_THRESHOLD_PROFILE=generated \
SKIP_BUILD=0 \
N_PREDICT=128 \
bash run.sh
```

Run the same dual-output workflow manually, step by step.
This is the explicit `swiglu+silu` version of the same process and writes both
the `silu` and `swiglu` artifact families under the same case directory:

```bash
CASE_FILTER=Qwen-3-8B \
RUN_KIND=swiglu-collect \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
N_PREDICT=128 \
bash run.sh

CASE_FILTER=Qwen-3-8B \
RUN_KIND=swiglu-generate \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
SWIGLU_TARGET_PROFILE_KIND=minimal \
SWIGLU_TARGET_SCALE=1.0 \
bash run.sh

CASE_FILTER=Qwen-3-8B \
RUN_KIND=perplexity \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
SWIGLU_THRESHOLD_ENABLE=1 \
SWIGLU_THRESHOLD_PROFILE=generated \
SKIP_BUILD=0 \
bash run.sh

CASE_FILTER=Qwen-3-8B \
RUN_KIND=cli \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
SWIGLU_THRESHOLD_ENABLE=1 \
SWIGLU_THRESHOLD_PROFILE=generated \
SKIP_BUILD=0 \
N_PREDICT=128 \
bash run.sh
```

Reuse existing dual artifacts and rerun only the apply steps:

```bash
CASE_FILTER=Llama-2-7B \
RUN_KIND=activation-flow \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
FLOW_REUSE_ARTIFACTS=1 \
SKIP_BUILD=1 \
BUILD_DIR=build \
bash run.sh
```

This reuse form assumes that `BUILD_DIR/bin` already contains every binary needed
by the remaining flow steps. With the default `collect,generate,perplexity,cli`
sequence, that means the same build tree must contain both `llama-perplexity`
and `llama-cli`.

### Prepare Q4_K_M models for local runs

Many `run.sh` examples pin `MODEL` explicitly.
If your local tree currently contains only `f16` GGUF files and you want a smaller
model for repeated `perplexity` or `cli` runs, build `llama-quantize` once and
quantize directly from the repository root.

This section intentionally documents the direct non-`imatrix` path.
It is the fastest local workflow when you want smaller test models and do not want
to spend additional time generating an importance matrix first.

Build the quantizer once:

```bash
cmake --build build --target llama-quantize -j$(nproc)
```

If the model is still stored as a local Hugging Face directory with `safetensors`
weights, convert it to an `f16` GGUF file first.
The current workspace includes `./models/Qwen/Qwen3-1___7B/`, so the direct local
conversion flow is:

```bash
python3 -m pip install -r requirements.txt

python3 convert_hf_to_gguf.py \
  ./models/Qwen/Qwen3-1___7B \
  --outfile ./models/Qwen/Qwen3-1___7B-f16.gguf \
  --outtype f16

./build/bin/llama-quantize \
  ./models/Qwen/Qwen3-1___7B-f16.gguf \
  ./models/Qwen/Qwen3-1___7B-Q6_K.gguf \
  Q6_K \
  $(nproc)
```

Important detail: run `convert_hf_to_gguf.py` against the unpacked directory
`./models/Qwen/Qwen3-1___7B/`, not the archive file `./models/Qwen/Qwen3-1___7B.tar`.

Current direct `Q4_K_M` conversions for the `models/` tree in this workspace:

```bash
./build/bin/llama-quantize \
  ./models/Llama-3.2-1B-Instruct-f16.gguf \
  ./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  Q4_K_M \
  $(nproc)

./build/bin/llama-quantize \
  ./models/Qwen/Qwen3-1.7B-Base-f16.gguf \
  ./models/Qwen/Qwen3-1.7B-Base-Q4_K_M.gguf \
  Q4_K_M \
  $(nproc)

./build/bin/llama-quantize \
  ./models/Qwen/Qwen3-8B-f16.gguf \
  ./models/Qwen/Qwen3-8B-Q5_K_M.gguf \
  Q5_K_M \
  $(nproc)

./build/bin/llama-quantize \
  ./models/hf/Llama-3.2-1B-Instruct-f16.gguf \
  ./models/hf/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  Q4_K_M \
  $(nproc)

./build/bin/llama-quantize \
  ./models/hf/Llama-3___2-3B-Instruct-f16.gguf \
  ./models/hf/Llama-3___2-3B-Instruct-Q5_K_M.gguf \
  Q5_K_M \
  $(nproc)

./build/bin/llama-quantize \
  ./models/hf/llama-2-7B-F16.gguf \
  ./models/hf/llama-2-7B-Q5_K_M.gguf \
  Q5_K_M \
  $(nproc)
```

If you want to quantize the whole current set in one pass, use the same inputs in
this loop:

```bash
for f in \
  ./models/Llama-3.2-1B-Instruct-f16.gguf \
  ./models/Qwen/Qwen3-1.7B-Base-f16.gguf \
  ./models/Qwen/Qwen3-8B-f16.gguf \
  ./models/hf/Llama-3.2-1B-Instruct-f16.gguf \
  ./models/hf/Llama-3___2-3B-Instruct-f16.gguf \
  ./models/hf/llama-2-7B-F16.gguf
do
  out="${f%.gguf}-Q4_K_M.gguf"
  out="${out/-f16-Q4_K_M/-Q4_K_M}"
  out="${out/-F16-Q4_K_M/-Q4_K_M}"
  ./build/bin/llama-quantize "$f" "$out" Q4_K_M $(nproc)
done
```

After quantization, point any `run.sh` experiment at the new file by overriding
`MODEL`, for example:

```bash
CASE_FILTER=Llama-3.2-1B \
MODEL=models/hf/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
RUN_KIND=perplexity \
SIM_Q4Q6=0 \
SIM_Q8Q8=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=0 \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
bash run.sh
```

If you want a different quantization level, keep the same input/output pattern and
replace both the output suffix and the quant type argument, for example `Q5_K_M`
or `Q6_K`.

### Prepare models with FFN gate/up in IQ2_S and FFN down left native

Use `llama-quantize --tensor-type` when you want only selected FFN parameter
matrices to use `IQ2_S` while every other tensor keeps the native assignment of
the base quantization type.

Do not use `--pure` for this workflow. The base quantization type, such as
`Q4_K_M`, is still the native mixed-quantization policy for all tensors that are
not matched by `--tensor-type`. The override below matches only `ffn_gate` and
`ffn_up`; `ffn_down` is intentionally not matched, so it keeps the base policy.

`IQ2_S` is a very low-bit type, so an importance matrix is required. The imatrix
file must contain entries for the matched FFN gate/up tensors. If you generated
or filtered the imatrix with `--include-weights` or `--exclude-weights`, make sure
those FFN tensors are still included.

Dense Transformer FFN override:

```bash
cmake --build build --target llama-quantize -j$(nproc)

./build/bin/llama-quantize \
  --imatrix ./imatrix.dat \
  --tensor-type '^blk\.[0-9]+\.ffn_(gate|up)\.weight$=iq2_s' \
  ./models/Qwen/Qwen3-8B-f16.gguf \
  ./models/Qwen/Qwen3-8B-Q4_K_M-ffn-gate-up-IQ2_S.gguf \
  Q4_K_M \
  $(nproc)
```

MoE FFN override, including expert, shared-expert, and chunk-expert FFN gate/up
matrices while still leaving `ffn_down*` native:

```bash
cmake --build build --target llama-quantize -j$(nproc)

./build/bin/llama-quantize \
  --imatrix ./imatrix.dat \
  --tensor-type '^blk\.[0-9]+\.ffn_(gate|up)(_exps|_shexp|_chexps)?\.weight$=iq2_s' \
  ./models/Qwen/Qwen3-MoE-f16.gguf \
  ./models/Qwen/Qwen3-MoE-Q4_K_M-ffn-gate-up-IQ2_S.gguf \
  Q4_K_M \
  $(nproc)
```

Keep `ffn_gate_inp` out of the regex. It is the MoE router/gate-input tensor,
not the ordinary FFN gate projection, and should normally remain under the base
quantization policy.

To use the generated model in a `run.sh` experiment, point `MODEL` at the new
GGUF file as usual:

```bash
CASE_FILTER=Qwen-3-8B \
MODEL=models/Qwen/Qwen3-8B-Q4_K_M-ffn-gate-up-IQ2_S.gguf \
RUN_KIND=perplexity \
SIM_Q4Q6=0 \
SIM_Q8Q8=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=0 \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
bash run.sh
```

You can replace `Q4_K_M` with another base type, such as `Q5_K_M`, `IQ3_M`, or
`IQ2_M`. For `IQ2_M`, this override is less useful because the base policy already
uses `IQ2_S` for many ordinary matrices; the main purpose of this workflow is to
start from a higher-quality native baseline and selectively compress only the FFN
gate/up matrices.

After converting the local `Qwen3-1___7B` directory this way, you can point
`run.sh` at the generated quantized model directly:

```bash
CASE_FILTER=Qwen-3-1.7B \
MODEL=models/Qwen/Qwen3-1___7B-Q4_K_M.gguf \
RUN_KIND=perplexity \
SIM_Q4Q6=0 \
SIM_Q8Q8=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=0 \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
bash run.sh
```

## 3. Supported run modes

Primary `RUN_KIND` values:

- `cli`: run `llama-cli`
- `perplexity`: run `llama-perplexity`
- `decode-stats`: run `llama-decode-stats`
- `swiglu-collect`: collect activation distributions
- `swiglu-generate`: generate threshold tables from collected distributions
- `activation-flow`: run a multi-step activation experiment pipeline

Accepted aliases:

- `collect` and `activation-collect` map to `swiglu-collect`
- `generate` and `activation-generate` map to `swiglu-generate`
- `flow` maps to `activation-flow`

The script keeps the historical `swiglu-*` names for compatibility, even though
the same pipeline now also supports `SWIGLU_THRESHOLD_KIND=silu`,
`SWIGLU_THRESHOLD_KIND=silu_input`, and `SWIGLU_THRESHOLD_KIND=swiglu+silu`.

## 4. Case selection

Cases are defined directly in `run.sh`.

The current script uses a shared helper, `make_standard_case_spec`, because most
cases differ only in case name and model path.

If the same variable is set both in a case and on the shell command line,
the shell command line wins. For example, `N_PREDICT=128 bash run.sh` overrides
any `N_PREDICT` value embedded in the selected case.

Case execution order is controlled by the `RUN_CASES` array near the top of the file.

`CASE_FILTER` is substring matching against both:

- the human-readable case name
- the sanitized case slug used for paths

Examples:

```bash
CASE_FILTER=Llama-3.2-1B bash run.sh
CASE_FILTER=Qwen-3.1 bash run.sh
CASE_FILTER=f8e3m4-normal bash run.sh
```

## 5. Build behavior

The script builds only the targets required by the chosen run mode.

Mode to build-target mapping:

- `cli` and `swiglu-collect` build `llama-cli`
- `perplexity` builds `llama-perplexity`
- `decode-stats` builds `llama-decode-stats`
- `swiglu-generate` is generator-only and does not build C++ targets
- `activation-flow` builds the union of all targets needed by its steps once per case

Main build knobs:

- `BUILD_DIR`: build tree path, default `build`
- `SKIP_BUILD=0`: rebuild the required targets for the case
- `SKIP_BUILD=1`: reuse an existing build tree and only verify required binaries exist

Examples:

```bash
SKIP_BUILD=0 BUILD_DIR=build bash run.sh
SKIP_BUILD=1 BUILD_DIR=build-baseline bash run.sh
```

Important detail:

- in a single-step mode, `SKIP_BUILD=0` recreates `BUILD_DIR` and builds only the target needed by that mode
- in `activation-flow`, `SKIP_BUILD=0` builds the union of all required targets once, which is usually the least manual option
- `SKIP_BUILD=1` only works if `BUILD_DIR/bin` already contains every binary required by the selected mode or flow
- with the default `activation-flow` step list, `SKIP_BUILD=1` therefore requires the same `BUILD_DIR/bin` to contain both `llama-cli` and `llama-perplexity`

## 6. Activation threshold knobs

These variables control the threshold experiment itself:

- `SWIGLU_THRESHOLD_KIND`: `swiglu`, `silu`, `silu_input`, or `swiglu+silu`
- `SWIGLU_THRESHOLD_ENABLE`: whether `cli` and `perplexity` should load/apply thresholds
- `SWIGLU_THRESHOLD_PROFILE`: threshold profile name, or `generated`
- `SWIGLU_THRESHOLD_CONFIG`: explicit threshold CSV path, or `auto`
- `SWIGLU_GENERATE_MODE`: `target-profile` or `channel-max`
- `SWIGLU_CHANNEL_THRESHOLD_RATIO`: only for `channel-max`; per-channel threshold = absolute channel max * ratio

Activation kind semantics:

- `swiglu`: truncate the post-gating SwiGLU output
- `silu`: truncate the SiLU branch output before the final multiply
- `silu_input`: truncate the raw gate input before SiLU using one-sided negative-tail thresholds
- `swiglu+silu`: apply SiLU-output truncation first, then apply SwiGLU-output truncation in the same run

In dual apply mode, runtime sparsity accounting is ordered, not independent:

- the `silu` runtime/report sees the tensor before the later `swiglu` truncation step, so its `final_zero_ratio` reflects the effect of the SiLU step alone
- the later `swiglu` runtime/report sees the tensor after SiLU has already zeroed some values, so its `original_zero_ratio` already includes zeros introduced by the earlier SiLU step
- in that later `swiglu` report, `truncated_nonzero_ratio` is only the additional zeros introduced by the SwiGLU step itself, while `final_zero_ratio` is the cumulative zero ratio after both steps

Auto-resolution rules:

- if `SWIGLU_THRESHOLD_CONFIG=auto` and `SWIGLU_THRESHOLD_PROFILE=generated`, the script uses
  `kv_dump_logs/<case>/<case>_<kind>_threshold_generated.csv`
- otherwise it uses `tools/<kind>-threshold-configs/<profile>/<case>.csv`
- in `SWIGLU_GENERATE_MODE=channel-max`, that generated config path keeps the same filename but the CSV format becomes `layer,channel,threshold`

Combined-mode note:

- when `SWIGLU_THRESHOLD_KIND=swiglu+silu`, the script resolves both families automatically
- in that mode, `SWIGLU_THRESHOLD_CONFIG`, `SWIGLU_COLLECT_PREFIX`, `SWIGLU_TARGET_PROFILE`, `SWIGLU_GENERATED_CONFIG`, and `SWIGLU_GENERATED_REPORT` must stay `auto`
- the primary runtime channel is `silu` and the secondary runtime channel is `swiglu`
- dual mode does not solve one shared threshold across both families; it runs calibration twice, once for `silu` and once for `swiglu`
- each family uses its own collected histogram pair and its own target-profile CSV; the default `minimal` profiles currently carry the same group ratios, but they are still resolved from separate `tools/silu-threshold-targets/...` and `tools/swiglu-threshold-targets/...` files

Important naming note:

- variable names keep the historical `SWIGLU_*` prefix
- the actual activation family is selected by `SWIGLU_THRESHOLD_KIND`
- `swiglu` generation is anchored at `prefill` `layer 1`: the generator finds one threshold that makes that reference slice hit its target added-zero ratio, then applies that same threshold to all layers and all stages
- `silu` generation is anchored at `prefill` `layer 1`: the generator finds one threshold that makes that reference slice hit its target added-zero ratio, then applies that same threshold to all layers and all stages
- `silu` with `SWIGLU_GENERATE_MODE=channel-max` bypasses target profiles: the collector writes per-stage per-layer per-channel absolute SiLU maxima, the generator takes `abs_channel_max = max(prefill_abs_max, decode_abs_max)`, and writes `channel_threshold = abs_channel_max * SWIGLU_CHANNEL_THRESHOLD_RATIO`
- in that `channel-max` mode, runtime still performs the SiLU-output truncation test on the output value itself, now with per-channel thresholds, using `|value| <= channel_threshold`
- `silu_input` stores positive threshold magnitudes in CSV, but runtime truncation is one-sided and triggers on `gate_value <= -threshold`

## 7. Collection and generation knobs

Collection controls:

- `SWIGLU_COLLECT_PREFIX`: artifact prefix, or `auto`
- `SWIGLU_COLLECT_BINS`: histogram bins
- `SWIGLU_COLLECT_LOG10_MIN`: lower histogram bound
- `SWIGLU_COLLECT_LOG10_MAX`: upper histogram bound

Generation controls:

- `SWIGLU_TARGET_PROFILE_KIND`: profile subdirectory, for example `minimal`
- `SWIGLU_TARGET_PROFILE`: explicit target-profile CSV, or `auto`
- `SWIGLU_TARGET_SCALE`: multiplier applied to target added-zero ratios before inversion
- `SWIGLU_GENERATED_CONFIG`: explicit generated threshold output path, or `auto`
- `SWIGLU_GENERATED_REPORT`: explicit generation report path, or `auto`
- `SWIGLU_GENERATE_MODE`: `target-profile` or `channel-max`
- `SWIGLU_CHANNEL_THRESHOLD_RATIO`: for `channel-max`, threshold = absolute channel max * ratio

Auto-resolution rules:

- collection prefix defaults to `kv_dump_logs/<case>/<case>_<kind>_collect`
- target profile defaults to `tools/<kind>-threshold-targets/<profile-kind>/<case>.csv`
- generated config defaults to `kv_dump_logs/<case>/<case>_<kind>_threshold_generated.csv`
- generated report defaults to `kv_dump_logs/<case>/<case>_<kind>_threshold_generated_summary.csv`

Collection semantics differ by activation kind:

- `swiglu` and `silu` collect absolute-value histograms of the target output tensor
- `silu` with `SWIGLU_GENERATE_MODE=channel-max` also records per-stage per-layer per-channel absolute maxima and writes `..._silu_collect_channel_max.csv`
- `silu_input` collects negative-tail magnitudes from the raw gate input tensor before SiLU so the generator can invert one-sided truncation targets
- `swiglu+silu` collects both the `silu` and `swiglu` families in one `swiglu-collect` run and writes two artifact pairs under the same case directory

Generation semantics also differ by activation kind:

- `swiglu`: use `prefill` `layer 1` as the anchor, invert that one target into a single global threshold, and write the same threshold to every layer
- `silu` + `target-profile`: use `prefill` `layer 1` as the anchor, invert that one target into a single global threshold, and write the same threshold to every layer
- `silu` + `channel-max`: ignore target profiles, read `..._silu_collect_channel_max.csv`, take the maximum absolute SiLU output magnitude per channel across prefill and decode, and emit `layer,channel,threshold` with `threshold = abs_channel_max * SWIGLU_CHANNEL_THRESHOLD_RATIO`
- `silu_input`: generate one threshold per layer from one-sided negative-tail histograms
- `swiglu+silu`: run the generator twice in one `swiglu-generate` step, producing both the `silu` and `swiglu` generated config/report pairs

In the current dual-output setup, the two stage distributions are selected like this:

- `silu`: read `..._silu_collect_summary.csv` and `..._silu_collect_hist.csv`, then look up the `prefill` target for `layer 1` from the SiLU target profile and invert that one reference slice into one global threshold
- `swiglu`: read `..._swiglu_collect_summary.csv` and `..._swiglu_collect_hist.csv`, then look up the `prefill` target for `layer 1` from the SwiGLU target profile and invert that one reference slice into one global threshold
- after that, each family applies its own global threshold to every layer and both stages
- the per-layer `decode_target` values and non-reference layers are still written into the generated summary report, but they are only used for estimation and comparison; they do not change the chosen final threshold in `swiglu` or `silu` mode

## 8. activation-flow

`RUN_KIND=activation-flow` is the main change for reducing manual work.

Default step sequence:

- `collect`
- `generate`
- `perplexity`
- `cli`

Customize the sequence with `FLOW_STEPS`:

```bash
RUN_KIND=activation-flow \
FLOW_STEPS=collect,generate,cli \
bash run.sh
```

Behavior of `activation-flow`:

- it builds the union of required targets once per case
- it automatically turns threshold application on for the `perplexity` and `cli` steps
- it keeps threshold application off for `collect` and `generate`
- if a case defines `N_PREDICT=-1`, the flow uses `FLOW_DECODE_N_PREDICT` for `collect`
- if a case defines `N_PREDICT=-1`, the flow also uses `FLOW_DECODE_N_PREDICT` for `cli`

Reuse existing artifacts instead of rerunning completed steps:

```bash
RUN_KIND=activation-flow \
FLOW_REUSE_ARTIFACTS=1 \
bash run.sh
```

This is useful when collect/generate outputs already exist and you only want to
finish the remaining steps.

Important detail:

- `FLOW_REUSE_ARTIFACTS=1` skips completed artifact-producing steps, but it does not relax the binary checks for the remaining flow steps
- if the remaining flow still includes `perplexity` and `cli`, the same `BUILD_DIR/bin` must contain both `llama-perplexity` and `llama-cli`

## 9. Other runtime knobs

General runtime inputs:

- `MODEL`
- `DATA`
- `PROMPT`
- `CTX`
- `THREADS`
- `N_PREDICT`
- `SEQ_ID`
- `BATCH`
- `UBATCH`
- `STRIDE`

Profiler and graph extras:

- `DUMP_DOT`
- `REDUCTION_PROD_PROFILE`
- `REDUCTION_PROD_PROFILE_BINS`
- `REDUCTION_PROD_PROFILE_HIST_MIN_LOG2`
- `REDUCTION_PROD_PROFILE_HIST_MAX_LOG2`
- `REDUCTION_PROD_PROFILE_SAMPLE_RATE`
- `REDUCTION_PROD_BLOCK_DROP_LOG2_N`
- `REDUCTION_PROD_PROFILE_MAX_SAMPLES`

Low-precision simulation controls remain available and are passed into the build as CMake flags.

- `SIM_FP8`, `SIM_FP_FORMAT`, `SIM_FP8_LAYOUT`, `SIM_FP8_*`: fp8-sim controls
- `SIM_Q4Q6`, `SIM_Q4Q6_APPLY_SRC0`, `SIM_Q4Q6_APPLY_SRC1`, `SIM_Q4Q6_SRC0_BLOCK`, `SIM_Q4Q6_SRC1_BLOCK`: standalone low-bit replay controls for `SIM_Q4Q6` (current canonical CPU wiring: `src0=Q6`, `src1=Q6`)
- `SIM_Q4Q6_SRC1_QMODE`: `0=symmetric Q6`, `1=asymmetric Q6+zp`, `2=logarithmic Q6-exp + BF16 block scale` for `src1` only
- `SIM_Q4Q6_SRC1_LOG_STEP`: positive exponent divisor used by `SIM_Q4Q6_SRC1_QMODE=2`; replay spacing follows `2^(q/step)`

Important interaction:

- `SWIGLU_THRESHOLD_ENABLE=1` affects `perplexity` and `cli` runs independently of the low-precision simulation flags
- for pure `SIM_Q4Q6` or pure fp8-sim measurements, set `SWIGLU_THRESHOLD_ENABLE=0` explicitly so FFN threshold truncation does not change the result

## 10. Output layout

Case outputs are written under:

```text
<OUT_DIR>/<case-slug>/
```

Common logs:

- `<case>_cli.log`
- `<case>_perplexity.log`
- `<case>_decode-stats.log`
- `<case>_swiglu-collect.log`
- `<case>_swiglu-generate.log`

Activation artifacts:

- `<case>_<kind>_collect_summary.csv`
- `<case>_<kind>_collect_hist.csv`
- `<case>_silu_collect_channel_max.csv` when `SWIGLU_THRESHOLD_KIND=silu` and `SWIGLU_GENERATE_MODE=channel-max`
- `<case>_<kind>_threshold_generated.csv`
- `<case>_<kind>_threshold_generated_summary.csv`
- `<case>_<kind>_threshold_cli.csv`
- `<case>_<kind>_threshold_perplexity.csv`

In `SWIGLU_THRESHOLD_KIND=swiglu+silu` mode, the same run writes both `<kind>=silu`
and `<kind>=swiglu` variants for the activation artifacts above.

In `SWIGLU_GENERATE_MODE=channel-max`, `<case>_silu_threshold_generated.csv` stores
`layer,channel,threshold` instead of `layer,threshold`.

Profiler / postprocessing artifacts:

- `<case>_fp8_sim_analysis.log`
- `<case>_fp8_interval_hist.csv`
- `<case>_fp8_interval_hist.png`
- `<case>_block_psum_relation.csv`
- `<case>_block_psum_relation.md`
- `block_over_psum_hist.png`
- `block_over_psum_percent.png`
- `block_over_psum_cdf.png`

At the output-root level, the script also tries to write:

- `reduction_block_drop_compare.csv`
- `reduction_block_drop_compare.md`

## 11. Typical workflows

### 11.1 Reuse an existing build

```bash
CASE_FILTER=Llama-3.2-1B \
RUN_KIND=cli \
SKIP_BUILD=1 \
BUILD_DIR=build \
bash run.sh
```

### 11.2 Run decode stats only

```bash
CASE_FILTER=Llama-3.2-1B \
RUN_KIND=decode-stats \
SKIP_BUILD=1 \
N_PREDICT=128 \
bash run.sh
```

### 11.3 Run one-command SwiGLU threshold pipeline

```bash
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=activation-flow \
SWIGLU_THRESHOLD_KIND=swiglu \
SWIGLU_TARGET_PROFILE_KIND=minimal \
SWIGLU_TARGET_SCALE=1.0 \
SKIP_BUILD=0 \
bash run.sh
```

### 11.4 Run pure `SIM_Q4Q6` perplexity

This is the recommended form when you want `ppl` for the standalone `SIM_Q4Q6` replay only,
without any FFN threshold sparsity effect mixed in.

```bash
CASE_FILTER=Llama-3.2-1B \
RUN_KIND=perplexity \
SIM_Q4Q6=1 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=1 \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
bash run.sh
```

### 11.5 Run one-command SiLU threshold pipeline with artifact reuse

```bash
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=activation-flow \
SWIGLU_THRESHOLD_KIND=silu \
FLOW_REUSE_ARTIFACTS=1 \
SKIP_BUILD=1 \
bash run.sh
```

### 11.6 Run one-command per-channel SiLU threshold pipeline

This mode bypasses target profiles and derives one runtime threshold per SiLU output channel.

```bash
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=activation-flow \
SWIGLU_THRESHOLD_KIND=silu \
SWIGLU_GENERATE_MODE=channel-max \
SWIGLU_CHANNEL_THRESHOLD_RATIO=0.10 \
SKIP_BUILD=0 \
bash run.sh
```

### 11.7 Run one-command dual SwiGLU + SiLU threshold pipeline

Use `SKIP_BUILD=0` on the first dual run so the same build tree is populated with
every binary required by the default flow.

```bash
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=activation-flow \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
SWIGLU_TARGET_PROFILE_KIND=minimal \
SKIP_BUILD=0 \
bash run.sh
```

### 11.8 Select thresholds first, then run ppl for `Q8/Q8` + FFN sparsity

Use the first command to run `collect,generate` only.
Inspect `*_threshold_generated_summary.csv` to see whether the estimated added-zero ratios
already match the sparsity you want; if not, change `SWIGLU_TARGET_SCALE` and rerun only
this first command.

```bash
CASE_FILTER=Qwen-3-8B \
MODEL=models/Qwen/Qwen3-8B-f16.gguf \
RUN_KIND=activation-flow \
FLOW_STEPS=collect,generate \
SIM_Q8Q8=1 \
SIM_Q8Q8_SRC0_BLOCK=32 \
SIM_Q8Q8_SRC1_BLOCK=32 \
SIM_Q4Q6=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=1 \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
SWIGLU_TARGET_PROFILE_KIND=minimal \
SWIGLU_TARGET_SCALE=1.0 \
SKIP_BUILD=0 \
bash run.sh
```

Then run `ppl` only after you are happy with the generated thresholds:

```bash
CASE_FILTER=Qwen-3-8B \
MODEL=models/Qwen/Qwen3-8B-f16.gguf \
RUN_KIND=perplexity \
SIM_Q8Q8=1 \
SIM_Q8Q8_SRC0_BLOCK=32 \
SIM_Q8Q8_SRC1_BLOCK=32 \
SIM_Q4Q6=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=1 \
SWIGLU_THRESHOLD_ENABLE=1 \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
SWIGLU_THRESHOLD_PROFILE=generated \
SKIP_BUILD=0 \
bash run.sh
```

### 11.9 Reuse dual artifacts and rerun only apply steps

```bash
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=activation-flow \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
FLOW_REUSE_ARTIFACTS=1 \
SKIP_BUILD=1 \
BUILD_DIR=build \
bash run.sh
```

### 11.10 Run dual SwiGLU + SiLU workflow manually, step by step

```bash
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=swiglu-collect \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
N_PREDICT=128 \
bash run.sh

CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=swiglu-generate \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
SWIGLU_TARGET_PROFILE_KIND=minimal \
SWIGLU_TARGET_SCALE=1.0 \
bash run.sh

CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=perplexity \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
SWIGLU_THRESHOLD_ENABLE=1 \
SWIGLU_THRESHOLD_PROFILE=generated \
SKIP_BUILD=0 \
bash run.sh

CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=cli \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
SWIGLU_THRESHOLD_ENABLE=1 \
SWIGLU_THRESHOLD_PROFILE=generated \
SKIP_BUILD=0 \
N_PREDICT=128 \
bash run.sh
```

## 12. Editing cases

For a new case with the same common settings, add one line like this near the top of `run.sh`:

```bash
CASE_MY_MODEL="$(make_standard_case_spec "My-Case-Name" "models/path/to/model.gguf")"
```

Then add it into `RUN_CASES`.

If a case needs different runtime or FP8 knobs, either:

- add those overrides after generating the standard spec, or
- replace that case with a custom here-doc block

## 13. Notes on compatibility

Compatibility choices intentionally preserved in the script:

- the threshold env vars still use the `SWIGLU_*` prefix
- `swiglu-collect` and `swiglu-generate` remain the canonical artifact names
- aliases like `collect`, `generate`, and `flow` are accepted for convenience

This keeps old local habits working while making the script structure clearer.
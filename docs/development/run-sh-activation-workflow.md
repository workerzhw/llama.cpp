# run.sh Activation Workflow

本文档说明仓库根目录下的本地实验驱动脚本 `run.sh`。

`run.sh` 主要用于本地 activation threshold 实验、decode 统计观察、
低精度 replay 实验，以及 reduction-product profiler 运行。它不是通用的
项目启动器。

## 1. 先看这里

大多数任务可以先按下面这张表选路径：

| 目标 | 推荐命令形态 |
| --- | --- |
| 用 `llama-cli` 快速跑一个 case | `CASE_FILTER=... RUN_KIND=cli bash run.sh` |
| 跑普通 perplexity 基线 | `RUN_KIND=perplexity`，并显式关闭 simulation 和 threshold |
| 跑纯 `SIM_Q4Q6`、`SIM_Q6Q6`、`SIM_Q6Q8` 或 `SIM_Q8Q8` replay | `RUN_KIND=perplexity`，开启 replay，设置 `SWIGLU_THRESHOLD_ENABLE=0` |
| 一条命令跑完整 activation threshold 流程 | `RUN_KIND=activation-flow` |
| 先生成 threshold，检查报告后再跑 `ppl` | 先跑 `RUN_KIND=activation-flow FLOW_STEPS=collect,generate`，再跑 `RUN_KIND=perplexity` |
| 分步调试 threshold 流程 | 依次跑 `swiglu-collect`、`swiglu-generate`，最后用 `perplexity` 或 `cli` apply |
| 复用已有 generated artifacts | `RUN_KIND=activation-flow FLOW_REUSE_ARTIFACTS=1` |
| 查看 decode-time tensor 统计 | `RUN_KIND=decode-stats` |

查看脚本内置帮助：

```bash
bash run.sh --help
```

下面所有命令都默认从仓库根目录运行。以 `\` 结尾的行是 Bash 续行，可以
整块粘贴执行，也可以去掉 `\` 后合并成一行执行。

构建行为要特别注意：

- `SKIP_BUILD=0` 会重建 `BUILD_DIR`，并构建当前 mode 所需的二进制。
- `SKIP_BUILD=1` 会复用 `BUILD_DIR`，只检查所需二进制是否已经存在。
- 第一次跑某个 mode，或者 build tree 可能已经过期时，先用 `SKIP_BUILD=0`。

## 2. 心智模型

`run.sh` 主要协调四件事：

1. 选择 case：从 `RUN_CASES` 中选择模型和运行参数。
2. 选择构建：把 simulation flags 传给 CMake，并构建所需工具。
3. 执行工具：运行 `llama-cli`、`llama-perplexity`、`llama-decode-stats`，或运行 threshold generator。
4. 归档产物：把日志、收集到的分布、生成的 threshold、报告和 profiler 输出放到对应 case 目录。

Activation threshold 流程可以理解成三步：

```text
collect activation distributions
  -> generate threshold CSV files
  -> apply thresholds in llama-perplexity and/or llama-cli
```

`RUN_KIND=activation-flow` 是这个流程的一键模式，默认 step list 是：

```text
collect,generate,perplexity,cli
```

可以用 `FLOW_STEPS` 改写这个列表。

在 `activation-flow` 中，`collect` 和 `generate` 会关闭 threshold apply；
`perplexity` 和 `cli` 会开启 threshold apply。如果某个 case 里
`N_PREDICT=-1`，flow 会用 `FLOW_DECODE_N_PREDICT` 作为 `collect` 和
`cli` 的有限 token 预算。

## 3. 常用工作流

### 3.1 快速跑一个 case

```bash
CASE_FILTER=Llama-3.2-1B \
RUN_KIND=cli \
SKIP_BUILD=0 \
bash run.sh
```

第一次成功构建后，可以复用同一个 build tree：

```bash
CASE_FILTER=Llama-3.2-1B \
RUN_KIND=cli \
SKIP_BUILD=1 \
BUILD_DIR=build \
bash run.sh
```

### 3.2 普通 native perplexity 基线

这个命令用于没有低精度 replay、没有 activation threshold truncation 的基线。
这里显式设置 `SIM_MATMUL_OUT_MODE=0` 很重要；否则默认 case 配置仍然会让
matmul output 路径走 BF16 round-trip。

```bash
CASE_FILTER=Qwen-3-8B \
MODEL=models/Qwen/Qwen3-8B-Q5_K_M.gguf \
RUN_KIND=perplexity \
SIM_Q4Q6=0 \
SIM_Q6Q6=0 \
SIM_Q6Q8=0 \
SIM_Q8Q8=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=0 \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
bash run.sh
```

### 3.3 纯 `SIM_Q4Q6` perplexity

`SIM_Q4Q6` 不能和 `SIM_Q6Q6`、`SIM_Q6Q8`、`SIM_Q8Q8` 或 `SIM_FP8` 同时开启。

当前 CPU 路径中，`src0` 使用 symmetric Q6 blocks。`src1` 默认也是
symmetric Q6，也可以通过 `SIM_Q4Q6_SRC1_QMODE=1` 切换到 asymmetric
Q6+zero-point，或通过 `SIM_Q4Q6_SRC1_QMODE=2` 切换到 logarithmic
Q6-exp plus BF16 block scale。

保持 `SWIGLU_THRESHOLD_ENABLE=0`，避免 FFN threshold truncation 混进 replay
测量结果。保持 `SIM_MATMUL_OUT_MODE=1`，因为 Q4Q6 replay 依赖 BF16 output
round-trip。

```bash
CASE_FILTER=Llama-2-7B \
MODEL=models/hf/llama-2-7B-Q5_K_M.gguf \
RUN_KIND=perplexity \
SIM_Q4Q6=1 \
SIM_Q4Q6_SRC1_QMODE=2 \
SIM_Q4Q6_SRC1_LOG_STEP=4 \
SIM_Q6Q6=0 \
SIM_Q6Q8=0 \
SIM_Q8Q8=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=1 \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
bash run.sh
```

### 3.4 纯 `SIM_Q6Q6` perplexity

`SIM_Q6Q6` 是 `SIM_Q8Q8` 的 Q6 版本：两个 source 都使用 symmetric uniform
Q6 replay，每个 block 使用一个 int8 power-of-2 block scale。除量化范围
`[-31, 31]` 外，其余结构和 `SIM_Q8Q8` 一致。它不能和 `SIM_Q4Q6`、
`SIM_Q6Q8`、`SIM_Q8Q8` 或 `SIM_FP8` 同时开启。

```bash
CASE_FILTER=Qwen-3-8B \
MODEL=models/Qwen/Qwen3-8B-Q5_K_M.gguf \
RUN_KIND=perplexity \
SIM_Q6Q6=1 \
SIM_Q6Q6_SRC0_BLOCK=32 \
SIM_Q6Q6_SRC1_BLOCK=32 \
SIM_Q4Q6=0 \
SIM_Q6Q8=0 \
SIM_Q8Q8=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=1 \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
bash run.sh
```

### 3.5 纯 `SIM_Q6Q8` perplexity

`SIM_Q6Q8` 用于“权重 Q6、激活 Q8”的 replay：`src0` 走 symmetric uniform
Q6，范围 `[-31, 31]`；`src1` 走 symmetric uniform Q8，范围 `[-127, 127]`。
两个 source 都使用 int8 power-of-2 block scale，输出仍走 BF16 round-trip。它
不能和 `SIM_Q4Q6`、`SIM_Q6Q6`、`SIM_Q8Q8` 或 `SIM_FP8` 同时开启。

```bash
CASE_FILTER=Qwen-3-8B \
MODEL=models/Qwen/Qwen3-8B-Q5_K_M.gguf \
RUN_KIND=perplexity \
SIM_Q6Q8=1 \
SIM_Q6Q8_SRC0_BLOCK=32 \
SIM_Q6Q8_SRC1_BLOCK=32 \
SIM_Q4Q6=0 \
SIM_Q6Q6=0 \
SIM_Q8Q8=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=1 \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
bash run.sh
```

### 3.6 纯 `SIM_Q8Q8` perplexity

`SIM_Q8Q8` 是独立 replay mode，不能和 `SIM_Q4Q6`、`SIM_Q6Q6`、`SIM_Q6Q8`
或 `SIM_FP8` 同时开启。
两个 source 都使用 symmetric uniform Q8 replay，每个 block 使用一个 int8
power-of-2 block scale。

```bash
CASE_FILTER=Qwen-3-8B \
MODEL=models/Qwen/Qwen3-8B-Q5_K_M.gguf \
RUN_KIND=perplexity \
SIM_Q8Q8=1 \
SIM_Q8Q8_SRC0_BLOCK=32 \
SIM_Q8Q8_SRC1_BLOCK=32 \
SIM_Q4Q6=0 \
SIM_Q6Q6=0 \
SIM_Q6Q8=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=1 \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
bash run.sh
```

### 3.7 一键 threshold pipeline

第一次跑时用 `SKIP_BUILD=0`，让脚本构建默认 flow 所需的全部二进制。

SwiGLU output threshold：

```bash
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=activation-flow \
SWIGLU_THRESHOLD_KIND=swiglu \
SWIGLU_TARGET_PROFILE_KIND=minimal \
SWIGLU_TARGET_SCALE=1.0 \
SKIP_BUILD=0 \
bash run.sh
```

SiLU output threshold：

```bash
CASE_FILTER=Llama-3.2-1B \
RUN_KIND=activation-flow \
SWIGLU_THRESHOLD_KIND=silu \
SKIP_BUILD=0 \
bash run.sh
```

SiLU input threshold。这个路径在 SiLU activation 之前，对 raw gate input 做
one-sided negative-tail truncation：

```bash
CASE_FILTER=Llama-3.2-1B \
RUN_KIND=activation-flow \
SWIGLU_THRESHOLD_KIND=silu_input \
SKIP_BUILD=0 \
bash run.sh
```

Dual SiLU output plus SwiGLU output threshold：

```bash
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=activation-flow \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
SWIGLU_TARGET_PROFILE_KIND=minimal \
SKIP_BUILD=0 \
bash run.sh
```

Per-channel SiLU output threshold。这个 mode 不使用 target-profile CSV，而是使用
`threshold = abs_channel_max * SWIGLU_CHANNEL_THRESHOLD_RATIO`：

```bash
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=activation-flow \
SWIGLU_THRESHOLD_KIND=silu \
SWIGLU_GENERATE_MODE=channel-max \
SWIGLU_CHANNEL_THRESHOLD_RATIO=0.10 \
SKIP_BUILD=0 \
bash run.sh
```

### 3.8 先选 threshold，再跑 `ppl`

当最终 `perplexity` 很贵时，可以先只做 collect/generate，检查生成的
summary report，再决定是否跑最终 apply。

对于 Q8/Q8 replay plus dual FFN sparsity，这里保持 `MODEL` 指向 f16 GGUF。
`SIM_Q8Q8=1` 会在运行时注入 Q8/Q8 replay，生成的 `silu/swiglu` thresholds
也会 apply 到同一条 runtime path 上。

第一条命令只做 collect 和 generate：

```bash
CASE_FILTER=Qwen-3-8B \
MODEL=models/Qwen/Qwen3-8B-f16.gguf \
RUN_KIND=activation-flow \
FLOW_STEPS=collect,generate \
SIM_Q8Q8=1 \
SIM_Q8Q8_SRC0_BLOCK=32 \
SIM_Q8Q8_SRC1_BLOCK=32 \
SIM_Q4Q6=0 \
SIM_Q6Q6=0 \
SIM_Q6Q8=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=1 \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
SWIGLU_TARGET_PROFILE_KIND=minimal \
SWIGLU_TARGET_SCALE=1.0 \
SKIP_BUILD=0 \
bash run.sh
```

检查生成的 summary files：

```text
kv_dump_logs/<case>/<case>_silu_threshold_generated_summary.csv
kv_dump_logs/<case>/<case>_swiglu_threshold_generated_summary.csv
```

最有用的列：

- `final_threshold`：最终写入 generated config 的 threshold。
- `prefill_final_estimated_added_zero_ratio`：prefill 上估计的 added-zero ratio。
- `decode_final_estimated_added_zero_ratio`：decode 上估计的 added-zero ratio。

如果估计 sparsity 太低或太高，调整 `SWIGLU_TARGET_SCALE` 后重新跑第一条命令。

满意后再跑最终 perplexity apply：

```bash
CASE_FILTER=Qwen-3-8B \
MODEL=models/Qwen/Qwen3-8B-f16.gguf \
RUN_KIND=perplexity \
SIM_Q8Q8=1 \
SIM_Q8Q8_SRC0_BLOCK=32 \
SIM_Q8Q8_SRC1_BLOCK=32 \
SIM_Q4Q6=0 \
SIM_Q6Q6=0 \
SIM_Q6Q8=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=1 \
SWIGLU_THRESHOLD_ENABLE=1 \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
SWIGLU_THRESHOLD_PROFILE=generated \
SKIP_BUILD=0 \
bash run.sh
```

### 3.9 手动分步 threshold pipeline

当你要调试中间 artifacts，或者只想复现某个阶段时，用手动分步模式。

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

单步模式下，`SKIP_BUILD=0` 只构建当前 mode 所需的二进制。如果后续用同一个
build tree 加 `SKIP_BUILD=1` 跑另一个 step，需要确认对应二进制已经存在。

### 3.10 复用 generated artifacts

当 collect/generate 输出已经存在，只想补跑或重跑 apply steps 时，使用 artifact reuse。

```bash
CASE_FILTER=Qwen-3-1.7B \
RUN_KIND=activation-flow \
SWIGLU_THRESHOLD_KIND=swiglu+silu \
FLOW_REUSE_ARTIFACTS=1 \
SKIP_BUILD=1 \
BUILD_DIR=build \
bash run.sh
```

`FLOW_REUSE_ARTIFACTS=1` 会跳过已经完成的 artifact-producing steps，但不会放宽
剩余 steps 对二进制的检查。

### 3.11 只跑 decode stats

```bash
CASE_FILTER=Llama-3.2-1B \
RUN_KIND=decode-stats \
SKIP_BUILD=1 \
N_PREDICT=128 \
bash run.sh
```

## 4. Run modes

主要 `RUN_KIND`：

| Mode | 运行内容 | Build target |
| --- | --- | --- |
| `cli` | `llama-cli` | `llama-cli` |
| `perplexity` | `llama-perplexity` | `llama-perplexity` |
| `decode-stats` | `llama-decode-stats` | `llama-decode-stats` |
| `swiglu-collect` | 带 collection 参数的 `llama-cli` | `llama-cli` |
| `swiglu-generate` | `tools/swiglu-threshold-configs/generate.py` | 无 |
| `activation-flow` | 多步骤 pipeline | 所有 step targets 的并集 |

可用 aliases：

| Alias | Normalized mode |
| --- | --- |
| `collect`, `activation-collect` | `swiglu-collect` |
| `generate`, `activation-generate` | `swiglu-generate` |
| `flow` | `activation-flow` |

脚本保留了历史上的 `swiglu-*` 名字，以兼容已有本地习惯。实际 pipeline 现在也支持
`silu`、`silu_input` 和 `swiglu+silu`。

## 5. Case selection

Cases 直接定义在 `run.sh` 中。大部分 case 通过 `make_standard_case_spec` 创建，
然后放进 `RUN_CASES`。

当前标准 cases：

| Case filter fragment | 默认 model |
| --- | --- |
| `Llama-3.2-1B` | `models/hf/Llama-3.2-1B-Instruct-f16.gguf` |
| `Qwen-3-1.7B` | `models/Qwen/Qwen3-1.7B-Base-f16.gguf` |
| `Llama-3.2-3B` | `models/hf/Llama-3___2-3B-Instruct-f16.gguf` |
| `Llama-2-7B` | `models/hf/llama-2-7B-F16.gguf` |
| `Qwen-3-8B` | `models/Qwen/Qwen3-8B-f16.gguf` |

`CASE_FILTER` 会同时对子 case name 和 sanitized case slug 做 substring matching。

```bash
CASE_FILTER=Llama-3.2-1B bash run.sh
CASE_FILTER=Qwen-3-1.7B bash run.sh
CASE_FILTER=f8e3m4-normal bash run.sh
```

如果某个变量同时在 case 和命令行中设置，命令行优先。例如：
`N_PREDICT=128 bash run.sh` 会覆盖 case 内的 `N_PREDICT`。

添加一个新的标准 case：

```bash
CASE_MY_MODEL="$(make_standard_case_spec "My-Case-Name" "models/path/to/model.gguf")"
```

然后把它加入 `RUN_CASES`。

## 6. Build behavior

`BUILD_DIR` 默认是 `build`。

单步 mode 在 `SKIP_BUILD=0` 时只构建自己需要的 target。`activation-flow`
会一次性构建全部 steps 所需 targets 的并集。默认 flow 下，这通常意味着同时需要
`llama-cli` 和 `llama-perplexity`。

示例：

```bash
SKIP_BUILD=0 BUILD_DIR=build bash run.sh
SKIP_BUILD=1 BUILD_DIR=build-baseline bash run.sh
```

低精度 simulation knobs 是 compile-time flags。如果修改了这些 knobs，请使用
`SKIP_BUILD=0`，或者指向一个使用相同 flags 构建出来的 `BUILD_DIR`。

## 7. Activation threshold 参考

主要 knobs：

| Variable | 含义 |
| --- | --- |
| `SWIGLU_THRESHOLD_KIND` | `swiglu`、`silu`、`silu_input` 或 `swiglu+silu` |
| `SWIGLU_THRESHOLD_ENABLE` | 在 `cli` 和 `perplexity` 中启用 threshold runtime |
| `SWIGLU_THRESHOLD_PROFILE` | profile 名字，或 `generated` |
| `SWIGLU_THRESHOLD_CONFIG` | 显式 threshold CSV 路径，或 `auto` |
| `SWIGLU_GENERATE_MODE` | `target-profile` 或 `channel-max` |
| `SWIGLU_CHANNEL_THRESHOLD_RATIO` | `channel-max` 使用的 per-channel ratio |

Activation kind 语义：

| Kind | Runtime 行为 |
| --- | --- |
| `swiglu` | 截断 post-gating SwiGLU output |
| `silu` | 在最终 multiply 之前截断 SiLU branch output |
| `silu_input` | 在 SiLU 之前对 raw gate input 做 one-sided negative-tail truncation |
| `swiglu+silu` | 先 apply SiLU-output truncation，再 apply SwiGLU-output truncation |

Dual mode 细节：

- `SWIGLU_THRESHOLD_KIND=swiglu+silu` 会自动解析两套 threshold family。
- `SWIGLU_THRESHOLD_CONFIG`、`SWIGLU_COLLECT_PREFIX`、`SWIGLU_TARGET_PROFILE`、
  `SWIGLU_GENERATED_CONFIG`、`SWIGLU_GENERATED_REPORT` 必须保持 `auto`。
- primary runtime channel 是 `silu`，secondary runtime channel 是 `swiglu`。
- dual mode 不会求一个共享 threshold，而是分别校准、分别 apply 两套 family。
- runtime sparsity accounting 是有顺序的：`silu` report 看到的是后续 `swiglu`
  truncation 之前的 tensor；`swiglu` report 看到的是已经被 SiLU 置零过的 tensor。

Auto-resolution 规则：

| Artifact | 默认路径 |
| --- | --- |
| collection prefix | `kv_dump_logs/<case>/<case>_<kind>_collect` |
| target profile | `tools/<kind>-threshold-targets/<profile-kind>/<case>.csv` |
| generated config | `kv_dump_logs/<case>/<case>_<kind>_threshold_generated.csv` |
| generated report | `kv_dump_logs/<case>/<case>_<kind>_threshold_generated_summary.csv` |
| `SWIGLU_THRESHOLD_PROFILE=generated` 时的 apply config | generated config path |
| named profile 的 apply config | `tools/<kind>-threshold-configs/<profile>/<case>.csv` |

命名说明：环境变量继续使用历史上的 `SWIGLU_*` 前缀，但实际 activation family
由 `SWIGLU_THRESHOLD_KIND` 决定。

## 8. Collection 和 generation 参考

Collection knobs：

| Variable | 含义 |
| --- | --- |
| `SWIGLU_COLLECT_PREFIX` | artifact prefix，或 `auto` |
| `SWIGLU_COLLECT_BINS` | histogram bin count |
| `SWIGLU_COLLECT_LOG10_MIN` | histogram 下界 |
| `SWIGLU_COLLECT_LOG10_MAX` | histogram 上界 |

Generation knobs：

| Variable | 含义 |
| --- | --- |
| `SWIGLU_TARGET_PROFILE_KIND` | target profile 子目录，例如 `minimal` |
| `SWIGLU_TARGET_PROFILE` | 显式 target-profile CSV，或 `auto` |
| `SWIGLU_TARGET_SCALE` | 对 target added-zero ratios 施加的 multiplier |
| `SWIGLU_GENERATED_CONFIG` | generated threshold 输出路径，或 `auto` |
| `SWIGLU_GENERATED_REPORT` | generated summary 输出路径，或 `auto` |

Collection 语义：

| Kind | 收集内容 |
| --- | --- |
| `swiglu` | SwiGLU output 的 absolute-value histogram |
| `silu` | SiLU output branch 的 absolute-value histogram |
| `silu_input` | raw gate input 在 SiLU 之前的 negative-tail magnitudes |
| `silu` plus `channel-max` | histogram，以及 per-stage per-layer per-channel absolute maxima |
| `swiglu+silu` | 同时写出 `silu` 和 `swiglu` 两套 artifact family |

Generation 语义：

| Mode | 行为 |
| --- | --- |
| `swiglu` target-profile | 使用 prefill layer 1 作为 reference slice，选择一个 global threshold，并写到所有 layer |
| `silu` target-profile | 和 `swiglu` 相同的 global-threshold 策略 |
| `silu_input` target-profile | 从 one-sided negative-tail histograms 中为每层生成 threshold |
| `silu` channel-max | 输出 `layer,channel,threshold`，其中 `threshold = abs_channel_max * ratio` |
| `swiglu+silu` | 对 `silu` 和 `swiglu` 各运行一次 generation |

在 target-profile mode 中，non-reference layers 和 decode targets 仍会写入
generated summary report，用于估计和对比。它们不会改变 `swiglu` 或 `silu`
最终选出的 threshold。

## 9. Low-precision 和 runtime knobs

通用 runtime 输入：

```text
MODEL DATA PROMPT OUT_DIR
CTX THREADS N_PREDICT SEQ_ID BATCH UBATCH STRIDE
```

Low-precision simulation controls：

| Variable group | 含义 |
| --- | --- |
| `SIM_FP8`, `SIM_FP_FORMAT`, `SIM_FP8_LAYOUT`, `SIM_FP8_*` | fp8-sim controls |
| `SIM_Q4Q6`, `SIM_Q4Q6_APPLY_SRC0`, `SIM_Q4Q6_APPLY_SRC1` | 启用并选择 Q4/Q6 replay inputs |
| `SIM_Q4Q6_SRC0_BLOCK`, `SIM_Q4Q6_SRC1_BLOCK` | Q4/Q6 replay block sizes |
| `SIM_Q4Q6_SRC1_QMODE` | `0=symmetric Q6`，`1=asymmetric Q6+zp`，`2=logarithmic Q6-exp + BF16 scale` |
| `SIM_Q4Q6_SRC1_LOG_STEP` | logarithmic Q6-exp replay 的 exponent divisor |
| `SIM_Q6Q6`, `SIM_Q6Q6_APPLY_SRC0`, `SIM_Q6Q6_APPLY_SRC1` | 启用并选择 Q6/Q6 power-of-two replay inputs |
| `SIM_Q6Q6_SRC0_BLOCK`, `SIM_Q6Q6_SRC1_BLOCK` | Q6/Q6 replay block sizes |
| `SIM_Q6Q8`, `SIM_Q6Q8_APPLY_SRC0`, `SIM_Q6Q8_APPLY_SRC1` | 启用并选择 Q6/Q8 power-of-two replay inputs；src0=权重 Q6，src1=激活 Q8 |
| `SIM_Q6Q8_SRC0_BLOCK`, `SIM_Q6Q8_SRC1_BLOCK` | Q6/Q8 replay block sizes |
| `SIM_Q8Q8`, `SIM_Q8Q8_APPLY_SRC0`, `SIM_Q8Q8_APPLY_SRC1` | 启用并选择 Q8/Q8 replay inputs |
| `SIM_Q8Q8_SRC0_BLOCK`, `SIM_Q8Q8_SRC1_BLOCK` | Q8/Q8 replay block sizes |
| `SIM_MATMUL_OUT_MODE` | `0=fp8-sim 开启时执行 FP8 output QDQ`，`1=BF16 output round-trip` |

重要交互：

- `SIM_Q4Q6`、`SIM_Q6Q6`、`SIM_Q6Q8`、`SIM_Q8Q8` 和 `SIM_FP8` 互斥。
- `SIM_Q4Q6=1` 要求 `SIM_MATMUL_OUT_MODE=1`。
- `SIM_Q6Q6=1` 要求 `SIM_MATMUL_OUT_MODE=1`。
- `SIM_Q6Q8=1` 要求 `SIM_MATMUL_OUT_MODE=1`。
- `SIM_Q8Q8=1` 要求 `SIM_MATMUL_OUT_MODE=1`。
- `SWIGLU_THRESHOLD_ENABLE=1` 会独立影响 `perplexity` 和 `cli`，不依赖低精度 simulation flags。
- 做纯低精度 replay 测量时，显式设置 `SWIGLU_THRESHOLD_ENABLE=0`。

Profiler 和 graph extras：

```text
DUMP_DOT
REDUCTION_PROD_PROFILE
REDUCTION_PROD_PROFILE_BINS
REDUCTION_PROD_PROFILE_HIST_MIN_LOG2
REDUCTION_PROD_PROFILE_HIST_MAX_LOG2
REDUCTION_PROD_PROFILE_SAMPLE_RATE
REDUCTION_PROD_BLOCK_DROP_LOG2_N
REDUCTION_PROD_PROFILE_MAX_SAMPLES
```

## 10. Output layout

Case 输出会写到：

```text
<OUT_DIR>/<case-slug>/
```

常见日志：

```text
<case>_cli.log
<case>_perplexity.log
<case>_decode-stats.log
<case>_swiglu-collect.log
<case>_swiglu-generate.log
```

Activation artifacts：

```text
<case>_<kind>_collect_summary.csv
<case>_<kind>_collect_hist.csv
<case>_silu_collect_channel_max.csv
<case>_<kind>_threshold_generated.csv
<case>_<kind>_threshold_generated_summary.csv
<case>_<kind>_threshold_cli.csv
<case>_<kind>_threshold_perplexity.csv
```

在 `SWIGLU_THRESHOLD_KIND=swiglu+silu` mode 中，同一次运行会同时写出
`<kind>=silu` 和 `<kind>=swiglu` 两套 variants。

在 `SWIGLU_GENERATE_MODE=channel-max` 中，`<case>_silu_threshold_generated.csv`
存储的是 `layer,channel,threshold`，不是 `layer,threshold`。

Profiler 和 postprocessing artifacts：

```text
<case>_fp8_sim_analysis.log
<case>_fp8_interval_hist.csv
<case>_fp8_interval_hist.png
<case>_block_psum_relation.csv
<case>_block_psum_relation.md
block_over_psum_hist.png
block_over_psum_percent.png
block_over_psum_cdf.png
```

在 output root 层级，脚本还可能写出：

```text
reduction_block_drop_compare.csv
reduction_block_drop_compare.md
```

## 11. Model preparation

本节是可选内容。只有当本地 `models/` tree 里还没有 workflow 需要的模型文件时，
才需要参考这里。

### 11.1 构建 quantizer

```bash
cmake --build build --target llama-quantize -j$(nproc)
```

### 11.2 把本地 Hugging Face 目录转换成 GGUF

如果模型仍然是包含 `safetensors` weights 的 Hugging Face 目录，需要先转换成 GGUF：

```bash
python3 -m pip install -r requirements.txt

python3 convert_hf_to_gguf.py \
  ./models/Qwen/Qwen3-1___7B \
  --outfile ./models/Qwen/Qwen3-1___7B-f16.gguf \
  --outtype f16
```

`convert_hf_to_gguf.py` 的输入应该是解包后的目录，不是 archive 文件。

### 11.3 直接本地 quantization 示例

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
  ./models/hf/llama-2-7B-F16.gguf \
  ./models/hf/llama-2-7B-Q5_K_M.gguf \
  Q5_K_M \
  $(nproc)
```

通过覆盖 `MODEL`，让实验使用新生成的文件：

```bash
CASE_FILTER=Llama-3.2-1B \
MODEL=models/hf/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
RUN_KIND=perplexity \
SIM_Q4Q6=0 \
SIM_Q6Q6=0 \
SIM_Q6Q8=0 \
SIM_Q8Q8=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=0 \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
bash run.sh
```

### 11.4 选择性把 FFN gate/up 量化成 `IQ2_S`

当你只希望部分 FFN gate/up matrices 使用 `IQ2_S`，而其他 tensor 继续使用
base quantization type 的 native assignment 时，使用
`llama-quantize --tensor-type`。

这个 workflow 不要使用 `--pure`。例如 base type 是 `Q4_K_M` 时，未被
`--tensor-type` 匹配到的 tensor 仍然使用 `Q4_K_M` 的 native mixed-quantization
策略。

`IQ2_S` 需要 importance matrix。请从原始 f16/bf16 GGUF 模型和有代表性的
calibration text 生成：

```bash
cmake --build build --target llama-imatrix llama-quantize -j$(nproc)

./build/bin/llama-imatrix \
  --model ./models/Qwen/Qwen3-8B-f16.gguf \
  -f ./models/hf/wiki.test.raw \
  -o ./models/Qwen/Qwen3-8B-imatrix.gguf \
  -c 1024 \
  -b 512 \
  --chunks 16
```

正式 quantization 运行时，建议使用更多 calibration chunks，例如 `64`、`128`，
或者省略 `--chunks` 来使用整个 calibration 文件。如果有可用 GPU backend，可以
添加常规 offload 选项，例如 `-ngl 99`。

Dense Transformer FFN override：

```bash
./build/bin/llama-quantize \
  --imatrix ./models/Qwen/Qwen3-8B-imatrix.gguf \
  --tensor-type '^blk\.[0-9]+\.ffn_(gate|up)\.weight$=iq2_s' \
  ./models/Qwen/Qwen3-8B-f16.gguf \
  ./models/Qwen/Qwen3-8B-Q4_K_M-ffn-gate-up-IQ2_S.gguf \
  Q4_K_M \
  $(nproc)
```

MoE FFN override，覆盖 expert、shared-expert 和 chunk-expert FFN gate/up
matrices，同时让 `ffn_down*` 保持 native：

```bash
./build/bin/llama-quantize \
  --imatrix ./models/Qwen/Qwen3-MoE-imatrix.gguf \
  --tensor-type '^blk\.[0-9]+\.ffn_(gate|up)(_exps|_shexp|_chexps)?\.weight$=iq2_s' \
  ./models/Qwen/Qwen3-MoE-f16.gguf \
  ./models/Qwen/Qwen3-MoE-Q4_K_M-ffn-gate-up-IQ2_S.gguf \
  Q4_K_M \
  $(nproc)
```

不要把 `ffn_gate_inp` 放进 regex。它是 MoE router/gate-input tensor，不是普通
FFN gate projection。

通过覆盖 `MODEL`，在 `run.sh` 实验中使用新模型：

```bash
CASE_FILTER=Qwen-3-8B \
MODEL=models/Qwen/Qwen3-8B-Q4_K_M-ffn-gate-up-IQ2_S.gguf \
RUN_KIND=perplexity \
SIM_Q4Q6=0 \
SIM_Q6Q6=0 \
SIM_Q6Q8=0 \
SIM_Q8Q8=0 \
SIM_FP8=0 \
SIM_MATMUL_OUT_MODE=0 \
SWIGLU_THRESHOLD_ENABLE=0 \
SKIP_BUILD=0 \
bash run.sh
```

## 12. Compatibility notes

脚本有意保留了这些兼容性选择：

- Threshold 环境变量仍然使用 `SWIGLU_*` 前缀。
- `swiglu-collect` 和 `swiglu-generate` 仍然是 canonical artifact names。
- `collect`、`generate`、`flow` 这类 aliases 仍然可用。

这些选择能保留已有本地命令习惯，同时通过 `SWIGLU_THRESHOLD_KIND` 暴露新的
SiLU、SiLU-input 和 dual-output 路径。

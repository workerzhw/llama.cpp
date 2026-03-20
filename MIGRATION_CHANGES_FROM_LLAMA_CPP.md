# llama.cpp 修改迁移总记录（相对原版）

本文档记录当前工作区分支相对原版 llama.cpp 的全部改动点，目标是支持后续在其他模型/分支上做等效迁移。

## 0. 基线与范围

- 对比基线（merge-base）：`8c0d6bb4559e15f1236035a0465781c1345324e5`
- 对比对象：当前工作树（包含已提交改动 + 本地未提交改动）
- 本次检测到的改动文件总数：17

> 说明：`run.sh` 当前有本地未提交改动（默认参数调整），已纳入本记录，避免遗漏。

---

## 1. 变更文件总清单（不遗漏）

### 1.1 新增文件（6）

1. `PPL统计.png`
2. `kv_dump_logs/kv_seq_0.txt`
3. `readmd-ppl.md`
4. `run.sh`
5. `scripts/kv_cache_dump.py`
6. `summary.sh`

### 1.2 修改文件（11）

1. `common/arg.cpp`
2. `common/common.h`
3. `ggml/src/ggml-cpu/ggml-cpu-impl.h`
4. `ggml/src/ggml-cpu/ggml-cpu.c`
5. `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`
6. `ggml/src/ggml-cpu/repack.cpp`
7. `ggml/src/ggml-cpu/vec.cpp`
8. `ggml/src/ggml-cpu/vec.h`
9. `src/llama-context.cpp`
10. `tools/main/main.cpp`
11. `tools/perplexity/perplexity.cpp`

---

## 2. 修改主题总览

本分支的改动可分为 5 大主题：

1. CPU matmul 路径数值模拟框架扩展（FP8/FP9、多布局、输出模式）
2. FP8 仿真统计与误差分析报告（含层级统计、PPL 闭环估算）
3. matmul 输出分布采样器（单独日志体系）
4. KV 单序列状态导入导出链路（main/perplexity + CLI 参数）
5. 实验自动化与离线解析脚本（run、summary、kv_dump）

---

## 3. 逐文件详细迁移说明

## 3.1 `common/common.h`

### 新增字段（`common_params`）

- `path_seq_state_in`：加载单序列状态文件路径
- `path_seq_state_out`：保存单序列状态文件路径
- `seq_state_in_id`：导入目标 seq id
- `seq_state_out_id`：导出源 seq id

### 迁移意义

- 为 CLI 和工具侧（main/perplexity）提供单序列 KV 状态 I/O 参数承载。

---

## 3.2 `common/arg.cpp`

### 新增参数

- `--seq-state-in FNAME`
- `--seq-state-in-id N`
- `--seq-state-out FNAME`
- `--seq-state-out-id N`

### 新增校验

- `--seq-state-in-id >= 0`
- `--seq-state-out-id >= 0`
- `--prompt-cache` 与 `--seq-state-in/--seq-state-out` 互斥

### 迁移意义

- CLI 层完整打通单序列状态导入/导出控制。

---

## 3.3 `tools/main/main.cpp`

### 主要新增行为

1. 程序启动阶段：
   - 支持从 `--seq-state-in` 文件加载单序列状态到 `seq_state_in_id`
   - 增加文件存在性与空文件检查
2. 推理过程中：
   - 增加 `track_seq_state_tokens`，在启用 seq-state-out 时也持续累计 token
3. 程序退出阶段：
   - 支持保存单序列状态到 `--seq-state-out`

### 迁移意义

- 可做“中间状态落盘/恢复”，便于 KV 调试、复现实验和跨进程衔接。

---

## 3.4 `tools/perplexity/perplexity.cpp`

### 主要新增行为

- 在 perplexity 运行中，如设置 `--seq-state-out`，会在最后一个 chunk 保存 KV seq-state。
- 保存调用：`llama_state_seq_save_file(...)`

### 迁移意义

- 将 perplexity 路径也接入 seq-state 导出能力，与 main 保持一致的调试手段。

---

## 3.5 `src/llama-context.cpp`

### 新增调试能力

- 读取环境变量 `LLAMA_DUMP_DOT`
- 在 decode 流程中一次性导出计算图 DOT 文件（只导出一次，避免重复写大文件）

### 迁移意义

- 支持图级调试（结构审计、路径确认）。

---

## 3.6 `ggml/src/ggml-cpu/ggml-cpu-impl.h`

### 新增声明

- `ggml_mm_dist_record_chunk_values(...)`
- `ggml_mm_dist_record_chunk_values_pair(...)`

### 迁移意义

- 为 CPU 内核/后处理提供统一的 matmul 分布统计采样 API 入口。

---

## 3.7 `ggml/src/ggml-cpu/vec.h`

该文件引入了 FP 仿真框架的核心编译期开关与 API 声明。

### A) 新增 trunc4 相关宏与 dot 接口

- `GGML_BF16_TRUNC4_MASK`
- `GGML_BF16_RNA_BIAS`
- `GGML_MULMAT_TRUNC4_SRC1`（默认 0）
- 新增函数声明：`ggml_vec_dot_bf16_trunc4(...)`

### B) 新增 FP 仿真总开关与格式开关

- `GGML_SIM_FP8E4M3`（0/1）
- `GGML_SIM_FP_FORMAT`
  - `8` = F8
  - `9` = F9(E4M4-like)
- `GGML_SIM_FP8_LAYOUT`（仅 F8 有效）
  - `0` = E4M3
  - `1` = E3M4
  - `2` = E3M4_NO_SUBNORM（纯规格化模式）

### C) 新增 scale 类型开关（输入/输出可分离）

- 旧兼容宏：`GGML_SIM_FP8E4M3_SCALE_TYPE`
- 新宏：
  - `GGML_SIM_FP8E4M3_SCALE_TYPE_IN`
  - `GGML_SIM_FP8E4M3_SCALE_TYPE_OUT`
- 取值：
  - `0` = int8 pow2
  - `1` = bf16 exact

### D) 新增作用域开关

- `GGML_SIM_FP8E4M3_APPLY_SRC0`
- `GGML_SIM_FP8E4M3_APPLY_SRC1`

### E) 新增输出仿真模式开关

- `GGML_SIM_MATMUL_OUT_MODE`
  - `0` = 输出走 FP8 QDQ
  - `1` = 输出走 BF16 round-trip

### F) 新增 API 声明

- `ggml_sim_fp8e4m3_block_quant_dequant_f32(...)`
- `ggml_sim_fp8e4m3_block_quant_dequant_f32_to_bf16(...)`
- `ggml_fp8_sim_stats_reset()`
- `ggml_fp8_sim_stats_report(...)`
- `ggml_sim_bf16_roundtrip_f32(...)`
- `ggml_sim_bf16_roundtrip_f32_array(...)`

### G) 其他新增

- `ggml_dequantize_row_to_f32(...)` 多量化类型统一解码工具（供 cast-to-bf16 路径使用）
- `GGML_MUL_MAT_LOG` / `GGML_MMLOG_LEVEL` / `GGML_MMLOG_CHUNK_BUDGET` 调试宏

---

## 3.8 `ggml/src/ggml-cpu/vec.cpp`

该文件是 FP8/FP9 模拟、统计与 BF16 trunc4 内核的核心实现。

### A) FP 仿真通用参数化

- 通过 `GGML_SIM_FP_FORMAT` + `GGML_SIM_FP8_LAYOUT` 运行时逻辑分支实现：
  - F8(E4M3)
  - F8(E3M4)
  - F8(E3M4_NO_SUBNORM)
  - F9(E4M4-like)
- 关键函数：
  - `ggml_sim_mantissa_bits()`
  - `ggml_sim_exponent_bits()`
  - `ggml_sim_support_subnormals()`
  - `ggml_sim_exponent_bias()`
  - `ggml_sim_min_norm_exp_unbiased()`
  - `ggml_sim_max_norm_exp_unbiased()`

### B) 单值量化回放逻辑

- `ggml_fp8e4m3_quant_dequant_one(float x)`
- 特性：
  - RNE 舍入
  - 溢出饱和到 max finite
  - 下溢阈值判定
  - 带/不带 subnormal 的分支
  - 纯规格化模式下对 min_norm 以下值的处理

### C) block scale 策略

- int8 pow2：`ggml_choose_k_for_block(...)`
- bf16 exact：`ggml_choose_scale_for_block_bf16(...)`
- 按 `src_id` 区分 in/out scale type：`ggml_fp8_scale_type_for_src(...)`

### D) 两个主 QDQ 接口实现

- `ggml_sim_fp8e4m3_block_quant_dequant_f32(...)`
- `ggml_sim_fp8e4m3_block_quant_dequant_f32_to_bf16(...)`

### E) 统计体系（FP8_SIM）

- 全局统计结构：`FP8SimSrcStats`
- 全局容器：
  - `g_fp8_stats[3]`（src0/src1/src2）
  - `g_fp8_layer_stats[3]`（按层/张量名）
- 采样：`FP8_SIM_STATS_SAMPLE`（默认 100）
- 退出自动报告：`atexit(fp8_stats_atexit_handler)`
- 报告函数：`ggml_fp8_sim_stats_report("fp8_sim_analysis.log")`

### F) 报告内容增强

- 配置与格式信息
- src0/src1/src2 分项 SQNR/MAE/RMSE/overflow/underflow/subnormal
- scale 分布直方图
- Per-layer 最差层排序
- PPL 闭环估算（含误差预算与启发式推导）

### G) 新增 BF16 trunc4 dot 内核

- `ggml_vec_dot_bf16_trunc4(...)`
- 包含 AVX512BF16 / AVX512F / AVX2 / 标量路径
- 支持仅截断 src0 或同时截断 src1（由 `GGML_MULMAT_TRUNC4_SRC1` 控制）

---

## 3.9 `ggml/src/ggml-cpu/ggml-cpu.c`

该文件是 CPU matmul 执行主路径改造核心。

### A) 新增 matmul 输出分布采样器（独立于 FP8_SIM 报告）

新增大块基础设施（全局统计 + atexit 报告）：

- 运行时开关：
  - `GGML_MATMUL_DIST=1`
  - `GGML_MATMUL_DIST_SAMPLE=N`
  - `GGML_MATMUL_DIST_FILE=...`
- 记录接口实现：
  - `ggml_mm_dist_record_chunk_values(...)`
  - `ggml_mm_dist_record_chunk_values_pair(...)`
- 报告包含：
  - summary 表
  - q/k/v focus
  - layer aggregate
  - op aggregate
  - matmul kind aggregate

### B) `ggml_compute_forward_mul_mat_one_chunk(...)` 改造

1. dot kernel 选择逻辑
   - 在 BF16 dot 时可切换 trunc4 dot（FP8_SIM 开时避免冲突）
2. src0/src1 wdata 布局认知增强
   - 支持 src1 与 src0 分段放置
3. 输出后处理
   - 支持输出 FP8 QDQ 或 BF16 round-trip
4. 采样接入
   - pre/post 值都记录到 matmul dist
5. 详细日志宏（`GGML_MUL_MAT_LOG`）

### C) `ggml_compute_forward_mul_mat(...)` 改造

1. 强制 `dot_type = GGML_TYPE_BF16`
2. Stage A：src1 cast/重排到 BF16 contiguous
   - 条件包括：类型不匹配、非 contiguous、要求 src1 FP8 模拟
3. Stage B：src0 cast 到 BF16 contiguous
   - 支持 src0 类型：F32/BF16/F16/quantized
   - quantized 先解码到 f32 再转 bf16
   - 可插入 src0 FP8 模拟
4. 注释掉 llamafile 快路径段（避免语义偏差）
5. one_chunk 调用改为传 `dot_type`

### D) `ggml_compute_forward_mul_mat_id_*` 改造

- 同步 mul_mat 的改造策略：
  - dot type 固定 BF16
  - src1/src0 wdata 准备
  - 输出 FP8/BF16 模拟分支
  - dist pre/post 采样
- 函数签名新增参数：`src0_nb01`、`src1_wdata_contig`

### E) `ggml_graph_plan(...)` workspace 估算改造

- 对 `MUL_MAT` 和 `MUL_MAT_ID` 节点按新布局重新预估：
  - src1 临时区（可选）
  - src0 临时区（可选）
- 条件与执行路径保持一致，避免 workspace 低估。

---

## 3.10 `ggml/src/ggml-cpu/repack.cpp`

### A) 输出模拟接入（repack GEMM/GEMV）

- 新增 `use_fp8sim_out` / `use_bf16sim_out` / `use_out_sim`
- 对输出 row 执行 QDQ 或 BF16 round-trip
- 记录 pre/post 分布：
  - `gemm_repack_pre/post`
  - `gemv_repack_pre/post`
  - `gemv_id_repack_pre/post`

### B) src1 模拟接入

- 在 src1 packing 前可先进行 FP8 block QDQ（`APPLY_SRC1`）

### C) fast path 语义保护

- `extra_buffer_type::supports_op(...)` 增加条件：
  - 若启用 `GGML_SIM_FP8E4M3 && GGML_SIM_FP8E4M3_APPLY_SRC0`，repack 内核返回不支持，回退通用路径，避免 src0 语义不一致。

---

## 3.11 `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`

### A) 头文件与依赖

- 新增 `<vector>`
- 引入 `vec.h`（调用仿真函数）

### B) f16/q4_0 kleidiai 路径接入模拟

1. src1 打包前支持 FP8 QDQ
2. src0（RHS 组织后）支持列级 QDQ
3. 输出支持 FP8 QDQ
4. 接入 pre/post 分布统计（gemv/gemm，f16/q40 分别命名）

### C) fast path 语义保护

- 当 `APPLY_SRC0` 打开时，禁用某些 kleidiai 快路径以避免语义不一致。

---

## 3.12 `run.sh`（新增，且当前有未提交本地改动）

### 脚本能力

- 一键 cmake 配置 + 构建 + 运行 `llama-perplexity`
- 通过环境变量注入全部仿真宏
- 自动设置：
  - `FP8_SIM_STATS_SAMPLE`
  - `GGML_MATMUL_DIST`
  - `GGML_MATMUL_DIST_SAMPLE`
  - `GGML_MATMUL_DIST_FILE`
- 支持可选 DOT 导出（`LLAMA_DUMP_DOT`）
- 归档日志文件到 `OUT_DIR`

### 参数校验

- 检查 `SIM_MATMUL_OUT_MODE` 取值
- 检查 `SIM_FP_FORMAT` 取值
- 检查 `SIM_FP8_LAYOUT` 取值
- 检查 `SIM_FP8_SCALE_TYPE_IN/OUT` 取值

### 当前本地未提交差异（相对 HEAD）

- `SIM_FP8_LAYOUT` 默认值：`1 -> 2`
- `SIM_MATMUL_OUT_MODE` 默认值：`0 -> 1`

即默认实验改为：

- F8 使用 `E3M4_NO_SUBNORM`
- 输出使用 BF16 round-trip

---

## 3.13 `scripts/kv_cache_dump.py`（新增）

### 功能

- 解析 `llama_state_seq_*` 单序列状态二进制格式
- 输出人类可读文本报告
- 支持：
  - 只看统计（默认）
  - 按 layer/cell 筛选
  - 只导 K 或 V
  - 自动处理 V 转置存储

### 关键点

- 支持多 ggml 类型映射（f32/f16/bf16 及多种 quant id）
- 对可解码类型给出 min/max/mean/std/absmax
- 输出 tokens、meta、K/V 明细与摘要

---

## 3.14 `summary.sh`（新增）

### 功能

- 从 `ppl_*.log` 提取最终 PPL 并写入 CSV
- 输出格式：`case,ppl`

---

## 3.15 `readmd-ppl.md`（新增）

### 内容

- 简要记录 wikitext-2 raw 数据获取和 `llama-perplexity` 调用示例。

---

## 3.16 `kv_dump_logs/kv_seq_0.txt`（新增）

### 性质

- 由 KV dump 工具输出的样例文本结果，属于实验产物。

---

## 3.17 `PPL统计.png`（新增）

### 性质

- PPL 分析图像产物，属于实验结果文件（非代码）。

---

## 4. 新增/扩展宏与环境变量总表

## 4.1 编译期宏（核心）

- `GGML_SIM_FP8E4M3`
- `GGML_SIM_FP_FORMAT`（8/9）
- `GGML_SIM_FP8_LAYOUT`（0/1/2）
- `GGML_SIM_FP8E4M3_APPLY_SRC0`
- `GGML_SIM_FP8E4M3_APPLY_SRC1`
- `GGML_SIM_FP8E4M3_SCALE_TYPE`
- `GGML_SIM_FP8E4M3_SCALE_TYPE_IN`
- `GGML_SIM_FP8E4M3_SCALE_TYPE_OUT`
- `GGML_SIM_FP8E4M3_BLOCK`
- `GGML_SIM_MATMUL_OUT_MODE`（0/1）
- `GGML_MULMAT_TRUNC4_SRC1`
- `GGML_MUL_MAT_LOG`
- `GGML_MMLOG_LEVEL`
- `GGML_MMLOG_CHUNK_BUDGET`

## 4.2 运行时环境变量（核心）

- `FP8_SIM_STATS_SAMPLE`
- `FP8_SIM_BASELINE_PPL`
- `GGML_MATMUL_DIST`
- `GGML_MATMUL_DIST_SAMPLE`
- `GGML_MATMUL_DIST_FILE`
- `LLAMA_DUMP_DOT`

---

## 5. 迁移时建议的落地顺序

1. 先迁移 `vec.h` + `vec.cpp`（格式、QDQ、统计 API）
2. 再迁移 `ggml-cpu.c`（mul_mat/mul_mat_id 主路径和 workspace）
3. 再迁移 `repack.cpp`、`kleidiai.cpp`（fast path 保持语义一致）
4. 再迁移 CLI 与工具层：`common/*` + `tools/main` + `tools/perplexity`
5. 最后迁移脚本与产物解析：`run.sh`、`summary.sh`、`scripts/kv_cache_dump.py`

---

## 6. 行为差异重点（迁移后需重点复核）

1. matmul dot 类型固定 BF16（原生路径行为变化）
2. src0/src1 可能被强制转为 BF16 contiguous 临时缓冲
3. 输出端新增 FP8/BF16 两种仿真后处理模式
4. fast path（repack/kleidiai）在某些宏组合下会主动回退
5. 工作区大小估算逻辑已改造，迁移时必须同步
6. 统计体系有两套：
   - FP8_SIM 报告（`fp8_sim_analysis.log`）
   - matmul 分布报告（`GGML_MATMUL_DIST_FILE`）

---

## 7. 非代码产物与可选迁移项

## 7.1 建议不直接迁移（可重生成）

- `PPL统计.png`
- `kv_dump_logs/kv_seq_0.txt`

## 7.2 建议迁移

- `run.sh`
- `summary.sh`
- `scripts/kv_cache_dump.py`
- `readmd-ppl.md`（可作为实验说明模板）

---

## 8. 一句话版本（总览）

本分支在 CPU matmul 体系中引入了可配置的 F8/F9 仿真与输出模式、完善的误差与分布统计、KV 单序列状态导入导出能力，以及配套实验/解析脚本；并对 fast path 与 workspace 规划做了语义一致性改造。以上 17 个文件即为与原版相比的完整改动集合。

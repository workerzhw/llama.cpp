# LLM in Flash: llama.cpp 实现计划 — Phase 1 基础设施

## Context

将论文 [arxiv:2312.11514](https://arxiv.org/pdf/2312.11514) 的 Flash 推理技术应用到 llama.cpp。
当前阶段：**仅实现 Phase 1 基础设施**，搭建 mmap 选择性读取 + DRAM 神经元缓存 + CLI 参数，验证 I/O 管道可行。

后续 Phase 2 的神经元选择策略已确定为 **Magnitude Top-K**。

---

## 实施步骤

### Step 1: CLI 参数与配置

添加 `--flash-inference` 和 `--flash-sparsity` 参数，贯穿配置层。

**修改文件:**
- [include/llama.h](include/llama.h) — 在 `llama_context_params` 中添加字段：
  ```cpp
  bool flash_inference;       // 启用 flash inference
  float flash_sparsity;       // 目标稀疏率 (0.1 = 保留 top 10% 神经元)
  size_t flash_cache_size;    // DRAM 缓存大小 (0 = 自动)
  ```
- [src/llama-cparams.h](src/llama-cparams.h) — 同步添加到 `llama_cparams`
- [common/common.h](common/common.h) — 添加到 `common_params`
- [common/arg.cpp](common/arg.cpp) — 添加 CLI 参数（参照 `--flash-attn` 在 line 985 的模式）

### Step 2: mmap 选择性读取

扩展现有 mmap 机制，添加按行级别的 `pread()` 读取能力。

**修改文件:**
- [src/llama-mmap.h](src/llama-mmap.h) — 添加方法：
  ```cpp
  // 从文件中读取指定范围的字节到目标缓冲区（使用 pread，线程安全）
  void read_range(size_t offset, size_t length, void * dst) const;
  // scatter-gather 版本，一次读取多个不连续区域
  void read_ranges(const std::vector<std::pair<size_t, size_t>> & ranges, void * dst) const;
  ```
- [src/llama-mmap.cpp](src/llama-mmap.cpp) — 用 `pread()`/`preadv()` 实现：
  - 保留 `file_id`（fd）引用用于 pread
  - 不修改现有 mmap 映射（仍用于 attention 等常驻权重）
  - pread 路径用于按需加载单个 FFN 神经元行

### Step 3: DRAM 神经元缓存

**新建文件:**
- `src/llama-flash-inference.h` — 缓存数据结构和 API
- `src/llama-flash-inference.cpp` — 缓存管理实现

核心结构：
```cpp
struct llama_flash_neuron_cache {
    // 配置
    int64_t n_neurons;          // FFN 神经元总数 (n_ff)
    int64_t n_embd;             // 嵌入维度
    int64_t n_cache_max;        // DRAM 最多缓存多少神经元
    size_t row_bytes_up;        // up 投影每行字节数（含量化）
    size_t row_bytes_gate;      // gate 投影每行字节数
    size_t row_bytes_down;      // down 投影每行字节数

    // DRAM 缓冲区（预分配）
    uint8_t * buf_up;           // [n_cache_max * row_bytes_up]
    uint8_t * buf_gate;         // [n_cache_max * row_bytes_gate]
    uint8_t * buf_down;         // [n_cache_max * row_bytes_down]

    // 索引映射
    std::vector<int64_t> slot_to_neuron;  // slot_idx → neuron_idx
    std::unordered_map<int64_t, int64_t> neuron_to_slot; // neuron_idx → slot_idx
    int64_t n_used;             // 当前缓存中的神经元数量

    // 方法
    void init(int64_t n_neurons, int64_t n_embd, int64_t n_cache,
              size_t rb_up, size_t rb_gate, size_t rb_down);
    bool has_neuron(int64_t neuron_idx) const;
    int64_t load_neuron(int64_t neuron_idx,
                        const uint8_t * src_up, const uint8_t * src_gate, const uint8_t * src_down);
    void evict_neuron(int64_t neuron_idx);
    void clear();
};
```

缓存管理策略：
- 预分配所有 DRAM，运行时无 malloc/realloc
- 驱逐时用末尾元素填充空洞，保持连续存储
- 新增直接追加到 `n_used` 末尾

### Step 4: 模型加载器行偏移 API

为 FFN 权重的每行计算精确的文件偏移量。

**修改文件:**
- [src/llama-model-loader.h](src/llama-model-loader.h) — 添加：
  ```cpp
  struct llama_tensor_row_info {
      size_t offset;      // 行在文件中的字节偏移
      size_t row_bytes;   // 行的字节大小
  };
  // 获取 tensor 每行的文件偏移信息
  std::vector<llama_tensor_row_info> get_tensor_row_offsets(const char * name) const;
  ```
- [src/llama-model-loader.cpp](src/llama-model-loader.cpp) — 实现：
  - 从 weights_map 中查找 tensor 的基础文件偏移
  - 根据 tensor 的 `nb[1]`（行步长）计算每行偏移

### Step 5: 模型加载集成

**修改文件:**
- [src/llama-model.h](src/llama-model.h) — 添加 flash inference 上下文：
  ```cpp
  struct llama_flash_inference_ctx {
      bool enabled = false;
      float sparsity = 0.1f;
      // 每层每行的文件偏移
      std::vector<std::vector<llama_tensor_row_info>> ffn_up_row_offsets;    // [layer][neuron]
      std::vector<std::vector<llama_tensor_row_info>> ffn_gate_row_offsets;  // [layer][neuron]
      std::vector<std::vector<llama_tensor_row_info>> ffn_down_row_offsets;  // [layer][neuron]
      // 每层的神经元缓存
      std::vector<llama_flash_neuron_cache> layer_caches;  // [layer]
  };
  ```
- [src/llama-model.cpp](src/llama-model.cpp) — 在 `load_tensors()` 中：
  - 当 flash inference 启用时，正常加载 attention/norm/embedding
  - 对 FFN up/gate/down 权重：不分配 DRAM buffer，记录行偏移
  - 预分配每层的神经元缓存

### Step 6: 构建系统集成

**修改文件:**
- [CMakeLists.txt](CMakeLists.txt) 或 [src/CMakeLists.txt](src/CMakeLists.txt) — 添加新源文件

---

## 关键设计决策

1. **pread vs mmap+madvise:** 选择 pread 因为：
   - 线程安全，无需 seek
   - 精确控制读取哪些字节
   - 不受内核页面大小限制
   - 现有 mmap 保留用于非 FFN 权重

2. **缓存预分配:** 所有 DRAM 在模型加载时一次性分配，推理期间零内存分配

3. **量化兼容:** `row_bytes` 直接取自 tensor 的 `nb[1]`，已包含量化后的行大小

---

## 验证方法

1. **编译验证:** `cmake --build build` 成功编译
2. **功能验证:** 编写简单测试：
   - `read_range()` 能正确读取指定范围的文件内容
   - 神经元缓存能 load/evict/查询神经元
   - `get_tensor_row_offsets()` 返回正确的行偏移
3. **集成验证:** 使用小模型（Qwen2.5-0.5B Q4_0）验证：
   - `--flash-inference` 参数被正确解析
   - 模型加载时 attention 权重常驻 DRAM，FFN 权重仅记录偏移
   - 神经元缓存被正确初始化
   - 标准 `./llama-cli` 能以 flash inference 模式启动（虽然 Phase 1 不修改推理路径，但可验证加载管道）

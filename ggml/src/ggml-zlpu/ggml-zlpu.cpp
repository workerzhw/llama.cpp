/*
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "ggml-zlpu.h"
#include "zlpu-dev.h"
#include "zlpu-ops.h"

#include <stdarg.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <queue>
#include <chrono>
#include <unordered_set>
#include <optional>

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-zlpu/common.h"
#include "ggml.h"

#define GGML_COMMON_DECL_C

#include "ggml-common.h"

#define GGML_ZLPU_NAME "ZLPU"

/**
 * @brief Handles ZLPU errors by printing an error message and aborting.
 *
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number where the error occurred.
 * @param msg The error message.
 */
[[noreturn]] void ggml_zlpu_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int32_t id = -1;

    GGML_LOG_ERROR("ZLPU error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("ZLPU error");
}

// Thread-local variable to record the current device of this thread.
thread_local int g_current_zlpu_device = -1;

/**
 * @brief Set the ZLPU device to be used.
 *
 * @param device The target device ID to set.
 */
void ggml_zlpu_set_device(const int32_t device) {

    if (device == g_current_zlpu_device) {
        return;
    }

    // Update the global device record.
    g_current_zlpu_device = device;
}

/**
 * @brief Retrieves the current device ID.
 *
 * @return The current device ID.
 */
// int32_t ggml_zlpu_get_device() {
// }

/**
 * @brief Get the value of the specified environment variable (name).
 *        if not empty, return a std::string object
 */
// std::optional<std::string> get_env(const std::string & name) {
// }

/**
 * @brief Verify whether the environment variable is a valid value.
 */

/**
 * @brief Parse a string as an integer, returning 0 if invalid.
 *
 * This function attempts to convert the input string `value` to an `int`.
 * If the string is not a valid integer or is out of the `int` range,
 * it returns 0.
 *
 * @param value The string to parse.
 * @return The parsed integer, or 0 if conversion fails.
 */
int parse_integer(const std::string & value) {
    try {
        return std::stoi(value);
    } catch (...) {
        return 0;
    }
}

/**
 * @brief Initialize the ZLPU device information.
 *
 * This function initializes the ZLPU device information by obtaining the
 * device count and setting the memory allocation granularity for each device.
 *
 * @return A structure containing the device information.
 */
static ggml_zlpu_device_info ggml_zlpu_init() {
    ggml_zlpu_device_info info = {};

    info.device_count = 1;

    GGML_ASSERT(info.device_count <= GGML_ZLPU_MAX_DEVICES);

    for (int id = 0; id < info.device_count; ++id) {
        ;
    }

    // TODO: add more device info later.
    return info;
}

/**
 * @brief Retrieve the ZLPU device information.
 *
 * This function returns a reference to a structure containing the ZLPU device
 * information. The device information is initialized once and reused on
 * subsequent calls.
 *
 * @return A reference to the structure containing the device information.
 */
const ggml_zlpu_device_info & ggml_zlpu_info() {
    static ggml_zlpu_device_info info = ggml_zlpu_init();
    return info;
}


// zlpu buffer
struct ggml_backend_zlpu_buffer_context {
    int32_t device;             ///< The device ID associated with this buffer context.
    void *  dev_ptr = nullptr;  ///< Pointer to the device memory allocated for the buffer.
    ggml_backend_zlpu_buffer_context(int32_t device, void * dev_ptr) : device(device), dev_ptr(dev_ptr) {}
    ~ggml_backend_zlpu_buffer_context() { zakccFree(dev_ptr); }
};

static void ggml_backend_zlpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_zlpu_buffer_context * ctx = (ggml_backend_zlpu_buffer_context *) buffer->context;
    delete ctx;
}

static void * ggml_backend_zlpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_zlpu_buffer_context * ctx = (ggml_backend_zlpu_buffer_context *) buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_zlpu_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (tensor->view_src != NULL && tensor->view_offs == 0) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    // TODO: zlpu backend doesn't support quantized yet. Just leave the code
    // here.
    if (ggml_is_quantized(tensor->type)) {
        // Initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size   = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            size_t memset_size = padded_size - original_size;
            zakccMemset((char *) tensor->data + original_size, memset_size, 0, memset_size);
        }
    }
    return GGML_STATUS_SUCCESS;
}
static void ggml_backend_zlpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *         tensor,
                                                const void *          data,
                                                size_t                offset,
                                                size_t                size) {
    ggml_backend_zlpu_buffer_context * ctx = (ggml_backend_zlpu_buffer_context *) buffer->context;

    ggml_zlpu_set_device(ctx->device);
    memcpy((char *) tensor->data, data, size);
    // memcpy((char *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

static void ggml_backend_zlpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *   tensor,
                                                void *                data,
                                                size_t                offset,
                                                size_t                size) {
    ggml_backend_zlpu_buffer_context * ctx = (ggml_backend_zlpu_buffer_context *) buffer->context;

    ggml_zlpu_set_device(ctx->device);

    // memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

static bool ggml_backend_zlpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *   src,
                                                ggml_tensor *         dst) {

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;
}
static void ggml_backend_zlpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_zlpu_buffer_context * ctx = (ggml_backend_zlpu_buffer_context *) buffer->context;

    ggml_zlpu_set_device(ctx->device);
    zakccMemset(ctx->dev_ptr, buffer->size, value, buffer->size);
}

static const ggml_backend_buffer_i ggml_backend_zlpu_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_zlpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_zlpu_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_zlpu_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_zlpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_zlpu_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_zlpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_zlpu_buffer_clear,
    /* .reset           = */ NULL,
};

// zlpu buffer type
/**
 * @brief Structure representing context information for a specific backend
 * buffer type.
 */
struct ggml_backend_zlpu_buffer_type_context {
    int32_t     device; /**< Device identifier associated with the buffer context. */
    std::string name;   /**< Name associated with the buffer context. */
};

static const char * ggml_backend_zlpu_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_zlpu_buffer_type_context * buft_ctx = (ggml_backend_zlpu_buffer_type_context *) buft->context;

    return buft_ctx->name.c_str();
}

/**
 * @brief Allocates a new ZLPU buffer of the specified type and size.
 *
 * This function allocates a new ZLPU buffer on the specified device with the
 * given size.
 *
 * @param buft Pointer to the buffer type context.
 * @param size Size in bytes of the buffer to allocate.
 * @return Pointer to the allocated buffer, or nullptr if allocation fails.
 */
static ggml_backend_buffer_t ggml_backend_zlpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_zlpu_buffer_type_context * buft_ctx = (ggml_backend_zlpu_buffer_type_context *) buft->context;

    ggml_zlpu_set_device(buft_ctx->device);

    const size_t alignment = ZLPU_MEM_ALIGENED;
    size                   = GGML_PAD(size, alignment);
    if (size == 0) {
        size = alignment;
    }
    void *   dev_ptr;
    zakErr err = zakccMalloc(&dev_ptr, size, ZAKCC_MEM_MALLOC_HUGE_FIRST);
    if (err != ZAK_SUCCESS) {
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: zakccMalloc failed: %s\n", __func__,
                       size / 1024.0 / 1024.0, buft_ctx->device, zakGetRecentErrMsg());
        return nullptr;
    }

    ggml_backend_zlpu_buffer_context * ctx = new ggml_backend_zlpu_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_zlpu_buffer_interface, ctx, size);
}

static size_t ggml_backend_zlpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return ZLPU_MEM_ALIGENED;

    GGML_UNUSED(buft);
}
static size_t ggml_backend_zlpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                           const ggml_tensor *        tensor) {
    size_t  size = ggml_nbytes(tensor);
#if 0
    int64_t ne0  = tensor->ne[0];

    // Only check env once.
    static bool weight_to_nz = parse_bool(get_env("GGML_ZLPU_WEIGHT_NZ").value_or("on"));

    // last line must bigger than 32, because every single op deal at
    // least 32 bytes.
    // TODO: quantized type?
    // int64_t line_size = ne0 * ggml_element_size(tensor);
    // int64_t line_size_align_32 = (line_size + 31) & ~31;
    // size += (line_size_align_32 - line_size);
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    } else if (weight_to_nz && is_matmul_weight((const ggml_tensor *) tensor)) {
        // NZ format weight are not support quantized yet.
        // If ND tensor transform to NZ, size may changed.
        int64_t shape[] = { tensor->ne[1], tensor->ne[0] };
        GGML_ASSERT(tensor->ne[2] == 1);
        GGML_ASSERT(tensor->ne[3] == 1);
        const aclIntArray * acl_shape = aclCreateIntArray(shape, 2);
        size_t              new_size;
        ACL_CHECK(aclnnCalculateMatmulWeightSizeV2(acl_shape, ggml_zlpu_type_mapping(tensor->type), &new_size));
        ACL_CHECK(aclDestroyIntArray(acl_shape));
        size = std::max(size, new_size);
    }
#endif
    return size;

    GGML_UNUSED(buft);
}

static bool ggml_backend_zlpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_zlpu_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_zlpu_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_zlpu_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_zlpu_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_zlpu_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_zlpu_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_zlpu_buffer_type(int32_t device) {
    static std::mutex           mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_zlpu_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_zlpu_buffer_types[GGML_ZLPU_MAX_DEVICES];

    static bool ggml_backend_zlpu_buffer_type_initialized = false;

    if (!ggml_backend_zlpu_buffer_type_initialized) {
        for (int32_t i = 0; i < ggml_zlpu_info().device_count; i++) {
            ggml_backend_zlpu_buffer_types[i] = {
                /* .iface    = */ ggml_backend_zlpu_buffer_type_interface,
                /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_zlpu_reg(), i),
                /* .context  = */
                new ggml_backend_zlpu_buffer_type_context{ i, "ZLPU" + std::to_string(i) },
            };
        }
        ggml_backend_zlpu_buffer_type_initialized = true;
    }

    return &ggml_backend_zlpu_buffer_types[device];
}

static const char * ggml_backend_zlpu_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return "ZLPU_Host";

    GGML_UNUSED(buft);
}

static void ggml_backend_zlpu_host_buffer_free(ggml_backend_buffer_t buffer) {
    zakccFreeHost(buffer->context);
}

static void * ggml_zlpu_host_malloc(size_t size) {
    if (getenv("GGML_ZLPU_NO_PINNED") != nullptr) {
        return nullptr;
    }

    const size_t alignment = ZLPU_MEM_ALIGENED;
    size                   = GGML_PAD(size, alignment);
    if (size == 0) {
        size = alignment;
    }

    void *   hostPtr = nullptr;
    zakErr err     = zakccMallocHost((void **) &hostPtr, size);
    if (err != ZAK_SUCCESS) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__, size / 1024.0 / 1024.0,
                      zakGetRecentErrMsg());
        return nullptr;
    }
    return hostPtr;
}

static ggml_backend_buffer_t ggml_backend_zlpu_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                             size_t                     size) {
    void * hostPtr = ggml_zlpu_host_malloc(size);

    if (hostPtr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(hostPtr, size);
    buffer->buft                 = buft;
    buffer->iface.free_buffer    = ggml_backend_zlpu_host_buffer_free;

    return buffer;
}


ggml_backend_buffer_type_t ggml_backend_zlpu_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_zlpu_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_zlpu_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_zlpu_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */
        ggml_backend_reg_dev_get(ggml_backend_zlpu_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_zlpu_buffer_type_host;
}

/**
 * @brief Computes the forward operation for a given tensor using ZLPU
 * operations.
 *
 * This function selects the appropriate ZLPU operation based on the type of
 * operation specified in the tensor and performs the computation.
 *
 * @param ctx The ZLPU context containing necessary resources and
 * configurations.
 * @param dst The destination tensor where the result of the computation will be
 * stored.
 * @return true if the computation was successful; false otherwise.
 */
static bool ggml_zlpu_compute_forward(ggml_backend_zlpu_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_REPEAT:
        case GGML_OP_GET_ROWS:
        case GGML_OP_SET_ROWS:
        case GGML_OP_DUP:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_ACC:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_STEP:
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU_QUICK:
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_CONCAT:
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RMS_NORM:
            break;
        case GGML_OP_MUL_MAT:
            ggml_zlpu_mul_mat(ctx, dst);
            break;
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_CLAMP:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
            ggml_zlpu_softmax(ctx, dst);
        case GGML_OP_ROPE:
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_2D:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ARGMAX:
        case GGML_OP_COS:
        case GGML_OP_SIN:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_LOG:
        case GGML_OP_MEAN:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_COUNT_EQUAL:
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_zlpu_flash_attn_ext(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
}

static const char * ggml_backend_zlpu_name(ggml_backend_t backend) {
    ggml_backend_zlpu_context * zlpu_ctx = (ggml_backend_zlpu_context *) backend->context;

    return zlpu_ctx->name.c_str();
}
static void ggml_backend_zlpu_free(ggml_backend_t backend) {
    ggml_backend_zlpu_context * zlpu_ctx = (ggml_backend_zlpu_context *) backend->context;

    zakccResetDevice(zlpu_ctx->device);
    delete zlpu_ctx;
    delete backend;
}


static void evaluate_and_capture_zlpu_graph(ggml_backend_zlpu_context * zlpu_ctx,
                                            ggml_cgraph *               cgraph,
                                            bool &                      use_zlpu_graph,
                                            bool &                      zlpu_graph_update_required) {
    // Only perform the graph execution if ZLPU graphs are not enabled, or we are capturing the graph.
    // With the use of ZLPU graphs, the execution will be performed by the graph launch.
    if (!use_zlpu_graph || zlpu_graph_update_required) {
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];

            if (ggml_is_empty(node) 
			    || node->op == GGML_OP_RESHAPE 
				|| node->op == GGML_OP_TRANSPOSE 
				|| node->op == GGML_OP_VIEW 
				|| node->op == GGML_OP_PERMUTE 
				|| node->op == GGML_OP_NONE) {
                continue;
            }

            // GGML_LOG_ERROR("%s: %s (%s)  ", __func__, node->name, ggml_op_name(node->op));
            // GGML_LOG_ERROR("ne: [%d %d %d %d] ", node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
            // GGML_LOG_ERROR("nb: [%d %d %d %d] ", node->nb[0], node->nb[1], node->nb[2], node->nb[3]);
            // if(node->src[0] != nullptr) GGML_LOG_ERROR("src: %s", node->src[0]->name);
            // if(node->src[1] != nullptr) GGML_LOG_ERROR(" + %s", node->src[1]->name);
            // GGML_LOG_ERROR("\n");
            bool ok = ggml_zlpu_compute_forward(*zlpu_ctx, node);
            if (!ok) {
                GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
            }
            GGML_ASSERT(ok);
        }
    }

}

/**
 * @brief Computes a computational graph using a ZLPU backend.
 *
 * This function computes the operations defined in the computational graph
 * using the specified ZLPU backend.
 *
 * @param backend Pointer to the ZLPU backend structure to use for computation.
 * @param cgraph Pointer to the computational graph structure containing nodes
 *               representing operations to be computed.
 * @return enum ggml_status Returns GGML_STATUS_SUCCESS if computation
 *         completes successfully, otherwise an appropriate error status.
 */
static enum ggml_status ggml_backend_zlpu_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_zlpu_context * zlpu_ctx = (ggml_backend_zlpu_context *) backend->context;


    bool use_zlpu_graph             = false;
    bool zlpu_graph_update_required = false;
    evaluate_and_capture_zlpu_graph(zlpu_ctx, cgraph, use_zlpu_graph, zlpu_graph_update_required);

    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_zlpu_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return true;
    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU_ERF:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return true;
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                        return true;
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q4_0:
                        // only support contiguous for quantized types.
                        return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
                    default:
                        return false;
                }
            }
        case GGML_OP_MUL_MAT_ID:
            switch (op->src[0]->type) {
                case GGML_TYPE_F16:
                case GGML_TYPE_F32:
                    return true;
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_0:
                    // only support contiguous for quantized types.
                    return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
                default:
                    return false;
            }
        // embedding
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q4_0:
                        return true;
                    default:
                        return false;
                }
            }
            break;
        case GGML_OP_SET_ROWS:
            {
                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                        return true;
                    default:
                        return false;
                }
            }
            break;
        case GGML_OP_CPY:
            {
                ggml_tensor * src = op->src[0];
                if ((op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) ||
                    (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16)) {
                    // only support F32 and F16.
                    return false;
                }
                return true;
            }
            break;
        case GGML_OP_CONT:
            {
                // TODO: support GGML_TYPE_BF16
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_ROPE:
            {
                // TODO: with ops-test v == 1
                // TODO: n_dims <= ne0
                if (op->src[0]->ne[0] != op->op_params[1]) {
                    return false;
                }

                const int mode = ((const int32_t *) op->op_params)[2];
                if (mode & GGML_ROPE_TYPE_MROPE) {
                    return false;
                }
                if (mode & GGML_ROPE_TYPE_VISION) {
                    return false;
                }
                return true;
            }
        case GGML_OP_UPSCALE:
            {
                return true;
            }
        case GGML_OP_POOL_2D:
        case GGML_OP_DUP:
        case GGML_OP_SUM:
        case GGML_OP_IM2COL:
        case GGML_OP_CONCAT:
        case GGML_OP_REPEAT:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NORM:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_CLAMP:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ACC:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_ARGMAX:
        case GGML_OP_COS:
        case GGML_OP_SIN:
        case GGML_OP_LOG:
        case GGML_OP_MEAN:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_COUNT_EQUAL:
            return true;
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_SCALE:
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                return true;
            }
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

/**
 * @brief Checks if the backend buffer type is associated with the ZLPU backend.
 *
 * This function checks whether the provided backend buffer type is associated
 * with the ZLPU backend based on the comparison of its name retrieval function
 * pointer.
 *
 * @param buft Pointer to the backend buffer type to check.
 * @return bool Returns true if the buffer type is associated with the ZLPU
 * backend, otherwise false.
 */
static bool ggml_backend_buft_is_zlpu(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_zlpu_buffer_type_name;
}
/* LLLL 50000----------------------------------------------- */
/**
 * @brief Determines if a tensor operation should be offloaded to the ZLPU
 * backend.
 *
 * This function checks if a given tensor operation should be offloaded to the
 * ZLPU backend based on the operation type and the size of the tensor. It
 * returns true if the second dimension (ne[1]) of the tensor is greater than or
 * equal to the minimum batch size and the operation is not GGML_OP_GET_ROWS.
 *
 * @param backend Pointer to the ZLPU backend.
 * @param op Pointer to the tensor operation to check.
 * @return bool Returns true if the operation should be offloaded, otherwise
 * false.
 */
static bool ggml_backend_zlpu_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    // const int min_batch_size = 32;
    GGML_UNUSED(dev);
    GGML_UNUSED(op);

    // return op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS;
    return true;
}

/**
 * @brief Records an event on the ZLPU backend stream.
 *
 * This function records the given event on the ACL runtime stream associated
 * with the backend context.
 *
 * @param event Pointer to the event structure to be recorded.
 */
static void ggml_backend_zlpu_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(event);
    GGML_UNUSED(backend);
}

/**
 * @brief Waits for a recorded event to complete on the ZLPU backend stream.
 *
 * This function makes the given backend wait for the event to complete on its
 * ACL runtime stream.
 *
 * @param backend Pointer to the backend structure.
 * @param event Pointer to the event structure that the backend needs to wait
 * for.
 */
static void ggml_backend_zlpu_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(backend);
    GGML_UNUSED(event);
}

/**
 * @brief Structure defining the interface for the ZLPU backend.
 *
 * This structure contains function pointers for various operations
 * supported by the ZLPU backend, including name retrieval, memory
 * management, tensor operations, synchronization, and event handling.
 */
static const ggml_backend_i ggml_backend_zlpu_interface = {
    /* .get_name                = */ ggml_backend_zlpu_name,
    /* .free                    = */ ggml_backend_zlpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_zlpu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

/**
 * @brief Return the hardcoded GUID for the ZLPU backend.
 *
 * This function returns a static GUID which uniquely identifies the ZLPU
 * backend.
 *
 * @return A pointer to the static GUID.
 */
static ggml_guid_t ggml_backend_zlpu_guid() {
    static ggml_guid guid = { 0xa1, 0x94, 0xaf, 0xac, 0xbd, 0x4f, 0x47, 0x34,
                              0xbe, 0x1a, 0x9e, 0x71, 0x1f, 0x9e, 0xed, 0x64 };
    return &guid;
}

// backend device
struct ggml_backend_zlpu_device_context {
    int         device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_zlpu_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_zlpu_device_context * ctx = (ggml_backend_zlpu_device_context *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_zlpu_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_zlpu_device_context * ctx = (ggml_backend_zlpu_device_context *) dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_zlpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_zlpu_device_context * ctx = (ggml_backend_zlpu_device_context *) dev->context;
    ggml_backend_zlpu_get_device_memory(ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_zlpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_zlpu_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_zlpu_device_get_name(dev);
    props->description = ggml_backend_zlpu_device_get_description(dev);
    props->type        = ggml_backend_zlpu_device_get_type(dev);
    ggml_backend_zlpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
}

static ggml_backend_t ggml_backend_zlpu_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_zlpu_device_context * ctx = (ggml_backend_zlpu_device_context *) dev->context;
    return ggml_backend_zlpu_init(ctx->device);
}

/**
 * @brief Checks if the ZLPU backend supports a specific backend buffer type.
 *
 * This function determines whether the ZLPU backend supports the given backend
 * buffer type by comparing the device context of the backend and buffer type.
 * It returns true if the devices are same between the backend context and
 * buffer type context.
 *
 * @param backend Pointer to the ZLPU backend.
 * @param buft Pointer to the backend buffer type to check.
 * @return bool Returns true if the ZLPU backend supports the buffer type,
 *              otherwise false.
 */
static bool ggml_backend_zlpu_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (ggml_backend_buft_is_zlpu(buft)) {
        ggml_backend_zlpu_device_context *      dev_ctx  = (ggml_backend_zlpu_device_context *) dev->context;
        ggml_backend_zlpu_buffer_type_context * buft_ctx = (ggml_backend_zlpu_buffer_type_context *) buft->context;
        return buft_ctx->device == dev_ctx->device;
    }
    return false;
}

static ggml_backend_buffer_type_t ggml_backend_zlpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_zlpu_device_context * ctx = (ggml_backend_zlpu_device_context *) dev->context;
    return ggml_backend_zlpu_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_zlpu_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_zlpu_host_buffer_type();
}

/**
 * @brief Creates a new event for the ZLPU backend device.
 *
 * This function initializes a new event for the ZLPU backend by setting the
 * device and creating an ACL runtime event. The created event is then wrapped
 * in a ggml_backend_event structure and returned.
 *
 * @param backend Pointer to the ZLPU backend.
 * @return ggml_backend_event_t Returns a pointer to the new event structure.
 */
static ggml_backend_event_t ggml_backend_zlpu_device_event_new(ggml_backend_dev_t dev) {
}

/**
 * @brief Frees a ZLPU backend event.
 *
 * This function destroys the ACL runtime event associated with the given ZLPU
 * backend event and then deletes the event structure itself.
 *
 * @param event Pointer to the event structure to be freed.
 */
static void ggml_backend_zlpu_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
}

/**
 * @brief Synchronizes the given event on the ZLPU backend.
 *
 * This function waits for the specified event to complete on the ACL runtime.
 *
 * @param event Pointer to the event structure to be synchronized.
 */
static void ggml_backend_zlpu_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
}
/* LZS */
static const ggml_backend_device_i ggml_backend_zlpu_device_interface = {
    /* .get_name                = */ ggml_backend_zlpu_device_get_name,
    /* .get_description         = */ ggml_backend_zlpu_device_get_description,
    /* .get_memory              = */ ggml_backend_zlpu_device_get_memory,
    /* .get_type                = */ ggml_backend_zlpu_device_get_type,
    /* .get_props               = */ ggml_backend_zlpu_device_get_props,
    /* .init_backend            = */ ggml_backend_zlpu_device_init,  // called for every card
    /* .get_buffer_type         = */ ggml_backend_zlpu_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_zlpu_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,  // not supported for ZLPU
    /* .supports_op             = */ ggml_backend_zlpu_supports_op,
    /* .supports_buft           = */ ggml_backend_zlpu_supports_buft,
    /* .offload_op              = */ ggml_backend_zlpu_offload_op,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

// backend reg
struct ggml_backend_zlpu_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_zlpu_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_ZLPU_NAME;
}

static size_t ggml_backend_zlpu_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_zlpu_reg_context * ctx = (ggml_backend_zlpu_reg_context *) reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_zlpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_zlpu_reg_context * ctx = (ggml_backend_zlpu_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void * ggml_backend_zlpu_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    // reserved for future use
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_zlpu_reg_interface = {
    /* .get_name          = */ ggml_backend_zlpu_reg_get_name,
    /* .get_device_count  = */ ggml_backend_zlpu_reg_get_device_count,
    /* .get_device        = */ ggml_backend_zlpu_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_zlpu_reg_get_proc_address,
};
// backend registry, called only once for zlpu backend
ggml_backend_reg_t ggml_backend_zlpu_reg() {
    static ggml_backend_reg reg;
    static bool             initialized = false;

    {
        static std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_zlpu_reg_context * ctx = new ggml_backend_zlpu_reg_context;

            for (int i = 0; i < ggml_zlpu_info().device_count; i++) {
                ggml_backend_zlpu_device_context * dev_ctx = new ggml_backend_zlpu_device_context();
                dev_ctx->description                       = "wmv701";
                dev_ctx->device                            = i;
                dev_ctx->name                              = GGML_ZLPU_NAME + std::to_string(i);
                ggml_zlpu_set_device(i);
                ggml_backend_dev_t dev = new ggml_backend_device{ /* .iface   = */ ggml_backend_zlpu_device_interface,
                                                                  /* .reg     = */ &reg,
                                                                  /* .context = */ dev_ctx };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{ /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_zlpu_reg_interface,
                                    /* .context     = */ ctx };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_zlpu_init(int32_t device) {
    if (device < 0 || device >= ggml_backend_zlpu_get_device_count()) {
        GGML_LOG_ERROR("%s: error: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_zlpu_context * ctx = new ggml_backend_zlpu_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return nullptr;
    }
    ggml_zlpu_set_device(ctx->device);
    ggml_backend_t zlpu_backend =
        new ggml_backend{ /* .guid      = */ ggml_backend_zlpu_guid(),
                          /* .interface = */ ggml_backend_zlpu_interface,
                          /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_zlpu_reg(), device),
                          /* .context   = */ ctx };

    return zlpu_backend;
}

bool ggml_backend_is_zlpu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_zlpu_guid());
}

int32_t ggml_backend_zlpu_get_device_count() {
    return ggml_zlpu_info().device_count;
}


void ggml_backend_zlpu_get_device_memory(int32_t device, size_t * free, size_t * total) {
    ggml_zlpu_set_device(device);
    zakccGetMemInfo(ZAKCC_HBM_MEM, free, total);
}

GGML_BACKEND_DL_IMPL(ggml_backend_zlpu_reg)

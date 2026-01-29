#ifndef __ZLPU_OPS_H__
#define __ZLPU_OPS_H__

#include "common.h"

extern void ggml_zlpu_flash_attn_ext(ggml_backend_zlpu_context & ctx, ggml_tensor * dst);
extern void ggml_zlpu_softmax(ggml_backend_zlpu_context & ctx, ggml_tensor * dst);
extern void ggml_zlpu_mul_mat(const ggml_backend_zlpu_context & ctx, ggml_tensor * dst);

#endif
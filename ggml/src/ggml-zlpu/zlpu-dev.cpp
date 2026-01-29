
#include <stdarg.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "ggml.h"
#include "zlpu-dev.h"


// zlpu device memory
const size_t zlpu_mem_total=2*1024*1024;
static size_t zlpu_mem_used=0;

zakErr zakccGetMemInfo(ZAK_MEM_TYPE type, size_t *free, size_t *total)
{
    *free = zlpu_mem_total - zlpu_mem_used;
    *total= zlpu_mem_total;
    return ZAK_SUCCESS;
}

zakErr zakccMalloc(void**ptr, size_t size, ZAK_MEM_TYPE type) {
    // *ptr = malloc(size);
        *ptr = aligned_alloc(ZLPU_MEM_ALIGENED, size);
    return ((*ptr != NULL)?ZAK_SUCCESS:ZAK_FAIL);
}


void zakccFree(void* ptr) {
    if (ptr != NULL) {  // 避免对 NULL 指针调用 free
        free(ptr);
    }
}

zakErr zakccMemset(void *ptr, size_t total, int value, size_t num)
{
    num = (num>total)?total:num;

    memset(ptr, value, num);
    return ZAK_SUCCESS;
}

zakErr zakccMallocHost(void**ptr, size_t size) {
    *ptr = malloc(size);
    return ((ptr != NULL)?ZAK_SUCCESS:ZAK_FAIL);
}
void zakccFreeHost(void* ptr) {
    if (ptr != NULL) {  // 避免对 NULL 指针调用 free
        free(ptr);
    }
}

void zakccResetDevice(int32_t device)
{
    //to do
    // ggml_backend_free
    ;
}



char * zakGetRecentErrMsg(void) {
    char *const err = "ZAK NONE";
    return err;
}

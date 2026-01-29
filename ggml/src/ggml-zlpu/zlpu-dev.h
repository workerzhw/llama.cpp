
#ifndef __ZLPU_DEV_H__
#define __ZLPU_DEV_H__

#define ZLPU_MEM_ALIGENED    256

enum zakErr {
    ZAK_SUCCESS,
    ZAK_FAIL,
    ZAK_IO_FAIL,
    ZAK_RAM_FAIL
};

enum ZAK_MEM_TYPE {
    ZAKCC_MEM_MALLOC_HUGE_FIRST,
    ZAKCC_MEM_MALLOC_NEAR_FIRST,
    ZAKCC_HBM_MEM
};


extern zakErr zakccGetMemInfo(ZAK_MEM_TYPE type, size_t *free, size_t *total);
extern zakErr zakccMalloc(void**ptr, size_t size, ZAK_MEM_TYPE type);
extern void zakccFree(void* ptr);
extern zakErr zakccMemset(void *ptr, size_t total, int value, size_t num);

extern char * zakGetRecentErrMsg(void);
extern zakErr zakccMallocHost(void**ptr, size_t size);
extern void zakccFreeHost(void* ptr);
extern void zakccResetDevice(int32_t device);

#endif /*__ZLPU_DEV_H__ */


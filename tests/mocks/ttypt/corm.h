#ifndef CORM_H
#define CORM_H

#include <stdint.h>
#include <stddef.h>

/*
 * Minimal corm API mirror used when compiling libqllm.c against the mock
 * layer (MOCK_BUILD). Handles are uint32_t ids, CM_MISS is the "not found"
 * sentinel, and corm_get() returns a pointer to the stored slot (or NULL
 * when the key is absent) exactly like the real library.
 */

#define CM_MISS ((uint32_t)-1)

enum corm_tbi {
	CM_PTR	= 0,
	CM_HNDL = 1,
	CM_STR	= 2,
	CM_U32	= 3,
};

#define CM_RECORD_FLAG 0x00010000u

#ifdef __cplusplus
extern "C" {
#endif

uint32_t corm_reg(size_t len);
uint32_t corm_open(const char *filename,
		   const char *database,
		   uint32_t ktype,
		   uint32_t vtype,
		   uint32_t mask,
		   uint32_t flags);
const void *corm_get(uint32_t hd, const void * const key);
uint32_t corm_put(uint32_t hd, const void * const key, const void * const value);
void corm_close(uint32_t hd);

#ifdef __cplusplus
}
#endif

#endif
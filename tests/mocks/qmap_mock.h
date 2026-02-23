#ifndef QMAP_MOCK_H
#define QMAP_MOCK_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QM_STR 1

void mock_qmap_init(void);
void mock_qmap_cleanup(void);

void mock_qmap_set_get_result(void *result);
void mock_qmap_set_put_fail(int should_fail);
void mock_qmap_enable_persistence(int enable);

uint32_t qmap_reg(size_t size);
uint32_t qmap_open(const char *path, const char *database, int ktype,
		   uint32_t vtype, uint32_t mask, int flags);
void *qmap_get(uint32_t hd, const char *key);
int qmap_put(uint32_t hd, const char *key, void *value);

#ifdef __cplusplus
}
#endif

#endif

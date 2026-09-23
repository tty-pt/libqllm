#ifndef CORM_MOCK_H
#define CORM_MOCK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void mock_corm_init(void);
void mock_corm_cleanup(void);
int mock_corm_count(uint32_t hd);

#ifdef __cplusplus
}
#endif

#endif
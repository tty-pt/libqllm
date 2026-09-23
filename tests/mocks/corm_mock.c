#include "ttypt/corm.h"
#include "corm_mock.h"
#include <stdlib.h>
#include <string.h>

#define MAX_CORM_MAPS 64
#define MAX_CORM_ENTRIES 256

typedef struct {
	char    key[1024];
	uint64_t value;
} corm_entry_t;

typedef struct {
	uint32_t hd;
	uint32_t vtype;
	int      n;
	corm_entry_t entries[MAX_CORM_ENTRIES];
} corm_map_t;

static corm_map_t _maps[MAX_CORM_MAPS];
static int        _map_used[MAX_CORM_MAPS];
static uint32_t   _next_handle = 1;

static corm_map_t *
map_by_hd(uint32_t hd)
{
	int i;

	for (i = 0; i < MAX_CORM_MAPS; i++)
		if (_map_used[i] && _maps[i].hd == hd)
			return &_maps[i];
	return NULL;
}

static uint32_t
new_map(void)
{
	int i;

	for (i = 0; i < MAX_CORM_MAPS; i++) {
		if (!_map_used[i]) {
			memset(&_maps[i], 0, sizeof(_maps[i]));
			_maps[i].hd = _next_handle++;
			_map_used[i] = 1;
			return _maps[i].hd;
		}
	}
	return CM_MISS;
}

static void
clear_map(corm_map_t *m)
{
	m->n = 0;
	memset(m->entries, 0, sizeof(m->entries));
}

uint32_t
corm_reg(size_t len)
{
	uint32_t hd = new_map();
	corm_map_t *m;

	(void)len;
	if (hd != CM_MISS) {
		m = map_by_hd(hd);
		m->vtype = CM_PTR;
	}
	return hd;
}

uint32_t
corm_open(const char *filename,
	  const char *database,
	  uint32_t ktype,
	  uint32_t vtype,
	  uint32_t mask,
	  uint32_t flags)
{
	uint32_t hd = new_map();
	corm_map_t *m;

	(void)filename;
	(void)database;
	(void)ktype;
	(void)mask;
	(void)flags;
	if (hd != CM_MISS) {
		m = map_by_hd(hd);
		m->vtype = vtype;
	}
	return hd;
}

const void *
corm_get(uint32_t hd, const void * const key)
{
	corm_map_t *m = map_by_hd(hd);
	const char *skey = key;
	int i;

	if (!m || !skey)
		return NULL;

	for (i = 0; i < m->n; i++)
		if (strcmp(m->entries[i].key, skey) == 0)
			return &m->entries[i].value;

	return NULL;
}

uint32_t
corm_put(uint32_t hd, const void * const key, const void * const value)
{
	corm_map_t *m = map_by_hd(hd);
	const char *skey = key;
	int i;

	if (!m || !skey || !value)
		return CM_MISS;

	for (i = 0; i < m->n; i++) {
		if (strcmp(m->entries[i].key, skey) == 0) {
			m->entries[i].value = 0;
			memcpy(&m->entries[i].value, value, sizeof(uint64_t));
			return 0;
		}
	}

	if (m->n >= MAX_CORM_ENTRIES)
		return CM_MISS;

	strncpy(m->entries[m->n].key, skey,
		sizeof(m->entries[m->n].key) - 1);
	m->entries[m->n].value = 0;
	memcpy(&m->entries[m->n].value, value, sizeof(uint64_t));
	m->n++;
	return 0;
}

void
corm_close(uint32_t hd)
{
	corm_map_t *m = map_by_hd(hd);

	if (!m)
		return;
	clear_map(m);
	_map_used[(int)(m - _maps)] = 0;
}

void
mock_corm_init(void)
{
	int i;

	for (i = 0; i < MAX_CORM_MAPS; i++) {
		if (_map_used[i])
			clear_map(&_maps[i]);
	}
}

void
mock_corm_cleanup(void)
{
	int i;

	for (i = 0; i < MAX_CORM_MAPS; i++) {
		if (_map_used[i])
			clear_map(&_maps[i]);
		_map_used[i] = 0;
	}
	_next_handle = 1;
}

int
mock_corm_count(uint32_t hd)
{
	corm_map_t *m = map_by_hd(hd);

	return m ? m->n : 0;
}
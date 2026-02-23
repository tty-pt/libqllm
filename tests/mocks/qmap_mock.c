#include "qmap_mock.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_ENTRIES 256
#define MAX_FILES 16

static void *_mock_get_result = NULL;
static int _mock_put_fail = 0;

struct qmap_entry {
	char key[256];
	void *value;
	int used;
};

/* File-based persistence simulation */
struct qmap_file {
	char filename[256];
	char database[256];
	struct qmap_entry entries[MAX_ENTRIES];
	int used;
};

static struct qmap_entry _entries[MAX_ENTRIES];
static struct qmap_file _files[MAX_FILES];
static uint32_t _next_type = 1;
static uint32_t _next_hd = 1;

/* Map handle to file index */
static int _hd_to_file[MAX_FILES];
static int _persistence_enabled = 0;

void
mock_qmap_init(void)
{
	memset(_entries, 0, sizeof(_entries));
	memset(_files, 0, sizeof(_files));
	memset(_hd_to_file, -1, sizeof(_hd_to_file));
	_mock_get_result = NULL;
	_mock_put_fail = 0;
	_next_type = 1;
	_next_hd = 1;
	_persistence_enabled = 0;
}

void
mock_qmap_cleanup(void)
{
}

void
mock_qmap_set_get_result(void *result)
{
	_mock_get_result = result;
}

void
mock_qmap_set_put_fail(int should_fail)
{
	_mock_put_fail = should_fail;
}

void
mock_qmap_enable_persistence(int enable)
{
	_persistence_enabled = enable;
}

uint32_t
qmap_reg(size_t size)
{
	return _next_type++;
}

uint32_t
qmap_open(const char *path, const char *database, int ktype,
	  uint32_t vtype, uint32_t mask, int flags)
{
	uint32_t hd = _next_hd++;
	int file_idx = -1;
	int i;
	
	/* If persistence is enabled and path is not NULL, find or create file entry */
	if (_persistence_enabled && path && path[0] != '\0') {
		/* Look for existing file */
		for (i = 0; i < MAX_FILES; i++) {
			if (_files[i].used && 
			    strcmp(_files[i].filename, path) == 0 &&
			    strcmp(_files[i].database, database) == 0) {
				file_idx = i;
				break;
			}
		}
		
		/* If not found, create new file entry */
		if (file_idx == -1) {
			for (i = 0; i < MAX_FILES; i++) {
				if (!_files[i].used) {
					strncpy(_files[i].filename, path, sizeof(_files[i].filename) - 1);
					strncpy(_files[i].database, database, sizeof(_files[i].database) - 1);
					memset(_files[i].entries, 0, sizeof(_files[i].entries));
					_files[i].used = 1;
					file_idx = i;
					break;
				}
			}
		}
		
		/* Map handle to file */
		if (file_idx != -1 && (hd - 1) < MAX_FILES) {
			_hd_to_file[hd - 1] = file_idx;
		}
	}
	
	return hd;
}

void *
qmap_get(uint32_t hd, const char *key)
{
    int i;
    int file_idx;
    struct qmap_entry *entry_list;

    if (_mock_get_result)
        return _mock_get_result;

    /* Determine which entry list to use */
    if (_persistence_enabled && (hd - 1) < MAX_FILES && (file_idx = _hd_to_file[hd - 1]) >= 0) {
        entry_list = _files[file_idx].entries;
    } else {
        entry_list = _entries;
    }

    for (i = 0; i < MAX_ENTRIES; i++) {
        if (entry_list[i].used && strcmp(entry_list[i].key, key) == 0)
            {
                return &entry_list[i].value;
            }
    }

    return NULL;
}

int
qmap_put(uint32_t hd, const char *key, void *value)
{
    int i;
    int file_idx;
    struct qmap_entry *entry_list;

    if (_mock_put_fail)
        return -1;

    /* Determine which entry list to use */
    if (_persistence_enabled && (hd - 1) < MAX_FILES && (file_idx = _hd_to_file[hd - 1]) >= 0) {
        entry_list = _files[file_idx].entries;
    } else {
        entry_list = _entries;
    }

    /* First check if key already exists and update it */
    for (i = 0; i < MAX_ENTRIES; i++) {
        if (entry_list[i].used && strcmp(entry_list[i].key, key) == 0) {
            entry_list[i].value = value;
            return 0;
        }
    }

    /* If not found, add new entry */
    for (i = 0; i < MAX_ENTRIES; i++) {
        if (!entry_list[i].used) {
            strncpy(entry_list[i].key, key, sizeof(entry_list[i].key) - 1);
            entry_list[i].value = value;
            entry_list[i].used = 1;
            return 0;
        }
    }

	return -1;
}

#include "mock_registry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static mock_entry_t _mock_registry[MOCK_REGISTRY_SIZE];
static int _mock_count = 0;

static mock_config_t _mock_configs[MOCK_REGISTRY_SIZE];
static int _mock_config_count = 0;

static mock_callback_t _mock_callbacks[MOCK_REGISTRY_SIZE];
static int _mock_callback_count = 0;

void
mock_init(void)
{
	memset(_mock_registry, 0, sizeof(_mock_registry));
	memset(_mock_configs, 0, sizeof(_mock_configs));
	memset(_mock_callbacks, 0, sizeof(_mock_callbacks));
	_mock_count = 0;
	_mock_config_count = 0;
	_mock_callback_count = 0;
}

void
mock_cleanup(void)
{
	mock_disable_all();
	_mock_count = 0;
	_mock_config_count = 0;
	_mock_callback_count = 0;
}

static int
find_mock(const char *name)
{
	int i;
	for (i = 0; i < _mock_count; i++) {
		if (strcmp(_mock_registry[i].name, name) == 0)
			return i;
	}
	return -1;
}

static int
find_or_create_mock(const char *name)
{
	int idx = find_mock(name);
	if (idx >= 0)
		return idx;

	if (_mock_count >= MOCK_REGISTRY_SIZE) {
		fprintf(stderr, "Mock registry full\n");
		return -1;
	}

	idx = _mock_count++;
	_mock_registry[idx].name = name;
	_mock_registry[idx].enabled = 0;
	_mock_registry[idx].call_count = 0;
	return idx;
}

void
mock_set(const char *name, void *original, void *replacement)
{
	int idx = find_or_create_mock(name);
	if (idx < 0)
		return;

	_mock_registry[idx].original = original;
	_mock_registry[idx].replacement = replacement;
	_mock_registry[idx].enabled = 1;
	_mock_registry[idx].call_count = 0;
}

void *
mock_get(const char *name)
{
	int idx = find_mock(name);
	if (idx < 0 || !_mock_registry[idx].enabled)
		return NULL;

	_mock_registry[idx].call_count++;

	if (_mock_callbacks[idx].on_call)
		_mock_callbacks[idx].on_call(_mock_callbacks[idx].user_data,
					     _mock_registry[idx].call_count);

	return _mock_registry[idx].replacement;
}

void
mock_enable(const char *name)
{
	int idx = find_mock(name);
	if (idx >= 0)
		_mock_registry[idx].enabled = 1;
}

void
mock_disable(const char *name)
{
	int idx = find_mock(name);
	if (idx >= 0)
		_mock_registry[idx].enabled = 0;
}

void
mock_disable_all(void)
{
	int i;
	for (i = 0; i < _mock_count; i++)
		_mock_registry[i].enabled = 0;
}

int
mock_call_count(const char *name)
{
	int idx = find_mock(name);
	if (idx < 0)
		return 0;
	return _mock_registry[idx].call_count;
}

void
mock_reset_call_counts(void)
{
	int i;
	for (i = 0; i < _mock_count; i++)
		_mock_registry[i].call_count = 0;
}

void
mock_configure(const char *name, mock_config_t *config)
{
	int i;
	int idx = -1;

	for (i = 0; i < _mock_config_count; i++) {
		if (strcmp(_mock_configs[i].name, name) == 0) {
			idx = i;
			break;
		}
	}

	if (idx < 0) {
		if (_mock_config_count >= MOCK_REGISTRY_SIZE)
			return;
		idx = _mock_config_count++;
	}

	_mock_configs[idx].name = name;
	_mock_configs[idx].return_value = config->return_value;
	_mock_configs[idx].should_fail = config->should_fail;
	_mock_configs[idx].fail_count = config->fail_count;
}

mock_config_t *
mock_get_config(const char *name)
{
	int i;
	for (i = 0; i < _mock_config_count; i++) {
		if (strcmp(_mock_configs[i].name, name) == 0)
			return &_mock_configs[i];
	}
	return NULL;
}

void
mock_set_callback(const char *name, mock_callback_t *callback)
{
	int i;
	int idx = -1;

	for (i = 0; i < _mock_callback_count; i++) {
		if (strcmp(_mock_callbacks[i].name, name) == 0) {
			idx = i;
			break;
		}
	}

	if (idx < 0) {
		if (_mock_callback_count >= MOCK_REGISTRY_SIZE)
			return;
		idx = _mock_callback_count;
		_mock_callback_count++;
	}

	if (callback) {
		_mock_callbacks[idx] = *callback;
	} else {
		memset(&_mock_callbacks[idx], 0, sizeof(mock_callback_t));
	}
}

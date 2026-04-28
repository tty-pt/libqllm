#include "ndc_mock.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>

static char *_capture_buf = NULL;
static size_t _capture_size = 0;
static size_t _capture_pos = 0;

static struct {
	char name[64];
	void *cb;
	int flags;
} _registered_cmds[32];
static int _n_registered = 0;

static int _last_status = 0;
static struct {
	char name[64];
	char value[256];
} _headers[16];
static int _n_headers = 0;

void
mock_ndc_init(void)
{
	_capture_buf = NULL;
	_capture_size = 0;
	_capture_pos = 0;
	_n_registered = 0;
	_last_status = 0;
	_n_headers = 0;
	memset(_registered_cmds, 0, sizeof(_registered_cmds));
	memset(_headers, 0, sizeof(_headers));
}

void
mock_ndc_cleanup(void)
{
}

void
mock_ndc_set_write_capture(char *buf, size_t size)
{
	_capture_buf = buf;
	_capture_size = size;
	_capture_pos = 0;
	if (buf && size > 0)
		buf[0] = '\0';
}

void
mock_ndc_get_written_data(char **buf, size_t *size)
{
	if (buf)
		*buf = _capture_buf;
	if (size)
		*size = _capture_pos;
}

void
mock_ndc_reset_write_capture(void)
{
	_capture_pos = 0;
	if (_capture_buf && _capture_size > 0)
		_capture_buf[0] = '\0';
}

int
mock_ndc_get_last_status(void)
{
	return _last_status;
}

const char*
mock_ndc_get_header(const char *name)
{
	for (int i = 0; i < _n_headers; i++) {
		if (strcasecmp(_headers[i].name, name) == 0)
			return _headers[i].value;
	}
	return NULL;
}

int
ndc_main(void)
{
	return 0;
}

int
ndc_accept(int fd)
{
	return 0;
}

void
ndc_disconnect(int fd)
{
}

void
ndc_write(int fd, const void *buf, size_t len)
{
	if (_capture_buf && _capture_size > 0) {
		size_t avail = _capture_size - _capture_pos - 1;
		if (len > avail)
			len = avail;
		if (len > 0) {
			memcpy(_capture_buf + _capture_pos, buf, len);
			_capture_pos += len;
			_capture_buf[_capture_pos] = '\0';
		}
	}
}

int
ndc_writef(int fd, const char *fmt, ...)
{
	char buf[8192];
	va_list ap;
	int n;

	va_start(ap, fmt);
	n = vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);

	if (n > 0)
		ndc_write(fd, buf, (size_t)n);

	return n;
}

int
ndc_exec(int fd, char **args, ndc_cmd_cb cb, void *user, int flags)
{
	return 0;
}

void
ndc_register(const char *name, void *cb, int flags)
{
	if (_n_registered < 32) {
		strncpy(_registered_cmds[_n_registered].name, name, 63);
		_registered_cmds[_n_registered].cb = cb;
		_registered_cmds[_n_registered].flags = flags;
		_n_registered++;
	}
}

void
ndc_register_handler(const char *path, ndc_http_handler handler)
{
	/* For mocks, we don't need to do much here unless we want to simulate routing */
}

void
ndc_respond(socket_t fd, int code, const char *body)
{
	_last_status = code;
	if (body) {
		ndc_write(fd, body, strlen(body));
	}
}

void
ndc_header_set(socket_t fd, const char *name, const char *value)
{
	if (_n_headers < 16) {
		strncpy(_headers[_n_headers].name, name, 63);
		strncpy(_headers[_n_headers].value, value, 255);
		_n_headers++;
	}
}

void
ndc_close(socket_t fd)
{
}

size_t
ndc_mmap(char **ptr, const char *path)
{
	*ptr = NULL;
	return 0;
}

void
ndc_certs_add(const char *path)
{
}

void
ndc_cert_add(const char *path)
{
}

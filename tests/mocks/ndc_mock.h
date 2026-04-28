#ifndef NDC_MOCK_H
#define NDC_MOCK_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NDC_DETACH 1
#define NDC_ROOT   2

#define CF_NOAUTH  1
#define CF_NOTRIM  2

struct ndc_config {
	int flags;
	int port;
	int ssl_port;
	const char *chroot;
};

typedef int socket_t;
typedef void (ndc_cb_t)(int fd, int argc, char *argv[]);
typedef int (*ndc_http_handler)(socket_t fd, char *body);

struct cmd_slot {
    char *name;
    ndc_cb_t *cb;
    int flags;
};

void mock_ndc_init(void);
void mock_ndc_cleanup(void);

void mock_ndc_set_write_capture(char *buf, size_t size);
void mock_ndc_get_written_data(char **buf, size_t *size);
void mock_ndc_reset_write_capture(void);

int mock_ndc_get_last_status(void);
const char* mock_ndc_get_header(const char *name);

typedef void (*ndc_cmd_cb)(int fd, char *buf, size_t len, int ofd);

int ndc_main(void);
int ndc_accept(int fd);
void ndc_disconnect(int fd);
void ndc_write(int fd, const void *buf, size_t len);
int ndc_writef(int fd, const char *fmt, ...);
int ndc_exec(int fd, char **args, ndc_cmd_cb cb, void *user, int flags);
void ndc_register(const char *name, void *cb, int flags);
void ndc_register_handler(const char *path, ndc_http_handler handler);
void ndc_respond(socket_t fd, int code, const char *body);
void ndc_header_set(socket_t fd, const char *name, const char *value);
size_t ndc_mmap(char **ptr, const char *path);
void ndc_certs_add(const char *path);
void ndc_cert_add(const char *path);
void ndc_close(socket_t fd);

#ifdef __cplusplus
}
#endif

#endif

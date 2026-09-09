#include <ttypt/axil.h>
#include <ttypt/qsys.h>
#include <ttypt/xy-mod.h>

#include "openai_chat.h"
#include "openai_embed.h"
#include "qllm-engine.h"
#include "qllm-pending.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REC_END "\r\n.\r\n"

/* Per-telnet-connection state. LLM state lives in the engine (sessions keyed
 * by the connection's fd); the loop thread only keeps the input line being
 * assembled (for "$ " command execution inside streamed output) and a busy
 * flag so overlapping ask turns are rejected at submit time. */
typedef struct fd_info {
	char			line_buf[BUFSIZ * 4];
	unsigned		line_pos;
	unsigned		busy;
	unsigned long		epoch;	/* bumps on each accept; guards stale jobs */
} fdi_t;

fdi_t fdis[FD_SETSIZE];

size_t crb_len = 0;
char *crb_mapped = NULL;
char *crb_system = NULL;	/* NUL-terminated heap copy of crb.txt */

static char qllm_model_path[BUFSIZ];

pending_t *
make_pending(void *handle, int mode, int sid)
{
	pending_t *p = malloc(sizeof(*p));

	if (p) {
		p->handle = handle;
		p->mode = mode;
		p->sid = sid;
		p->epoch = 0;
	}
	return p;
}

static inline void
append_to_line(fdi_t *fdi, const char *s, size_t len)
{
	size_t space;

	if (len == 0)
		return;

	if (fdi->line_pos >= sizeof(fdi->line_buf) - 1)
		return;

	space = sizeof(fdi->line_buf) - 1 - fdi->line_pos;
	if (len > space)
		len = space;

	memcpy(fdi->line_buf + fdi->line_pos, s, len);
	fdi->line_pos += len;
	fdi->line_buf[fdi->line_pos] = '\0';
}

static inline void
line_reset(fdi_t *fdi)
{
	fdi->line_pos = 0;
	memset(fdi->line_buf, 0, sizeof(fdi->line_buf));
}

static void
cmd_cb(
	int fd,
	char *buf,
	size_t len,
	int ofd __attribute__((unused)))
{
	/* Just echo command output back to client. */
	axil_write(fd, buf, len);
}

static inline void
cmd_exec(int fd, fdi_t *fdi)
{
	char argsbuf[BUFSIZ], *space;
	int argc = 0;
	char *args[8];
	char *pound;

	if (!fdi->line_pos)
		return;

	pound = strstr(fdi->line_buf, "$ ");
	if (!pound)
		return;

	snprintf(argsbuf, sizeof(argsbuf), "%s", pound + 2);
	space = argsbuf;

	do {
		args[argc] = space;
		argc++;
		space = strchr(space, ' ');
		if (!space)
			break;
		*space = '\0';
		space++;
	} while (argc < 7);

	args[argc] = NULL;
	space = strchr(args[argc - 1], '\n');
	if (space)
		*space = '\0';

	axil_exec(fd, args, cmd_cb, NULL, 0);
	axil_write(fd, "\n", 1);
}

static void
do_ASK(int fd, int argc, char *argv[])
{
	fdi_t *fdi = &fdis[fd];
	char *user;
	size_t total = 1;
	int i;

	if (fdi->busy) {
		axil_write(fd, "Busy\n", 5);
		return;
	}

	for (i = 1; i < argc; ++i)
		total += strlen(argv[i]) + (i > 1 ? 1 : 0);

	user = malloc(total);
	if (!user) {
		axil_writef(fd, "Out of memory\n");
		return;
	}

	user[0] = '\0';
	for (i = 1; i < argc; ++i) {
		if (i > 1)
			strcat(user, " ");
		strcat(user, argv[i]);
	}

	fdi->busy = 1;
	{
		pending_t *p = make_pending(NULL, 0, fd);

		if (!p) {
			fdi->busy = 0;
			axil_writef(fd, "Out of memory\n");
			free(user);
			return;
		}
		p->epoch = fdi->epoch;
		if (qllm_engine_chat(fd, user, p, NULL) != 0) {
			free(p);
			fdi->busy = 0;
			axil_writef(fd, "Engine busy\n");
		}
	}
	free(user);
}

static void
do_CHAT(int fd, int argc __attribute__((unused)),
    char *argv[] __attribute__((unused)))
{
	/* Clear the connection's session history + KV in the engine. */
	if (qllm_engine_session_reset(fd, NULL, NULL) != 0)
		qsyslog(QLOG_ERR, "chat reset queue full\n");
}

/* Feed a streamed chunk's newlines through "$ " command execution. */
static void
chat_handle_chunk(int fd, const char *chunk, size_t len)
{
	fdi_t *fdi = &fdis[fd];
	size_t i;

	for (i = 0; i < len; ++i) {
		append_to_line(fdi, &chunk[i], 1);
		if (chunk[i] == '\n') {
			cmd_exec(fd, fdi);
			line_reset(fdi);
		}
	}
}

/* Loop-thread completion drain: complete deferred responses from the engine.
 * Owns/frees the pending_t attached to the job's ud. Meta results (open/reset/
 * close) carry ud = NULL and need no response. */
static void
drain(void *cbud __attribute__((unused)),
    const struct qllm_engine_result *r)
{
	pending_t *p = r->ud;

	if (!p)
		return;

	if (r->kind == QE_JOB_EMBED) {
		if (r->err)
			axil_respond_defer_abort(p->handle);
		else
			axil_respond_defer_finish(p->handle, r->payload);
		free(p);
		return;
	}

	if (r->kind == QE_JOB_CHAT) {
		int fd = p->sid;

		/* The connection moved on (closed and its fd reused): these
		 * results belong to a dead generation — drop them entirely. */
		if (fdis[fd].epoch != p->epoch) {
			free(p);
			return;
		}

		if (r->more) {
			size_t len = r->payload ? strlen(r->payload) : 0;

			if (len)
				axil_write(fd, r->payload, len);
			chat_handle_chunk(fd, r->payload ? r->payload : "", len);
			return;   /* keep pending until the final result */
		}

		/* Final: end the streaming body; the chunks are already sent. */
		axil_write(fd, (void *)REC_END, strlen(REC_END));
		fdis[fd].busy = 0;
		free(p);
		return;
	}

	if (r->kind == QE_JOB_CHAT_ONESHOT) {
		socket_t fd = (socket_t)(intptr_t)p->handle;

		if (p->mode && r->more) {
			axil_write(fd, "data: ", 6);
			if (r->payload)
				axil_write(fd, r->payload, strlen(r->payload));
			axil_write(fd, "\n\n", 2);
			return;   /* keep pending until the final result */
		}

		if (p->mode) {
			if (r->payload) {
				axil_write(fd, "data: ", 6);
				axil_write(fd, r->payload, strlen(r->payload));
				axil_write(fd, "\n\n", 2);
			}
			axil_write(fd, "data: [DONE]\n\n", 15);
			axil_respond_defer_done(p->handle);
		} else if (r->err) {
			axil_respond_defer_abort(p->handle);
		} else {
			axil_respond_defer_finish(p->handle, r->payload);
		}
		free(p);
		return;
	}

	free(p);
}

/* The library's terms (this module) implement the axil host hooks. The engine
 * worker wakes the loop via qllm_engine_wake_fd(); every readable fd also
 * drains whatever completed (idempotent on the loop thread). */
XY_IMPL(int, on_axil_connect, socket_t, fd)
{
	fdis[fd].busy = 0;
	fdis[fd].epoch++;
	line_reset(&fdis[fd]);
	return 0;
}

XY_IMPL(int, on_axil_disconnect, socket_t, fd)
{
	fdi_t *fdi = &fdis[fd];

	/* Release the engine session; the worker frees ctx/history. */
	if (qllm_engine_session_close(fd, NULL, NULL) != 0)
		qsyslog(QLOG_ERR, "session close queue full\n");

	fdi->busy = 0;
	line_reset(fdi);
	return 0;
}

XY_IMPL(int, on_axil_tick, socket_t, fd)
{
	(void)fd;
	qllm_engine_poll(drain, NULL);
	return 0;
}

static void
setup_crb(const char *path)
{
	ssize_t r;

	r = axil_mmap(&crb_mapped, (char *)path);
	if (r > 0) {
		crb_len = (size_t)r;
		crb_system = strndup(crb_mapped, crb_len);
	}
}

static int
qllm_init(void)
{
	const char *model_path = getenv("QLLM_MODEL_PATH");
	const char *crb_path = getenv("QLLM_CRB_PATH");

	if (!model_path || !model_path[0]) {
		qsyslog(QLOG_ERR, "QLLM_MODEL_PATH not set; qllm engine disabled\n");
		return -1;
	}

	snprintf(qllm_model_path, sizeof(qllm_model_path), "%s", model_path);

	setup_crb(crb_path && crb_path[0] ? crb_path : "crb.txt");

	axil_register("ask", do_ASK, CF_NOAUTH | CF_NOTRIM);
	axil_register("chat", do_CHAT, CF_NOAUTH | CF_NOTRIM);

	if (qllm_engine_init(qllm_model_path) != 0) {
		qsyslog(QLOG_ERR, "Failed to init engine\n");
		return -1;
	}

	qllm_engine_set_system(crb_system);

	axil_fd_watch(qllm_engine_wake_fd());

	openai_embed_init(qllm_model_path);
	openai_chat_init(qllm_model_path);

	return 0;
}

XY_MODULE_API void
xy_install(void)
{
	if (qllm_init() != 0)
		qsyslog(QLOG_ERR, "libaxil-qllm: init failed\n");
}

XY_MODULE_API void
xy_shutdown(void)
{
	qllm_engine_shutdown();
	openai_embed_shutdown();
	openai_chat_shutdown();
	free(crb_system);
	crb_system = NULL;
}
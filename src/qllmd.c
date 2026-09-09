#include <ttypt/axil.h>
#include <ttypt/qsys.h>

#include "openai_chat.h"
#include "openai_embed.h"
#include "qllm-engine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

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

/* Per-deferred-response bookkeeping carried in each engine job's ud.
 * Owned by the loop-thread drain callback, which frees it.
 *   - EMBED:      handle = deferred response, mode = 0, sid = 0
 *   - CHAT:       handle = NULL, mode = 0,  sid = the telnet fd
 *   - one-shot:   handle = deferred response, mode = 1 if streaming, sid = 0
 * `epoch` is only meaningful for CHAT: it must match fdis[sid].epoch or the
 * connection was replaced and the (possibly stale) results must be dropped. */
typedef struct pending {
	void 		*handle;
	int   		mode;
	int   		sid;
	unsigned long	epoch;
} pending_t;

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

fdi_t fdis[FD_SETSIZE];

size_t crb_len = 0;
char *crb_mapped = NULL;
char *crb_system = NULL;	/* NUL-terminated heap copy of crb.txt */

static char qllm_model_path[BUFSIZ];

unsigned n_contexts = 1;
unsigned n_ctx = 0;

struct axil_config axil_config = {
	.flags = AXIL_DETACH,
	.port = 4242,
};

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

void cmd_cb(
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

int
axil_accept(int fd)
{
	fdis[fd].busy = 0;
	fdis[fd].epoch++;
	line_reset(&fdis[fd]);
	return 0;
}

void
axil_disconnect(int fd __attribute__((unused)))
{
	fdi_t *fdi = &fdis[fd];

	/* Release the engine session; the worker frees ctx/history. */
	if (qllm_engine_session_close(fd, NULL, NULL) != 0)
		qsyslog(QLOG_ERR, "session close queue full\n");

	fdi->busy = 0;
	line_reset(fdi);
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

/* Strong override of the weak axil_fd_tick (loop thread): the engine's
 * self-pipe woke the loop, so drain whatever completed. */
void
axil_fd_tick(socket_t fd)
{
	(void)fd;
	qllm_engine_poll(drain, NULL);
}

static void
usage(char *prog)
{
	fprintf(stderr, "Usage: %s [-dr?] [-C PATH] [-u USER] [-k PATH] [-c PATH] [-p PORT] MODEL\n", prog);
	fprintf(stderr, "    Options:\n");
	fprintf(stderr, "        -C PATH   changes directory to PATH before starting up.\n");
	fprintf(stderr, "        -u USER   login as USER before starting up.\n");
	fprintf(stderr, "        -k PATH   specify SSL certificate 'key' file\n");
	fprintf(stderr, "        -c PATH   specify SSL certificate 'crt' file\n");
	fprintf(stderr, "        -p PORT   specify server port (defaults to 4242)\n");
	fprintf(stderr, "        -d        don't detach\n");
	fprintf(stderr, "        -r        root multiplex mode\n");
	fprintf(stderr, "        -c SIZE   specify n_ctx (0 - auto)\n");
	fprintf(stderr, "        -n NUM    specify an estimation of concurrent sessions (2)\n");
	fprintf(stderr, "        -?        display this message.\n");
}

static void
setup(const char *model_path)
{
	ssize_t r;

	snprintf(qllm_model_path, sizeof(qllm_model_path), "%s", model_path);

	r = axil_mmap(&crb_mapped, "crb.txt");
	if (r > 0) {
		crb_len = (size_t)r;
		crb_system = strndup(crb_mapped, crb_len);
	}
}

int
main(int argc, char *argv[])
{
	register char c;
	struct stat st;
	char *arg_model;
	char model_path[BUFSIZ];
	FILE *fp;
	char cmd[BUFSIZ];
	char *nl;
	int ret;

	qsys_openlog("qllmd");
	axil_config.port = 4242;

	while ((c = getopt(argc, argv, "?dK:k:C:rp:s:n:c:")) != -1) switch (c) {
		case 'd':
			axil_config.flags &= ~AXIL_DETACH;
			break;

		case 'K':
		case 'k':
			break;

		case 'C':
			axil_config.chroot = strdup(optarg);
			break;

		case 'r':
			axil_config.flags |= AXIL_ROOT;
			break;

		case 'p':
			axil_config.port = atoi(optarg);
			break;

		case 's':
			axil_config.ssl_port = atoi(optarg);
			break;

		case 'n':
			n_contexts = atoi(optarg);
			break;

		case 'c':
			n_ctx = atoi(optarg);
			break;

		default:
			usage(*argv);
			return 1;
	}

	optind = 1;

	while ((c = getopt(argc, argv, "?dK:k:C:rp:s:n:c:")) != -1) switch (c) {
		case 'K':
			axil_certs_add(optarg);
			break;

		case 'k':
			axil_cert_add(optarg);
			break;

		default:
			break;
	}

	arg_model = argv[argc - 1];

	if (stat(arg_model, &st) == 0 && S_ISREG(st.st_mode)) {
		snprintf(model_path, sizeof(model_path), "%s", arg_model);
	} else {
		snprintf(cmd, sizeof(cmd), "qllm-path %s", arg_model);
		fp = popen(cmd, "r");
		CBUG(!fp || !fgets(model_path, sizeof(model_path), fp),
				"Couldn't resolve model\n");
		pclose(fp);

		nl = strchr(model_path, '\n');
		if (nl)
			*nl = '\0';

		arg_model = model_path;
	}

	axil_register("ask", do_ASK, CF_NOAUTH | CF_NOTRIM);
	axil_register("chat", do_CHAT, CF_NOAUTH | CF_NOTRIM);

	setup(arg_model);

	if (qllm_engine_init(qllm_model_path) != 0) {
		qsyslog(QLOG_ERR, "Failed to init engine\n");
		return 1;
	}

	qllm_engine_set_system(crb_system);

	axil_fd_watch(qllm_engine_wake_fd());

	openai_embed_init(qllm_model_path);
	openai_chat_init(qllm_model_path);

	ret = axil_main();

	qllm_engine_shutdown();

	openai_embed_shutdown();
	openai_chat_shutdown();
	free(crb_system);

	return ret;
}
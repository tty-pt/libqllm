#include <ttypt/axil.h>
#include <ttypt/qllm.h>
#include <ttypt/qmap.h>
#include <ttypt/qsys.h>

#include "openai_embed.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_MSGS 64
#define REC_END "\r\n.\r\n"

typedef struct fd_info {
	char			line_buf[BUFSIZ * 4];
	unsigned		line_pos;
	struct qllm_context *	ctx;
	struct qllm_message	msgs[MAX_MSGS];	/* content heap-owned except msgs[0] (crb) */
	size_t			n_msgs;
	char			*prev_prompt;	/* malloc'd: last rendered prompt */
	char			*assistant_buf;	/* malloc'd: in-progress reply */
	size_t			assist_len;
} fdi_t;

typedef struct gen_state {
	int	fd;
	fdi_t *fdi;
} gen_state_t;

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

/*
 * Evict the oldest messages to make room for a new one, keeping the
 * system message (crb) at msgs[0] and the newest MAX_MSGS-1 entries.
 * Forces prev_prompt to NULL so the next turn re-primes from the
 * surviving suffix.
 */
static void
fdi_evict_oldest(fdi_t *fdi)
{
	size_t keep = crb_system ? 1 : 0;
	size_t want = MAX_MSGS - 1;
	size_t evict, i;

	if (fdi->n_msgs < MAX_MSGS)
		return;

	evict = fdi->n_msgs - want;
	if (evict > fdi->n_msgs - keep)
		evict = fdi->n_msgs - keep;

	for (i = keep; i < keep + evict; ++i)
		free((void *)fdi->msgs[i].content);

	for (i = 0; i + keep + evict < fdi->n_msgs; ++i)
		fdi->msgs[keep + i] = fdi->msgs[keep + evict + i];

	fdi->n_msgs -= evict;

	free(fdi->prev_prompt);
	fdi->prev_prompt = NULL;
}

static int
push_msg(fdi_t *fdi, const char *role, const char *content)
{
	char *dup;

	if (fdi->n_msgs >= MAX_MSGS)
		fdi_evict_oldest(fdi);

	dup = strdup(content ? content : "");
	if (!dup)
		return -1;

	fdi->msgs[fdi->n_msgs].role = role;
	fdi->msgs[fdi->n_msgs].content = dup;
	fdi->n_msgs++;
	return 0;
}

/* Free the message contents (keeping the crb system message) and all
 * session buffers; re-establish the system message. */
static inline void
fdi_clear_msgs(fdi_t *fdi)
{
	size_t i;

	for (i = 0; i < fdi->n_msgs; ++i)
		if (fdi->msgs[i].content != crb_system)
			free((void *)fdi->msgs[i].content);

	fdi->n_msgs = 0;
	if (crb_system) {
		fdi->msgs[0].role = "system";
		fdi->msgs[0].content = crb_system;
		fdi->n_msgs = 1;
	}

	free(fdi->prev_prompt);
	fdi->prev_prompt = NULL;
	free(fdi->assistant_buf);
	fdi->assistant_buf = NULL;
	fdi->assist_len = 0;
}

static inline void
fdi_reset(fdi_t *fdi)
{
	fdi_clear_msgs(fdi);
	line_reset(fdi);
}

static inline int
fdi_ensure_ctx(fdi_t *fdi)
{
	struct qllm_config cfg;

	if (fdi->ctx)
		return 0;

	cfg.model_path = qllm_model_path;
	cfg.n_ctx = n_ctx;
	cfg.n_threads = 0;
	cfg.max_offload_bytes = 0;
	cfg.n_contexts = n_contexts;

	fdi->ctx = qllm_create(&cfg);
	if (!fdi->ctx) {
		qsyslog(QLOG_ERR, "Failed to init qllm context\n");
		return -1;
	}

	return 0;
}

/*
 * Echo a generated chunk to the client, buffer it as the current
 * assistant reply, and process newlines for "$ " command execution.
 */
static void
qllm_chat_cb(void *user, const char *chunk, size_t len)
{
	gen_state_t *st = user;
	fdi_t *fdi = st->fdi;
	int fd = st->fd;
	size_t i;

	axil_write(fd, (void *)chunk, len);

	if (len) {
		char *nb = realloc(fdi->assistant_buf,
		    fdi->assist_len + len + 1);
		if (nb) {
			memcpy(nb + fdi->assist_len, chunk, len);
			fdi->assist_len += len;
			nb[fdi->assist_len] = '\0';
			fdi->assistant_buf = nb;
		}
	}

	for (i = 0; i < len; ++i) {
		append_to_line(fdi, &chunk[i], 1);
		if (chunk[i] == '\n') {
			cmd_exec(fd, fdi);
			line_reset(fdi);
		}
	}
}

static void
do_ASK(int fd, int argc, char *argv[])
{
	fdi_t *fdi = &fdis[fd];
	gen_state_t st;
	char *user;
	size_t total = 1;
	char *rendered = NULL;
	size_t rlen = 0;
	int i, ret;

	if (fdi_ensure_ctx(fdi) != 0)
		return;

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

	if (push_msg(fdi, "user", user) != 0) {
		free(user);
		axil_writef(fd, "Out of memory\n");
		return;
	}
	free(user);

	st.fd = fd;
	st.fdi = fdi;

	ret = qllm_chat(fdi->ctx, fdi->msgs, fdi->n_msgs,
	    fdi->prev_prompt, qllm_chat_cb, &st);
	if (ret != 0)
		qsyslog(QLOG_ERR, "qllm_chat failed\n");

	/* Fold the completed reply into the conversation history. */
	if (fdi->assistant_buf)
		push_msg(fdi, "assistant", fdi->assistant_buf);
	free(fdi->assistant_buf);
	fdi->assistant_buf = NULL;
	fdi->assist_len = 0;

	/* Recompute prev_prompt as the exact state after this turn. */
	if (qllm_render(fdi->ctx, fdi->msgs, fdi->n_msgs,
	    false, &rendered, &rlen) == 0) {
		free(fdi->prev_prompt);
		fdi->prev_prompt = rendered;
	}

	axil_write(fd, (void *)REC_END, strlen(REC_END));
}

static void
do_CHAT(int fd, int argc __attribute__((unused)),
    char *argv[] __attribute__((unused)))
{
	fdi_t *fdi = &fdis[fd];

	/* Clear history, keep the context (KV is reset below). */
	fdi_reset(fdi);

	/* Drop the previous KV cache but keep the shared model handle. */
	if (fdi->ctx)
		qllm_reset(fdi->ctx);
}

int
axil_accept(int fd)
{
	fdi_reset(&fdis[fd]);
	return 0;
}

void
axil_disconnect(int fd __attribute__((unused)))
{
	fdi_t *fdi = &fdis[fd];

	if (fdi->ctx)
		qllm_free(fdi->ctx);
	fdi->ctx = NULL;
	fdi_reset(fdi);
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
	int ret, i;

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

	openai_embed_init(qllm_model_path);

	ret = axil_main();

	for (i = 0; i < FD_SETSIZE; ++i) {
		if (fdis[i].ctx) {
			qllm_free(fdis[i].ctx);
			fdis[i].ctx = NULL;
		}
	}

	openai_embed_shutdown();
	free(crb_system);

	return ret;
}
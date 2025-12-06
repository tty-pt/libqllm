#include <ttypt/ndc.h>
#include <ttypt/qmap.h>
#include <ttypt/qsys.h>
#include "./../include/ttypt/qllm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define DEFAULT_SEQ_MAX 4
#define MAX_TOKENS 1024
#define MAX_MEMORY (MAX_TOKENS * 10)
#define MAX_TOKEN_LEN 64

struct qllm_context;

typedef struct fd_info {
	char			line_buf[BUFSIZ * 4];
	struct qllm_context *	ctx;
	unsigned		end_pos;
	unsigned		line_pos;
} fdi_t;

fdi_t fdis[FD_SETSIZE], general;

const char *start = "<|im_start|>";

const char *end = "<|im_end|>";
const unsigned end_len = 10;

size_t crb_len = 0;
char *crb = NULL;

static char qllm_model_path[BUFSIZ];

typedef struct gen_state {
	int	fd;
	fdi_t *fdi;
	int	stop;
} gen_state_t;

struct ndc_config ndc_config = {
	.flags = NDC_DETACH,
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
reset_fdi(fdi_t *fdi)
{
	fdi->line_pos = 0;
	fdi->end_pos = 0;
	memset(fdi->line_buf, 0, sizeof(fdi->line_buf));
}

void cmd_cb(
	int fd,
	char *buf,
	size_t len,
	int ofd __attribute__((unused)))
{
	/* Just echo command output back to client. */
	ndc_write(fd, buf, len);
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

	ndc_exec(fd, args, cmd_cb, NULL, 0);
	ndc_write(fd, "\n", 1);
}

/*
 * Process a text chunk from qllm and stream it to the client,
 * handling:
 *  - detection of the "<|im_end|>" marker (not printed),
 *  - line buffering and command execution on newline.
 */
static inline void
process_chunk(gen_state_t *st, const char *chunk, size_t len)
{
	fdi_t *fdi = st->fdi;
	int fd = st->fd;

	for (size_t i = 0; i < len; ++i) {
		char ch = chunk[i];

		/* Normal character */
		ndc_write(fd, &ch, 1);
		append_to_line(fdi, &ch, 1);

        /* Command execution on newline */
		if (ch == '\n') {
			cmd_exec(fd, fdi);
			reset_fdi(fdi);
		}
	}
}


static inline int
inference(int fd, fdi_t *fdi)
{
	char	buf[MAX_TOKEN_LEN];
	int	ret;
	size_t	buflen;
	char	*eoim;

	/* Get next piece of text from qllm */
	ret = qllm_next(fdi->ctx, buf, sizeof(buf));
	if (ret < 0)
		return 0;	/* error -> stop */

	if (ret == 0)
		return 0;	/* EOS -> stop */

	buflen = (size_t)ret;

	/* Mesma lógica de antes para detectar "<|im_end|>" */
	eoim = strchr(buf, *(end + fdi->end_pos));
	if (eoim && (eoim - buf) <= end_len - fdi->end_pos) {
		size_t clen = buflen - (size_t)(eoim - buf);

		if (strncmp(eoim, end + fdi->end_pos, clen))
			goto end;

		fdi->end_pos += clen;

		if (fdi->end_pos < end_len)
			return 1;

		fdi->end_pos = 0;
		return 0;
	}

end:
	if (fdi->end_pos) {
		ndc_write(fd, (void *)end, fdi->end_pos);
		append_to_line(fdi, end, fdi->end_pos);
		fdi->end_pos = 0;
	}

	ndc_write(fd, buf, buflen);
	append_to_line(fdi, buf, buflen);

	if (strrchr(buf, '\n')) {
		cmd_exec(fd, fdi);
		fdi->line_pos = 0;
	}

	return 1;
}

void
generate(int fd, const char *prompt)
{
	fdi_t	*fdi = &fdis[fd];
	int	 step;
	int	 max_gen = MAX_TOKEN_LEN;

	/* Prime qllm context with the full prompt */
	if (qllm_prime(fdi->ctx, prompt) < 0) {
		qsyslog(QLOG_ERR, "qllm_prime failed\n");
		return;
	}

	fdi->line_pos = 0;
	fdi->end_pos = 0;

	for (step = 0;
	     step < max_gen && inference(fd, fdi);
	     ++step)
		;

	cmd_exec(fd, fdi);
	fdi->line_pos = 0;
}

void
do_ASK(int fd, int argc, char *argv[])
{
	fdi_t *fdi __attribute__((unused)) = &fdis[fd];
	char buf[BUFSIZ * 2], *b = buf;
	int i, ret;

	b += snprintf(b, sizeof(buf) - (b - buf), "%suser\n", start);
	for (i = 1; i < argc; i++) {
		ret = snprintf(b, sizeof(buf) - (b - buf), " %s", argv[i]);
		if (ret < 0 || (size_t)ret >= sizeof(buf) - (size_t)(b - buf)) {
			ndc_writef(fd, "Buffer size exceeded\n");
			return;
		}
		b += ret;
	}
	b += snprintf(b, sizeof(buf) - (b - buf), "%s\n%sassistant\n ", end, start);

	generate(fd, buf);
	ndc_writef(fd, "%s\n", end);
}

static inline void
fdi_init(fdi_t *fdi)
{
	struct qllm_config cfg = {
		.model_path = qllm_model_path,
		.n_ctx = 0,
		.n_threads = 0,
		.auto_ngl_max = MAX_MEMORY,
	};

	if (fdi->ctx && fdi->ctx != general.ctx)
		qllm_free(fdi->ctx);

	fdi->ctx = qllm_create(&cfg);
	/* fdi->ctx = general.ctx; */
	if (!fdi->ctx)
		qsyslog(QLOG_ERR, "Failed to init qllm context\n");

	reset_fdi(fdi);
}

void
do_CHAT(int fd, int argc __attribute__((unused)), char *argv[] __attribute__((unused)))
{
	fdi_init(&fdis[fd]);
}

struct cmd_slot cmds[] = {
	{
		.name = "ask",
		.cb = &do_ASK,
		.flags = CF_NOAUTH | CF_NOTRIM,
	}, {
		.name = "chat",
		.cb = &do_CHAT,
		.flags = CF_NOAUTH | CF_NOTRIM,
	}, {
		.name = NULL
	}
};

int
ndc_accept(int fd)
{
	fdis[fd].ctx = general.ctx;
	reset_fdi(&fdis[fd]);
	return 0;
}

void
ndc_disconnect(int fd __attribute__((unused)))
{
	fdi_t *fdi = &fdis[fd];

	if (fdi->ctx && fdi->ctx != general.ctx)
		qllm_free(fdi->ctx);

	fdi->ctx = NULL;
	reset_fdi(fdi);
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
	fprintf(stderr, "        -?        display this message.\n");
}

static void
setup(const char *model_path)
{
	struct qllm_config cfg = {
		.model_path = model_path,
		.n_ctx = 0,		/* use qllm defaults */
		.n_threads = 0,		/* use qllm defaults */
	};

	snprintf(qllm_model_path, sizeof(qllm_model_path), "%s", model_path);

	general.ctx = qllm_create(&cfg);
	CBUG(!general.ctx,
			"Failed to create qllm context\n");

	reset_fdi(&general);

	crb_len = (size_t)ndc_mmap(&crb, "crb.txt");
	(void)crb_len;
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

	while ((c = getopt(argc, argv, "?dK:k:C:rp:s:")) != -1) switch (c) {
		case 'd':
			ndc_config.flags &= ~NDC_DETACH;
			break;

		case 'K':
		case 'k':
			break;

		case 'C':
			ndc_config.chroot = strdup(optarg);
			break;

		case 'r':
			ndc_config.flags |= NDC_ROOT;
			break;

		case 'p':
			ndc_config.port = atoi(optarg);
			break;

		case 's':
			ndc_config.ssl_port = atoi(optarg);
			break;

		default:
			usage(*argv);
			return 1;
	}

	optind = 1;

	while ((c = getopt(argc, argv, "?dK:k:C:rp:s:")) != -1) switch (c) {
		case 'K':
			ndc_certs_add(optarg);
			break;

		case 'k':
			ndc_cert_add(optarg);
			break;

		default:
			break;
	}

	arg_model = argv[argc - 1];

	if (stat(arg_model, &st) == 0 && S_ISREG(st.st_mode)) {
		snprintf(model_path, sizeof(model_path), "%s", arg_model);
	} else {
		snprintf(cmd, sizeof(cmd), "llm-path %s", arg_model);
		fp = popen(cmd, "r");
		CBUG(!fp || !fgets(model_path, sizeof(model_path), fp),
				"Couldn't resolve model\n");
		pclose(fp);

		nl = strchr(model_path, '\n');
		if (nl)
			*nl = '\0';

		arg_model = model_path;
	}

	ndc_register("ask", do_ASK, CF_NOAUTH | CF_NOTRIM);
	ndc_register("chat", do_CHAT, CF_NOAUTH | CF_NOTRIM);

	setup(arg_model);

	ret = ndc_main();

	if (general.ctx)
		qllm_free(general.ctx);

	return ret;
}

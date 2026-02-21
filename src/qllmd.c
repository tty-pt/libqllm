#include <ttypt/ndc.h>
#include <ttypt/qmap.h>
#include <ttypt/qsys.h>
#include "./../include/ttypt/qllm.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define DEFAULT_SEQ_MAX 4
#define MAX_TOKENS 1024
#define MAX_MEMORY (MAX_TOKENS * 10)
#define FEAT_GENERAL 0

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
static char qllm_model_id[BUFSIZ];

typedef struct gen_state {
	int	fd;
	fdi_t *fdi;
	int	stop;
} gen_state_t;

struct ndc_config ndc_config = {
	.flags = NDC_DETACH,
	.port = 4242,
};

#if FEAT_GENERAL
unsigned n_contexts = 2;
#else
unsigned n_contexts = 1;
#endif
unsigned n_ctx = 0;

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
	char	buf[MAX_MEMORY];
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
	int	 max_gen = MAX_MEMORY;

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
		.n_ctx = n_ctx,
		.n_threads = 0,
		.n_contexts = n_contexts,
	};

	if (fdi->ctx && fdi->ctx != general.ctx)
		qllm_free(fdi->ctx);

	fprintf(stderr, "N_CONTEXTS! %d\n", cfg.n_contexts);
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

/* ---- Minimal JSON helpers for OpenAI-compatible endpoint ---- */

static const char *
json_skip_ws(const char *p)
{
	while (*p && isspace((unsigned char)*p))
		p++;
	return p;
}

static const char *
json_str_end(const char *p)
{
	while (*p) {
		if (*p == '\\') {
			p++;
			if (*p) p++;
		} else if (*p == '"') {
			return p;
		} else {
			p++;
		}
	}
	return NULL;
}

static void
json_unescape(char *s, size_t len)
{
	char *r = s, *w = s, *end = s + len;

	while (r < end) {
		if (*r == '\\' && r + 1 < end) {
			r++;
			switch (*r) {
			case '"':  *w++ = '"';  break;
			case '\\': *w++ = '\\'; break;
			case '/':  *w++ = '/';  break;
			case 'n':  *w++ = '\n'; break;
			case 'r':  *w++ = '\r'; break;
			case 't':  *w++ = '\t'; break;
			case 'b':  *w++ = '\b'; break;
			case 'f':  *w++ = '\f'; break;
			case 'u':
				if (r + 4 < end) r += 4;
				*w++ = '?';
				break;
			default:   *w++ = *r;   break;
			}
			r++;
		} else {
			*w++ = *r++;
		}
	}
	*w = '\0';
}

static int
json_get_str(const char *json, size_t len, const char *key,
	     char *out, size_t outsz)
{
	char search[128];
	const char *p = json, *end = json + len;
	const char *vstart, *vend;
	size_t vlen;

	snprintf(search, sizeof(search), "\"%s\"", key);
	while (p < end && (p = strstr(p, search)) != NULL) {
		if (p >= end) break;
		p += strlen(search);
		p = json_skip_ws(p);
		if (p >= end || *p != ':') continue;
		p++;
		p = json_skip_ws(p);
		if (p >= end || *p != '"') continue;
		vstart = p + 1;
		vend = json_str_end(vstart);
		if (!vend) continue;
		vlen = (size_t)(vend - vstart);
		if (vlen >= outsz) vlen = outsz - 1;
		memcpy(out, vstart, vlen);
		out[vlen] = '\0';
		json_unescape(out, vlen);
		return 1;
	}
	return 0;
}

static int
json_get_bool(const char *json, size_t len, const char *key)
{
	char search[128];
	const char *p = json, *end = json + len;

	snprintf(search, sizeof(search), "\"%s\"", key);
	while (p < end && (p = strstr(p, search)) != NULL) {
		if (p >= end) break;
		p += strlen(search);
		p = json_skip_ws(p);
		if (p >= end || *p != ':') continue;
		p++;
		p = json_skip_ws(p);
		return strncmp(p, "true", 4) == 0;
	}
	return 0;
}

#define OPENAI_MAX_MSG     32
#define OPENAI_MAX_CONTENT 16384

struct openai_msg {
	char role[32];
	char content[OPENAI_MAX_CONTENT];
};

static int
json_get_messages(const char *json, size_t json_len,
		  struct openai_msg *msgs, int maxmsgs)
{
	const char *p = json, *end = json + json_len;
	const char *obj_start;
	int depth, n = 0;
	int found = 0;

	while (p < end && (p = strstr(p, "\"messages\"")) != NULL) {
		p += strlen("\"messages\"");
		p = json_skip_ws(p);
		if (p >= end || *p != ':') continue;
		p++;
		p = json_skip_ws(p);
		if (p >= end || *p != '[') continue;
		p++;
		found = 1;
		break;
	}
	if (!found || !p || p >= end) return 0;

	while (p < end && n < maxmsgs) {
		p = json_skip_ws(p);
		if (p >= end || *p == ']') break;
		if (*p != '{') { p++; continue; }
		obj_start = p + 1;
		depth = 1;
		p++;
		while (p < end && depth > 0) {
			if (*p == '\\') { p += 2; continue; }
			if (*p == '"') {
				p++;
				while (p < end) {
					if (*p == '\\') {
						p++;
						if (p < end) p++;
						continue;
					}
					if (*p++ == '"') break;
				}
				continue;
			}
			if (*p == '{') depth++;
			else if (*p == '}') depth--;
			p++;
		}

		size_t objlen = (size_t)(p - 1 - obj_start);
		if (json_get_str(obj_start, objlen, "role",
				 msgs[n].role, sizeof(msgs[n].role))) {
			if (!json_get_str(obj_start, objlen, "content",
					  msgs[n].content,
					  sizeof(msgs[n].content))) {
				/* content may be an array; find first "text" */
				const char *ck = strstr(obj_start,
							"\"content\"");
				if (ck && ck < obj_start + objlen) {
					ck += strlen("\"content\"");
					ck = json_skip_ws(ck);
					if (ck < obj_start + objlen &&
					    *ck == ':') {
						ck++;
						ck = json_skip_ws(ck);
						if (ck < obj_start + objlen &&
						    *ck == '[') {
							size_t arrlen =
							    (size_t)(obj_start +
							    objlen - ck);
							json_get_str(ck, arrlen,
							    "text",
							    msgs[n].content,
							    sizeof(msgs[n]
							    .content));
						}
					}
				}
			}
			n++;
		}
		p = json_skip_ws(p);
		if (p < end && *p == ',') p++;
	}
	return n;
}

static int
build_chatml_prompt(struct openai_msg *msgs, int n,
		    char *buf, size_t bufsz)
{
	char *b = buf;
	size_t rem = bufsz;
	int i, ret;

	for (i = 0; i < n; i++) {
		ret = snprintf(b, rem, "<|im_start|>%s\n%s<|im_end|>\n",
			       msgs[i].role, msgs[i].content);
		if (ret < 0 || (size_t)ret >= rem)
			return -1;
		b += ret;
		rem -= (size_t)ret;
	}
	ret = snprintf(b, rem, "<|im_start|>assistant\n");
	if (ret < 0 || (size_t)ret >= rem)
		return -1;
	return 0;
}

struct openai_stream {
	int	 fd;
	char	 id[64];
	time_t	 created;
};

/* JSON-escape src[0..len) into dst (capacity dstsz). Returns bytes written. */
static size_t
json_escape(const char *src, size_t len, char *dst, size_t dstsz)
{
	size_t i, j = 0;

	for (i = 0; i < len && j < dstsz - 2; i++) {
		unsigned char ch = (unsigned char)src[i];
		if      (ch == '"')  { dst[j++] = '\\'; dst[j++] = '"'; }
		else if (ch == '\\') { dst[j++] = '\\'; dst[j++] = '\\'; }
		else if (ch == '\n') { dst[j++] = '\\'; dst[j++] = 'n'; }
		else if (ch == '\r') { dst[j++] = '\\'; dst[j++] = 'r'; }
		else if (ch == '\t') { dst[j++] = '\\'; dst[j++] = 't'; }
		else                 { dst[j++] = (char)ch; }
	}
	dst[j] = '\0';
	return j;
}

static void
openai_sse_cb(void *user, const char *chunk, size_t len)
{
	struct openai_stream *st = user;
	char escaped[1024];

	json_escape(chunk, len, escaped, sizeof(escaped));

	ndc_writef(st->fd,
		"data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
		"\"created\":%ld,\"model\":\"%s\","
		"\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},"
		"\"finish_reason\":null}]}\n\n",
		st->id, (long)st->created, qllm_model_id, escaped);
}

static void
handle_get_models(socket_t fd, char *body __attribute__((unused)))
{
	char resp[512];
	int resplen;

	resplen = snprintf(resp, sizeof(resp),
		"{\"object\":\"list\",\"data\":[{\"id\":\"%s\","
		"\"object\":\"model\",\"owned_by\":\"user\"}]}\n",
		qllm_model_id);
	ndc_writef(fd,
		"HTTP/1.1 200 OK\r\n"
		"Content-Type: application/json\r\n"
		"Access-Control-Allow-Origin: *\r\n"
		"Content-Length: %d\r\n"
		"Connection: close\r\n"
		"\r\n",
		resplen);
	ndc_write(fd, resp, resplen);
	ndc_close(fd);
}

static void
handle_chat_completions(socket_t fd, char *body)
{
	struct openai_msg *msgs;
	char *prompt;
	int n_msgs, is_stream;
	size_t bodylen, prompt_cap;
	fdi_t *fdi = &fdis[fd];

	if (!body || !*body) {
		ndc_writef(fd, "HTTP/1.1 400 Bad Request\r\n"
			"Content-Length: 0\r\n\r\n");
		ndc_close(fd);
		return;
	}

	msgs = calloc(OPENAI_MAX_MSG, sizeof(*msgs));
	prompt_cap = (size_t)OPENAI_MAX_MSG * OPENAI_MAX_CONTENT + 256;
	prompt = malloc(prompt_cap);
	if (!msgs || !prompt) {
		free(msgs);
		free(prompt);
		ndc_writef(fd, "HTTP/1.1 500 Internal Server Error\r\n"
			"Content-Length: 0\r\n\r\n");
		ndc_close(fd);
		return;
	}

	bodylen = strlen(body);
	n_msgs = json_get_messages(body, bodylen, msgs, OPENAI_MAX_MSG);
	if (n_msgs <= 0) {
		free(msgs);
		free(prompt);
		ndc_writef(fd, "HTTP/1.1 400 Bad Request\r\n"
			"Content-Length: 0\r\n\r\n");
		ndc_close(fd);
		return;
	}

	is_stream = json_get_bool(body, bodylen, "stream");

	if (build_chatml_prompt(msgs, n_msgs, prompt, prompt_cap) < 0) {
		free(msgs);
		free(prompt);
		ndc_writef(fd, "HTTP/1.1 400 Bad Request\r\n"
			"Content-Length: 0\r\n\r\n");
		ndc_close(fd);
		return;
	}
	free(msgs);

	if (!fdi->ctx)
		fdi_init(fdi);

	if (!fdi->ctx) {
		free(prompt);
		ndc_writef(fd, "HTTP/1.1 500 Internal Server Error\r\n"
			"Content-Length: 0\r\n\r\n");
		ndc_close(fd);
		return;
	}

	if (is_stream) {
		struct openai_stream sst;
		time_t now = time(NULL);

		ndc_writef(fd,
			"HTTP/1.1 200 OK\r\n"
			"Content-Type: text/event-stream\r\n"
			"Cache-Control: no-cache\r\n"
			"Access-Control-Allow-Origin: *\r\n"
			"Connection: close\r\n"
			"\r\n");

		snprintf(sst.id, sizeof(sst.id), "chatcmpl-%ld", (long)now);
		sst.fd = fd;
		sst.created = now;

		/* initial delta with role */
		ndc_writef(fd,
			"data: {\"id\":\"%s\",\"object\":"
			"\"chat.completion.chunk\","
			"\"created\":%ld,\"model\":\"%s\","
			"\"choices\":[{\"index\":0,"
			"\"delta\":{\"role\":\"assistant\"},"
			"\"finish_reason\":null}]}\n\n",
			sst.id, (long)sst.created, qllm_model_id);

		qllm_generate_stream(fdi->ctx, prompt, openai_sse_cb, &sst);

		ndc_writef(fd,
			"data: {\"id\":\"%s\",\"object\":"
			"\"chat.completion.chunk\","
			"\"created\":%ld,\"model\":\"%s\","
			"\"choices\":[{\"index\":0,\"delta\":{},"
			"\"finish_reason\":\"stop\"}]}\n\n"
			"data: [DONE]\n\n",
			sst.id, (long)sst.created, qllm_model_id);
	} else {
		char *out = malloc(MAX_MEMORY);
		long outlen;
		size_t esc_len;
		char *esc;
		time_t now;
		int pfx_len, body_len;
		const char *sfx =
			"\"},\"finish_reason\":\"stop\"}]}\n";

		if (!out) {
			free(prompt);
			ndc_writef(fd,
				"HTTP/1.1 500 Internal Server Error\r\n"
				"Content-Length: 0\r\n\r\n");
			ndc_close(fd);
			return;
		}

		outlen = qllm_generate(fdi->ctx, prompt, out, MAX_MEMORY);
		if (outlen < 0) {
			free(prompt);
			free(out);
			ndc_writef(fd,
				"HTTP/1.1 500 Internal Server Error\r\n"
				"Content-Length: 0\r\n\r\n");
			ndc_close(fd);
			return;
		}

		/* worst-case: every byte expands to 2 chars */
		esc = malloc((size_t)outlen * 2 + 1);
		if (!esc) {
			free(prompt);
			free(out);
			ndc_writef(fd,
				"HTTP/1.1 500 Internal Server Error\r\n"
				"Content-Length: 0\r\n\r\n");
			ndc_close(fd);
			return;
		}

		esc_len = json_escape(out, (size_t)outlen,
				      esc, (size_t)outlen * 2 + 1);
		free(out);

		now = time(NULL);
		pfx_len = snprintf(NULL, 0,
			"{\"id\":\"chatcmpl-%ld\","
			"\"object\":\"chat.completion\","
			"\"created\":%ld,\"model\":\"%s\","
			"\"choices\":[{\"index\":0,"
			"\"message\":{\"role\":\"assistant\","
			"\"content\":\"",
			(long)now, (long)now, qllm_model_id);
		body_len = pfx_len + (int)esc_len + (int)strlen(sfx);

		ndc_writef(fd,
			"HTTP/1.1 200 OK\r\n"
			"Content-Type: application/json\r\n"
			"Access-Control-Allow-Origin: *\r\n"
			"Content-Length: %d\r\n"
			"Connection: close\r\n"
			"\r\n",
			body_len);
		ndc_writef(fd,
			"{\"id\":\"chatcmpl-%ld\","
			"\"object\":\"chat.completion\","
			"\"created\":%ld,\"model\":\"%s\","
			"\"choices\":[{\"index\":0,"
			"\"message\":{\"role\":\"assistant\","
			"\"content\":\"",
			(long)now, (long)now, qllm_model_id);
		ndc_write(fd, esc, esc_len);
		ndc_write(fd, (void *)sfx, strlen(sfx));
		free(esc);
	}

	free(prompt);
	ndc_close(fd);
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
#if FEAT_GENERAL
	fdis[fd].ctx = general.ctx;
#else
	reset_fdi(&fdis[fd]);
#endif
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
	fprintf(stderr, "        -c SIZE   specify n_ctx (0 - auto)\n");
	fprintf(stderr, "        -n NUM    specify an estimation of concurrent sessions (2)\n");
	fprintf(stderr, "        -?        display this message.\n");
}

static void
setup(const char *model_path)
{
#if FEAT_GENERAL
	struct qllm_config cfg = {
		.model_path = model_path,
		.n_ctx = n_ctx,
		.n_threads = 0,
		.n_contexts = n_contexts,
	};

	general.ctx = qllm_create(&cfg);
	CBUG(!general.ctx,
			"Failed to create qllm context\n");

	reset_fdi(&general);
#endif

	snprintf(qllm_model_path, sizeof(qllm_model_path), "%s", model_path);

	/* Derive model ID: basename without .gguf extension */
	{
		const char *base = strrchr(model_path, '/');
		char *ext;

		base = base ? base + 1 : model_path;
		snprintf(qllm_model_id, sizeof(qllm_model_id), "%s", base);
		ext = strrchr(qllm_model_id, '.');
		if (ext && strcmp(ext, ".gguf") == 0)
			*ext = '\0';
	}

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
	ndc_config.port = 4242;

	while ((c = getopt(argc, argv, "?dK:k:C:rp:s:n:c:")) != -1) switch (c) {
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

	ndc_register("ask", do_ASK, CF_NOAUTH | CF_NOTRIM);
	ndc_register("chat", do_CHAT, CF_NOAUTH | CF_NOTRIM);
	ndc_register("GET", do_GET, CF_NOAUTH | CF_NOTRIM);
	ndc_register("POST", do_POST, CF_NOAUTH | CF_NOTRIM);
	ndc_register_handler("/v1/models", handle_get_models);
	ndc_register_handler("/v1/chat/completions", handle_chat_completions);

	setup(arg_model);

	ret = ndc_main();

	if (general.ctx)
		qllm_free(general.ctx);

	return ret;
}

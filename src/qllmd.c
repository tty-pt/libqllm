#define _GNU_SOURCE  /* for strcasestr */
#include <ttypt/ndc.h>
#include <ttypt/qmap.h>
#include <ttypt/qsys.h>
#include "./../include/ttypt/qllm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>  /* for strcasestr on some systems */
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <cJSON.h>

#define DEFAULT_SEQ_MAX 4
#define MAX_TOKENS 1024
#define MAX_MEMORY 512  /* Reasonable max output tokens */
/* FEAT_GENERAL disabled due to crash in llama_batch_free with shared context.
 * For now, only one connection at a time is supported with Vulkan backend. */
#define FEAT_GENERAL 0

struct qllm_context;

typedef struct {
	unsigned tail;
} token_queue_t;

typedef struct fd_info {
	char			line_buf[BUFSIZ * 4];
	struct qllm_context *	ctx;
	unsigned		line_pos;
	token_queue_t		queue;
} fdi_t;

fdi_t fdis[FD_SETSIZE], general;

const char delimiter = 4;

size_t crb_len = 0;
char *crb = NULL;

static char qllm_model_path[BUFSIZ];
static char system_prompt[BUFSIZ] = "";  /* Optional system prompt */

/* Chat template types */
typedef enum {
	TEMPLATE_GENERIC,
	TEMPLATE_PHI3,
	TEMPLATE_MISTRAL,
	TEMPLATE_GEMMA,
} chat_template_t;

static chat_template_t model_template = TEMPLATE_GENERIC;

/* Template names for info command */
static const char *template_names[] = {
	[TEMPLATE_GENERIC] = "generic",
	[TEMPLATE_PHI3]    = "phi3",
	[TEMPLATE_MISTRAL] = "mistral",
	[TEMPLATE_GEMMA]   = "gemma",
};

/* Detect model type from path */
static chat_template_t
detect_model_template(const char *path)
{
	if (!path)
		return TEMPLATE_GENERIC;
	
	/* Simple detection based on model filename */
	if (strcasestr(path, "phi-3") || strcasestr(path, "phi3"))
		return TEMPLATE_PHI3;
	else if (strcasestr(path, "mistral"))
		return TEMPLATE_MISTRAL;
	else if (strcasestr(path, "gemma"))
		return TEMPLATE_GEMMA;
	
	return TEMPLATE_GENERIC;
}

/* Format prompt according to model's chat template */
static int
format_prompt(char *buf, size_t bufsize, const char *user_msg, chat_template_t template)
{
	int ret;
	int has_system = (system_prompt[0] != '\0');
	
	switch (template) {
	case TEMPLATE_PHI3:
		/* Phi-3 format with optional system: <|system|>\n{sys}<|end|>\n<|user|>\n{msg}<|end|>\n<|assistant|>\n */
		if (has_system)
			ret = snprintf(buf, bufsize, "<|system|>\n%s<|end|>\n<|user|>\n%s<|end|>\n<|assistant|>\n", 
			              system_prompt, user_msg);
		else
			ret = snprintf(buf, bufsize, "<|user|>\n%s<|end|>\n<|assistant|>\n", user_msg);
		break;
		
	case TEMPLATE_MISTRAL:
		/* Mistral format: [INST] {sys}\n\n{msg} [/INST] */
		if (has_system)
			ret = snprintf(buf, bufsize, "[INST] %s\n\n%s [/INST]", system_prompt, user_msg);
		else
			ret = snprintf(buf, bufsize, "[INST] %s [/INST]", user_msg);
		break;
		
	case TEMPLATE_GEMMA:
		/* Gemma format: <start_of_turn>user\n{sys}\n\n{msg}<end_of_turn>\n<start_of_turn>model\n */
		if (has_system)
			ret = snprintf(buf, bufsize, "<start_of_turn>user\n%s\n\n%s<end_of_turn>\n<start_of_turn>model\n", 
			              system_prompt, user_msg);
		else
			ret = snprintf(buf, bufsize, "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", user_msg);
		break;
		
	case TEMPLATE_GENERIC:
	default:
		/* Generic format: \n<delimiter>\n{sys?}user:\n{msg}<delimiter>\nassistant:\n */
		if (has_system)
			ret = snprintf(buf, bufsize, "%c\nsystem:\n%s%c\nuser:\n%s%c\nassistant:\n", 
			              delimiter, system_prompt, delimiter, user_msg, delimiter);
		else
			ret = snprintf(buf, bufsize, "%c\nuser:\n%s%c\nassistant:\n", delimiter, user_msg, delimiter);
		break;
	}
	
	if (ret < 0 || (size_t)ret >= bufsize)
		return -1;
	
	return ret;
}

/* Calculate required buffer size for conversation formatting */
static size_t
calc_conversation_size(cJSON *messages, cJSON *tools, chat_template_t template)
{
	size_t total_size = 0;
	int num_messages = cJSON_GetArraySize(messages);
	int i;
	
	/* Sum content lengths */
	for (i = 0; i < num_messages; i++) {
		cJSON *msg = cJSON_GetArrayItem(messages, i);
		cJSON *content = cJSON_GetObjectItem(msg, "content");
		if (content && content->valuestring) {
			total_size += strlen(content->valuestring);
		}
	}
	
	/* Add template overhead (200 bytes per message for markers) */
	total_size += num_messages * 200;
	
	/* Add global system prompt if present */
	if (system_prompt[0] != '\0') {
		total_size += strlen(system_prompt) + 100;
	}
	
	/* Add tools size if present */
	if (tools && cJSON_IsArray(tools)) {
		int num_tools = cJSON_GetArraySize(tools);
		total_size += num_tools * 512; /* Estimate */
	}
	
	/* Add 20% safety margin, minimum 4KB */
	total_size = (total_size * 120) / 100;
	if (total_size < 4096)
		total_size = 4096;
	
	return total_size;
}

/* Format tools to text description for prompt injection
 * Returns: allocated buffer with formatted tools, or NULL if no tools
 * Caller must free() the returned buffer
 */
static char *
format_tools(cJSON *tools)
{
	char *buf;
	size_t bufsize, pos = 0;
	int num_tools, i;
	
	if (!tools || !cJSON_IsArray(tools))
		return NULL;
	
	num_tools = cJSON_GetArraySize(tools);
	if (num_tools == 0)
		return NULL;
	
	/* Estimate size: each tool needs ~500 chars */
	bufsize = num_tools * 512 + 256;
	buf = malloc(bufsize);
	if (!buf)
		return NULL;
	
	pos += snprintf(buf + pos, bufsize - pos,
		"\n\nYou have access to functions. To call a function, use:\n"
		"<tool_call>function_name|arguments</tool_call>\n\n"
		"Available tools:\n");
	
	for (i = 0; i < num_tools; i++) {
		cJSON *tool = cJSON_GetArrayItem(tools, i);
		cJSON *func = cJSON_GetObjectItem(tool, "function");
		cJSON *name = NULL;
		cJSON *desc = NULL;
		cJSON *params = NULL;
		
		if (!func)
			continue;
		
		name = cJSON_GetObjectItem(func, "name");
		desc = cJSON_GetObjectItem(func, "description");
		params = cJSON_GetObjectItem(func, "parameters");
		
		if (!name || !name->valuestring)
			continue;
		
		pos += snprintf(buf + pos, bufsize - pos, "%d. %s\n", i + 1, name->valuestring);
		
		if (desc && desc->valuestring) {
			pos += snprintf(buf + pos, bufsize - pos, "   %s\n", desc->valuestring);
		}
		
		if (params && cJSON_IsObject(params)) {
			cJSON *props = cJSON_GetObjectItem(params, "properties");
			if (props && cJSON_IsObject(props)) {
				pos += snprintf(buf + pos, bufsize - pos, "   Parameters: ");
				/* Print first few properties */
				int prop_count = 0;
				cJSON *prop = props->child;
				while (prop && prop_count < 3) {
					pos += snprintf(buf + pos, bufsize - pos, "%s%s",
						prop_count > 0 ? ", " : "",
						prop->string);
					prop = prop->next;
					prop_count++;
				}
				pos += snprintf(buf + pos, bufsize - pos, "\n");
			}
		}
		pos += snprintf(buf + pos, bufsize - pos, "\n");
		
		if (pos >= bufsize - 100)
			break;
	}
	
	return buf;
}

/* Format multi-turn conversation with chat template
 * Returns: allocated buffer with formatted conversation, or NULL on error
 * Caller must free() the returned buffer
 */
static char *
format_conversation(cJSON *messages, cJSON *tools, chat_template_t template, int skip_global_system)
{
	char *buf;
	size_t bufsize, pos = 0;
	int num_messages, i;
	int has_json_system = 0;
	const char *json_system_content = NULL;
	
	if (!messages || !cJSON_IsArray(messages))
		return NULL;
	
	num_messages = cJSON_GetArraySize(messages);
	if (num_messages == 0)
		return NULL;
	
	/* Allocate buffer */
	bufsize = calc_conversation_size(messages, tools, template);
	buf = malloc(bufsize);
	if (!buf)
		return NULL;
	
	/* Format tools if present */
	char *tools_text = NULL;
	if (tools && cJSON_IsArray(tools)) {
		tools_text = format_tools(tools);
	}
	
	/* Check for system message in JSON */
	for (i = 0; i < num_messages; i++) {
		cJSON *msg = cJSON_GetArrayItem(messages, i);
		cJSON *role = cJSON_GetObjectItem(msg, "role");
		if (role && role->valuestring && strcmp(role->valuestring, "system") == 0) {
			cJSON *content = cJSON_GetObjectItem(msg, "content");
			if (content && content->valuestring) {
				has_json_system = 1;
				json_system_content = content->valuestring;
				break;
			}
		}
	}
	
	/* Build formatted conversation based on template */
	switch (template) {
	case TEMPLATE_PHI3: {
		/* Phi-3 format: <|system|>\n{sys}<|end|>\n<|user|>\n{msg}<|end|>\n... */
		
		/* Add system prompt first (global or JSON) */
		if (!skip_global_system && system_prompt[0] != '\0') {
			pos += snprintf(buf + pos, bufsize - pos, "<|system|>\n%s<|end|>\n", system_prompt);
		} else if (has_json_system && json_system_content) {
			pos += snprintf(buf + pos, bufsize - pos, "<|system|>\n%s<|end|>\n", json_system_content);
		}
		
		/* Add tools description if present */
		if (tools_text) {
			pos += snprintf(buf + pos, bufsize - pos, "%s", tools_text);
		}
		
		/* Process all messages */
		for (i = 0; i < num_messages; i++) {
			cJSON *msg = cJSON_GetArrayItem(messages, i);
			cJSON *role = cJSON_GetObjectItem(msg, "role");
			cJSON *content = cJSON_GetObjectItem(msg, "content");
			
			if (!role || !role->valuestring || !content || !content->valuestring)
				continue;
			
			if (strcmp(role->valuestring, "system") == 0) {
				/* Skip, already handled above */
				continue;
			} else if (strcmp(role->valuestring, "user") == 0) {
				pos += snprintf(buf + pos, bufsize - pos, 
				               "<|user|>\n%s<|end|>\n", content->valuestring);
			} else if (strcmp(role->valuestring, "assistant") == 0) {
				pos += snprintf(buf + pos, bufsize - pos, 
				               "<|assistant|>\n%s<|end|>\n", content->valuestring);
			}
			
			if (pos >= bufsize - 100) break;  /* Safety check */
		}
		
		/* End with assistant marker (ready for generation) */
		pos += snprintf(buf + pos, bufsize - pos, "<|assistant|>\n");
		break;
	}
	
	case TEMPLATE_MISTRAL: {
		/* Mistral format: [INST] {sys?}\n\n{user1} [/INST] {asst1} [INST] {user2} [/INST] */
		int in_user = 0;
		
		/* Add tools description if present */
		if (tools_text) {
			pos += snprintf(buf + pos, bufsize - pos, "%s", tools_text);
		}
		
		for (i = 0; i < num_messages; i++) {
			cJSON *msg = cJSON_GetArrayItem(messages, i);
			cJSON *role = cJSON_GetObjectItem(msg, "role");
			cJSON *content = cJSON_GetObjectItem(msg, "content");
			
			if (!role || !role->valuestring || !content || !content->valuestring)
				continue;
			
			if (strcmp(role->valuestring, "system") == 0) {
				/* Add system at start of first [INST] block */
				if (!in_user) {
					pos += snprintf(buf + pos, bufsize - pos, "[INST] %s\n\n", content->valuestring);
					in_user = 1;
				}
			} else if (strcmp(role->valuestring, "user") == 0) {
				if (!in_user) {
					/* Start new [INST] block */
					if (!skip_global_system && system_prompt[0] != '\0' && pos == 0) {
						/* Add global system prompt at very start */
						pos += snprintf(buf + pos, bufsize - pos, "[INST] %s\n\n", system_prompt);
					} else {
						pos += snprintf(buf + pos, bufsize - pos, "[INST] ");
					}
					in_user = 1;
				}
				pos += snprintf(buf + pos, bufsize - pos, "%s [/INST]", content->valuestring);
				in_user = 0;
			} else if (strcmp(role->valuestring, "assistant") == 0) {
				pos += snprintf(buf + pos, bufsize - pos, " %s ", content->valuestring);
			}
			
			if (pos >= bufsize - 100) break;
		}
		break;
	}
	
	case TEMPLATE_GEMMA: {
		/* Gemma format: <start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n{resp}<end_of_turn>\n */
		
		/* Add system in first user turn if present */
		int added_system = 0;
		
		/* Add tools description if present */
		if (tools_text) {
			pos += snprintf(buf + pos, bufsize - pos, "%s", tools_text);
		}
		
		for (i = 0; i < num_messages; i++) {
			cJSON *msg = cJSON_GetArrayItem(messages, i);
			cJSON *role = cJSON_GetObjectItem(msg, "role");
			cJSON *content = cJSON_GetObjectItem(msg, "content");
			
			if (!role || !role->valuestring || !content || !content->valuestring)
				continue;
			
			if (strcmp(role->valuestring, "system") == 0) {
				continue;  /* Will be added with first user message */
			} else if (strcmp(role->valuestring, "user") == 0) {
				pos += snprintf(buf + pos, bufsize - pos, "<start_of_turn>user\n");
				
				/* Add system prompt with first user message */
				if (!added_system) {
					if (!skip_global_system && system_prompt[0] != '\0') {
						pos += snprintf(buf + pos, bufsize - pos, "%s\n\n", system_prompt);
					} else if (has_json_system && json_system_content) {
						pos += snprintf(buf + pos, bufsize - pos, "%s\n\n", json_system_content);
					}
					added_system = 1;
				}
				
				pos += snprintf(buf + pos, bufsize - pos, "%s<end_of_turn>\n", content->valuestring);
			} else if (strcmp(role->valuestring, "assistant") == 0) {
				pos += snprintf(buf + pos, bufsize - pos, 
				               "<start_of_turn>model\n%s<end_of_turn>\n", content->valuestring);
			}
			
			if (pos >= bufsize - 100) break;
		}
		
		/* End with model marker */
		pos += snprintf(buf + pos, bufsize - pos, "<start_of_turn>model\n");
		break;
	}
	
	case TEMPLATE_GENERIC:
	default: {
		/* Generic format with delimiter */
		
		/* Add system if present */
		if (!skip_global_system && system_prompt[0] != '\0') {
			pos += snprintf(buf + pos, bufsize - pos, "%c\nsystem:\n%s", delimiter, system_prompt);
		} else if (has_json_system && json_system_content) {
			pos += snprintf(buf + pos, bufsize - pos, "%c\nsystem:\n%s", delimiter, json_system_content);
		}
		
		/* Add tools description if present */
		if (tools_text) {
			pos += snprintf(buf + pos, bufsize - pos, "%s", tools_text);
		}
		
		/* Process all messages */
		for (i = 0; i < num_messages; i++) {
			cJSON *msg = cJSON_GetArrayItem(messages, i);
			cJSON *role = cJSON_GetObjectItem(msg, "role");
			cJSON *content = cJSON_GetObjectItem(msg, "content");
			
			if (!role || !role->valuestring || !content || !content->valuestring)
				continue;
			
			if (strcmp(role->valuestring, "system") == 0) {
				continue;  /* Already handled */
			} else if (strcmp(role->valuestring, "user") == 0) {
				pos += snprintf(buf + pos, bufsize - pos, "%c\nuser:\n%s", delimiter, content->valuestring);
			} else if (strcmp(role->valuestring, "assistant") == 0) {
				pos += snprintf(buf + pos, bufsize - pos, "%c\nassistant:\n%s", delimiter, content->valuestring);
			}
			
			if (pos >= bufsize - 100) break;
		}
		
		/* End with assistant marker */
		pos += snprintf(buf + pos, bufsize - pos, "%c\nassistant:\n", delimiter);
		break;
	}
	}
	
	/* Free tools_text */
	free(tools_text);
	
	return buf;
}

typedef struct gen_state {
	int	fd;
	fdi_t	*fdi;
	int	stop;
} gen_state_t;

struct ndc_config ndc_config = {
	.flags = NDC_DETACH,
	.port = 4242,
};

struct qllm_config cfg = {
	.model_path = qllm_model_path,
	.n_ctx = 2048,  /* Increased from 512 */
	.n_threads = 0,
	.n_contexts = 1,  /* Only 1 sequence - connections will share KV cache */
	/* Sampling parameters - good defaults for general use */
	.temperature = 0.7f,
	.top_k = 40,
	.top_p = 0.95f,
	.repeat_penalty = 1.1f,
	.repeat_last_n = 64,
	/* Feature flags */
	.enable_embeddings = 0,  /* Don't need embeddings for chat */
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

	ret = qllm_next(fdi->ctx, buf, sizeof(buf));
	if (ret < 0)
		return 0; /* error -> stop */

	if (ret == 0)
		return 0; /* EOS -> stop */

	fdi->queue.tail++;

	buflen = (size_t)ret;

	eoim = memchr(buf, delimiter, buflen);
	if (eoim) {
		size_t n = (size_t)((char *)eoim - buf);

		if (n) {
			ndc_write(fd, buf, n);
			append_to_line(fdi, buf, n);
		}

		return 0;
	}

	ndc_write(fd, buf, buflen);
	append_to_line(fdi, buf, buflen);

	if (strrchr(buf, '\n')) {
		cmd_exec(fd, fdi);
		fdi->line_pos = 0;
	}

	qllm_compress(fdi->ctx, cfg.n_ctx * 4 / 5);

	return 1;
}

void
generate(int fd, const char *prompt)
{
	fdi_t	*fdi = &fdis[fd];
	int	 step;
	int	 max_gen = MAX_MEMORY;

	qllm_anchor_start(fdi->ctx);
	/* Prime qllm context with the full prompt */
	if (qllm_prime(fdi->ctx, prompt) < 0) {
		qsyslog(QLOG_ERR, "qllm_prime failed\n");
		return;
	}
	qllm_anchor_end(fdi->ctx);

	fdi->line_pos = 0;

	for (step = 0;
	     step < max_gen && inference(fd, fdi);
	     ++step)
		;

	cmd_exec(fd, fdi);
	fdi->line_pos = 0;
}

/* Streaming callback context */
typedef struct {
	int fd;
	fdi_t *fdi;
	char line_buf[BUFSIZ];
	size_t line_pos;
} stream_ctx_t;

static int
stream_token_cb(void *ctx, const char *token, size_t len)
{
	stream_ctx_t *sctx = (stream_ctx_t *)ctx;
	int fd = sctx->fd;
	
	/* Output in SSE format */
	char sse_buf[BUFSIZ];
	int n = snprintf(sse_buf, sizeof(sse_buf), "data: {\"type\":\"chunk\",\"delta\":\"%.*s\"}\n\n", (int)len, token);
	ndc_write(fd, sse_buf, n);
	
	/* Also append to line buffer for command execution */
	size_t space = sizeof(sctx->line_buf) - 1 - sctx->line_pos;
	if (len < space) {
		memcpy(sctx->line_buf + sctx->line_pos, token, len);
		sctx->line_pos += len;
		sctx->line_buf[sctx->line_pos] = '\0';
		
		/* Check for newline to execute commands */
		if (memchr(token, '\n', len)) {
			cmd_exec(fd, sctx->fdi);
			sctx->line_pos = 0;
		}
	}
	
	return 0; /* Continue generation */
}

void
generate_stream(int fd, const char *prompt)
{
	fdi_t	*fdi = &fdis[fd];
	stream_ctx_t sctx;
	int step;
	int max_gen = MAX_MEMORY;
	char buf[MAX_MEMORY];
	int ret;
	size_t buflen;
	char *eoim;
	
	/* Send start event */
	ndc_writef(fd, "data: {\"type\":\"start\",\"id\":\"chatcmpl-%ld\"}\n\n", (long)time(NULL));
	
	/* Initialize stream context */
	sctx.fd = fd;
	sctx.fdi = fdi;
	sctx.line_pos = 0;
	memset(sctx.line_buf, 0, sizeof(sctx.line_buf));
	
	qllm_anchor_start(fdi->ctx);
	/* Prime qllm context with the full prompt */
	if (qllm_prime(fdi->ctx, prompt) < 0) {
		qsyslog(QLOG_ERR, "qllm_prime failed\n");
		ndc_writef(fd, "data: {\"type\":\"stop\",\"finish_reason\":\"error\"}\n\n");
		return;
	}
	qllm_anchor_end(fdi->ctx);

	fdi->line_pos = 0;

	for (step = 0; step < max_gen; step++) {
		ret = qllm_next(fdi->ctx, buf, sizeof(buf));
		if (ret < 0)
			break; /* error */
		if (ret == 0)
			break; /* EOS */

		fdi->queue.tail++;
		buflen = (size_t)ret;

		/* Check for delimiter (end of message) */
		eoim = memchr(buf, delimiter, buflen);
		if (eoim) {
			size_t n = (size_t)((char *)eoim - buf);
			if (n) {
				char sse_buf[BUFSIZ];
				int sn = snprintf(sse_buf, sizeof(sse_buf), "data: {\"type\":\"chunk\",\"delta\":\"%.*s\"}\n\n", (int)n, buf);
				ndc_write(fd, sse_buf, sn);
			}
			break;
		}

		/* Output in SSE format */
		char sse_buf[BUFSIZ];
		int sn = snprintf(sse_buf, sizeof(sse_buf), "data: {\"type\":\"chunk\",\"delta\":\"%.*s\"}\n\n", (int)buflen, buf);
		ndc_write(fd, sse_buf, sn);
		
		/* Check for newline to execute commands */
		if (strrchr(buf, '\n')) {
			cmd_exec(fd, fdi);
			fdi->line_pos = 0;
		}

		qllm_compress(fdi->ctx, cfg.n_ctx * 4 / 5);
	}

	cmd_exec(fd, fdi);
	fdi->line_pos = 0;
	
	/* Send stop event */
	ndc_writef(fd, "data: {\"type\":\"stop\",\"finish_reason\":\"stop\"}\n\n");
}

void
do_ASK(int fd, int argc, char *argv[])
{
	fdi_t *fdi __attribute__((unused)) = &fdis[fd];
	char user_msg[BUFSIZ], buf[BUFSIZ * 2];
	char *u = user_msg;
	int i, ret;

	/* Build user message from arguments */
	*u = '\0';
	for (i = 1; i < argc; i++) {
		ret = snprintf(u, sizeof(user_msg) - (u - user_msg), "%s%s", 
		               (i > 1) ? " " : "", argv[i]);
		if (ret < 0 || (size_t)ret >= sizeof(user_msg) - (size_t)(u - user_msg)) {
			ndc_writef(fd, "Message too long\n");
			return;
		}
		u += ret;
	}

	/* Format with appropriate chat template */
	ret = format_prompt(buf, sizeof(buf), user_msg, model_template);
	if (ret < 0) {
		ndc_writef(fd, "Failed to format prompt\n");
		return;
	}

	generate(fd, buf);
	ndc_writef(fd, "%c\n", delimiter);
}

void
do_MESSAGES(int fd, int argc, char *argv[])
{
	fdi_t *fdi __attribute__((unused)) = &fdis[fd];
	cJSON *root = NULL;
	cJSON *messages = NULL;
	cJSON *tools = NULL;
	cJSON *stream_val = NULL;
	char *formatted_prompt = NULL;
	char *json_str = NULL;
	size_t total_len = 0;
	char *p;
	int i;
	int stream = 0;
	
	/* Reconstruct JSON string from argv (may be split by spaces) */
	for (i = 1; i < argc; i++) {
		total_len += strlen(argv[i]) + 1; /* +1 for space */
	}
	
	json_str = malloc(total_len + 1);
	if (!json_str) {
		ndc_writef(fd, "{\"error\":\"Out of memory\"}%c\n", delimiter);
		return;
	}
	
	p = json_str;
	for (i = 1; i < argc; i++) {
		if (i > 1) *p++ = ' ';
		strcpy(p, argv[i]);
		p += strlen(argv[i]);
	}
	*p = '\0';
	
	/* Parse JSON */
	root = cJSON_Parse(json_str);
	if (!root) {
		const char *err = cJSON_GetErrorPtr();
		ndc_writef(fd, "{\"error\":\"Invalid JSON\",\"detail\":\"%s\"}%c\n", 
		          err ? err : "unknown", delimiter);
		free(json_str);
		return;
	}
	
	/* Extract messages array */
	messages = cJSON_GetObjectItem(root, "messages");
	if (!messages || !cJSON_IsArray(messages)) {
		ndc_writef(fd, "{\"error\":\"messages must be an array\"}%c\n", delimiter);
		cJSON_Delete(root);
		free(json_str);
		return;
	}
	
	/* Extract stream flag (optional) */
	stream_val = cJSON_GetObjectItem(root, "stream");
	if (stream_val && cJSON_IsBool(stream_val)) {
		stream = cJSON_IsTrue(stream_val);
	}
	
	/* Extract tools (optional) */
	tools = cJSON_GetObjectItem(root, "tools");
	
	/* Format conversation (with tools if present) */
	formatted_prompt = format_conversation(messages, tools, model_template, 0);
	if (!formatted_prompt) {
		ndc_writef(fd, "{\"error\":\"Failed to format conversation\"}%c\n", delimiter);
		cJSON_Delete(root);
		free(json_str);
		return;
	}
	
	/* Generate response - streaming or blocking */
	if (stream) {
		generate_stream(fd, formatted_prompt);
	} else {
		generate(fd, formatted_prompt);
		ndc_writef(fd, "%c\n", delimiter);
	}
	
	/* Cleanup */
	free(formatted_prompt);
	cJSON_Delete(root);
	free(json_str);
}

void
do_INFO(int fd, int argc __attribute__((unused)), char *argv[] __attribute__((unused)))
{
	const char *model_name = "unknown";
	
	/* Extract model name from qllm_model_path */
	if (qllm_model_path[0] != '\0') {
		const char *basename = strrchr(qllm_model_path, '/');
		if (basename) {
			model_name = basename + 1;
		} else {
			model_name = qllm_model_path;
		}
	}
	
	/* Return JSON response */
	ndc_writef(fd, "{\"model\":\"%s\",\"template\":\"%s\"}\n",
	          model_name, template_names[model_template]);
}

static inline void
fdi_init(fdi_t *fdi)
{
	/* Use the shared context instead of creating a new one */
	/* This avoids the Vulkan multi-context crash */
	fdi->ctx = general.ctx;
	
	if (!fdi->ctx) {
		qsyslog(QLOG_ERR, "Shared context is NULL\n");
	}

	fdi->queue.tail = 0;
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
		.name = "messages",
		.cb = &do_MESSAGES,
		.flags = CF_NOAUTH | CF_NOTRIM,
	}, {
		.name = "info",
		.cb = &do_INFO,
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
	fprintf(stderr, "Usage: %s [-dr?] [-C PATH] [-u USER] [-k PATH] [-c PATH] [-p PORT] [-S PROMPT] MODEL\n", prog);
	fprintf(stderr, "    Options:\n");
	fprintf(stderr, "        -C PATH   changes directory to PATH before starting up.\n");
	fprintf(stderr, "        -u USER   login as USER before starting up.\n");
	fprintf(stderr, "        -k PATH   specify SSL certificate 'key' file\n");
	fprintf(stderr, "        -c PATH   specify SSL certificate 'crt' file\n");
	fprintf(stderr, "        -p PORT   specify server port (defaults to 4242)\n");
	fprintf(stderr, "        -d        don't detach\n");
	fprintf(stderr, "        -r        root multiplex mode\n");
	fprintf(stderr, "        -c SIZE   specify n_ctx (default 2048)\n");
	fprintf(stderr, "        -n NUM    specify an estimation of concurrent sessions (default 1)\n");
	fprintf(stderr, "        -S PROMPT set system prompt for all conversations\n");
	fprintf(stderr, "        -?        display this message.\n");
}

static void
setup(const char *model_path)
{
	/* Set model path FIRST so qllm_create can use it */
	snprintf(qllm_model_path, sizeof(qllm_model_path), "%s", model_path);
	
	/* Detect model template type */
	model_template = detect_model_template(model_path);
	
	fprintf(stderr, "qllmd: Detected model template: ");
	switch (model_template) {
	case TEMPLATE_PHI3:
		fprintf(stderr, "Phi-3\n");
		break;
	case TEMPLATE_MISTRAL:
		fprintf(stderr, "Mistral\n");
		break;
	case TEMPLATE_GEMMA:
		fprintf(stderr, "Gemma\n");
		break;
	case TEMPLATE_GENERIC:
	default:
		fprintf(stderr, "Generic\n");
		break;
	}

	/* Create a SINGLE shared context that all connections will use */
	/* This avoids the Vulkan multi-context issue */
	fprintf(stderr, "qllmd: Creating shared context\n");
	general.ctx = qllm_create(&cfg);
	if (!general.ctx) {
		fprintf(stderr, "qllmd: FATAL - Failed to create shared context\n");
		exit(1);
	}
	fprintf(stderr, "qllmd: Shared context created successfully\n");

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
	
	/* Initialize fdis array to zeros */
	memset(fdis, 0, sizeof(fdis));

	while ((c = getopt(argc, argv, "?dK:k:C:rp:s:n:c:S:")) != -1) switch (c) {
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
			cfg.n_contexts = atoi(optarg);
			break;

		case 'c':
			cfg.n_ctx = atoi(optarg);
			break;

		case 'S':
			/* System prompt */
			snprintf(system_prompt, sizeof(system_prompt), "%s", optarg);
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
	ndc_register("messages", do_MESSAGES, CF_NOAUTH | CF_NOTRIM);
	ndc_register("info", do_INFO, CF_NOAUTH | CF_NOTRIM);

	setup(arg_model);

	ret = ndc_main();

	if (general.ctx)
		qllm_free(general.ctx);

	return ret;
}

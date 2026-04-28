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
#include <stdarg.h>
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
	struct llama_sampler *  sampler;
	unsigned		line_pos;
	token_queue_t		queue;
} fdi_t;

fdi_t fdis[FD_SETSIZE], general;

static void fdi_init(fdi_t *fdi, int fd);

const char delimiter = 4;

/* GBNF grammar to enforce tool call syntax: <tool_call>name|args</tool_call>
 * This ensures the model always produces syntactically correct tool calls.
 */
static const char *tool_call_grammar = 
    "root ::= (text | tool_call)*\n"
    "text ::= [^<]+\n"
    "tool_call ::= \"<tool_call>\" name \"|\" args \"</tool_call>\"\n"
    "name ::= [a-zA-Z0-9_]+\n"
    "args ::= [^<]+\n";

size_t crb_len = 0;
char *crb = NULL;

char qllm_model_path[BUFSIZ];
static char system_prompt[BUFSIZ] = "";  /* Optional system prompt */

/* Chat template types */
typedef enum {
	TEMPLATE_GENERIC,
	TEMPLATE_PHI3,
	TEMPLATE_MISTRAL,
	TEMPLATE_GEMMA,
	TEMPLATE_CHATML,
} chat_template_t;

/* FIM tokens for autocomplete */
const char *fim_prefix = NULL;
const char *fim_suffix = NULL;
const char *fim_middle = NULL;

static chat_template_t model_template = TEMPLATE_GENERIC;

/* Template names for info command */
static const char *template_names[] = {
	[TEMPLATE_GENERIC] = "generic",
	[TEMPLATE_PHI3]    = "phi3",
	[TEMPLATE_MISTRAL] = "mistral",
	[TEMPLATE_GEMMA]   = "gemma",
	[TEMPLATE_CHATML]  = "chatml",
};

typedef struct message_view {
	const char *role;
	const char *content;
} message_view_t;

static int
appendf(char *buf, size_t bufsize, size_t *pos, const char *fmt, ...)
{
	va_list ap;
	int n;

	if (*pos >= bufsize)
		return -1;

	va_start(ap, fmt);
	n = vsnprintf(buf + *pos, bufsize - *pos, fmt, ap);
	va_end(ap);
	if (n < 0)
		return -1;

	if ((size_t)n >= bufsize - *pos) {
		*pos = bufsize;
		return -1;
	}

	*pos += (size_t)n;
	return 0;
}

static int
message_view_get(cJSON *msg, message_view_t *view)
{
	cJSON *role;
	cJSON *content;

	if (!msg || !view)
		return 0;

	role = cJSON_GetObjectItem(msg, "role");
	content = cJSON_GetObjectItem(msg, "content");
	if (!role || !role->valuestring || !content || !content->valuestring)
		return 0;

	view->role = role->valuestring;
	view->content = content->valuestring;
	return 1;
}

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
	else if (strcasestr(path, "qwen"))
		return TEMPLATE_CHATML;
	
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
		
	case TEMPLATE_CHATML:
		/* ChatML format: <|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n */
		if (has_system)
			ret = snprintf(buf, bufsize, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", 
			              system_prompt, user_msg);
		else
			ret = snprintf(buf, bufsize, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_msg);
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
	(void)template;
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
	
	appendf(buf, bufsize, &pos,
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
		
		appendf(buf, bufsize, &pos, "%d. %s\n", i + 1, name->valuestring);
		
		if (desc && desc->valuestring) {
			appendf(buf, bufsize, &pos, "   %s\n", desc->valuestring);
		}
		
		if (params && cJSON_IsObject(params)) {
			cJSON *props = cJSON_GetObjectItem(params, "properties");
			if (props && cJSON_IsObject(props)) {
				appendf(buf, bufsize, &pos, "   Parameters: ");
				/* Print first few properties */
				int prop_count = 0;
				cJSON *prop = props->child;
				while (prop && prop_count < 3) {
					appendf(buf, bufsize, &pos, "%s%s",
						prop_count > 0 ? ", " : "",
						prop->string);
					prop = prop->next;
					prop_count++;
				}
				appendf(buf, bufsize, &pos, "\n");
			}
		}
		appendf(buf, bufsize, &pos, "\n");
		
		if (pos >= bufsize - 100)
			break;
	}
	
	return buf;
}

/* Format multi-turn conversation with chat template
 * Returns: allocated buffer with formatted conversation, or NULL on error
 * Caller must free() the returned buffer
 */
extern struct qllm_config cfg;

static int
calculate_max_system(cJSON *messages, cJSON *tools, int stream)
{
	int n_ctx = cfg.n_ctx > 0 ? cfg.n_ctx : 2048;
	int response_buffer = stream ? 256 : 512;
	int available = n_ctx - response_buffer;
	
	/* Rough token estimation: 4 chars per token */
	char *messages_json = cJSON_PrintUnformatted(messages);
	int messages_tokens = messages_json ? (int)strlen(messages_json) / 4 : 0;
	free(messages_json);
	
	if (tools && cJSON_IsArray(tools) && cJSON_GetArraySize(tools) > 0) {
		char *tools_json = cJSON_PrintUnformatted(tools);
		int tools_tokens = tools_json ? (int)strlen(tools_json) / 4 : 0;
		free(tools_json);
		
		int total_needed = messages_tokens + tools_tokens;
		if (total_needed > available) {
			int remaining = available - tools_tokens;
			return remaining > 256 ? remaining * 4 : 0;
		}
	}
	
	int sys_limit = (available - messages_tokens) * 4;
	return sys_limit > 0 ? sys_limit : 4096;
}

static char *
format_conversation(cJSON *messages, cJSON *tools, chat_template_t template, int skip_global_system, int max_system_len)
{
	char *buf;
	size_t bufsize, pos = 0;
	int num_messages, i;
	int has_json_system = 0;
	const char *json_system_content = NULL;
	char *truncated_system = NULL;
	
	if (!messages || !cJSON_IsArray(messages))
		return NULL;
	
	num_messages = cJSON_GetArraySize(messages);
	if (num_messages == 0)
		return NULL;

	/* Check for system message in JSON and handle truncation */
	for (i = 0; i < num_messages; i++) {
		cJSON *msg = cJSON_GetArrayItem(messages, i);
		message_view_t mv;

		if (message_view_get(msg, &mv) && strcmp(mv.role, "system") == 0) {
			has_json_system = 1;
			json_system_content = mv.content;
			
			if (max_system_len > 0 && (int)strlen(json_system_content) > max_system_len) {
				const char *suffix = "\n\n[System prompt truncated by qllmd]";
				truncated_system = malloc((size_t)max_system_len + strlen(suffix) + 1);
				if (truncated_system) {
					strncpy(truncated_system, json_system_content, (size_t)max_system_len);
					truncated_system[max_system_len] = '\0';
					strcat(truncated_system, suffix);
					json_system_content = truncated_system;
				}
			}
			break;
		}
	}
	
	/* Allocate buffer */
	bufsize = calc_conversation_size(messages, tools, template);
	/* Adjust buffer size for truncation if needed */
	if (truncated_system) {
		bufsize += strlen(truncated_system);
	}
	buf = malloc(bufsize);
	if (!buf) {
		free(truncated_system);
		return NULL;
	}
	
	/* Format tools if present */
	char *tools_text = NULL;
	if (tools && cJSON_IsArray(tools)) {
		tools_text = format_tools(tools);
	}

	/* Build formatted conversation based on template */
	switch (template) {
	case TEMPLATE_PHI3: {
		/* Phi-3 format: <|system|>\n{sys}<|end|>\n<|user|>\n{msg}<|end|>\n... */
		
		/* Add system prompt first (global or JSON) */
		if (!skip_global_system && system_prompt[0] != '\0') {
			appendf(buf, bufsize, &pos, "<|system|>\n%s<|end|>\n", system_prompt);
		} else if (has_json_system && json_system_content) {
			appendf(buf, bufsize, &pos, "<|system|>\n%s<|end|>\n", json_system_content);
		}
		
		/* Add tools description if present */
		if (tools_text) {
			appendf(buf, bufsize, &pos, "%s", tools_text);
		}
		
		/* Process all messages */
		for (i = 0; i < num_messages; i++) {
			cJSON *msg = cJSON_GetArrayItem(messages, i);
			message_view_t mv;
			
			if (!message_view_get(msg, &mv))
				continue;
			
			if (strcmp(mv.role, "system") == 0) {
				/* Skip, already handled above */
				continue;
			} else if (strcmp(mv.role, "user") == 0) {
				appendf(buf, bufsize, &pos, "<|user|>\n%s<|end|>\n", mv.content);
			} else if (strcmp(mv.role, "assistant") == 0) {
				appendf(buf, bufsize, &pos, "<|assistant|>\n%s<|end|>\n", mv.content);
			} else if (strcmp(mv.role, "tool") == 0) {
				appendf(buf, bufsize, &pos, "<|user|>\ntool result:\n%s<|end|>\n", mv.content);
			}
			
			if (pos >= bufsize - 100) break;  /* Safety check */
		}
		
		/* End with assistant marker (ready for generation) */
		appendf(buf, bufsize, &pos, "<|assistant|>\n");
		break;
	}
	
	case TEMPLATE_MISTRAL: {
		/* Mistral format: [INST] {sys?}\n\n{user1} [/INST] {asst1} [INST] {user2} [/INST] */
		int in_user = 0;
		
		/* Add tools description if present */
		if (tools_text) {
			appendf(buf, bufsize, &pos, "%s", tools_text);
		}
		
		for (i = 0; i < num_messages; i++) {
			cJSON *msg = cJSON_GetArrayItem(messages, i);
			message_view_t mv;
			
			if (!message_view_get(msg, &mv))
				continue;
			
			if (strcmp(mv.role, "system") == 0) {
				/* Add system at start of first [INST] block */
				if (!in_user) {
					appendf(buf, bufsize, &pos, "[INST] %s\n\n", mv.content);
					in_user = 1;
				}
			} else if (strcmp(mv.role, "user") == 0) {
				if (!in_user) {
					/* Start new [INST] block */
					if (!skip_global_system && system_prompt[0] != '\0' && pos == 0) {
						/* Add global system prompt at very start */
						appendf(buf, bufsize, &pos, "[INST] %s\n\n", system_prompt);
					} else {
						appendf(buf, bufsize, &pos, "[INST] ");
					}
					in_user = 1;
				}
				appendf(buf, bufsize, &pos, "%s [/INST]", mv.content);
				in_user = 0;
			} else if (strcmp(mv.role, "assistant") == 0) {
				appendf(buf, bufsize, &pos, " %s ", mv.content);
			} else if (strcmp(mv.role, "tool") == 0) {
				if (!in_user) {
					appendf(buf, bufsize, &pos, "[INST] ");
					in_user = 1;
				}
				appendf(buf, bufsize, &pos, "tool result:\n%s [/INST]", mv.content);
				in_user = 0;
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
			appendf(buf, bufsize, &pos, "%s", tools_text);
		}
		
		for (i = 0; i < num_messages; i++) {
			cJSON *msg = cJSON_GetArrayItem(messages, i);
			message_view_t mv;
			
			if (!message_view_get(msg, &mv))
				continue;
			
			if (strcmp(mv.role, "system") == 0) {
				continue;  /* Will be added with first user message */
			} else if (strcmp(mv.role, "user") == 0) {
				appendf(buf, bufsize, &pos, "<start_of_turn>user\n");
				
				/* Add system prompt with first user message */
				if (!added_system) {
					if (!skip_global_system && system_prompt[0] != '\0') {
						appendf(buf, bufsize, &pos, "%s\n\n", system_prompt);
					} else if (has_json_system && json_system_content) {
						appendf(buf, bufsize, &pos, "%s\n\n", json_system_content);
					}
					added_system = 1;
				}
				
				appendf(buf, bufsize, &pos, "%s<end_of_turn>\n", mv.content);
			} else if (strcmp(mv.role, "assistant") == 0) {
				appendf(buf, bufsize, &pos,
				        "<start_of_turn>model\n%s<end_of_turn>\n", mv.content);
			} else if (strcmp(mv.role, "tool") == 0) {
				appendf(buf, bufsize, &pos,
				        "<start_of_turn>user\ntool result:\n%s<end_of_turn>\n", mv.content);
			}
			
			if (pos >= bufsize - 100) break;
		}
		
		/* End with model marker */
		appendf(buf, bufsize, &pos, "<start_of_turn>model\n");
		break;
	}
	
	case TEMPLATE_CHATML: {
		/* ChatML format: <|im_start|>role\ncontent<|im_end|>\n */
		
		/* Add system prompt first (global or JSON) */
		if (!skip_global_system && system_prompt[0] != '\0') {
			appendf(buf, bufsize, &pos, "<|im_start|>system\n%s<|im_end|>\n", system_prompt);
		} else if (has_json_system && json_system_content) {
			appendf(buf, bufsize, &pos, "<|im_start|>system\n%s<|im_end|>\n", json_system_content);
		}
		
		/* Add tools description if present */
		if (tools_text) {
			appendf(buf, bufsize, &pos, "<|im_start|>system\n%s<|im_end|>\n", tools_text);
		}
		
		/* Process all messages */
		for (i = 0; i < num_messages; i++) {
			cJSON *msg = cJSON_GetArrayItem(messages, i);
			message_view_t mv;
			
			if (!message_view_get(msg, &mv))
				continue;
			
			if (strcmp(mv.role, "system") == 0) {
				/* Skip, already handled above */
				continue;
			} else if (strcmp(mv.role, "user") == 0) {
				appendf(buf, bufsize, &pos, "<|im_start|>user\n%s<|im_end|>\n", mv.content);
			} else if (strcmp(mv.role, "assistant") == 0) {
				appendf(buf, bufsize, &pos, "<|im_start|>assistant\n%s<|im_end|>\n", mv.content);
			} else if (strcmp(mv.role, "tool") == 0) {
				appendf(buf, bufsize, &pos, "<|im_start|>user\ntool result:\n%s<|im_end|>\n", mv.content);
			}
			
			if (pos >= bufsize - 100) break;
		}
		
		/* End with assistant marker */
		appendf(buf, bufsize, &pos, "<|im_start|>assistant\n");
		break;
	}
	
	case TEMPLATE_GENERIC:
	default: {
		/* Generic format with delimiter */
		
		/* Add system if present */
		if (!skip_global_system && system_prompt[0] != '\0') {
			appendf(buf, bufsize, &pos, "%c\nsystem:\n%s", delimiter, system_prompt);
		} else if (has_json_system && json_system_content) {
			appendf(buf, bufsize, &pos, "%c\nsystem:\n%s", delimiter, json_system_content);
		}
		
		/* Add tools description if present */
		if (tools_text) {
			appendf(buf, bufsize, &pos, "%s", tools_text);
		}
		
		/* Process all messages */
		for (i = 0; i < num_messages; i++) {
			cJSON *msg = cJSON_GetArrayItem(messages, i);
			message_view_t mv;
			
			if (!message_view_get(msg, &mv))
				continue;
			
			if (strcmp(mv.role, "system") == 0) {
				continue;  /* Already handled */
			} else if (strcmp(mv.role, "user") == 0) {
				appendf(buf, bufsize, &pos, "%c\nuser:\n%s", delimiter, mv.content);
			} else if (strcmp(mv.role, "assistant") == 0) {
				appendf(buf, bufsize, &pos, "%c\nassistant:\n%s", delimiter, mv.content);
			} else if (strcmp(mv.role, "tool") == 0) {
				appendf(buf, bufsize, &pos, "%c\ntool:\n%s", delimiter, mv.content);
			}
			
			if (pos >= bufsize - 100) break;
		}
		
		/* End with assistant marker */
		appendf(buf, bufsize, &pos, "%c\nassistant:\n", delimiter);
		break;
	}
	}
	
	/* Free tools_text and truncated_system */
	free(tools_text);
	free(truncated_system);
	
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
    .n_ctx = 0,  /* 0 = auto-detect from model */
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
    .n_gpu_layers = 0, /* 0 = auto */
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


static int
check_stop_sequences(const char *buf, cJSON *stop)
{
	int i, num_stops;
	if (!buf || !stop || !cJSON_IsArray(stop))
		return 0;

	num_stops = cJSON_GetArraySize(stop);
	for (i = 0; i < num_stops; i++) {
		cJSON *s = cJSON_GetArrayItem(stop, i);
		if (cJSON_IsString(s) && s->valuestring[0] != '\0') {
			if (strstr(buf, s->valuestring))
				return 1;
		}
	}
	return 0;
}

static inline int
inference(int fd, fdi_t *fdi)
{
	char	buf[MAX_MEMORY];
	int	ret;
	size_t	buflen;
	char	*eoim;

	ret = qllm_next(fdi->ctx, NULL, buf, sizeof(buf));
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

#if 0
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
#endif

static char *
json_escape(const char *str)
{
	if (!str)
		return NULL;

	size_t len = strlen(str);
	char *escaped = malloc(len * 2 + 1);
	if (!escaped)
		return NULL;

	char *p = escaped;
	const char *s = str;
	while (*s) {
		switch (*s) {
		case '"':  *p++ = '\\'; *p++ = '"';  break;
		case '\\': *p++ = '\\'; *p++ = '\\'; break;
		case '\n': *p++ = '\\'; *p++ = 'n';  break;
		case '\r': *p++ = '\\'; *p++ = 'r';  break;
		case '\t': *p++ = '\\'; *p++ = 't';  break;
		default:   *p++ = *s; break;
		}
		s++;
	}
	*p = '\0';
	return escaped;
}

typedef struct tool_call {
	char *content;
	char *name;
	char *arguments;
} tool_call_t;

void
tool_call_free(tool_call_t *tc)
{
	if (tc) {
		free(tc->content);
		free(tc->name);
		free(tc->arguments);
		memset(tc, 0, sizeof(*tc));
	}
}

int
parse_tool_call(const char *output, tool_call_t *tc)
{

	const char *start, *bar, *end;
	size_t content_len, name_len, args_len;

	if (!output || !tc)
		return 0;

	memset(tc, 0, sizeof(*tc));

	start = strstr(output, "<tool_call>");
	if (!start)
		return 0;

	bar = strchr(start + 11, '|');
	end = strstr(start + 11, "</tool_call>");
	if (!bar || !end || bar > end)
		return 0;

	content_len = (size_t)(start - output);
	name_len = (size_t)(bar - (start + 11));
	args_len = (size_t)(end - (bar + 1));
	if (!name_len || !args_len)
		return 0;

	tc->content = strndup(output, content_len);
	tc->name = strndup(start + 11, name_len);
	tc->arguments = strndup(bar + 1, args_len);
	if (!tc->content || !tc->name || !tc->arguments) {
		tool_call_free(tc);
		return 0;
	}

	return 1;
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
		ret = qllm_next(fdi->ctx, NULL, buf, sizeof(buf));
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
				char *escaped = json_escape(buf);
				if (escaped) {
					char sse_buf[BUFSIZ];
					int sn = snprintf(sse_buf, sizeof(sse_buf), "data: {\"type\":\"chunk\",\"delta\":\"%s\"}\n\n", escaped);
					ndc_write(fd, sse_buf, sn);
					free(escaped);
				}
			}
			break;
		}

		/* Output in SSE format */
		char *escaped = json_escape(buf);
		if (escaped) {
			char sse_buf[BUFSIZ];
			int sn = snprintf(sse_buf, sizeof(sse_buf), "data: {\"type\":\"chunk\",\"delta\":\"%s\"}\n\n", escaped);
			ndc_write(fd, sse_buf, sn);
			free(escaped);
		}
		
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
	
	/* Send delimiter to signal end of stream */
	ndc_write(fd, "\x04", 1);
}

/* HTTP handlers for OpenAI-compatible API */

static void
http_json(socket_t fd, int code, const char *body)
{
	ndc_header_set(fd, "Content-Type", "application/json");
	ndc_respond(fd, code, body);
}

static void
http_error(socket_t fd, int code, const char *type, const char *message)
{
	char *escaped_type = json_escape(type ? type : "internal_error");
	char *escaped_message = json_escape(message ? message : "Internal error");
	char *response = NULL;

	if (!escaped_type || !escaped_message ||
	    asprintf(&response, "{\"error\":{\"message\":\"%s\",\"type\":\"%s\"}}",
	             escaped_message, escaped_type) < 0) {
		http_json(fd, 500, "{\"error\":{\"message\":\"Out of memory\",\"type\":\"internal_error\"}}");
		free(escaped_type);
		free(escaped_message);
		return;
	}

	http_json(fd, code, response);
	free(response);
	free(escaped_type);
	free(escaped_message);
}

static void
http_sse_start(socket_t fd)
{
	ndc_header_set(fd, "Content-Type", "text/event-stream");
	ndc_header_set(fd, "Cache-Control", "no-cache");
	ndc_header_set(fd, "Connection", "close");
	ndc_respond(fd, 200, NULL);
}

static void
oai_stream_role(socket_t fd, const char *completion_id)
{
	ndc_writef(fd,
		"data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"index\":0,\"finish_reason\":null}]}\n\n",
		completion_id);
}

static void
oai_stream_content(socket_t fd, const char *completion_id, const char *content)
{
	char *escaped = json_escape(content);

	if (!escaped)
		return;

	ndc_writef(fd,
		"data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"%s\"},\"index\":0,\"finish_reason\":null}]}\n\n",
		completion_id, escaped);
	free(escaped);
}

static void
oai_stream_tool_call(socket_t fd, const char *completion_id, const tool_call_t *tool_call)
{
	char *escaped_name;
	char *escaped_args;

	if (!tool_call)
		return;

	escaped_name = json_escape(tool_call->name);
	escaped_args = json_escape(tool_call->arguments);
	if (escaped_name && escaped_args) {
		ndc_writef(fd,
			"data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_%ld_0\",\"type\":\"function\",\"function\":{\"name\":\"%s\",\"arguments\":\"%s\"}}]},\"index\":0,\"finish_reason\":null}]}\n\n",
			completion_id, (long)time(NULL), escaped_name, escaped_args);
	}

	free(escaped_name);
	free(escaped_args);
}

static void
oai_stream_stop(socket_t fd, const char *completion_id, const char *finish_reason)
{
	ndc_writef(fd,
		"data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"%s\"}]}\n\n",
		completion_id, finish_reason);
}

static char *
oai_completion_json(
	const char *completion_id,
	const char *model_name,
	const char *content,
	const tool_call_t *tool_call,
	int prompt_tokens,
	int completion_tokens)
{
	char *escaped_model = NULL;
	char *escaped_content = NULL;
	char *response = NULL;
	const char *short_name;

	short_name = strrchr(model_name, '/');
	short_name = short_name ? short_name + 1 : model_name;

	escaped_model = json_escape(short_name);
	escaped_content = json_escape(content ? content : "");
	if (!escaped_model || !escaped_content)
		goto out;

	if (tool_call) {
		char *escaped_name = json_escape(tool_call->name);
		char *escaped_args = json_escape(tool_call->arguments);

		if (escaped_name && escaped_args) {
			if (asprintf(&response,
				"{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\",\"tool_calls\":[{\"id\":\"call_%ld_0\",\"type\":\"function\",\"function\":{\"name\":\"%s\",\"arguments\":\"%s\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
				completion_id, (long)time(NULL), escaped_model, escaped_content,
				(long)time(NULL), escaped_name, escaped_args,
				prompt_tokens, completion_tokens,
				prompt_tokens + completion_tokens) < 0)
				response = NULL;
		}

		free(escaped_name);
		free(escaped_args);
	} else {
		if (asprintf(&response,
			"{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
			completion_id, (long)time(NULL), escaped_model, escaped_content,
			prompt_tokens, completion_tokens,
			prompt_tokens + completion_tokens) < 0)
			response = NULL;
	}

out:
	free(escaped_model);
	free(escaped_content);
	return response;
}

int
handle_health(socket_t fd, char *body __attribute__((unused)))
{
	http_json(fd, 200, "{\"status\":\"ok\"}");
	return 1;
}

int
handle_v1_completions(socket_t fd, char *body)
{
	cJSON *root = NULL;
	cJSON *prompt_val = NULL;
	cJSON *suffix_val = NULL;
	cJSON *stop = NULL;
	cJSON *model_val = NULL;
	char *final_prompt = NULL;
	char completion_id[64];

	snprintf(completion_id, sizeof(completion_id), "cmpl-%ld", (long)time(NULL));

	if (!body || !*body) {
		http_error(fd, 400, "invalid_request_error", "Missing request body");
		return 1;
	}

	root = cJSON_Parse(body);
	if (!root) {
		http_error(fd, 400, "invalid_request_error", "Invalid JSON");
		return 1;
	}

	prompt_val = cJSON_GetObjectItem(root, "prompt");
	suffix_val = cJSON_GetObjectItem(root, "suffix");
	stop = cJSON_GetObjectItem(root, "stop");
	model_val = cJSON_GetObjectItem(root, "model");

	if (!prompt_val || !cJSON_IsString(prompt_val)) {
		http_error(fd, 400, "invalid_request_error", "prompt must be a string");
		cJSON_Delete(root);
		return 1;
	}

	/* Handle FIM (Fill-In-The-Middle) if suffix is provided */
	if (suffix_val && cJSON_IsString(suffix_val) && suffix_val->valuestring[0] != '\0') {
		if (asprintf(&final_prompt, "%s%s%s%s%s", 
		             fim_prefix, prompt_val->valuestring, 
		             fim_suffix, suffix_val->valuestring, 
		             fim_middle) < 0) {
			http_error(fd, 500, "internal_error", "Out of memory");
			cJSON_Delete(root);
			return 1;
		}
	} else {
		final_prompt = strdup(prompt_val->valuestring);
	}

	/* Initialize fdi and context */
	fdi_init(&fdis[fd], fd);
	qllm_set_grammar(fdis[fd].ctx, NULL); /* No grammar for raw completions */

	/* Generate response (non-streaming for now as FIM is usually fast) */
	fdi_t *fdi = &fdis[fd];
	char buf[4096];
	char full_response[BUFSIZ * 16];
	size_t full_len = 0;
	int ret, step;
	int max_gen = 128; /* Autocomplete usually wants short bursts */

	full_response[0] = '\0';
	
	qllm_anchor_start(fdi->ctx);
	if (qllm_prime(fdi->ctx, final_prompt) < 0) {
		http_error(fd, 500, "internal_error", "Generation failed");
		free(final_prompt);
		cJSON_Delete(root);
		return 1;
	}
	qllm_anchor_end(fdi->ctx);
	
	for (step = 0; step < max_gen; step++) {
		ret = qllm_next(fdi->ctx, fdi->sampler, buf, sizeof(buf));
		if (ret <= 0) break;
		
		char *delim = memchr(buf, delimiter, ret);
		if (delim) {
			*delim = '\0';
			ret = delim - buf;
		}
		
		if (ret > 0) {
			size_t copy_len = (size_t)ret;
			if (copy_len > sizeof(full_response) - full_len - 1)
				copy_len = sizeof(full_response) - full_len - 1;
			memcpy(full_response + full_len, buf, copy_len);
			full_len += copy_len;
			full_response[full_len] = '\0';
		}
		
		if (delim || check_stop_sequences(full_response, stop))
			break;
	}

	/* Format OpenAI Completion response */
	char *model_name = model_val && cJSON_IsString(model_val) ? model_val->valuestring : "local-model";
	char *escaped_content = json_escape(full_response);
	char *response_json = NULL;

	if (asprintf(&response_json, 
		"{\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,\"model\":\"%s\",\"choices\":[{\"text\":\"%s\",\"index\":0,\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
		completion_id, (long)time(NULL), model_name, escaped_content ? escaped_content : "",
		(int)strlen(final_prompt)/4, (int)full_len/4, ((int)strlen(final_prompt) + (int)full_len)/4) >= 0) {
		
		http_json(fd, 200, response_json);
		free(response_json);
	} else {
		http_error(fd, 500, "internal_error", "Out of memory");
	}

	free(escaped_content);
	free(final_prompt);
	cJSON_Delete(root);
	return 1;
}

int
handle_v1_models(socket_t fd, char *body __attribute__((unused)))
{
	char *model_name = *qllm_model_path ? qllm_model_path : "unknown";
	char *escaped_model = json_escape(model_name);
	char *response_json = NULL;

	if (!escaped_model) {
		http_json(fd, 500, "{\"error\":{\"message\":\"Out of memory\",\"type\":\"internal_error\"}}");
		return 1;
	}
	
	if (asprintf(&response_json,
		"{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\",\"created\":%ld,\"owned_by\":\"local\"}]}",
		escaped_model, (long)time(NULL)) < 0) {
		free(escaped_model);
		http_json(fd, 500, "{\"error\":{\"message\":\"Out of memory\",\"type\":\"internal_error\"}}");
		return 1;
	}

	http_json(fd, 200, response_json);
	free(response_json);
	free(escaped_model);
	return 1;
}

int
handle_v1_chat_completions(socket_t fd, char *body)
{
	/* Force context initialization */
	fdis[fd].ctx = general.ctx;
	
	cJSON *root = NULL;
	cJSON *messages = NULL;
	cJSON *tools = NULL;
	cJSON *stop = NULL;
	cJSON *stream_val = NULL;
	cJSON *model_val = NULL;
	char *formatted_prompt = NULL;
	int stream = 0;
	char *model_id = NULL;
	char completion_id[64];
	
	snprintf(completion_id, sizeof(completion_id), "chatcmpl-%ld", (long)time(NULL));
	
	if (!body || !*body) {
		http_error(fd, 400, "invalid_request_error", "Missing request body");
		return 1;
	}
	
	/* Parse JSON */
	root = cJSON_Parse(body);
	if (!root) {
		http_error(fd, 400, "invalid_request_error", "Invalid JSON");
		return 1;
	}
	
	/* Extract messages array */
	messages = cJSON_GetObjectItem(root, "messages");
	if (!messages || !cJSON_IsArray(messages)) {
		http_error(fd, 400, "invalid_request_error", "messages must be an array");
		cJSON_Delete(root);
		return 1;
	}
	
	/* Extract stream flag (optional, default false) */
	stream_val = cJSON_GetObjectItem(root, "stream");
	if (stream_val && cJSON_IsBool(stream_val)) {
		stream = cJSON_IsTrue(stream_val);
	}

	/* Extract stream_options - ignored for now as we always send usage */
	cJSON_GetObjectItem(root, "stream_options");

	
	/* Extract model (optional) */
	model_val = cJSON_GetObjectItem(root, "model");
	if (model_val && cJSON_IsString(model_val)) {
		model_id = model_val->valuestring;
	}

	/* Extract tools (optional, OpenAI/OpenCode-compatible) */
	tools = cJSON_GetObjectItem(root, "tools");
	
	/* Extract stop sequences (optional) */
	stop = cJSON_GetObjectItem(root, "stop");
	if (stop && cJSON_IsString(stop)) {
		/* Convert single string stop to array for internal consistency */
		cJSON *arr = cJSON_CreateArray();
		cJSON_AddItemToArray(arr, cJSON_CreateString(stop->valuestring));
		cJSON_ReplaceItemInObject(root, "stop", arr);
		stop = arr;
	}
	
	/* Format conversation - truncate system if it exceeds context budget
	 * Character budget is roughly 2x context size tokens */
	int max_sys = calculate_max_system(messages, tools, stream);
	formatted_prompt = format_conversation(messages, tools, model_template, 0, max_sys);
	if (!formatted_prompt) {
		http_error(fd, 500, "internal_error", "Failed to format conversation");
		cJSON_Delete(root);
		return 1;
	}
	
	if (stream) {
		/* Send streaming response in SSE format */
		http_sse_start(fd);
		
		/* Send start */
		oai_stream_role(fd, completion_id);
		
		/* Initialize fdi with shared context */
		fdi_init(&fdis[fd], fd);
		
		/* Apply isolated grammar sampler if tools are present to ensure valid tool calls */
		if (tools && cJSON_IsArray(tools) && cJSON_GetArraySize(tools) > 0) {
			fdis[fd].sampler = qllm_sampler_create(fdis[fd].ctx, &cfg);
			if (fdis[fd].sampler) {
				qllm_sampler_add_grammar(fdis[fd].ctx, fdis[fd].sampler, tool_call_grammar);
			}
		}

		/* Stream the response */
		fdi_t *fdi = &fdis[fd];
		char buf[4096];
		char generated[BUFSIZ * 16];
		size_t generated_len = 0;
		int ret;
		int step;
		int max_gen = 512;
		int completion_tokens = 0;

		generated[0] = '\0';
		
		qllm_anchor_start(fdi->ctx);
		if (qllm_prime(fdi->ctx, formatted_prompt) < 0) {
			ndc_writef(fd, "data: {\"error\":\"Generation failed\"}\n\n");
			ndc_writef(fd, "data: [DONE]\n\n");
			free(formatted_prompt);
			cJSON_Delete(root);
			ndc_close(fd);
			return 1;
		}
		qllm_anchor_end(fdi->ctx);
		
		for (step = 0; step < max_gen; step++) {
			ret = qllm_next(fdi->ctx, fdi->sampler, buf, sizeof(buf));
			if (ret <= 0)
				break;
			
			completion_tokens++;

			/* Check for delimiter */
			char *delim = memchr(buf, delimiter, ret);
			if (delim) {
				*delim = '\0';
				ret = delim - buf;
			}
			
			if (ret > 0) {
				size_t copy_len = (size_t)ret;
				if (copy_len > sizeof(generated) - generated_len - 1)
					copy_len = sizeof(generated) - generated_len - 1;
				if (copy_len) {
					memcpy(generated + generated_len, buf, copy_len);
					generated_len += copy_len;
					generated[generated_len] = '\0';
				}

				oai_stream_content(fd, completion_id, buf);
			}
			
			if (delim || check_stop_sequences(generated, stop))
				break;
		}
		
		tool_call_t tool_call;
		int has_tool_call = parse_tool_call(generated, &tool_call);
		if (has_tool_call)
			oai_stream_tool_call(fd, completion_id, &tool_call);

		/* Send stop */
		oai_stream_stop(fd, completion_id, has_tool_call ? "tool_calls" : "stop");

		/* Final usage chunk - always send for compatibility */
		int prompt_tokens = (int)strlen(formatted_prompt) / 4;
		ndc_writef(fd, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}\n\n",
			completion_id, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);

		ndc_writef(fd, "data: [DONE]\n\n");

		tool_call_free(&tool_call);
		ndc_close(fd);
		
	} else {
		/* Non-streaming response */
		/* Initialize fdi with shared context */
		fdi_init(&fdis[fd], fd);
		
		/* Apply isolated grammar sampler if tools are present */
		if (tools && cJSON_IsArray(tools) && cJSON_GetArraySize(tools) > 0) {
			fdis[fd].sampler = qllm_sampler_create(fdis[fd].ctx, &cfg);
			if (fdis[fd].sampler) {
				qllm_sampler_add_grammar(fdis[fd].ctx, fdis[fd].sampler, tool_call_grammar);
			}
		}

		if (!fdis[fd].ctx) {
			fprintf(stderr, "DEBUG: fdi ctx is NULL after fdi_init!\n");
			fflush(stderr);
			http_error(fd, 500, "internal_error", "Context not initialized");
			free(formatted_prompt);
			cJSON_Delete(root);
			return 1;
		}
		
		/* Generate response */
		fdi_t *fdi = &fdis[fd];
		char buf[4096];
		char full_response[BUFSIZ * 16];
		size_t full_len = 0;
		int ret;
		int step;
		int max_gen = 512;

		full_response[0] = '\0';
		
		qllm_anchor_start(fdi->ctx);
		if (qllm_prime(fdi->ctx, formatted_prompt) < 0) {
			http_error(fd, 500, "internal_error", "Generation failed");
			free(formatted_prompt);
			cJSON_Delete(root);
			return 1;
		}
		qllm_anchor_end(fdi->ctx);
		
		for (step = 0; step < max_gen; step++) {
			ret = qllm_next(fdi->ctx, fdi->sampler, buf, sizeof(buf));
			if (ret <= 0)
				break;
			
			/* Check for delimiter */
			char *delim = memchr(buf, delimiter, ret);
			if (delim) {
				*delim = '\0';
				ret = delim - buf;
			}
			
			if (ret > 0) {
				size_t copy_len = (size_t)ret;
				if (copy_len > sizeof(full_response) - full_len - 1)
					copy_len = sizeof(full_response) - full_len - 1;
				if (copy_len) {
					memcpy(full_response + full_len, buf, copy_len);
					full_len += copy_len;
					full_response[full_len] = '\0';
				}
			}
			
			if (delim)
				break;

			if (check_stop_sequences(full_response, stop)) {
				break;
			}
		}
		
		tool_call_t tool_call;
		int has_tool_call = parse_tool_call(full_response, &tool_call);
		char *model_name = model_id ? model_id : (*qllm_model_path ? qllm_model_path : "unknown");
		char *response_json = NULL;
		int prompt_tokens = (int)strlen(formatted_prompt) / 4;
		int completion_tokens = (int)full_len / 4;
		
		response_json = oai_completion_json(
			completion_id,
			model_name,
			has_tool_call ? tool_call.content : full_response,
			has_tool_call ? &tool_call : NULL,
			prompt_tokens,
			completion_tokens);
		tool_call_free(&tool_call);

		if (!response_json) {
			http_error(fd, 500, "internal_error", "Out of memory");
			free(formatted_prompt);
			cJSON_Delete(root);
			return 1;
		}
		
		http_json(fd, 200, response_json);
		free(response_json);
	}
	
	free(formatted_prompt);
	cJSON_Delete(root);
	return 1;
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
	int max_sys = calculate_max_system(messages, tools, stream);
	formatted_prompt = format_conversation(messages, tools, model_template, 0, max_sys);
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
	ndc_writef(fd, "{\"model\":\"%s\",\"template\":\"%s\",\"n_ctx\":%d}\n",
	          model_name, template_names[model_template], cfg.n_ctx);
}

static inline void
fdi_init(fdi_t *fdi, int fd)
{
	/* Use the shared context instead of creating a new one */
	/* This avoids the Vulkan multi-context crash */
	fdi->ctx = general.ctx;
	
	if (!fdi->ctx) {
		qsyslog(QLOG_ERR, "Shared context is NULL\n");
	} else {
		uint32_t seq_id = 0;

		if (cfg.n_contexts > 1)
			seq_id = (uint32_t)(fd % cfg.n_contexts);
		qllm_set_seq(fdi->ctx, seq_id);
	}

	fdi->sampler = NULL;
	fdi->queue.tail = 0;
	reset_fdi(fdi);
}

void
do_CHAT(int fd, int argc __attribute__((unused)), char *argv[] __attribute__((unused)))
{
	fdi_init(&fdis[fd], fd);
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

#ifndef MOCK_BUILD
int
__attribute__((used))
ndc_accept(int fd)
{
	fprintf(stderr, "DEBUG ndc_accept: fd=%d\n", fd);
	fflush(stderr);
#if FEAT_GENERAL
	fdis[fd].ctx = general.ctx;
	fprintf(stderr, "DEBUG ndc_accept: ctx=%p general.ctx=%p\n", fdis[fd].ctx, general.ctx);
	fflush(stderr);
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

	if (fdi->sampler)
		qllm_sampler_free(fdi->sampler);

	fdi->ctx = NULL;
	fdi->sampler = NULL;
	reset_fdi(fdi);
}
#endif

static void
usage(char *prog)
{
	fprintf(stderr, "Usage: %s [-dr?] [-C PATH] [-u USER] [-k PATH] [-c PATH] [-p PORT] [-S PROMPT] [-g LAYERS] MODEL\n", prog);
	fprintf(stderr, "    Options:\n");
	fprintf(stderr, "        -C PATH   changes directory to PATH before starting up.\n");
	fprintf(stderr, "        -u USER   login as USER before starting up.\n");
	fprintf(stderr, "        -k PATH   specify SSL certificate 'key' file\n");
	fprintf(stderr, "        -c PATH   specify SSL certificate 'crt' file\n");
	fprintf(stderr, "        -p PORT   specify server port (defaults to 4242)\n");
	fprintf(stderr, "        -d        don't detach\n");
	fprintf(stderr, "        -r        root multiplex mode\n");
	fprintf(stderr, "        -c SIZE   specify n_ctx (default auto-detect)\n");
	fprintf(stderr, "        -n NUM    specify max concurrent sequences (default 1)\n");
	fprintf(stderr, "        -S PROMPT set system prompt for all conversations\n");
    fprintf(stderr, "        -g LAYERS specify max GPU layers (0=auto, default 0)\n");
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

	/* Initialize FIM tokens */
	if (model_template == TEMPLATE_CHATML) {
		fim_prefix = "<|fim_prefix|>";
		fim_suffix = "<|fim_suffix|>";
		fim_middle = "<|fim_middle|>";
	} else {
		/* Default/Llama-style FIM tokens */
		fim_prefix = "<PRE>";
		fim_suffix = "<SUF>";
		fim_middle = "<MID>";
	}

	/* Create a SINGLE shared context that all connections will use */
	/* This avoids the Vulkan multi-context issue */
	fprintf(stderr, "qllmd: Creating shared context\n");
	general.ctx = qllm_create(&cfg);
	if (!general.ctx) {
		fprintf(stderr, "qllmd: FATAL - Failed to create shared context\n");
		exit(1);
	}
	cfg.n_ctx = qllm_n_ctx(general.ctx);
	fprintf(stderr, "qllmd: Shared context created successfully (n_ctx=%d)\n", cfg.n_ctx);

	crb_len = (size_t)ndc_mmap(&crb, "crb.txt");
	(void)crb_len;
}

#ifndef MOCK_BUILD
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

    while ((c = getopt(argc, argv, "?dK:k:C:rp:s:n:c:S:g:")) != -1) switch (c) {
        case 'g':
            cfg.n_gpu_layers = atoi(optarg);
            break;
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

    while ((c = getopt(argc, argv, "?dK:k:C:rp:s:n:c:g:")) != -1) switch (c) {
        case 'g':
            cfg.n_gpu_layers = atoi(optarg);
            break;
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

	/* Register OpenAI-compatible HTTP handlers */
	/* Format is METHOD:path (e.g., GET:/health, POST:/v1/chat/completions) */
	ndc_register_handler("GET:/v1/models", handle_v1_models);
	ndc_register_handler("POST:/v1/chat/completions", handle_v1_chat_completions);
	ndc_register_handler("POST:/v1/completions", handle_v1_completions);
	ndc_register_handler("GET:/health", handle_health);

	setup(arg_model);

	ret = ndc_main();

	if (general.ctx)
		qllm_free(general.ctx);

	return ret;
}
#endif

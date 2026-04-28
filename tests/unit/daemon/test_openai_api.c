#include "../framework/test_utils.h"
#include "../mocks/ndc_mock.h"
#include "../mocks/llama_mock.h"
#include "ttypt/qllm.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* Forward declarations for functions in qllmd.c that are not static or 
 * that we want to test. */
int handle_health(socket_t fd, char *body);
int handle_v1_models(socket_t fd, char *body);
int handle_v1_chat_completions(socket_t fd, char *body);
int handle_v1_completions(socket_t fd, char *body);

/* Tool call support */
typedef struct {
    char *content;
    char *name;
    char *arguments;
} tool_call_t;
int parse_tool_call(const char *output, tool_call_t *tc);
void tool_call_free(tool_call_t *tc);

/* We need to mock some global state that qllmd.c expects */
extern struct qllm_config cfg;
extern char qllm_model_path[1024];

TEST(openai_health_check)
{
    char buf[1024] = {0};
    mock_ndc_init();
    mock_ndc_set_write_capture(buf, sizeof(buf));

    int ret = handle_health(0, NULL);
    
    ASSERT_EQ(ret, 1);
    ASSERT_EQ(mock_ndc_get_last_status(), 200);
    ASSERT_STR_EQ(mock_ndc_get_header("Content-Type"), "application/json");
    ASSERT_TRUE(strstr(buf, "{\"status\":\"ok\"}") != NULL);

    mock_ndc_cleanup();
}

TEST(openai_models_list)
{
    char buf[1024] = {0};
    mock_ndc_init();
    mock_ndc_set_write_capture(buf, sizeof(buf));
    
    strcpy(qllm_model_path, "/path/to/test-model.gguf");

    int ret = handle_v1_models(0, NULL);
    
    ASSERT_EQ(ret, 1);
    ASSERT_EQ(mock_ndc_get_last_status(), 200);
    ASSERT_TRUE(strstr(buf, "test-model.gguf") != NULL);
    ASSERT_TRUE(strstr(buf, "\"object\":\"model\"") != NULL);

    mock_ndc_cleanup();
}

TEST(openai_chat_completions_invalid_json)
{
    char buf[1024] = {0};
    mock_ndc_init();
    mock_ndc_set_write_capture(buf, sizeof(buf));

    int ret = handle_v1_chat_completions(0, "invalid json");
    
    ASSERT_EQ(ret, 1);
    ASSERT_EQ(mock_ndc_get_last_status(), 400);
    ASSERT_TRUE(strstr(buf, "Invalid JSON") != NULL);

    mock_ndc_cleanup();
}

TEST(openai_chat_completions_missing_messages)
{
    char buf[1024] = {0};
    mock_ndc_init();
    mock_ndc_set_write_capture(buf, sizeof(buf));

    int ret = handle_v1_chat_completions(0, "{\"model\":\"test\"}");
    
    ASSERT_EQ(ret, 1);
    ASSERT_EQ(mock_ndc_get_last_status(), 400);
    ASSERT_TRUE(strstr(buf, "messages must be an array") != NULL);

    mock_ndc_cleanup();
}

/* Helper to setup a dummy qllm_context in the global general struct 
 * which handle_v1_chat_completions uses. */
typedef struct {
    unsigned tail;
} token_queue_t;

typedef struct fd_info {
    char            line_buf[BUFSIZ * 4];
    struct qllm_context *   ctx;
    struct llama_sampler *  sampler;
    unsigned        line_pos;
    token_queue_t        queue;
} fdi_t;

extern fdi_t general;
extern const char *fim_prefix;
extern const char *fim_suffix;
extern const char *fim_middle;

static int
count_substr(const char *haystack, const char *needle)
{
    int count = 0;
    const char *p = haystack;

    while ((p = strstr(p, needle)) != NULL) {
        count++;
        p += strlen(needle);
    }

    return count;
}

TEST(openai_chat_completions_success)
{
    char buf[4096] = {0};
    mock_ndc_init();
    mock_llama_init();
    mock_ndc_set_write_capture(buf, sizeof(buf));

    /* Setup mock qllm context */
    struct qllm_config qcfg = {
        .model_path = "mock_model.gguf",
        .n_ctx = 512
    };
    general.ctx = qllm_create(&qcfg);
    ASSERT_NOT_NULL(general.ctx);
    cfg.n_ctx = 512;

    /* Mock tokens to be returned by LLM */
    llama_token tokens[] = { 100, 101, 102, 2 }; /* 2 is usually EOS in our mocks */
    mock_llama_set_next_tokens(tokens, 4);

    const char *payload = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":false}";
    int ret = handle_v1_chat_completions(0, (char*)payload);
    
    ASSERT_EQ(ret, 1);
    ASSERT_EQ(mock_ndc_get_last_status(), 200);
    ASSERT_TRUE(strstr(buf, "\"object\":\"chat.completion\"") != NULL);
    ASSERT_TRUE(strstr(buf, "\"role\":\"assistant\"") != NULL);
    
    qllm_free(general.ctx);
    general.ctx = NULL;
    mock_llama_cleanup();
    mock_ndc_cleanup();
}

TEST(openai_chat_completions_high_fd_uses_valid_sequence)
{
    char buf[4096] = {0};
    mock_ndc_init();
    mock_llama_init();
    mock_ndc_set_write_capture(buf, sizeof(buf));

    struct qllm_config qcfg = {
        .model_path = "mock_model.gguf",
        .n_ctx = 512,
        .n_contexts = 1
    };
    general.ctx = qllm_create(&qcfg);
    ASSERT_NOT_NULL(general.ctx);
    cfg.n_ctx = 512;
    cfg.n_contexts = 1;

    llama_token tokens[] = { 200, 201, 2 };
    mock_llama_set_next_tokens(tokens, 3);

    const char *payload = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":false}";
    int ret = handle_v1_chat_completions(30, (char*)payload);

    ASSERT_EQ(ret, 1);
    ASSERT_EQ(mock_ndc_get_last_status(), 200);
    ASSERT_TRUE(strstr(buf, "\"object\":\"chat.completion\"") != NULL);

    qllm_free(general.ctx);
    general.ctx = NULL;
    cfg.n_contexts = 1;
    mock_llama_cleanup();
    mock_ndc_cleanup();
}

TEST(openai_chat_completions_generation_failure)
{
    char buf[4096] = {0};
    mock_ndc_init();
    mock_llama_init();
    mock_ndc_set_write_capture(buf, sizeof(buf));

    struct qllm_config qcfg = {
        .model_path = "mock_model.gguf",
        .n_ctx = 512
    };
    general.ctx = qllm_create(&qcfg);
    ASSERT_NOT_NULL(general.ctx);
    cfg.n_ctx = 512;

    mock_llama_set_decode_fail(1);

    const char *payload = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":false}";
    int ret = handle_v1_chat_completions(0, (char*)payload);

    ASSERT_EQ(ret, 1);
    ASSERT_EQ(mock_ndc_get_last_status(), 500);
    ASSERT_TRUE(strstr(buf, "Generation failed") != NULL);

    mock_llama_set_decode_fail(0);
    qllm_free(general.ctx);
    general.ctx = NULL;
    mock_llama_cleanup();
    mock_ndc_cleanup();
}

TEST(openai_chat_completions_stop_sequence)
{
    char buf[4096] = {0};
    mock_ndc_init();
    mock_llama_init();
    mock_ndc_set_write_capture(buf, sizeof(buf));

    struct qllm_config qcfg = {
        .model_path = "mock_model.gguf",
        .n_ctx = 512
    };
    general.ctx = qllm_create(&qcfg);
    ASSERT_NOT_NULL(general.ctx);
    cfg.n_ctx = 512;

    llama_token tokens[] = { 200, 201, 202, 2 };
    mock_llama_set_next_tokens(tokens, 4);

    const char *payload = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stop\":[\"e\"],\"stream\":false}";
    int ret = handle_v1_chat_completions(0, (char*)payload);

    ASSERT_EQ(ret, 1);
    ASSERT_EQ(mock_ndc_get_last_status(), 200);
    ASSERT_TRUE(strstr(buf, "\"content\":\"de\"") != NULL);
    ASSERT_TRUE(strstr(buf, "\"content\":\"def\"") == NULL);

    qllm_free(general.ctx);
    general.ctx = NULL;
    mock_llama_cleanup();
    mock_ndc_cleanup();
}

TEST(openai_completions_missing_prompt)
{
    char buf[1024] = {0};
    mock_ndc_init();
    mock_ndc_set_write_capture(buf, sizeof(buf));

    int ret = handle_v1_completions(0, "{\"model\":\"test\"}");

    ASSERT_EQ(ret, 1);
    ASSERT_EQ(mock_ndc_get_last_status(), 400);
    ASSERT_TRUE(strstr(buf, "prompt must be a string") != NULL);

    mock_ndc_cleanup();
}

TEST(openai_completions_fim)
{
    char buf[4096] = {0};
    mock_ndc_init();
    mock_llama_init();
    mock_ndc_set_write_capture(buf, sizeof(buf));

    struct qllm_config qcfg = {
        .model_path = "mock_model.gguf",
        .n_ctx = 512
    };
    general.ctx = qllm_create(&qcfg);
    ASSERT_NOT_NULL(general.ctx);

    /* Initialize FIM tokens which are normally set in setup() */
    fim_prefix = "<PRE>";
    fim_suffix = "<SUF>";
    fim_middle = "<MID>";

    llama_token tokens[] = { 200, 201, 2 };
    mock_llama_set_next_tokens(tokens, 3);

    const char *payload = "{\"prompt\":\"def hello():\", \"suffix\":\"return True\", \"model\":\"test\"}";
    int ret = handle_v1_completions(0, (char*)payload);
    
    ASSERT_EQ(ret, 1);
    ASSERT_EQ(mock_ndc_get_last_status(), 200);
    ASSERT_TRUE(strstr(buf, "\"object\":\"text_completion\"") != NULL);
    
    qllm_free(general.ctx);
    general.ctx = NULL;
    mock_llama_cleanup();
    mock_ndc_cleanup();
}

TEST(parse_tool_call_basic)
{
    tool_call_t tc;
    const char *output = "Sure, I can help with that. <tool_call>get_weather|{\"city\":\"London\"}</tool_call>";
    
    int ret = parse_tool_call(output, &tc);
    
    ASSERT_EQ(ret, 1);
    ASSERT_STR_EQ(tc.name, "get_weather");
    ASSERT_STR_EQ(tc.arguments, "{\"city\":\"London\"}");
    ASSERT_STR_EQ(tc.content, "Sure, I can help with that. ");
    
    tool_call_free(&tc);
}

TEST(parse_tool_call_invalid)
{
    tool_call_t tc;
    ASSERT_EQ(parse_tool_call("no tool call here", &tc), 0);
    ASSERT_EQ(parse_tool_call("<tool_call>incomplete|", &tc), 0);
    ASSERT_EQ(parse_tool_call("<tool_call>name_only</tool_call>", &tc), 0);
}

TEST(openai_chat_completions_streaming_usage)
{
    char buf[8192] = {0};
    mock_ndc_init();
    mock_llama_init();
    mock_ndc_set_write_capture(buf, sizeof(buf));

    struct qllm_config qcfg = {
        .model_path = "mock_model.gguf",
        .n_ctx = 512
    };
    general.ctx = qllm_create(&qcfg);
    ASSERT_NOT_NULL(general.ctx);

    llama_token tokens[] = { 100, 101, 2 };
    mock_llama_set_next_tokens(tokens, 3);

    const char *payload = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"stream\":true,\"stream_options\":{\"include_usage\":true}}";
    int ret = handle_v1_chat_completions(0, (char*)payload);
    
    ASSERT_EQ(ret, 1);
    /* Should contain usage info at the end of the stream */
    ASSERT_TRUE(strstr(buf, "\"usage\":{\"prompt_tokens\":") != NULL);
    ASSERT_TRUE(strstr(buf, "data: [DONE]") != NULL);
    ASSERT_EQ(count_substr(buf, "data: [DONE]"), 1);
    ASSERT_TRUE(strstr(buf, "\"usage\":{\"prompt_tokens\":") < strstr(buf, "data: [DONE]"));
    
    qllm_free(general.ctx);
    general.ctx = NULL;
    mock_llama_cleanup();
    mock_ndc_cleanup();
}

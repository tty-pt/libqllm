#include "../framework/test_utils.h"
#include "../mocks/llama_mock.h"
#include "../mocks/qmap_mock.h"
#include "../mocks/gguf_mock.h"
#include "ttypt/qllm.h"
#include <stdlib.h>
#include <string.h>

/* Redefine struct qllm_context for white-box testing */
struct qllm_context {
	uint32_t		 magic;
	struct llama_model	*model;
	struct llama_context	*ctx;
	struct llama_sampler	*sampler;
	struct llama_sampler	*grammar_sampler;
	struct llama_sampler	*sampler_children[8];
	int32_t			sampler_children_n;
	struct llama_context_params params;
    const struct llama_vocab *vocab;
    char *model_path;
	int32_t			 n_embd;
	int32_t			 max_tokens;
	llama_pos		 anchor_start, anchor_end;
	llama_token		*token_buf;
	llama_seq_id		*seq_ids;
	llama_seq_id		 current_seq;
	int32_t gen_tokens;
	int32_t eos_start;
	float   eos_bias_max;
};

TEST(qllm_n_ctx_basic)
{
    struct qllm_config cfg = {
        .model_path = "/fake/model.gguf",
        .n_ctx = 1024
    };
    mock_llama_init();
    mock_qmap_init();
    
    struct qllm_context *ctx = qllm_create(&cfg);
    ASSERT_NOT_NULL(ctx);
    ASSERT_EQ(qllm_n_ctx(ctx), 1024);
    
    qllm_free(ctx);
}

TEST(qllm_set_seq_basic)
{
    struct qllm_config cfg = {
        .model_path = "/fake/model.gguf",
        .n_ctx = 512
    };
    mock_llama_init();
    mock_qmap_init();
    
    struct qllm_context *ctx = qllm_create(&cfg);
    ASSERT_NOT_NULL(ctx);
    
    qllm_set_seq(ctx, 42);
    ASSERT_EQ(ctx->current_seq, 42);
    
    qllm_free(ctx);
}

TEST(qllm_set_grammar_basic)
{
    struct qllm_config cfg = {
        .model_path = "/fake/model.gguf",
        .n_ctx = 512
    };
    mock_llama_init();
    mock_qmap_init();
    
    struct qllm_context *ctx = qllm_create(&cfg);
    ASSERT_NOT_NULL(ctx);
    ASSERT_NULL(ctx->grammar_sampler);
    
    /* Set a grammar */
    int ret = qllm_set_grammar(ctx, "root ::= [a-z]+");
    ASSERT_EQ(ret, 0);
    ASSERT_NOT_NULL(ctx->grammar_sampler);
    
    /* Clear grammar */
    ret = qllm_set_grammar(ctx, NULL);
    ASSERT_EQ(ret, 0);
    ASSERT_NULL(ctx->grammar_sampler);
    
    qllm_free(ctx);
}

TEST(qllm_set_grammar_replace)
{
    struct qllm_config cfg = {
        .model_path = "/fake/model.gguf",
        .n_ctx = 512
    };
    mock_llama_init();
    mock_qmap_init();
    
    struct qllm_context *ctx = qllm_create(&cfg);
    ASSERT_NOT_NULL(ctx);
    
    qllm_set_grammar(ctx, "root ::= [0-9]+");
    struct llama_sampler *first = ctx->grammar_sampler;
    ASSERT_NOT_NULL(first);
    
    /* Setting new grammar should free old one and set new one */
    qllm_set_grammar(ctx, "root ::= [a-f]+");
    ASSERT_NOT_NULL(ctx->grammar_sampler);
    ASSERT_TRUE(ctx->grammar_sampler != first);
    
    qllm_free(ctx);
}

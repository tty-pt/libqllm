#include "../framework/test_utils.h"
#include "../mocks/llama_mock.h"
#include "../mocks/corm_mock.h"
#include "../mocks/gguf_mock.h"
#include "../../include/ttypt/qllm.h"
#include <stdlib.h>
#include <string.h>

/* Redefine struct qllm_context for white-box testing; must mirror the
 * layout in src/libqllm.c exactly (offsets are relied upon). */
struct qllm_context {
	uint32_t		 magic;
	struct llama_model	*model;
	struct llama_context	*ctx;
	struct llama_sampler	*sampler;
	struct llama_sampler	*grammar_sampler;
	struct llama_sampler	*sampler_children[8];
	int32_t			 sampler_children_n;
	struct llama_context_params params;
	const struct llama_vocab *vocab;
	char			*model_path;
	int32_t			 n_embd;
	int32_t			 max_tokens;
	llama_pos		 anchor_start, anchor_end;
	llama_seq_id		 current_seq;
	llama_token		*token_buf;
	llama_seq_id		*seq_ids;
	int32_t			 gen_tokens;
	int32_t			 eos_start;
	float			 eos_bias_max;
};

TEST(qllm_n_ctx_basic)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = 1024,
		.n_threads = 1,
	};

	mock_llama_init();
	mock_corm_init();

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NOT_NULL(ctx);
	ASSERT_EQ(qllm_n_ctx(ctx), 1024);

	qllm_free(ctx);
}

TEST(qllm_set_seq_basic)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = 512,
		.n_threads = 1,
	};

	mock_llama_init();
	mock_corm_init();

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
		.n_ctx = 512,
		.n_threads = 1,
	};

	mock_llama_init();
	mock_corm_init();

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NOT_NULL(ctx);
	ASSERT_NULL(ctx->grammar_sampler);

	int ret = qllm_set_grammar(ctx, "root ::= [a-z]+");
	ASSERT_EQ(ret, 0);
	ASSERT_NOT_NULL(ctx->grammar_sampler);

	ret = qllm_set_grammar(ctx, NULL);
	ASSERT_EQ(ret, 0);
	ASSERT_NULL(ctx->grammar_sampler);

	qllm_free(ctx);
}

TEST(qllm_set_grammar_replace)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = 512,
		.n_threads = 1,
	};

	mock_llama_init();
	mock_corm_init();
	mock_llama_reset_counts();

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NOT_NULL(ctx);

	qllm_set_grammar(ctx, "root ::= [0-9]+");
	struct llama_sampler *first = ctx->grammar_sampler;
	ASSERT_NOT_NULL(first);
	ASSERT_EQ(mock_llama_get_grammar_init_count(), 1);

	qllm_set_grammar(ctx, "root ::= [a-f]+");
	ASSERT_NOT_NULL(ctx->grammar_sampler);
	/* The mock allocator may reuse the freed address, so verify the replace
	 * via the init counter instead of pointer identity. */
	ASSERT_EQ(mock_llama_get_grammar_init_count(), 2);

	qllm_free(ctx);
}

TEST(qllm_sampler_api_invalid_args)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = 512,
		.n_threads = 1,
	};

	mock_llama_init();
	mock_corm_init();

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NOT_NULL(ctx);

	struct llama_sampler *smpl = qllm_sampler_create(NULL, &cfg);
	ASSERT_NULL(smpl);

	smpl = qllm_sampler_create(ctx, &cfg);
	ASSERT_NOT_NULL(smpl);

	ASSERT_EQ(qllm_sampler_add_grammar(ctx, NULL, "root ::= [a-z]+"), -1);
	ASSERT_EQ(qllm_sampler_add_grammar(ctx, smpl, NULL), -1);
	ASSERT_EQ(qllm_sampler_add_grammar(ctx, smpl, "root ::= [a-z]+"), 0);

	qllm_sampler_free(smpl);
	qllm_free(ctx);
}

TEST(qllm_sampler_create_sampling_config)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = 512,
		.n_threads = 1,
		.temperature = 0.8f,
		.top_k = 50,
		.top_p = 0.9f,
		.repeat_penalty = 1.2f,
		.repeat_last_n = 32,
	};

	mock_llama_init();
	mock_corm_init();

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NOT_NULL(ctx);

	/* The create-time chain is populated from the config; with all
	 * sampling knobs set we expect eos_bias + penalties + top_k +
	 * top_p + temp + dist to be registered. */
	ASSERT_TRUE(ctx->sampler_children_n >= 6);

	qllm_free(ctx);
}
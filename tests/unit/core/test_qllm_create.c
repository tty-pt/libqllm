#include "../framework/test_utils.h"
#include "../mocks/llama_mock.h"
#include "../mocks/qmap_mock.h"
#include "../mocks/gguf_mock.h"
#include "../../include/ttypt/qllm.h"

TEST(qllm_create_null_config)
{
	struct qllm_context *ctx = qllm_create(NULL);
	ASSERT_NULL(ctx);
}

TEST(qllm_create_null_model_path)
{
	struct qllm_config cfg = {
		.model_path = NULL,
		.n_ctx = 512,
		.n_threads = 1,
	};

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NULL(ctx);
}

TEST(qllm_create_empty_model_path)
{
	struct qllm_config cfg = {
		.model_path = "",
		.n_ctx = 512,
		.n_threads = 1,
	};

	mock_llama_set_model_load_fail(1);
	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NULL(ctx);
}

TEST(qllm_create_model_load_fail)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = 512,
		.n_threads = 1,
	};

	mock_llama_init();
	mock_qmap_init();
	mock_llama_set_model_load_fail(1);

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NULL(ctx);
}

TEST(qllm_create_context_create_fail)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = 512,
		.n_threads = 1,
	};

	mock_llama_init();
	mock_qmap_init();
	mock_llama_set_model_load_fail(0);
	mock_llama_set_context_create_fail(1);

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NULL(ctx);
}

TEST(qllm_create_sampler_create_fail)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = 512,
		.n_threads = 1,
	};

	mock_llama_init();
	mock_qmap_init();
	mock_llama_set_model_load_fail(0);
	mock_llama_set_context_create_fail(0);
	mock_llama_set_sampler_create_fail(1);

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NULL(ctx);
}

TEST(qllm_create_success_basic)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = 512,
		.n_threads = 1,
	};

	mock_llama_init();
	mock_qmap_init();
	mock_llama_set_model_load_fail(0);
	mock_llama_set_context_create_fail(0);
	mock_llama_set_sampler_create_fail(0);

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NOT_NULL(ctx);

	qllm_free(ctx);
}

TEST(qllm_create_default_n_ctx)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = 0,
		.n_threads = 1,
	};

	mock_llama_init();
	mock_qmap_init();

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NOT_NULL(ctx);

	qllm_free(ctx);
}

TEST(qllm_create_default_n_threads)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = 512,
		.n_threads = 0,
	};

	mock_llama_init();
	mock_qmap_init();

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NOT_NULL(ctx);

	qllm_free(ctx);
}

TEST(qllm_free_null)
{
	qllm_free(NULL);
	ASSERT_TRUE(1);
}

TEST(qllm_create_negative_n_ctx)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = -1,
		.n_threads = 1,
	};

	mock_llama_init();
	mock_qmap_init();

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NOT_NULL(ctx);

	qllm_free(ctx);
}

TEST(qllm_create_negative_n_threads)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = 512,
		.n_threads = -1,
	};

	mock_llama_init();
	mock_qmap_init();

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NOT_NULL(ctx);

	qllm_free(ctx);
}

TEST(qllm_create_max_offload_zero)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = 512,
		.n_threads = 1,
		.max_offload_bytes = 0,
	};

	mock_llama_init();
	mock_qmap_init();

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NOT_NULL(ctx);

	qllm_free(ctx);
}

TEST(qllm_create_n_contexts_zero)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = 512,
		.n_threads = 1,
		.n_contexts = 0,
	};

	mock_llama_init();
	mock_qmap_init();

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NOT_NULL(ctx);

	qllm_free(ctx);
}

TEST(qllm_create_n_contexts_negative)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = 512,
		.n_threads = 1,
		.n_contexts = -5,
	};

	mock_llama_init();
	mock_qmap_init();

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NOT_NULL(ctx);

	qllm_free(ctx);
}

TEST(qllm_free_double_free)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = 512,
		.n_threads = 1,
	};

	mock_llama_init();
	mock_qmap_init();

	struct qllm_context *ctx = qllm_create(&cfg);
	ASSERT_NOT_NULL(ctx);

	qllm_free(ctx);
	qllm_free(ctx);

	ASSERT_TRUE(1);
}

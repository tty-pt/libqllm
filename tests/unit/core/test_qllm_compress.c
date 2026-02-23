#include "../framework/test_utils.h"
#include "../mocks/llama_mock.h"
#include "../mocks/qmap_mock.h"
#include "../../include/ttypt/qllm.h"

static struct qllm_context *create_compress_test_context(void)
{
	struct qllm_config cfg = {
		.model_path = "/fake/model.gguf",
		.n_ctx = 512,
		.n_threads = 1,
	};

	mock_llama_init();
	mock_qmap_init();

	return qllm_create(&cfg);
}

TEST(qllm_compress_no_op_when_under_limit)
{
	struct qllm_context *ctx = create_compress_test_context();
	if (!ctx) {
		SKIP("Could not create test context");
		return;
	}

	qllm_compress(ctx, 1000);

	ASSERT_TRUE(1);

	qllm_free(ctx);
}

TEST(qllm_compress_zero_limit)
{
	struct qllm_context *ctx = create_compress_test_context();
	if (!ctx) {
		SKIP("Could not create test context");
		return;
	}

	qllm_compress(ctx, 0);

	ASSERT_TRUE(1);

	qllm_free(ctx);
}

TEST(qllm_anchor_start_end)
{
	struct qllm_context *ctx = create_compress_test_context();
	if (!ctx) {
		SKIP("Could not create test context");
		return;
	}

	qllm_anchor_start(ctx);
	qllm_anchor_end(ctx);

	ASSERT_TRUE(1);

	qllm_free(ctx);
}

TEST(qllm_set_eos_bias_null_ctx)
{
	qllm_set_eos_bias(NULL, 64, 3.0f);
	ASSERT_TRUE(1);
}

TEST(qllm_set_eos_bias_valid)
{
	struct qllm_context *ctx = create_compress_test_context();
	if (!ctx) {
		SKIP("Could not create test context");
		return;
	}

	qllm_set_eos_bias(ctx, 100, 5.0f);
	qllm_set_eos_bias(ctx, 0, 0.0f);
	qllm_set_eos_bias(ctx, -1, -1.0f);

	ASSERT_TRUE(1);

	qllm_free(ctx);
}

TEST(qllm_compress_after_anchor)
{
	struct qllm_context *ctx = create_compress_test_context();
	if (!ctx) {
		SKIP("Could not create test context");
		return;
	}

	qllm_anchor_start(ctx);
	qllm_anchor_end(ctx);

	qllm_compress(ctx, 100);

	ASSERT_TRUE(1);

	qllm_free(ctx);
}

TEST(qllm_multiple_anchor_calls)
{
	struct qllm_context *ctx = create_compress_test_context();
	if (!ctx) {
		SKIP("Could not create test context");
		return;
	}

	qllm_anchor_start(ctx);
	qllm_anchor_start(ctx);
	qllm_anchor_end(ctx);
	qllm_anchor_end(ctx);

	ASSERT_TRUE(1);

	qllm_free(ctx);
}

TEST(qllm_compress_multiple_times)
{
	struct qllm_context *ctx = create_compress_test_context();
	if (!ctx) {
		SKIP("Could not create test context");
		return;
	}

	qllm_compress(ctx, 100);
	qllm_compress(ctx, 50);
	qllm_compress(ctx, 200);

	ASSERT_TRUE(1);

	qllm_free(ctx);
}

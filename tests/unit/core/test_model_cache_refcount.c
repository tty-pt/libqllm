#include "../framework/test_utils.h"
#include "../mocks/llama_mock.h"
#include "../mocks/qmap_mock.h"
#include "../../include/ttypt/qllm.h"

TEST(model_cache_refcount_same_model)
{
    struct qllm_config cfg = {
        .model_path = "/fake/model-a.gguf",
        .n_ctx = 512,
        .n_threads = 1,
    };

    mock_llama_init();
    mock_qmap_init();
    mock_llama_set_model_load_fail(0);
    mock_llama_set_context_create_fail(0);
    mock_llama_set_sampler_create_fail(0);
    mock_llama_reset_counts();

    struct qllm_context *ctx1 = qllm_create(&cfg);
    ASSERT_NOT_NULL(ctx1);

    struct qllm_context *ctx2 = qllm_create(&cfg);
    ASSERT_NOT_NULL(ctx2);

    qllm_free(ctx1);
    ASSERT_EQ(mock_llama_get_model_free_count(), 0);

    qllm_free(ctx2);
    ASSERT_EQ(mock_llama_get_model_free_count(), 1);
}

TEST(model_cache_refcount_different_models)
{
    struct qllm_config cfg1 = { .model_path = "/fake/model-a.gguf", .n_ctx = 512, .n_threads = 1 };
    struct qllm_config cfg2 = { .model_path = "/fake/model-b.gguf", .n_ctx = 512, .n_threads = 1 };

    mock_llama_init();
    mock_qmap_init();
    mock_llama_set_model_load_fail(0);
    mock_llama_set_context_create_fail(0);
    mock_llama_set_sampler_create_fail(0);
    mock_llama_reset_counts();

    struct qllm_context *a1 = qllm_create(&cfg1);
    ASSERT_NOT_NULL(a1);
    struct qllm_context *b1 = qllm_create(&cfg2);
    ASSERT_NOT_NULL(b1);

    qllm_free(a1);
    ASSERT_EQ(mock_llama_get_model_free_count(), 1);

    qllm_free(b1);
    ASSERT_EQ(mock_llama_get_model_free_count(), 2);
}

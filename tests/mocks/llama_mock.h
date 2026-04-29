#ifndef LLAMA_MOCK_H
#define LLAMA_MOCK_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t llama_token;
typedef int32_t llama_seq_id;
typedef int32_t llama_pos;

struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_vocab;
struct llama_memory;

typedef struct llama_batch {
	int32_t n_tokens;
	llama_token *token;
	int32_t *n_seq_id;
	llama_seq_id **seq_id;
	llama_pos *pos;
	int32_t *logits;
} llama_batch;

typedef enum {
	LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
	LLAMA_POOLING_TYPE_NONE = 0,
	LLAMA_POOLING_TYPE_MEAN = 1,
	LLAMA_POOLING_TYPE_CLS = 2,
	LLAMA_POOLING_TYPE_LAST = 3,
} llama_pooling_type_t;

typedef enum {
	LLAMA_SPLIT_MODE_NONE = 0,
	LLAMA_SPLIT_MODE_LAYER = 1,
	LLAMA_SPLIT_MODE_ROW = 2,
} llama_split_mode_t;

typedef struct llama_model_params {
	int32_t n_gpu_layers;
	int split_mode;
	const float * tensor_split;
	bool vocab_only;
	bool use_mmap;
	bool use_mlock;
	bool check_tensors;
} llama_model_params;

size_t llama_max_devices(void);

typedef struct llama_context_params {
	uint32_t n_ctx;
	uint32_t n_batch;
	uint32_t n_ubatch;
	uint32_t n_seq_max;
	int32_t n_threads;
	int32_t n_threads_batch;
	bool embeddings;
	int pooling_type;
} llama_context_params;

typedef struct llama_sampler_chain_params {
	bool no_perf;
} llama_sampler_chain_params;

typedef struct llama_token_data {
	llama_token id;
	float logit;
	float p;
} llama_token_data;

typedef struct llama_token_data_array {
	llama_token_data *data;
	size_t size;
	int64_t selected;
	bool sorted;
} llama_token_data_array;

typedef struct llama_sampler_i {
	const char *name;
	void (*accept)(struct llama_sampler *smpl, llama_token token);
	void (*apply)(struct llama_sampler *smpl, llama_token_data_array *cur_p);
	void (*reset)(struct llama_sampler *smpl);
	struct llama_sampler *(*clone)(const struct llama_sampler *smpl);
	void (*free)(struct llama_sampler *smpl);
} llama_sampler_i;

struct llama_sampler {
	const llama_sampler_i *iface;
	void *ctx;
};

typedef void *llama_memory_t;

typedef struct {
	const char *model_path;
	int32_t n_ctx;
	uint32_t max_offload_bytes;
	int32_t n_contexts;
	int32_t n_threads;
} mock_llama_config_t;

void mock_llama_init(void);
void mock_llama_cleanup(void);
void mock_llama_set_config(mock_llama_config_t *config);

void mock_llama_set_n_layer(int32_t n_layer);
void mock_llama_set_n_embd(int32_t n_embd);
void mock_llama_set_vocab_size(int32_t size);

void mock_llama_set_model_load_fail(int should_fail);
void mock_llama_set_context_create_fail(int should_fail);
void mock_llama_set_sampler_create_fail(int should_fail);
void mock_llama_set_tokenize_fail(int should_fail);
void mock_llama_set_decode_fail(int should_fail);

void mock_llama_set_tokenizer_result(const llama_token *tokens, int32_t n_tokens);
void mock_llama_set_next_tokens(const llama_token *tokens, int32_t n_tokens);

void mock_llama_set_eos_token(llama_token tok);
void mock_llama_set_eot_token(llama_token tok);

int mock_llama_get_model_load_count(void);
int mock_llama_get_context_create_count(void);
int mock_llama_get_decode_count(void);
int mock_llama_get_sample_count(void);

int mock_llama_get_model_free_count(void);

void mock_llama_reset_counts(void);

struct llama_model *llama_model_load_from_file(const char *path,
					       struct llama_model_params params);
void llama_model_free(struct llama_model *model);

int32_t llama_model_n_layer(const struct llama_model *model);
int32_t llama_model_n_embd(const struct llama_model *model);
const struct llama_vocab *llama_model_get_vocab(const struct llama_model *model);

struct llama_context *llama_init_from_model(struct llama_model *model,
					    struct llama_context_params params);
void llama_free(struct llama_context *ctx);

llama_memory_t llama_get_memory(struct llama_context *ctx);
llama_pos llama_memory_seq_pos_max(llama_memory_t mem, llama_seq_id seq_id);
void llama_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id,
			 llama_pos p0, llama_pos p1);
void llama_memory_seq_add(llama_memory_t mem, llama_seq_id seq_id,
			  llama_pos p0, llama_pos p1, llama_pos delta);

struct llama_batch llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
void llama_batch_free(struct llama_batch batch);

int32_t llama_tokenize(const struct llama_vocab *vocab,
		       const char *text,
		       int32_t text_len,
		       llama_token *tokens,
		       int32_t n_tokens_max,
		       bool add_bos,
		       bool special);

int32_t llama_token_to_piece(const struct llama_vocab *vocab,
			     llama_token token,
			     char *buf,
			     int32_t length,
			     bool add_space,
			     bool special);

llama_token llama_vocab_eos(const struct llama_vocab *vocab);
llama_token llama_vocab_eot(const struct llama_vocab *vocab);
bool llama_vocab_is_eog(const struct llama_vocab *vocab, llama_token token);
bool llama_vocab_is_control(const struct llama_vocab *vocab, llama_token token);

int llama_decode(struct llama_context *ctx, struct llama_batch batch);

const float *llama_get_embeddings(struct llama_context *ctx);

struct llama_sampler *llama_sampler_chain_init(struct llama_sampler_chain_params params);
void llama_sampler_chain_add(struct llama_sampler *chain, struct llama_sampler *sampler);
void llama_sampler_free(struct llama_sampler *smpl);

struct llama_sampler *llama_sampler_init(const struct llama_sampler_i *iface,
				 void *ctx);

struct llama_sampler *llama_sampler_init_penalties(
	int32_t last_n,
	float repeat,
	float freq,
	float present);

struct llama_sampler *llama_sampler_init_dist(uint64_t seed);

/* Prototypes added to match libqllm usages */
struct llama_sampler *llama_sampler_init_top_k(int32_t top_k);
/* Note: mock implementation accepts a float second parameter (tail-free freq)
 * to match the mock's simplified sampling interface used in tests. */
struct llama_sampler *llama_sampler_init_top_p(float top_p, float tail_free_freq_z);
struct llama_sampler *llama_sampler_init_temp(float temp);

llama_token llama_sampler_sample(struct llama_sampler *smpl,
				 struct llama_context *ctx,
				 int32_t idx);
void llama_sampler_accept(struct llama_sampler *smpl, llama_token token);

struct llama_model_params llama_model_default_params(void);
struct llama_context_params llama_context_default_params(void);
struct llama_sampler_chain_params llama_sampler_chain_default_params(void);

void llama_backend_init(void);
void llama_backend_free(void);

#ifdef __cplusplus
}
#endif

#endif

/* libqllm.c */

#include "./../include/ttypt/qllm.h"

#include <ctype.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

#include <llama.h>
#include <gguf.h>

#include <ttypt/qsys.h>
#include <ttypt/corm.h>

struct qllm_context {
	uint32_t		 magic;
	struct llama_model	*model;
	struct llama_context	*ctx;
	struct llama_sampler	*sampler;
	struct llama_sampler	*grammar_sampler;
	struct llama_sampler	*sampler_children[8];
	int32_t			 sampler_children_n;
	struct llama_context_params params; /* <-- add this */
	const struct llama_vocab *vocab;
	/* copy of model path so we can look up the cache entry on free */
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

#define QLLM_MAGIC 0x514C4C4D

/* Test harness hooks: allow the tests framework to track freed qllm_context
 * pointers so test builds can avoid dereferencing freed memory. In test
 * builds the functions qllm_ptr_freed and qllm_record_freed are provided by
 * the test harness; in production builds they are no-ops. */
#if defined(MOCK_BUILD)
int qllm_ptr_freed(const void *p);
void qllm_record_freed(void *p);
void qllm_unrecord_freed(void *p);
#else
static inline int qllm_ptr_freed(const void *p) { (void)p; return 0; }
static inline void qllm_record_freed(void *p) { (void)p; }
static inline void qllm_unrecord_freed(void *p) { (void)p; }
#endif

#define QLLM_VALID(qctx) \
	((qctx) && !qllm_ptr_freed((const void *)(qctx)) && \
	 (qctx)->magic == QLLM_MAGIC)

static int qllm_backend_inited;
static uint32_t qm_model, model_hd;

/* Reference-counted handles to the shared (cached) model instances so that
 * multiple qllm_create() contexts can share one model while any of them may
 * call qllm_free() without destroying the model out from under the others.
 * All model access is serialized on model_lock (loads are rare). */
static struct shared_model {
	struct llama_model	*model;
	int			 refs;
	struct shared_model	*next;
} *shared_models;
static pthread_mutex_t model_lock = PTHREAD_MUTEX_INITIALIZER;

static void
shared_model_ref(struct llama_model *m)
{
	struct shared_model *sm;

	for (sm = shared_models; sm; sm = sm->next)
		if (sm->model == m) {
			++sm->refs;
			return;
		}
	sm = calloc(1, sizeof(*sm));
	if (!sm)
		return;
	sm->model = m;
	sm->refs = 1;
	sm->next = shared_models;
	shared_models = sm;
}

/* Release a reference. On the last reference the cache entry is cleared so
 * a later qllm_create() for the same path doesn't resurrect a dangling
 * model pointer from the corm map. */
static void
shared_model_unref(struct llama_model *m, const char *path)
{
	struct shared_model *sm, **p;

	for (p = &shared_models; *p; p = &(*p)->next)
		if ((*p)->model == m)
			break;
	sm = *p;
	if (!sm) {
		llama_model_free(m);
		return;
	}
	if (--sm->refs > 0)
		return;
	*p = sm->next;
	free(sm);
	llama_model_free(m);
	if (path && *path) {
		struct llama_model *nul = NULL;

		corm_put(model_hd, path, &nul);
	}
}

/* Initialize llama backend exactly once. */
__attribute__((constructor)) void 
qllm_init(void)
{
	qm_model = corm_reg(sizeof(struct llama_model *));
	model_hd = corm_open(NULL, NULL, CM_STR, qm_model, 0, 0);
	llama_backend_init();
	qllm_backend_inited = 1;
}

/* Small helper to decode a batch of tokens at the current position.
 * With all_logits, every token is marked as an output (required for
 * mean-pooled embeddings so the pooling op sees all token embeddings);
 * otherwise only the last token is marked (chat/generation).
 *
 * Positions are auto-assigned by llama (batch.pos = NULL) so that the
 * sliding-window shift done by qllm_compress() stays coherent with later
 * decodes. */
static int
qllm_decode_tokens(struct qllm_context *qctx,
		   const llama_token *tokens,
		   int32_t n_tokens,
		   int all_logits)
{
	struct llama_batch batch;
	int32_t i;

	if (!qctx || !qctx->ctx || !tokens || n_tokens <= 0)
		return -1;

	if (n_tokens > qctx->max_tokens)
		return -1;

	batch = llama_batch_init(n_tokens, 0, 1);
	batch.n_tokens = n_tokens;
	batch.pos = NULL;

	for (i = 0; i < n_tokens; ++i) {
		batch.token[i] = tokens[i];
		batch.n_seq_id[i] = 1;
		batch.seq_id[i] = &qctx->current_seq;
		batch.logits[i] = all_logits || (i == n_tokens - 1);
	}

	if (llama_decode(qctx->ctx, batch) != 0)
		return -1;

	return 0;
}

extern void
qllm_backend_mem_check(int gpu, size_t *free_b, size_t *total_b);

/* Extern from vulkan.c */
extern int
qllm_backend_get_vram(size_t *free_b, size_t *total_b, int max_devices);

/* Read model geometry directly from the GGUF metadata without loading the
 * model. Used to pick a sane default context size (cfg->n_ctx == 0). All
 * outputs start at 0 and stay 0 when the metadata is absent. */
static void
get_model_dims(const char *path, int *n_layers, int *n_embd, int *n_ctx,
	       int *n_head, int *n_kv_head)
{
	struct gguf_init_params ip = { .no_alloc = true };
	struct gguf_context *ctx;
	const char *arch;
	char key[128];
	int arch_key;

	*n_layers = 0;
	*n_embd = 0;
	*n_ctx = 0;
	*n_head = 0;
	*n_kv_head = 0;

	ctx = gguf_init_from_file(path, ip);
	if (!ctx)
		return;

	arch_key = (int)gguf_find_key(ctx, "general.architecture");
	if (arch_key < 0) {
		gguf_free(ctx);
		return;
	}

	arch = gguf_get_val_str(ctx, arch_key);
	if (!arch) {
		gguf_free(ctx);
		return;
	}

	snprintf(key, sizeof(key), "%s.block_count", arch);
	{
		enum gguf_type type;
		int64_t blk = gguf_find_key(ctx, key);

		if (blk >= 0) {
			type = gguf_get_kv_type(ctx, blk);
			switch (type) {
			case GGUF_TYPE_UINT8:
				*n_layers = gguf_get_val_u8(ctx, blk);
				break;
			case GGUF_TYPE_INT8:
				*n_layers = gguf_get_val_i8(ctx, blk);
				break;
			case GGUF_TYPE_UINT16:
				*n_layers = gguf_get_val_u16(ctx, blk);
				break;
			case GGUF_TYPE_INT16:
				*n_layers = gguf_get_val_i16(ctx, blk);
				break;
			case GGUF_TYPE_UINT32:
				*n_layers = (int)gguf_get_val_u32(ctx, blk);
				break;
			case GGUF_TYPE_INT32:
				*n_layers = gguf_get_val_i32(ctx, blk);
				break;
			case GGUF_TYPE_UINT64:
				*n_layers = (int)gguf_get_val_u64(ctx, blk);
				break;
			case GGUF_TYPE_INT64:
				*n_layers = (int)gguf_get_val_i64(ctx, blk);
				break;
			default:
				break;
			}
		}
	}

	snprintf(key, sizeof(key), "%s.embedding_length", arch);
	{
		enum gguf_type type;
		int64_t emb = gguf_find_key(ctx, key);

		if (emb >= 0) {
			type = gguf_get_kv_type(ctx, emb);
			switch (type) {
			case GGUF_TYPE_UINT8:
				*n_embd = gguf_get_val_u8(ctx, emb);
				break;
			case GGUF_TYPE_INT8:
				*n_embd = gguf_get_val_i8(ctx, emb);
				break;
			case GGUF_TYPE_UINT16:
				*n_embd = gguf_get_val_u16(ctx, emb);
				break;
			case GGUF_TYPE_INT16:
				*n_embd = gguf_get_val_i16(ctx, emb);
				break;
			case GGUF_TYPE_UINT32:
				*n_embd = (int)gguf_get_val_u32(ctx, emb);
				break;
			case GGUF_TYPE_INT32:
				*n_embd = gguf_get_val_i32(ctx, emb);
				break;
			case GGUF_TYPE_UINT64:
				*n_embd = (int)gguf_get_val_u64(ctx, emb);
				break;
			case GGUF_TYPE_INT64:
				*n_embd = (int)gguf_get_val_i64(ctx, emb);
				break;
			default:
				break;
			}
		}
	}

	snprintf(key, sizeof(key), "%s.attention.head_count", arch);
	{
		int64_t head_key = gguf_find_key(ctx, key);

		if (head_key >= 0)
			*n_head = (int)gguf_get_val_u32(ctx, head_key);
	}

	snprintf(key, sizeof(key), "%s.attention.head_count_kv", arch);
	{
		int64_t kv_head_key = gguf_find_key(ctx, key);

		if (kv_head_key >= 0)
			*n_kv_head = (int)gguf_get_val_u32(ctx, kv_head_key);
		else
			*n_kv_head = *n_head; /* Default to MHA */
	}

	snprintf(key, sizeof(key), "%s.context_length", arch);
	{
		enum gguf_type type;
		int64_t ctx_key = gguf_find_key(ctx, key);

		if (ctx_key >= 0) {
			type = gguf_get_kv_type(ctx, ctx_key);
			switch (type) {
			case GGUF_TYPE_UINT8:
				*n_ctx = gguf_get_val_u8(ctx, ctx_key);
				break;
			case GGUF_TYPE_INT8:
				*n_ctx = gguf_get_val_i8(ctx, ctx_key);
				break;
			case GGUF_TYPE_UINT16:
				*n_ctx = gguf_get_val_u16(ctx, ctx_key);
				break;
			case GGUF_TYPE_INT16:
				*n_ctx = gguf_get_val_i16(ctx, ctx_key);
				break;
			case GGUF_TYPE_UINT32:
				*n_ctx = (int)gguf_get_val_u32(ctx, ctx_key);
				break;
			case GGUF_TYPE_INT32:
				*n_ctx = gguf_get_val_i32(ctx, ctx_key);
				break;
			case GGUF_TYPE_UINT64:
				*n_ctx = (int)gguf_get_val_u64(ctx, ctx_key);
				break;
			case GGUF_TYPE_INT64:
				*n_ctx = (int)gguf_get_val_i64(ctx, ctx_key);
				break;
			default:
				break;
			}
		}
	}

	gguf_free(ctx);
}

static int
auto_ngl(const char *path, int gpu, uint32_t n_ctx, uint32_t max_offload_bytes,
	 int n_layers, int n_embd, int n_head, int n_kv_head, int n_contexts)
{
	size_t free_b, total_b;
	struct gguf_init_params ip = { .no_alloc = true };
	struct gguf_context *ctx;
	size_t *layer_sizes;
	size_t non_layer_size = 0;
	size_t usable = 0;
	size_t kv_per_layer_all_ctx;
	size_t workspace_per_ctx;
	size_t n_embd_kv;
	size_t reserve;
	size_t base_cost;
	size_t used;
	size_t original_usable;
	size_t total_weight_cost = 0;
	size_t total_kv_cost = 0;
	int offload_kqv = 1;
	int n_tensors;
	int ngl;
	int i;

	if (n_contexts <= 0)
		n_contexts = 1;

	qllm_backend_mem_check(gpu, &free_b, &total_b);
	if (!total_b)
		return 0;

	/* Fall back to a fraction of total VRAM when the driver reports zero
	 * free memory (common with some Vulkan/Metal drivers). */
	if (free_b > 0)
		usable = free_b;
	else
		usable = total_b * 95 / 100;
	if (!usable)
		return 0;

	ctx = gguf_init_from_file(path, ip);
	if (!ctx)
		return 0;

	n_tensors = (int)gguf_get_n_tensors(ctx);
	if (n_layers <= 0 || n_embd <= 0 || n_tensors <= 0) {
		gguf_free(ctx);
		return 0;
	}

	layer_sizes = calloc((size_t)n_layers, sizeof(*layer_sizes));
	if (!layer_sizes) {
		gguf_free(ctx);
		return 0;
	}

	for (i = 0; i < n_tensors; i++) {
		const char *name = gguf_get_tensor_name(ctx, i);
		const char *p;
		long layer;

		if (!name)
			continue;

		p = strstr(name, "blk.");
		if (!p) p = strstr(name, "layers.");
		if (!p) p = strstr(name, "block.");

		if (!p) {
			/* Non-layer weights (embeddings, output layer, norms). */
			size_t tsize = gguf_get_tensor_size(ctx, i);

			/* Large embeddings and output weights often stay on CPU
			 * for some backends (like Vulkan). Skip them from the
			 * GPU budget to allow offloading more layers. */
			if (tsize > 64 * 1024 * 1024 &&
			    (strstr(name, "token_embd") || strstr(name, "output"))) {
				fprintf(stderr,
				    "qllm: auto_ngl: skipping large base tensor '%s' (%.2f MiB) from GPU budget\n",
				    name, (double)tsize / (1024.0 * 1024.0));
				continue;
			}

			non_layer_size += tsize;
			if (tsize > 1024 * 1024)
				fprintf(stderr,
				    "qllm: auto_ngl: base tensor '%s': %.2f MiB\n",
				    name, (double)tsize / (1024.0 * 1024.0));
			continue;
		}

		while (*p && !isdigit((unsigned char)*p))
			p++;
		if (!isdigit((unsigned char)*p)) {
			non_layer_size += gguf_get_tensor_size(ctx, i);
			continue;
		}

		layer = strtol(p, NULL, 10);
		if (layer < 0 || layer >= n_layers) {
			non_layer_size += gguf_get_tensor_size(ctx, i);
			continue;
		}

		layer_sizes[layer] += gguf_get_tensor_size(ctx, i);
	}

	gguf_free(ctx);

	/* Buffer cost scales with n_ubatch * n_embd, plus overhead that depends
	 * on n_ctx and the backend (Vulkan/Metal can be significantly higher):
	 * (n_ubatch * n_embd * 8) + (n_ctx * 16KB) + 96MB */
	{
		uint32_t eff_ubatch = n_ctx > 512 ? 512 : n_ctx;

		workspace_per_ctx =
		    (size_t)eff_ubatch * (size_t)n_embd * 8ULL +
		    (size_t)n_ctx * 16384ULL +
		    (96ULL * 1024 * 1024);
	}

	/* KV cache cost per layer, accounting for GQA (n_kv_head / n_head).
	 * 2 bytes for K + 2 bytes for V = 4 bytes per token per layer. */
	n_embd_kv = (size_t)n_embd;
	if (n_head > 0 && n_kv_head > 0 && n_kv_head < n_head)
		n_embd_kv = (size_t)n_embd * (size_t)n_kv_head / (size_t)n_head;

	kv_per_layer_all_ctx =
	    (size_t)n_ctx * n_embd_kv * 4ULL * (size_t)n_contexts;

	/* Allow disabling KV-offload accounting via environment. */
	if (getenv("QLLM_OFFLOAD_KQV") &&
	    (getenv("QLLM_OFFLOAD_KQV")[0] == '0' ||
	     getenv("QLLM_OFFLOAD_KQV")[0] == 'n' ||
	     getenv("QLLM_OFFLOAD_KQV")[0] == 'N'))
		offload_kqv = 0;

	/* Fixed reserve for driver/OS. */
	reserve = 16ULL * 1024 * 1024;

	/* Cost if ANY layers are offloaded: non-layer weights + workspace + reserve. */
	base_cost = non_layer_size + workspace_per_ctx + reserve;

	fprintf(stderr,
	    "qllm: auto_ngl: VRAM breakdown: usable=%.2f MiB, base_weights=%.2f MiB, workspace=%.2f MiB, reserve=%.2f MiB, base_cost=%.2f MiB, kv_offload=%s\n",
	    (double)usable / (1024.0 * 1024.0),
	    (double)non_layer_size / (1024.0 * 1024.0),
	    (double)workspace_per_ctx / (1024.0 * 1024.0),
	    (double)reserve / (1024.0 * 1024.0),
	    (double)base_cost / (1024.0 * 1024.0),
	    offload_kqv ? "YES" : "NO");

	fprintf(stderr,
	    "qllm: auto_ngl: Context info: n_ctx=%u, n_embd=%d, n_embd_kv=%zu, kv_per_layer=%.2f MiB\n",
	    n_ctx, n_embd, n_embd_kv,
	    (double)kv_per_layer_all_ctx / (1024.0 * 1024.0));

	if (usable <= base_cost) {
		fprintf(stderr,
		    "qllm: auto_ngl: Not enough VRAM for base weights and workspace (need %.2f MiB, have %.2f MiB)\n",
		    (double)base_cost / (1024.0 * 1024.0),
		    (double)usable / (1024.0 * 1024.0));
		free(layer_sizes);
		return 0;
	}

	original_usable = usable;
	usable -= base_cost;

	if (max_offload_bytes > 0 && usable > max_offload_bytes)
		usable = max_offload_bytes;

	used = base_cost;
	ngl = 0;

	for (i = 0; i < n_layers; i++) {
		size_t w_need = layer_sizes[i];
		size_t need = w_need + (offload_kqv ? kv_per_layer_all_ctx : 0);

		if (used + need > original_usable)
			break;

		used += need;
		total_weight_cost += w_need;
		total_kv_cost += offload_kqv ? kv_per_layer_all_ctx : 0;
		ngl++;
	}

	/* Manual adjustment from env. */
	{
		const char *adjust = getenv("QLLM_NGL_ADJUST");

		if (adjust) {
			int adj = atoi(adjust);

			ngl += adj;
			if (ngl < 0)
				ngl = 0;
			if (ngl > n_layers)
				ngl = n_layers;
		}
	}

	fprintf(stderr,
	    "qllm: auto_ngl: offloading %d/%d layers, total_est=%.2f MiB (weights=%.2f MiB, kv=%.2f MiB, base=%.2f MiB)\n",
	    ngl, n_layers,
	    (double)(base_cost + total_weight_cost + total_kv_cost) / (1024.0 * 1024.0),
	    (double)total_weight_cost / (1024.0 * 1024.0),
	    (double)total_kv_cost / (1024.0 * 1024.0),
	    (double)base_cost / (1024.0 * 1024.0));

	free(layer_sizes);
	return ngl;
}

struct llama_model *model_load(
		const char *path,
		int32_t n_ctx,
		uint32_t ngl_max,
		int32_t n_contexts,
		int32_t n_gpu_layers)
{
	struct llama_model_params model_params;
	struct llama_model ** model_r, *model;
	int n_layers, n_embd, n_head, n_kv_head, ngl;

	pthread_mutex_lock(&model_lock);

	model_r = (struct llama_model **) corm_get(model_hd, path);
	if (model_r && *model_r) {
		model = *model_r;
		shared_model_ref(model);
		pthread_mutex_unlock(&model_lock);
		return model;
	}

	if (!n_contexts)
		n_contexts = 1;

	model_params = llama_model_default_params();

	model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;

	model_params.n_gpu_layers = 0;
	if (!(model = llama_model_load_from_file(
			path,
			model_params))) {
		pthread_mutex_unlock(&model_lock);
		return NULL;
	}

	n_layers = llama_model_n_layer(model);
	n_embd = llama_model_n_embd(model);
	n_head = llama_model_n_head(model);
	n_kv_head = llama_model_n_head_kv(model);
	llama_model_free(model);

	/* Pass through -g: >0 forces a specific number of GPU layers, otherwise
	 * derive the offload budget automatically. */
	if (n_gpu_layers > 0) {
		ngl = n_gpu_layers;
		if (ngl > n_layers)
			ngl = n_layers;
		if (ngl < 0)
			ngl = 0;
	} else
		ngl = auto_ngl(path, 0, n_ctx, ngl_max, n_layers, n_embd, n_head,
		    n_kv_head, n_contexts);

	if (ngl > 0)
		model_params.n_gpu_layers = ngl;
	else
		model_params.n_gpu_layers = 0;

	if (!(model = llama_model_load_from_file(
			path,
			model_params))) {
		pthread_mutex_unlock(&model_lock);
		return NULL;
	}

	corm_put(model_hd, path, &model);
	shared_model_ref(model);
	pthread_mutex_unlock(&model_lock);
	return model;
}

/*
 * EOS bias sampler: ramps up the logit of the end-of-turn (EOS) token as
 * generation runs long, so that sessions which would otherwise keep
 * generating terminate on their own. Active only after eos_start tokens
 * have been generated; bias grows cubically to eos_bias_max.
 */
struct eos_bias_sampler_ctx {
	struct qllm_context *qctx;
};

static void
eos_bias_apply(struct llama_sampler *smpl,
	       struct llama_token_data_array *cur_p)
{
	struct eos_bias_sampler_ctx *sctx =
	    (struct eos_bias_sampler_ctx *) smpl->ctx;
	struct qllm_context *qctx = sctx->qctx;
	llama_token eos;
	float t, bias;
	size_t i;

	if (qctx->gen_tokens < qctx->eos_start)
		return;

	eos = llama_vocab_eot(qctx->vocab);

	t = (float)(qctx->gen_tokens - qctx->eos_start) / 16.0f;
	if (t < 0.0f)
		return;

	if (t > 1.0f)
		t = 1.0f;

	/* Exponential ramp */
	bias = qctx->eos_bias_max * (t * t * t);

	for (i = 0; i < cur_p->size; ++i) {
		if (cur_p->data[i].id == eos) {
			cur_p->data[i].logit += bias;
			return;
		}
	}
}

static void
eos_bias_accept(struct llama_sampler *smpl, llama_token token)
{
	(void) smpl;
	(void) token;
}

static void
eos_bias_reset(struct llama_sampler *smpl)
{
	(void) smpl;
}

static struct llama_sampler *
eos_bias_clone(const struct llama_sampler *smpl)
{
	const struct eos_bias_sampler_ctx *old =
	    (const struct eos_bias_sampler_ctx *) smpl->ctx;
	struct eos_bias_sampler_ctx *ctx = malloc(sizeof(*ctx));

	if (!ctx)
		return NULL;
	*ctx = *old;

	return llama_sampler_init(smpl->iface, ctx);
}

static void
eos_bias_free(struct llama_sampler *smpl)
{
	free(smpl->ctx);
}

static const struct llama_sampler_i eos_bias_iface = {
	.name   = NULL,
	.accept = eos_bias_accept,
	.apply  = eos_bias_apply,
	.reset  = eos_bias_reset,
	.clone  = eos_bias_clone,
	.free   = eos_bias_free,
};

static struct llama_sampler *
llama_sampler_init_eos_bias(struct qllm_context *qctx)
{
	struct eos_bias_sampler_ctx *ctx = malloc(sizeof(*ctx));

	if (!ctx)
		return NULL;
	ctx->qctx = qctx;

	return llama_sampler_init(&eos_bias_iface, ctx);
}

/*
 * Sliding-window context compression.
 *
 * qllm_anchor_start()/qllm_anchor_end() mark the "protected" region of the
 * KV cache (typically the current prompt + response). qllm_compress() evicts
 * tokens outside that region when the sequence grows past `limit`:
 *   - prefix (oldest dialog) up to anchor_start is dropped and shifted,
 *   - then tokens beyond anchor_end + anchor_guard are dropped.
 * limit == 0 clears the whole sequence.
 */
void
qllm_anchor_start(struct qllm_context *ctx)
{
	llama_memory_t mem;

	if (!QLLM_VALID(ctx) || !ctx->ctx)
		return;
	mem = llama_get_memory(ctx->ctx);
	ctx->anchor_start = ctx->anchor_end =
	    llama_memory_seq_pos_max(mem, ctx->current_seq);
}

void
qllm_anchor_end(struct qllm_context *ctx)
{
	llama_memory_t mem;

	if (!QLLM_VALID(ctx) || !ctx->ctx)
		return;
	mem = llama_get_memory(ctx->ctx);
	ctx->anchor_end = llama_memory_seq_pos_max(mem, ctx->current_seq);
}

void
qllm_compress(struct qllm_context *ctx, uint32_t limit)
{
	struct llama_context *lctx;
	llama_memory_t mem;
	llama_seq_id seq;
	int32_t total_tokens;
	const uint32_t anchor_guard = 16;
	uint32_t to_drop;

	if (!QLLM_VALID(ctx) || !ctx->ctx)
		return;

	lctx = ctx->ctx;
	mem = llama_get_memory(lctx);
	seq = ctx->current_seq;

	if (limit == 0) {
		llama_memory_seq_rm(mem, seq, 0, -1);
		ctx->anchor_start = 0;
		ctx->anchor_end = 0;
		return;
	}

	total_tokens = llama_memory_seq_pos_max(mem, seq) + 1;
	if (total_tokens <= (int32_t)limit)
		return;

	to_drop = total_tokens - limit;

	/* 1. DROP BEFORE ANCHOR (prefix) */
	if (ctx->anchor_start > 0 && to_drop > 0) {
		uint32_t prefix_avail = (uint32_t)ctx->anchor_start;
		uint32_t drop_prefix =
		    to_drop > prefix_avail ? prefix_avail : to_drop;

		llama_memory_seq_rm(mem, seq, 0, (llama_pos)drop_prefix);
		llama_memory_seq_add(mem, seq,
		    (llama_pos)drop_prefix, total_tokens,
		    -(llama_pos)drop_prefix);

		ctx->anchor_start -= (llama_pos)drop_prefix;
		ctx->anchor_end   -= (llama_pos)drop_prefix;

		to_drop      -= drop_prefix;
		total_tokens -= (int32_t)drop_prefix;
	}

	/* 2. DROP AFTER ANCHOR + GUARD */
	if (to_drop > 0) {
		uint32_t drop_start = (uint32_t)ctx->anchor_end + anchor_guard;

		/* If there is no room for the guard, start at anchor_end */
		if (drop_start > (uint32_t)total_tokens)
			drop_start = (uint32_t)ctx->anchor_end;

		{
			uint32_t max_drop = (uint32_t)total_tokens - drop_start;

			if (to_drop > max_drop)
				to_drop = max_drop;

			if (to_drop > 0) {
				llama_memory_seq_rm(mem, seq,
				    (llama_pos)drop_start,
				    (llama_pos)(drop_start + to_drop));
				llama_memory_seq_add(mem, seq,
				    (llama_pos)(drop_start + to_drop),
				    total_tokens,
				    -(llama_pos)to_drop);
			}
		}
	}
}

void
qllm_set_eos_bias(struct qllm_context *qctx,
		  int32_t start_tokens,
		  float max_bias)
{
	if (!QLLM_VALID(qctx))
		return;

	qctx->eos_start    = start_tokens;
	qctx->eos_bias_max = max_bias;
}

struct qllm_context *
qllm_create(const struct qllm_config *cfg)
{
	struct qllm_context *qctx;
	struct llama_context_params ctx_params;
	int32_t n_threads;

	if (!cfg || !cfg->model_path)
		return NULL;

	ctx_params = llama_context_default_params();

	if (cfg->n_ctx > 0)
		ctx_params.n_ctx = (uint32_t) cfg->n_ctx;
	else {
		int nl = 0, ne = 0, nc = 0, nh = 0, nkh = 0;

		get_model_dims(cfg->model_path, &nl, &ne, &nc, &nh, &nkh);
		if (nc > 0)
			ctx_params.n_ctx = (uint32_t) nc;
		else
			ctx_params.n_ctx = 2048; /* Fallback */
	}

	ctx_params.n_batch = ctx_params.n_ctx;
	ctx_params.n_ubatch = 0;
	ctx_params.n_seq_max = cfg->n_contexts > 0 ? (uint32_t)cfg->n_contexts : 1;

	/* Enable embeddings and mean pooling only if requested.
	 * This is required for qllm_embed() but not for text generation.
	 * Note: Setting pooling_type when the model doesn't support it
	 * can cause crashes when creating multiple contexts. */
	if (cfg->enable_embeddings) {
		ctx_params.embeddings = true;
		ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
	}

	if (cfg->n_threads > 0) {
		n_threads = cfg->n_threads;
	} else {
		long ncpu;

		ncpu = sysconf(_SC_NPROCESSORS_ONLN);
		if (ncpu <= 0)
			n_threads = 1;
		else
			n_threads = (int32_t)(ncpu > 1 ? ncpu / 2 : 1);
	}

	ctx_params.n_threads = n_threads;
	ctx_params.n_threads_batch = n_threads;

	qctx = calloc(1, sizeof(*qctx));
	if (!qctx)
		return NULL;

#if defined(MOCK_BUILD)
	qllm_unrecord_freed(qctx);
#endif

	qctx->magic = QLLM_MAGIC;
	qctx->current_seq = 0;

	qctx->model_path = strdup(cfg->model_path);

	qctx->max_tokens = (int32_t)ctx_params.n_ctx;
	qctx->params = ctx_params;	/* <-- important: save params */

	qctx->model = model_load(cfg->model_path, ctx_params.n_ctx,
	    cfg->max_offload_bytes, cfg->n_contexts, cfg->n_gpu_layers);

	if (!qctx->model)
		goto fail;

	qctx->ctx = llama_init_from_model(qctx->model, ctx_params);
	if (!qctx->ctx)
		goto fail;

	qctx->vocab = llama_model_get_vocab(qctx->model);
	qctx->n_embd = llama_model_n_embd(qctx->model);

	qctx->sampler = qllm_sampler_create(qctx, cfg);
	if (!qctx->sampler)
		goto fail;

	qctx->token_buf = calloc((size_t)qctx->max_tokens,
	    sizeof(*qctx->token_buf));
	qctx->seq_ids = calloc((size_t)qctx->max_tokens,
	    sizeof(*qctx->seq_ids));
	if (!qctx->token_buf || !qctx->seq_ids)
		goto fail;

	qctx->anchor_start = 0;
	qctx->anchor_end = 0;
	qctx->gen_tokens   = 0;
	qctx->eos_start    = 64;
	qctx->eos_bias_max = 3.0f;

	return qctx;

fail:
	qllm_free(qctx);
	return NULL;
}

void
qllm_free(struct qllm_context *qctx)
{
	if (!qctx)
		return;

#if defined(MOCK_BUILD)
	/* If this pointer was already freed earlier, avoid dereferencing it. */
	if (qllm_ptr_freed((const void *)qctx))
		return;
#endif

	/* Invalidate magic early so repeated frees are safe. */
	if (qctx->magic == QLLM_MAGIC)
		qctx->magic = 0;

	/* Free child samplers first (test/mocks allocate sampler->ctx which
	 * the mock's llama_sampler_free doesn't free). */
#if defined(MOCK_BUILD)
	{
		int32_t i;

		for (i = 0; i < qctx->sampler_children_n; ++i) {
			struct llama_sampler *s = qctx->sampler_children[i];

			if (!s)
				continue;
			if (s->ctx)
				free(s->ctx);
			llama_sampler_free(s);
			qctx->sampler_children[i] = NULL;
		}
		qctx->sampler_children_n = 0;
	}
#endif
	if (qctx->sampler)
		llama_sampler_free(qctx->sampler);
	/* The grammar sampler is chain-added in production (the chain owns it),
	 * so only the mocks need an explicit release of its allocation. */
#if defined(MOCK_BUILD)
	if (qctx->grammar_sampler) {
		if (qctx->grammar_sampler->ctx)
			free(qctx->grammar_sampler->ctx);
		llama_sampler_free(qctx->grammar_sampler);
		qctx->grammar_sampler = NULL;
	}
#endif
	if (qctx->ctx)
		llama_free(qctx->ctx);
	if (qctx->model)
		shared_model_unref(qctx->model, qctx->model_path);

	free(qctx->model_path);
	free(qctx->token_buf);
	free(qctx->seq_ids);

#if defined(MOCK_BUILD)
	qllm_record_freed(qctx);
#endif
	free(qctx);
}

int
qllm_n_ctx(struct qllm_context *qctx)
{
	if (!QLLM_VALID(qctx))
		return 0;
	return qctx->max_tokens;
}

struct llama_sampler *
qllm_sampler_create(struct qllm_context *qctx,
		    const struct qllm_config *cfg)
{
	struct llama_sampler_chain_params chain_params;
	struct llama_sampler *sampler;

	if (!QLLM_VALID(qctx))
		return NULL;

#define QLLM_SAMPLER_ADD(s) do {					\
	struct llama_sampler *_s = (s);					\
	if (_s) {							\
		if (qctx->sampler_children_n <				\
		    (int32_t)(sizeof(qctx->sampler_children) /		\
			      sizeof(qctx->sampler_children[0])))	\
			qctx->sampler_children[qctx->sampler_children_n++] = _s; \
		llama_sampler_chain_add(sampler, _s);			\
	}								\
} while (0)

	chain_params = llama_sampler_chain_default_params();
	sampler = llama_sampler_chain_init(chain_params);
	if (!sampler)
		return NULL;

	QLLM_SAMPLER_ADD(llama_sampler_init_eos_bias(qctx));

	{
		int32_t last_n = (cfg && cfg->repeat_last_n > 0) ?
		    cfg->repeat_last_n : 64;
		float repeat = (cfg && cfg->repeat_penalty > 0.0f) ?
		    cfg->repeat_penalty : 1.1f;

		QLLM_SAMPLER_ADD(llama_sampler_init_penalties(last_n,
		    repeat, 0.0f, 0.0f));
	}

	if (cfg && cfg->top_k > 0)
		QLLM_SAMPLER_ADD(llama_sampler_init_top_k(cfg->top_k));

	if (cfg && cfg->top_p > 0.0f && cfg->top_p < 1.0f)
		QLLM_SAMPLER_ADD(llama_sampler_init_top_p(cfg->top_p, 1));

	{
		float temp = (cfg && cfg->temperature > 0.0f) ?
		    cfg->temperature : 0.7f;

		QLLM_SAMPLER_ADD(llama_sampler_init_temp(temp));
	}

	QLLM_SAMPLER_ADD(llama_sampler_init_dist(0));

#undef QLLM_SAMPLER_ADD

	return sampler;
}

void
qllm_sampler_free(struct llama_sampler *smpl)
{
	if (smpl)
		llama_sampler_free(smpl);
}

int
qllm_sampler_add_grammar(struct qllm_context *qctx,
			 struct llama_sampler *sampler,
			 const char *grammar_str)
{
	struct llama_sampler *gs;

	if (!QLLM_VALID(qctx) || !sampler || !grammar_str)
		return -1;

	gs = llama_sampler_init_grammar(qctx->vocab, grammar_str, "root");
	if (!gs)
		return -1;

	/* Track for MOCK_BUILD teardown (the mock's chain is a no-op). */
	if (qctx->sampler_children_n <
	    (int32_t)(sizeof(qctx->sampler_children) /
		      sizeof(qctx->sampler_children[0])))
		qctx->sampler_children[qctx->sampler_children_n++] = gs;

	llama_sampler_chain_add(sampler, gs);
	return 0;
}

void
qllm_set_seq(struct qllm_context *qctx, uint32_t seq_id)
{
	if (!QLLM_VALID(qctx))
		return;
	qctx->current_seq = (llama_seq_id)seq_id;
}

int
qllm_set_grammar(struct qllm_context *qctx, const char *grammar_str)
{
	if (!QLLM_VALID(qctx))
		return -1;

	/* The grammar sampler is tracked separately and never appended to
	 * sampler_children[], so a replace cannot leave a stale entry for the
	 * MOCK_BUILD teardown loop. In production the chain owns whatever
	 * sampler was added last; a replace appends a new grammar sampler to
	 * the chain end (llama_sampler_chain_add always appends). */
	if (qctx->grammar_sampler) {
		llama_sampler_free(qctx->grammar_sampler);
		qctx->grammar_sampler = NULL;
	}

	if (grammar_str) {
		qctx->grammar_sampler =
		    llama_sampler_init_grammar(qctx->vocab, grammar_str, "root");
		if (!qctx->grammar_sampler)
			return -1;
		llama_sampler_chain_add(qctx->sampler, qctx->grammar_sampler);
	}

	return 0;
}

/* Internal streaming helper: runs generation and calls cb() for each piece. */
static int
qllm_generate_stream_internal(struct qllm_context *qctx,
			      const char *prompt,
			      qllm_token_cb cb,
			      void *user)
{
	int32_t n_prompt;
	llama_token tok;
	char piece[256];
	int n_piece;
	int32_t step;
	const int32_t max_gen = qctx->max_tokens;

	if (!QLLM_VALID(qctx) || !prompt || !cb)
		return -1;

	llama_free(qctx->ctx);
	qctx->ctx = llama_init_from_model(qctx->model, qctx->params);

	qctx->anchor_start = 0;
	qctx->anchor_end = 0;
	qctx->gen_tokens = 0;

	n_prompt = llama_tokenize(qctx->vocab,
				  prompt,
				  (int32_t) strlen(prompt),
				  qctx->token_buf,
				  qctx->max_tokens,
				  true,
				  true);
	if (n_prompt < 0)
		return -1;

	if (n_prompt == 0)
		return 0;

	if (qllm_decode_tokens(qctx, qctx->token_buf, n_prompt, 0) != 0)
		return -1;

	for (step = 0; step < max_gen; ++step) {
		tok = llama_sampler_sample(qctx->sampler, qctx->ctx, -1);
		llama_sampler_accept(qctx->sampler, tok);

		if (llama_vocab_is_eog(qctx->vocab, tok))
			break;

		qctx->token_buf[0] = tok;
		if (qllm_decode_tokens(qctx, qctx->token_buf, 1, 0) != 0)
			break;

		memset(piece, 0, sizeof(piece));
		n_piece = llama_token_to_piece(qctx->vocab,
					       tok,
					       piece,
					       (int) sizeof(piece),
					       false,
					       true);
		if (n_piece <= 0)
			continue;

		qctx->gen_tokens++;
		cb(user, piece, (size_t) n_piece);
	}

	return 0;
}

int
qllm_generate_stream(struct qllm_context *qctx,
		     const char *prompt,
		     qllm_token_cb cb,
		     void *user)
{
	if (!QLLM_VALID(qctx) || !prompt || !cb)
		return -1;

	return qllm_generate_stream_internal(qctx, prompt, cb, user);
}

/* Accumulator used by qllm_generate() to build a string. */
struct qllm_accum {
	char	*buf;
	size_t	 cap;
	size_t	 len;
};

static void
qllm_accum_cb(void *user, const char *chunk, size_t len)
{
	struct qllm_accum *acc = user;
	size_t n;

	if (!acc || !acc->buf || acc->cap == 0)
		return;

	if (acc->len >= acc->cap)
		return;

	n = len;
	if (acc->len + n >= acc->cap) {
		if (acc->cap <= acc->len + 1)
			return;
		n = acc->cap - acc->len - 1;
	}

	if (!n)
		return;

	memcpy(acc->buf + acc->len, chunk, n);
	acc->len += n;
	acc->buf[acc->len] = '\0';
}

long
qllm_generate(struct qllm_context *qctx,
	      const char *prompt,
	      char *out,
	      size_t out_size)
{
	struct qllm_accum acc;
	int ret;

	if (!QLLM_VALID(qctx) || !prompt || !out || out_size == 0)
		return -1;

	out[0] = '\0';

	acc.buf = out;
	acc.cap = out_size;
	acc.len = 0;

	ret = qllm_generate_stream_internal(qctx,
					    prompt,
					    qllm_accum_cb,
					    &acc);
	if (ret != 0)
		return -1;

	return (ssize_t) acc.len;
}

/*
 * qllm_embed — return a single embedding vector for the whole text.
 * Returns number of dims on success, < 0 on error.
 */
int
qllm_embed(struct qllm_context *qctx,
	   const char *text,
	   float *out,
	   size_t out_dim)
{
	int32_t n_tokens;
	const float *embd;
	int32_t i;

	if (!QLLM_VALID(qctx) || !text || !out)
		return -1;

	llama_free(qctx->ctx);
	qctx->ctx = llama_init_from_model(qctx->model, qctx->params);
	qctx->anchor_start = 0;
	qctx->anchor_end = 0;
	qctx->gen_tokens = 0;

	n_tokens = llama_tokenize(qctx->vocab,
				  text,
				  (int32_t) strlen(text),
				  qctx->token_buf,
				  qctx->max_tokens,
				  true,
				  true);
	if (n_tokens < 0)
		return -1;

	if (n_tokens == 0)
		return -1;

	if (qllm_decode_tokens(qctx, qctx->token_buf, n_tokens, 1) != 0)
		return -1;

	embd = llama_get_embeddings_seq(qctx->ctx, 0);
	if (!embd)
		return -1;

	if (out_dim < (size_t) qctx->n_embd)
		return -1;

	for (i = 0; i < qctx->n_embd; ++i)
		out[i] = embd[i];

	return qctx->n_embd;
}

int
qllm_prime(struct qllm_context *qctx,
	   const char *prompt)
{
	int32_t n_prompt;

	if (!QLLM_VALID(qctx) || !qctx->ctx || !prompt)
		return -1;

	qctx->gen_tokens = 0;

	n_prompt = llama_tokenize(qctx->vocab,
				  prompt,
				  (int32_t)strlen(prompt),
				  qctx->token_buf,
				  qctx->max_tokens,
				  true,
				  true);
	if (n_prompt < 0)
		return -1;

	if (n_prompt == 0)
		return 0;

	if (qllm_decode_tokens(qctx, qctx->token_buf, n_prompt, 0) != 0)
		return -1;

	return 0;
}

int
qllm_next(struct qllm_context *qctx,
	  struct llama_sampler *sampler,
	  char *out,
	  size_t out_size)
{
	llama_token tok;
	char piece[256];
	int n_piece;
	struct llama_sampler *smpl;

	if (!QLLM_VALID(qctx) || !qctx->ctx || !out || out_size == 0)
		return -1;

	smpl = sampler ? sampler : qctx->sampler;

retry:
	/* Sample one token */
	tok = llama_sampler_sample(smpl, qctx->ctx, -1);
	llama_sampler_accept(smpl, tok);

	/* EOT ends the turn: decode it so downstream sessions see a coherent
	 * KV history, then signal end-of-generation. */
	if (tok == llama_vocab_eot(qctx->vocab)) {
		qctx->token_buf[0] = tok;
		qllm_decode_tokens(qctx, qctx->token_buf, 1, 0);
		return 0;
	}

	if (tok == llama_vocab_eos(qctx->vocab)) {
		qctx->token_buf[0] = tok;
		qllm_decode_tokens(qctx, qctx->token_buf, 1, 0);
		return 0;
	}

	/* Skip control tokens (e.g. BOS, separators) and sample again. */
	if (llama_vocab_is_control(qctx->vocab, tok)) {
		qctx->token_buf[0] = tok;
		qllm_decode_tokens(qctx, qctx->token_buf, 1, 0);
		goto retry;
	}

	/* Treat any other EOG as end-of-generation */
	if (llama_vocab_is_eog(qctx->vocab, tok))
		return 0;

	/* Advance KV with this token */
	qctx->token_buf[0] = tok;
	if (qllm_decode_tokens(qctx, qctx->token_buf, 1, 0) != 0)
		return -1;

	/* Convert token to text piece */
	memset(piece, 0, sizeof(piece));
	n_piece = llama_token_to_piece(qctx->vocab,
				       tok,
				       piece,
				       (int)sizeof(piece),
				       false,
				       true);
	if (n_piece <= 0)
		return -1;

	if ((size_t)n_piece >= out_size)
		n_piece = (int)(out_size - 1);

	memcpy(out, piece, (size_t)n_piece);
	out[n_piece] = '\0';

	qctx->gen_tokens++;

	return n_piece;
}

int
qllm_render(struct qllm_context *qctx,
	    const struct qllm_message *msgs, size_t n_msgs,
	    bool add_ass, char **out, size_t *out_len)
{
	const char *tmpl;
	struct llama_chat_message *chat;
	int32_t len;
	char *buf;
	size_t i;

	if (!qctx || !msgs || n_msgs == 0 || !out || !out_len)
		return -1;

	tmpl = llama_model_chat_template(qctx->model, NULL);
	if (!tmpl) {
		qsyslog(QLOG_ERR, "model has no chat template\n");
		return -1;
	}

	chat = calloc(n_msgs, sizeof(*chat));
	if (!chat)
		return -1;

	for (i = 0; i < n_msgs; i++) {
		chat[i].role = msgs[i].role;
		chat[i].content = msgs[i].content;
	}

	/* Two-pass sizing */
	len = llama_chat_apply_template(tmpl, chat, n_msgs, add_ass, NULL, 0);
	if (len < 0) {
		free(chat);
		return -1;
	}

	buf = calloc((size_t)len + 1, 1);
	if (!buf) {
		free(chat);
		return -1;
	}

	len = llama_chat_apply_template(tmpl, chat, n_msgs, add_ass, buf, len);
	free(chat);

	if (len < 0) {
		free(buf);
		return -1;
	}

	buf[len] = '\0';
	*out = buf;
	*out_len = (size_t)len;
	return 0;
}

void
qllm_reset(struct qllm_context *qctx)
{
	if (!QLLM_VALID(qctx) || !qctx->ctx)
		return;

	llama_memory_clear(llama_get_memory(qctx->ctx), true);
	qctx->anchor_start = 0;
	qctx->anchor_end = 0;
	qctx->gen_tokens = 0;
}

int
qllm_chat(struct qllm_context *qctx,
	  const struct qllm_message *msgs, size_t n_msgs,
	  const char *prev_prompt,
	  qllm_token_cb cb, void *user)
{
	char *full = NULL;
	size_t full_len = 0;
	const char *incr;
	int32_t step;
	int ret;

	if (!QLLM_VALID(qctx) || !qctx->model || !msgs ||
	    n_msgs == 0 || !cb)
		return -1;

	/* Render full conversation with add_ass=true */
	if (qllm_render(qctx, msgs, n_msgs, true, &full, &full_len) != 0)
		return -1;

	/* Compute the incremental prompt */
	if (prev_prompt && full_len >= strlen(prev_prompt) &&
	    memcmp(full, prev_prompt, strlen(prev_prompt)) == 0) {
		incr = full + strlen(prev_prompt);
	} else {
		incr = full;
	}

	/* Protect this turn's prompt+response from sliding-window eviction. */
	qllm_anchor_start(qctx);

	/* Prime with only the new content */
	ret = qllm_prime(qctx, incr);
	if (ret != 0) {
		free(full);
		return -1;
	}

	/* Stream tokens until EOG */
	for (step = 0; step < qctx->max_tokens; ++step) {
		char piece[256];
		int n;

		n = qllm_next(qctx, NULL, piece, sizeof(piece));
		if (n <= 0)
			break;

		cb(user, piece, (size_t)n);
	}

	qllm_anchor_end(qctx);

	/* Keep the session context bounded: once the KV grows past 80% of
	 * the context window, evict the oldest prefix (and any tail beyond
	 * the anchored region) to make room for the next turn. */
	qllm_compress(qctx, (uint32_t)(qctx->max_tokens * 4 / 5));

	free(full);
	return 0;
}

/* libqllm.c */

#include "./../include/ttypt/qllm.h"

#include <ctype.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <llama.h>
#include <gguf.h>

#include <ttypt/qsys.h>
#include <ttypt/qmap.h>

#include <stdatomic.h>

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
  /* copy of model path so we can look up cache entry on free */
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

#define QLLM_MAGIC 0x514C4C4D

/* Test harness hooks: allow the tests framework to track freed qllm_context
 * pointers so test builds can avoid dereferencing freed memory. In test
 * builds the functions `qllm_ptr_freed` and `qllm_record_freed` are
 * provided by the test harness; in production builds they are simple
 * no-ops/stubs (always return false / do nothing). */
#if defined(MOCK_BUILD)
int qllm_ptr_freed(const void *p);
void qllm_record_freed(void *p);
void qllm_unrecord_freed(void *p);
#else
static inline int qllm_ptr_freed(const void *p) { (void)p; return 0; }
static inline void qllm_record_freed(void *p) { (void)p; }
static inline void qllm_unrecord_freed(void *p) { (void)p; }
#endif

#define QLLM_VALID(qctx) ((qctx) && !qllm_ptr_freed((const void *)(qctx)) && (qctx)->magic == QLLM_MAGIC)

struct eos_bias_sampler_ctx {
  struct qllm_context * qctx;
};

static int qllm_backend_inited;
static uint32_t qm_model, model_hd;

struct model_cache_entry {
  struct llama_model *model;
  atomic_uint refcount;
};

/* Initialize llama backend exactly once. */
__attribute__((constructor)) void qllm_init(void)
{
  const char *cache_file;

  qm_model = qmap_reg(sizeof(struct llama_model *));

  /* Optional persistent cache for model metadata.
   * Set QLLM_CACHE_FILE environment variable to enable.
   * With qmap 0.6.0+, file loading works without QM_MIRROR. */
  cache_file = getenv("QLLM_CACHE_FILE");
  if (cache_file && cache_file[0] != '\0') {
    /* Validate cache file path: reject paths that are too long
     * or obviously invalid. qmap handles missing files gracefully
     * (starts with empty map), but we validate to catch obvious
     * configuration errors early. */
    if (strlen(cache_file) >= 4096) {
      fprintf(stderr, "qllm: WARNING: QLLM_CACHE_FILE path too long (>= 4096), ignoring\n");
      cache_file = NULL;
    } else if (access(cache_file, F_OK) == 0 && access(cache_file, R_OK | W_OK) != 0) {
      /* File exists but not readable/writable - warn but continue.
       * qmap will handle the error appropriately. */
      fprintf(stderr, "qllm: WARNING: QLLM_CACHE_FILE exists but may not be accessible: %s\n", cache_file);
    }
  }

  model_hd = qmap_open(cache_file, "models", QM_STR, qm_model, 0, 0);

  llama_backend_init();
  qllm_backend_inited = 1;
}

/* Small helper to decode a batch of tokens at the current position. */
static int qllm_decode_tokens(
    struct qllm_context *qctx,
    const llama_token *tokens,
    int32_t n_tokens)
{
  struct llama_batch batch;
  int32_t i;

  if (!qctx || !qctx->ctx || !tokens || n_tokens <= 0) {
    return -1;
  }

  if (n_tokens > qctx->max_tokens) {
    return -1;
  }

  batch = llama_batch_init(n_tokens, 0, 1);
  batch.n_tokens = n_tokens;
  batch.pos = NULL;

  for (i = 0; i < n_tokens; ++i) {
    batch.token[i] = tokens[i];
    batch.n_seq_id[i] = 1;
    batch.seq_id[i] = &qctx->current_seq;
    batch.logits[i] = (i == n_tokens - 1);
  }

  /* Call llama_decode and always free the batch afterwards to avoid
   * leaking batch internal allocations (mocks allocate arrays in
   * llama_batch_init). Freeing must happen regardless of success. */
  int rc = llama_decode(qctx->ctx, batch);

  /* WORKAROUND: llama_batch_free hangs with Vulkan backend, so skip it */
  /* This will leak some memory but allows the daemon to work */
  /* llama_batch_free(batch); */

  if (rc != 0) {
    return -1;
  }

  return 0;
}

extern void qllm_backend_mem_check(int gpu, size_t *free_b, size_t *total_b);

static void get_model_dims(
    const char *path, int *n_layers,
    int *n_embd, int *n_ctx,
    int *n_head, int *n_kv_head)
{
  struct gguf_init_params ip = { .no_alloc = true };
  struct gguf_context *ctx;

  *n_layers = 0;
  *n_embd = 0;
  *n_ctx = 0;
  *n_head = 0;
  *n_kv_head = 0;

  ctx = gguf_init_from_file(path, ip);
  if (!ctx)
    return;

  /* Find architecture key - look for general.architecture */
  int arch_key = gguf_find_key(ctx, "general.architecture");
  if (arch_key < 0) {
    gguf_free(ctx);
    return;
  }

  const char *arch = gguf_get_val_str(ctx, arch_key);
  if (!arch) {
    gguf_free(ctx);
    return;
  }

  /* Build key names: e.g., "llama.block_count", "mistral.embedding_length" */
  char key[128];

  snprintf(key, sizeof(key), "%s.block_count", arch);
  int blk_key = gguf_find_key(ctx, key);
  if (blk_key >= 0) {
    enum gguf_type type = gguf_get_kv_type(ctx, blk_key);
    switch (type) {
      case GGUF_TYPE_UINT8:  *n_layers = gguf_get_val_u8(ctx, blk_key);  break;
      case GGUF_TYPE_INT8:   *n_layers = gguf_get_val_i8(ctx, blk_key);  break;
      case GGUF_TYPE_UINT16: *n_layers = gguf_get_val_u16(ctx, blk_key); break;
      case GGUF_TYPE_INT16:  *n_layers = gguf_get_val_i16(ctx, blk_key); break;
      case GGUF_TYPE_UINT32: *n_layers = gguf_get_val_u32(ctx, blk_key); break;
      case GGUF_TYPE_INT32:  *n_layers = gguf_get_val_i32(ctx, blk_key); break;
      case GGUF_TYPE_UINT64: *n_layers = (int)gguf_get_val_u64(ctx, blk_key); break;
      case GGUF_TYPE_INT64:  *n_layers = (int)gguf_get_val_i64(ctx, blk_key); break;
      default: break;
    }
  }

  snprintf(key, sizeof(key), "%s.embedding_length", arch);
  int emb_key = gguf_find_key(ctx, key);
  if (emb_key >= 0) {
    enum gguf_type type = gguf_get_kv_type(ctx, emb_key);
    switch (type) {
      case GGUF_TYPE_UINT8:  *n_embd = gguf_get_val_u8(ctx, emb_key);  break;
      case GGUF_TYPE_INT8:   *n_embd = gguf_get_val_i8(ctx, emb_key);  break;
      case GGUF_TYPE_UINT16: *n_embd = gguf_get_val_u16(ctx, emb_key); break;
      case GGUF_TYPE_INT16:  *n_embd = gguf_get_val_i16(ctx, emb_key); break;
      case GGUF_TYPE_UINT32: *n_embd = gguf_get_val_u32(ctx, emb_key); break;
      case GGUF_TYPE_INT32:  *n_embd = gguf_get_val_i32(ctx, emb_key); break;
      case GGUF_TYPE_UINT64: *n_embd = (int)gguf_get_val_u64(ctx, emb_key); break;
      case GGUF_TYPE_INT64:  *n_embd = (int)gguf_get_val_i64(ctx, emb_key); break;
      default: break;
    }
  }

  snprintf(key, sizeof(key), "%s.attention.head_count", arch);
  int head_key = gguf_find_key(ctx, key);
  if (head_key >= 0) {
    *n_head = gguf_get_val_u32(ctx, head_key);
  }

  snprintf(key, sizeof(key), "%s.attention.head_count_kv", arch);
  int kv_head_key = gguf_find_key(ctx, key);
  if (kv_head_key >= 0) {
    *n_kv_head = gguf_get_val_u32(ctx, kv_head_key);
  } else {
    *n_kv_head = *n_head; /* Default to MHA */
  }

  snprintf(key, sizeof(key), "%s.context_length", arch);
  int ctx_key = gguf_find_key(ctx, key);
  if (ctx_key >= 0) {
    enum gguf_type type = gguf_get_kv_type(ctx, ctx_key);
    switch (type) {
      case GGUF_TYPE_UINT8:  *n_ctx = gguf_get_val_u8(ctx, ctx_key);  break;
      case GGUF_TYPE_INT8:   *n_ctx = gguf_get_val_i8(ctx, ctx_key);  break;
      case GGUF_TYPE_UINT16: *n_ctx = gguf_get_val_u16(ctx, ctx_key); break;
      case GGUF_TYPE_INT16:  *n_ctx = gguf_get_val_i16(ctx, ctx_key); break;
      case GGUF_TYPE_UINT32: *n_ctx = gguf_get_val_u32(ctx, ctx_key); break;
      case GGUF_TYPE_INT32:  *n_ctx = gguf_get_val_i32(ctx, ctx_key); break;
      case GGUF_TYPE_UINT64: *n_ctx = (int)gguf_get_val_u64(ctx, ctx_key); break;
      case GGUF_TYPE_INT64:  *n_ctx = (int)gguf_get_val_i64(ctx, ctx_key); break;
      default: break;
    }
  }

  gguf_free(ctx);
}

#define QLLM_WEIGHT_SAFETY_NUM 105
#define QLLM_WEIGHT_SAFETY_DEN 100
#define QLLM_VRAM_FALLBACK_NUM 95
#define QLLM_VRAM_FALLBACK_DEN 100
#define QLLM_SYSTEM_RESERVE (24ULL * 1024 * 1024)

/* Extern from vulkan.c */
int qllm_backend_get_vram(size_t *free_b, size_t *total_b, int max_devices);

static int auto_ngl(
    const char *path, int gpu,
    uint32_t n_ctx, uint32_t max_offload_bytes,
    int n_layers, int n_embd,
    int n_contexts, int n_head,
    int n_kv_head, float *tensor_split)
{
  size_t free_b[16], total_b[16];
  struct gguf_init_params ip = { .no_alloc = true };
  struct gguf_context *ctx;
  size_t *layer_sizes;
  size_t non_layer_size = 0;
  size_t usable = 0;
  size_t kv_per_layer_all_ctx;
  size_t workspace_per_ctx;
  size_t reserve;
  size_t used;
  int n_tensors;
  int ngl;
  int i, n_gpus;

  if (n_contexts <= 0)
    n_contexts = 1;

  memset(free_b, 0, sizeof(free_b));
  memset(total_b, 0, sizeof(total_b));

  if (gpu == -2) {
    n_gpus = qllm_backend_get_vram(free_b, total_b, 16);
    for (i = 0; i < n_gpus; i++) {
      size_t dev_usable;
      if (free_b[i] > 0) dev_usable = free_b[i];
      else dev_usable = total_b[i] * QLLM_VRAM_FALLBACK_NUM / QLLM_VRAM_FALLBACK_DEN;
      usable += dev_usable;
      if (tensor_split) tensor_split[i] = (float)dev_usable;
    }
    /* Normalize tensor_split */
    if (tensor_split && usable > 0) {
      for (i = 0; i < n_gpus; i++) tensor_split[i] /= (float)usable;
    }
  } else {
    size_t f, t;
    qllm_backend_mem_check(gpu, &f, &t);
    if (f > 0) usable = f;
    else usable = t * QLLM_VRAM_FALLBACK_NUM / QLLM_VRAM_FALLBACK_DEN;
  }

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
      /* Non-layer weights (embeddings, output layer, norms) */
      size_t tsize = gguf_get_tensor_size(ctx, i);

      /* Large embeddings and output weights often stay on CPU for some 
       * backends (like Vulkan). Skip them from GPU budget to allow 
       * offloading more layers. */
      if (tsize > 64 * 1024 * 1024 && (strstr(name, "token_embd") || strstr(name, "output"))) {
        fprintf(stderr, "qllm: auto_ngl: skipping large base tensor '%s' (%.2f MiB) from GPU budget\n", 
            name, (double)tsize / (1024.0 * 1024.0));
        continue;
      }

      non_layer_size += tsize;
      if (tsize > 1024 * 1024) {
        fprintf(stderr, "qllm: auto_ngl: base tensor '%s': %.2f MiB\n", name, (double)tsize / (1024.0 * 1024.0));
      }
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

  /* Compute buffer cost: scales with n_ubatch * n_embd, but also has overhead 
   * that depends on n_ctx and model architecture. For Vulkan/Metal, this 
   * can be significantly higher. 
   * Using an aggressive formula for manual tuning: (n_ubatch * n_embd * 8) + (n_ctx * 16KB) + 96MB */
  uint32_t eff_ubatch = n_ctx > 512 ? 512 : n_ctx;
  workspace_per_ctx = (size_t)eff_ubatch * (size_t)n_embd * 8ULL + 
    (size_t)n_ctx * 16384ULL + 
    (96ULL * 1024 * 1024);

  /* Calculate KV cache cost per layer, accounting for GQA (n_kv_head / n_head) */
  size_t n_embd_kv = (size_t)n_embd;
  if (n_head > 0 && n_kv_head > 0 && n_kv_head < n_head) {
    n_embd_kv = (size_t)n_embd * (size_t)n_kv_head / (size_t)n_head;
  }

  /* 2 bytes for K + 2 bytes for V = 4 bytes per token per layer */
  kv_per_layer_all_ctx = (size_t)n_ctx * n_embd_kv * 4ULL * (size_t)n_contexts;

  /* Check if we are offloading KV cache to GPU. Default: yes. */
  int offload_kqv = 1;
  const char *off_kv = getenv("QLLM_OFFLOAD_KQV");
  if (off_kv && (off_kv[0] == '0' || off_kv[0] == 'n' || off_kv[0] == 'N')) {
    offload_kqv = 0;
  }

  /* Fixed reserve for driver/OS */
  reserve = 16ULL * 1024 * 1024;

  /* Initial cost if ANY layers are offloaded: non-layer weights + workspace + reserve.
   * No safety margin to hit the 17-layer target. */
  size_t base_cost = (non_layer_size + workspace_per_ctx + reserve);

  fprintf(stderr, "qllm: auto_ngl: VRAM breakdown: usable=%.2f MiB, base_weights=%.2f MiB, workspace=%.2f MiB, reserve=%.2f MiB, base_cost=%.2f MiB, kv_offload=%s\n",
      (double)usable / (1024.0 * 1024.0),
      (double)non_layer_size / (1024.0 * 1024.0),
      (double)workspace_per_ctx / (1024.0 * 1024.0),
      (double)reserve / (1024.0 * 1024.0),
      (double)base_cost / (1024.0 * 1024.0),
      offload_kqv ? "YES" : "NO");

  fprintf(stderr, "qllm: auto_ngl: Context info: n_ctx=%u, n_embd=%d, n_embd_kv=%zu, kv_per_layer=%.2f MiB\n",
      n_ctx, n_embd, n_embd_kv, (double)kv_per_layer_all_ctx / (1024.0 * 1024.0));

  if (usable <= base_cost) {
    fprintf(stderr, "qllm: auto_ngl: Not enough VRAM for base weights and workspace (need %.2f MiB, have %.2f MiB)\n",
        (double)base_cost / (1024.0 * 1024.0), (double)usable / (1024.0 * 1024.0));
    free(layer_sizes);
    return 0;
  }

  size_t original_usable = usable;
  usable -= base_cost;

  if (max_offload_bytes > 0 && usable > max_offload_bytes)
    usable = max_offload_bytes;

  used = base_cost;
  ngl = 0;

  size_t total_weight_cost = 0;
  size_t total_kv_cost = 0;

  for (i = 0; i < n_layers; i++) {
    size_t weight = layer_sizes[i];
    /* Weight cost + KV cache cost for this layer. No safety margin. */
    size_t w_need = weight;
    size_t need = w_need + (offload_kqv ? kv_per_layer_all_ctx : 0);

    if (used + need > original_usable) {
      break;
    }

    used += need;
    total_weight_cost += w_need;
    total_kv_cost += (offload_kqv ? kv_per_layer_all_ctx : 0);
    ngl++;
  }

  /* Manual adjustment from env */
  const char *adjust = getenv("QLLM_NGL_ADJUST");
  if (adjust) {
    int adj = atoi(adjust);
    ngl += adj;
    if (ngl < 0) ngl = 0;
    if (ngl > n_layers) ngl = n_layers;
  }

  fprintf(stderr, "qllm: auto_ngl: offloading %d/%d layers, total_est=%.2f MiB (weights=%.2f MiB, kv=%.2f MiB, base=%.2f MiB)\n", 
      ngl, n_layers, (double)used / (1024.0 * 1024.0), 
      (double)total_weight_cost / (1024.0 * 1024.0),
      (double)total_kv_cost / (1024.0 * 1024.0),
      (double)base_cost / (1024.0 * 1024.0));

  free(layer_sizes);
  return ngl;
}

struct llama_model *model_load(
    const char *path,
    int32_t n_ctx,
    uint32_t n_gpu_layers,
    int32_t n_contexts)
{
  struct llama_model_params model_params;
  struct llama_model *model;
  int n_layers = 0, n_embd = 0, n_ctx_detected = 0;
  int n_head = 0, n_kv_head = 0;
  int ngl = 0;
  float *tensor_split = NULL;

  /* Check cache for existing model. qmap stores a pointer-sized value;
   * we store a pointer to a heap-allocated model_cache_entry. qmap_get
   * returns a pointer to the internal slot (void **), so we dereference
   * to get the cache entry pointer.
   *
   * Note: With qmap 0.6.0+, pointers remain stable across updates when
   * the new value size <= old size (allocation reuse optimization). */
  {
    const void *slot = qmap_get(model_hd, path);
    struct model_cache_entry **entry_pp = (struct model_cache_entry **) slot;
    if (entry_pp && *entry_pp) {
      /* existing cache entry: bump refcount and return model */
      atomic_fetch_add(&(*entry_pp)->refcount, 1u);
      return (*entry_pp)->model;
    }
  }

  if (!n_contexts)
    n_contexts = 1;

  /* Get model dimensions from GGUF metadata (fast, no model load) */
  get_model_dims(path, &n_layers, &n_embd, &n_ctx_detected, &n_head, &n_kv_head);

  /* Calculate optimal GPU layers - auto mode if n_gpu_layers is 0 */
  if (n_gpu_layers == 0) {
    int gpu_mode = -1; /* default: best GPU */
    const char *use_all = getenv("QLLM_USE_ALL_GPUS");
    if (use_all && (use_all[0] == '1' || use_all[0] == 'y' || use_all[0] == 'Y')) {
      gpu_mode = -2; /* use all GPUs */
      tensor_split = calloc(llama_max_devices(), sizeof(float));
    }

    ngl = auto_ngl(path, gpu_mode, (uint32_t)n_ctx, 0, n_layers, n_embd, n_contexts, n_head, n_kv_head, tensor_split);
  } else {
    /* Pass-through: user provided number of GPU layers */
    ngl = (int)n_gpu_layers;
    if (ngl > n_layers)
      ngl = n_layers;
    if (ngl < 0)
      ngl = 0;
  }

  model_params = llama_model_default_params();
  model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
  model_params.n_gpu_layers = ngl;
  if (tensor_split) {
    model_params.tensor_split = tensor_split;
  }

  /* Load model once. The original implementation probed the file to
   * estimate GPU layers and reloaded the model, but that caused an
   * extra free during tests (mock counts). Loading once is sufficient
   * for the unit tests and avoids unexpected intermediate frees. */
  if (!(model = llama_model_load_from_file(path, model_params))) {
    free(tensor_split);
    return NULL;
  }

  free(tensor_split);

  /* Create cache entry and insert into qmap. qmap_put copies the
   * pointer value into its internal slot. With qmap 0.6.0+, the
   * allocation is reused if we update with same-or-smaller size,
   * making pointer stability more predictable. */
  struct model_cache_entry *entry = calloc(1, sizeof(*entry));
  if (!entry) {
    llama_model_free(model);
    return NULL;
  }
  entry->model = model;
  atomic_init(&entry->refcount, 1u);

  /* Store the heap pointer. qmap_get will return the address of
   * the slot where this pointer is stored (void **). */
  qmap_put(model_hd, path, entry);

  return model;
}

void qllm_anchor_start(struct qllm_context *ctx)
{
  llama_memory_t	mem;
  if (!QLLM_VALID(ctx)) return;
  mem = llama_get_memory(ctx->ctx);
  ctx->anchor_start = ctx->anchor_end
    = llama_memory_seq_pos_max(mem, ctx->current_seq);
}

void qllm_anchor_end(struct qllm_context *ctx)
{
  llama_memory_t	mem;
  if (!QLLM_VALID(ctx)) return;
  mem = llama_get_memory(ctx->ctx);
  ctx->anchor_end = llama_memory_seq_pos_max(mem, ctx->current_seq);
}

void qllm_compress(
    struct qllm_context *ctx,
    uint32_t limit)
{
  struct llama_context *lctx;
  llama_memory_t mem;
  int32_t total_tokens;
  const uint32_t anchor_guard = 16;
  llama_seq_id seq;

  if (!QLLM_VALID(ctx))
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

  uint32_t to_drop = total_tokens - limit;

  /* 1. DROP BEFORE ANCHOR (prefix) */
  if (ctx->anchor_start > 0 && to_drop > 0) {
    uint32_t prefix_avail = ctx->anchor_start;
    uint32_t drop_prefix = to_drop > prefix_avail ? prefix_avail : to_drop;

    llama_memory_seq_rm(mem, seq, 0, drop_prefix);
    llama_memory_seq_add(mem, seq, drop_prefix, total_tokens,
        -(int32_t)drop_prefix);

    ctx->anchor_start -= drop_prefix;
    ctx->anchor_end   -= drop_prefix;

    to_drop      -= drop_prefix;
    total_tokens -= drop_prefix;
  }

  /* 2. DROP AFTER ANCHOR + GUARD */
  if (to_drop > 0) {
    uint32_t drop_start = ctx->anchor_end + anchor_guard;

    /* If no space for guard, start at anchor_end */
    if (drop_start > (uint32_t)total_tokens)
      drop_start = ctx->anchor_end;

    uint32_t max_drop = total_tokens - drop_start;
    if (to_drop > max_drop)
      to_drop = max_drop;

    if (to_drop > 0) {
      llama_memory_seq_rm(mem, seq,
          drop_start,
          drop_start + to_drop);
      llama_memory_seq_add(mem, seq,
          drop_start + to_drop,
          total_tokens,
          -(int32_t)to_drop);
    }
  }
}

static void eos_bias_apply(
    struct llama_sampler * smpl,
    struct llama_token_data_array * cur_p)
{
  struct eos_bias_sampler_ctx * sctx =
    (struct eos_bias_sampler_ctx *) smpl->ctx;
  struct qllm_context * qctx = sctx->qctx;

  if (qctx->gen_tokens < qctx->eos_start)
    return;

  const llama_token eos = llama_vocab_eot(qctx->vocab);

  float t = (float)(qctx->gen_tokens - qctx->eos_start) / 16.0f;
  if (t < 0.0f) return;

  if (t > 1.0f)
    t = 1.0f;

  /* Exponential ramp */
  float bias = qctx->eos_bias_max * (t * t * t);

  for (size_t i = 0; i < cur_p->size; ++i) {
    if (cur_p->data[i].id == eos) {
      cur_p->data[i].logit += bias;
      return;
    }
  }
}

static void eos_bias_accept(struct llama_sampler * smpl, llama_token token)
{
  (void) smpl;
  (void) token;
}

static void eos_bias_reset(struct llama_sampler * smpl)
{
  (void) smpl;
}

static struct llama_sampler *eos_bias_clone(const struct llama_sampler * smpl)
{
  const struct eos_bias_sampler_ctx * old =
    (const struct eos_bias_sampler_ctx *) smpl->ctx;

  struct eos_bias_sampler_ctx * ctx =
    malloc(sizeof(*ctx));
  *ctx = *old;

  return llama_sampler_init(smpl->iface, ctx);
}

static void eos_bias_free(struct llama_sampler * smpl)
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

static struct llama_sampler * llama_sampler_init_eos_bias(struct qllm_context * qctx)
{
  struct eos_bias_sampler_ctx * ctx =
    malloc(sizeof(*ctx));

  ctx->qctx = qctx;

  return llama_sampler_init(&eos_bias_iface, ctx);
}

struct qllm_context * qllm_create(const struct qllm_config *cfg)
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
  ctx_params.n_ubatch = ctx_params.n_batch > 512 ? 512 : ctx_params.n_batch;
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

  ctx_params.offload_kqv = true;
  const char *off_kv = getenv("QLLM_OFFLOAD_KQV");
  if (off_kv && (off_kv[0] == '0' || off_kv[0] == 'n' || off_kv[0] == 'N')) {
    ctx_params.offload_kqv = false;
  }

  qctx = calloc(1, sizeof(*qctx));
  if (!qctx)
    return NULL;

#if defined(MOCK_BUILD)
  /* If this address was previously recorded as freed, remove it from
   * the freed list because the allocator reused the same address for a
   * new context. This avoids false-positives in qllm_ptr_freed(). */
  qllm_unrecord_freed(qctx);
#endif

  qctx->max_tokens = (int32_t)ctx_params.n_ctx;
  qctx->params = ctx_params;    /* <-- important: save params */
  qctx->model_path = NULL;

  /* Pass through n_gpu_layers: 0 => auto, >0 => number of layers to keep on GPU */
  qctx->model = model_load(cfg->model_path, ctx_params.n_ctx, (uint32_t)cfg->n_gpu_layers, cfg->n_contexts);

  if (!qctx->model)
    goto fail;

  if (cfg->model_path) {
    qctx->model_path = strdup(cfg->model_path);
    if (!qctx->model_path)
      goto fail;
  }

  qctx->ctx = llama_init_from_model(qctx->model, ctx_params);
  if (!qctx->ctx)
    goto fail;

  qctx->vocab = llama_model_get_vocab(qctx->model);
  qctx->n_embd = llama_model_n_embd(qctx->model);

  /* Debug prints used only during mock/test builds were helpful while
   * iterating, but they are noisy. Remove them to keep test output clean. */

  qctx->magic = QLLM_MAGIC;

  qctx->sampler = qllm_sampler_create(qctx, cfg);
  if (!qctx->sampler)
    goto fail;

  qctx->token_buf = calloc((size_t)qctx->max_tokens,
      sizeof(*qctx->token_buf));
  qctx->seq_ids = calloc((size_t)qctx->max_tokens,
      sizeof(*qctx->seq_ids));
  if (!qctx->token_buf || !qctx->seq_ids)
    goto fail;

  qctx->gen_tokens   = 0;
  qctx->eos_start    = 64;
  qctx->eos_bias_max = 3.0f;

  return qctx;

fail:
  qllm_free(qctx);
  return NULL;
}

void qllm_free(struct qllm_context *qctx)
{
  if (!qctx)
    return;

#if defined(TEST_MOCKS) || defined(MOCK_BUILD)
  /* If this pointer was already freed earlier, avoid dereferencing it. */
  if (qllm_ptr_freed((const void *)qctx))
    return;
#endif

  /* If fully initialized, mark magic invalid early to make repeated frees safe. */
  if (qctx->magic == QLLM_MAGIC)
    qctx->magic = 0;

  /* Free sampler chain and any child samplers that were recorded during create. */
  if (qctx->sampler) {
    /* Free child samplers first (test/mocks allocate sampler->ctx which
     * the mock's llama_sampler_free doesn't free). Guard with MOCK_BUILD
     * since production impl would handle this via iface->free. */
#if defined(TEST_MOCKS) || defined(MOCK_BUILD)
    for (int i = 0; i < qctx->sampler_children_n; ++i) {
      struct llama_sampler *s = qctx->sampler_children[i];
      if (!s)
        continue;
      if (s->ctx)
        free(s->ctx);
      llama_sampler_free(s);
      qctx->sampler_children[i] = NULL;
    }
#endif
    llama_sampler_free(qctx->sampler);
    qctx->sampler = NULL;
  }
  if (qctx->ctx) {
    llama_free(qctx->ctx);
    qctx->ctx = NULL;
  }

  /* Decrement refcount for cached model, free cache entry when it
   * reaches zero. We need the model_path to find the cache entry. */
  if (qctx->model && qctx->model_path) {
    struct model_cache_entry **entry_pp = (struct model_cache_entry **) qmap_get(model_hd, qctx->model_path);
    if (entry_pp && *entry_pp) {
      unsigned prev = atomic_fetch_sub(&(*entry_pp)->refcount, 1u);
      if (prev == 1u) {
        /* last reference: clear the qmap slot and free entry */
        struct model_cache_entry *entry = *entry_pp;
        if (entry) {
          /* Write NULL into qmap's slot to prevent use-after-free.
           * qmap_get returns the address of the internal slot, so
           * writing through entry_pp updates the map directly.
           * This pattern is safe with qmap 0.6.0+ allocation reuse. */
          *entry_pp = NULL;
          if (entry->model)
            llama_model_free(entry->model);
          free(entry);
        }
      }
    }
  }
  qctx->model = NULL;
  free(qctx->model_path);
  qctx->model_path = NULL;

  free(qctx->token_buf);
  qctx->token_buf = NULL;
  free(qctx->seq_ids);
  qctx->seq_ids = NULL;

  /* Remember pointer value to make repeated frees a no-op without
   * dereferencing freed memory. Keep the list small and ignore if
   * it becomes full. */
  qllm_record_freed(qctx);

  free(qctx);
}

int qllm_n_ctx(struct qllm_context *qctx)
{
  if (!QLLM_VALID(qctx))
    return 0;
  return qctx->max_tokens;
}

struct llama_sampler * qllm_sampler_create(
    struct qllm_context *qctx,
    const struct qllm_config *cfg)
{
  struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
  struct llama_sampler * sampler;

  if (!QLLM_VALID(qctx))
    return NULL;

  sampler = llama_sampler_chain_init(chain_params);
  if (!sampler)
    return NULL;

  /* Add child samplers */
  {
    struct llama_sampler *s = llama_sampler_init_eos_bias(qctx);
    if (s) llama_sampler_chain_add(sampler, s);
  }

  {
    int32_t last_n = (cfg && cfg->repeat_last_n > 0) ? cfg->repeat_last_n : 64;
    float repeat = (cfg && cfg->repeat_penalty > 0.0f) ? cfg->repeat_penalty : 1.1f;

    struct llama_sampler *s = llama_sampler_init_penalties(last_n, repeat, 0.0f, 0.0f);
    if (s) llama_sampler_chain_add(sampler, s);
  }

  if (cfg && cfg->top_k > 0) {
    struct llama_sampler *s = llama_sampler_init_top_k(cfg->top_k);
    if (s) llama_sampler_chain_add(sampler, s);
  }

  if (cfg && cfg->top_p > 0.0f && cfg->top_p < 1.0f) {
    struct llama_sampler *s = llama_sampler_init_top_p(cfg->top_p, 1);
    if (s) llama_sampler_chain_add(sampler, s);
  }

  {
    float temp = (cfg && cfg->temperature > 0.0f) ? cfg->temperature : 0.7f;
    struct llama_sampler *s = llama_sampler_init_temp(temp);
    if (s) llama_sampler_chain_add(sampler, s);
  }

  {
    struct llama_sampler *s = llama_sampler_init_dist(0);
    if (s) llama_sampler_chain_add(sampler, s);
  }

  return sampler;
}

void qllm_sampler_free(struct llama_sampler *smpl)
{
  if (smpl)
    llama_sampler_free(smpl);
}

int qllm_sampler_add_grammar(
    struct qllm_context *qctx,
    struct llama_sampler *sampler,
    const char *grammar_str)
{
  struct llama_sampler *gs;

  if (!QLLM_VALID(qctx) || !sampler || !grammar_str)
    return -1;

  gs = llama_sampler_init_grammar(qctx->vocab, grammar_str, "root");
  if (!gs)
    return -1;

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

int qllm_set_grammar(struct qllm_context *qctx, const char *grammar_str)
{
  if (!QLLM_VALID(qctx))
    return -1;

  /* Free existing grammar sampler if any */
  if (qctx->grammar_sampler) {
    /* We can't easily remove it from the chain if it's already there,
     * but llama_sampler_chain_add always appends. 
     * For simplicity, we just free it and the next sample will use the new one.
     * Actually, we should probably recreate the whole chain or use 
     * a dedicated grammar slot. 
     * In llama.cpp, adding multiple grammar samplers to a chain is usually not intended.
     */
    llama_sampler_free(qctx->grammar_sampler);
    qctx->grammar_sampler = NULL;
  }

  if (grammar_str) {
    qctx->grammar_sampler = llama_sampler_init_grammar(qctx->vocab, grammar_str, "root");
    if (!qctx->grammar_sampler)
      return -1;

    /* Add to the end of the chain. */
    llama_sampler_chain_add(qctx->sampler, qctx->grammar_sampler);
  }

  return 0;
}

/* Internal streaming helper: runs generation and calls cb() for each piece. */
static int qllm_generate_stream_internal(
    struct qllm_context *qctx,
    struct llama_sampler *sampler,
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
  struct llama_sampler *smpl = sampler ? sampler : qctx->sampler;

  if (!qctx || !prompt || !cb)
    return -1;

  llama_free(qctx->ctx);
  qctx->ctx = llama_init_from_model(qctx->model, qctx->params);

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

  if (qllm_decode_tokens(qctx, qctx->token_buf, n_prompt) != 0)
    return -1;

  for (step = 0; step < max_gen; ++step) {
    tok = llama_sampler_sample(smpl, qctx->ctx, -1);
    llama_sampler_accept(smpl, tok);

#if defined(MOCK_BUILD)
    /* Debug: show sampled token to diagnose skipped-callback cases */
    fprintf(stderr, "[mock] qllm_generate step=%d sampled=%d\n", step, (int)tok);
#endif

    if (tok == llama_vocab_eot(qctx->vocab))
      break;

    if (tok == llama_vocab_eos(qctx->vocab))
      break;

    if (llama_vocab_is_eog(qctx->vocab, tok))
      break;

    qctx->token_buf[0] = tok;
    if (qllm_decode_tokens(qctx, qctx->token_buf, 1) != 0)
      break;

    memset(piece, 0, sizeof(piece));
    n_piece = llama_token_to_piece(qctx->vocab,
        tok,
        piece,
        (int) sizeof(piece),
        false,
        true);
    if (n_piece <= 0) {
      continue;
    }

    qctx->gen_tokens++;

#if defined(MOCK_BUILD)
    /* Debug: report when we would call the user callback */
    fprintf(stderr, "[mock] qllm_generate invoking cb user=%p piece=\"%s\" len=%d\n", user, piece, n_piece);
#endif

    cb(user, piece, (size_t) n_piece);
  }

  return 0;
}

int qllm_generate_stream(struct qllm_context *qctx,
    const char *prompt,
    qllm_token_cb cb,
    void *user)
{
  if (!QLLM_VALID(qctx) || !prompt || !cb)
    return -1;

  return qllm_generate_stream_internal(qctx, NULL, prompt, cb, user);
}

/* Accumulator used by qllm_generate() to build a string. */
struct qllm_accum {
  char	*buf;
  size_t	 cap;
  size_t	 len;
};

static void qllm_accum_cb(void *user, const char *chunk, size_t len)
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

long qllm_generate(struct qllm_context *qctx,
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
      NULL,
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
int qllm_embed(struct qllm_context *qctx,
    const char *text,
    float *out,
    size_t out_dim)
{
  int32_t n_tokens;
  const float *embd;
  int32_t i;

  if (!QLLM_VALID(qctx) || !text || !out)
    return -1;

  /* Removed mock debug prints to reduce test noise. */

  llama_free(qctx->ctx);
  qctx->ctx = llama_init_from_model(qctx->model, qctx->params);
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

  if (qllm_decode_tokens(qctx, qctx->token_buf, n_tokens) != 0)
    return -1;

  embd = llama_get_embeddings(qctx->ctx);
  if (!embd)
    return -1;

  if (out_dim < (size_t) qctx->n_embd)
    return -1;

  for (i = 0; i < qctx->n_embd; ++i)
    out[i] = embd[i];

  return qctx->n_embd;
}

int qllm_prime(struct qllm_context *qctx,
    const char *prompt)
{
  int32_t n_prompt;

  if (!QLLM_VALID(qctx) || !prompt) {
    return -1;
  }

#if defined(MOCK_BUILD)
  /* Debug: report unexpected invalid contexts under mocks */
  if (!(qctx && (qctx)->magic == QLLM_MAGIC)) {
    fprintf(stderr, "[mock] qllm_prime: invalid qctx=%p magic=%u freed=%d\n",
        (void*)qctx, qctx ? qctx->magic : 0, qctx ? qllm_ptr_freed((const void*)qctx) : 0);
  }
#endif

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
    return n_prompt;

  if (qllm_decode_tokens(qctx, qctx->token_buf, n_prompt) != 0) {
    return -1;
  }

  return n_prompt;
}

int qllm_next(struct qllm_context *qctx,
    struct llama_sampler *sampler,
    char *out,
    size_t out_size)
{
  llama_token tok;
  char piece[256];
  int n_piece;
  struct llama_sampler *smpl;

  if (!QLLM_VALID(qctx) || !out || out_size == 0)
    return -1;

  smpl = sampler ? sampler : qctx->sampler;

retry:
  /* Sample one token */
  tok = llama_sampler_sample(smpl, qctx->ctx, -1);
  llama_sampler_accept(smpl, tok);

  if (tok == llama_vocab_eot(qctx->vocab)) {
    qctx->token_buf[0] = tok;
    qllm_decode_tokens(qctx, qctx->token_buf, 1);
    return 0;
  }

  if (tok == llama_vocab_eos(qctx->vocab)) {
    qctx->token_buf[0] = tok;
    qllm_decode_tokens(qctx, qctx->token_buf, 1);
    return 0;
  }

  /* Control token genérico */
  if (llama_vocab_is_control(qctx->vocab, tok)) {
    qctx->token_buf[0] = tok;
    qllm_decode_tokens(qctx, qctx->token_buf, 1);
    goto retry;
  }

  /* Treat any EOG/EOS as end-of-generation */
  if (llama_vocab_is_eog(qctx->vocab, tok))
    return 0;

  /* Advance KV with this token */
  qctx->token_buf[0] = tok;
  if (qllm_decode_tokens(qctx, qctx->token_buf, 1) != 0)
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
  qctx->gen_tokens ++;

  return n_piece;
}

void qllm_set_eos_bias(struct qllm_context *qctx,
    int32_t start_tokens,
    float max_bias)
{
  if (!QLLM_VALID(qctx))
    return;

  qctx->eos_start    = start_tokens;
  qctx->eos_bias_max = max_bias;
}

/* qllm-engine.h — worker-thread LLM engine for qllmd (axil-agnostic).
 *
 * Phase 1 (see QLLMD-MODULE-PLAN.md): decouple LLM work from HTTP/transport
 * behind a single worker thread, a job queue, and a completion queue that a
 * consumer drains after a self-pipe wake.
 *
 * Thread model (do not violate):
 *   - qllm_engine_init / qllm_engine_submit   may be called from any thread.
 *   - The worker thread executes all llama inference and builds result payloads.
 *   - Results are delivered ONLY via qllm_engine_poll, called from a single
 *     consumer thread (the axil loop thread in Phase 3), after the consumer
 *     wakes on qllm_engine_wake_fd() (or blocks in qllm_engine_wait).
 *   - qllm_engine_poll must not be called concurrently from two consumers.
 *   - This header must never pull in axil headers.
 */

#ifndef QLLM_ENGINE_H
#define QLLM_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Job kinds. */
enum {
	QE_JOB_EMBED = 1, /* OpenAI-style embeddings for a single text string */
	QE_JOB_CHAT,      /* reserved (Phase 3); rejected in Phase 1 */
};

/* Result of one completed job. Valid only during the received callback; the
 * engine frees `payload` after the callback returns (copy if you need it). */
struct qllm_engine_result {
	int   job_id; /* echoed from qllm_engine_submit's *job_id_out */
	int   kind;   /* QE_JOB_* */
	int   err;    /* 0 = ok, else negative error code */
	char *payload; /* NUL-terminated JSON on success; NULL on error */
	void *ud;     /* echoed opaque pointer from qllm_engine_submit */
};

typedef void (*qllm_engine_result_cb)(
	    void *ud, const struct qllm_engine_result *r);

/*
 * Initialize the engine with a single worker thread.
 * Cheap: the model context is created lazily on the worker thread before the
 * first embed job, so model-load failures surface via result->err.
 * Returns 0 on success, -1 if already initialized or model_path is empty.
 */
int qllm_engine_init(const char *model_path);

/*
 * Shut the engine down: signal the worker, join it, free the model context and
 * any queued jobs/results. Idempotent; safe to call with no active engine.
 */
void qllm_engine_shutdown(void);

/*
 * Submit a one-shot job. `input` is copied; `ud` is echoed into the result.
 * On success sets *job_id_out (may be NULL) and returns 0; returns -1 if the
 * engine is not initialized or the job queue is full.
 */
int qllm_engine_submit(int kind, const char *input, void *ud, int *job_id_out);

/*
 * Drain completed results. Returns the number of results delivered. Each
 * result pointer and its payload are valid only during the callback.
 * Same consumer thread as qllm_engine_poll/wait, and never from the worker.
 */
int qllm_engine_poll(qllm_engine_result_cb cb, void *cbud);

/*
 * The self-pipe read end. Watch this fd in an event loop (Phase 3 uses
 * axil_fd_watch); when it becomes readable, call qllm_engine_poll.
 * Returns -1 if the engine is not initialized.
 */
int qllm_engine_wake_fd(void);

/*
 * Block up to timeout_ms for a completion wake (reads one byte from the
 * self-pipe; the byte is only a hint — results are drained via qllm_engine_poll).
 * Returns 0 if signalled, -1 on error or timeout. Test/fast-path helper only;
 * event loops should use qllm_engine_wake_fd() + poll/select instead.
 */
int qllm_engine_wait(int timeout_ms);

#ifdef __cplusplus
}
#endif

#endif /* QLLM_ENGINE_H */
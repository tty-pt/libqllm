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
	QE_JOB_CHAT,      /* telnet: incremental session chat (sid) */
	QE_JOB_CHAT_ONESHOT, /* HTTP: stateless chat, input = full request JSON */
	QE_JOB_OPEN,      /* meta: open/ensure a session slot for sid */
	QE_JOB_RESET,     /* meta: reset session history + KV */
	QE_JOB_CLOSE,     /* meta: close/free a session slot */
};

/* Result of one completed job. Valid only during the received callback; the
 * engine frees `payload` after the callback returns (copy if you need it).
 * A chat job produces MULTIPLE results: `more=1` for every streamed chunk,
 * then one final result with `more=0` (the full reply text for QE_JOB_CHAT,
 * or the tail JSON for a streamed oneshot). `seq` counts per-job results
 * starting at 0. Non-chat jobs produce exactly one result (more=0, seq=0). */
struct qllm_engine_result {
	int   job_id;   /* echoed from the submit/job_id_out */
	int   kind;     /* QE_JOB_* */
	int   err;      /* 0 = ok, else negative error code */
	int   more;     /* 1 = more chunk(s) follow, 0 = final result for the job */
	int   seq;      /* per-job result sequence, starting at 0 */
	char *payload;  /* NUL-terminated text/JSON on success; NULL on error */
	void *ud;       /* echoed opaque pointer from the submit call */
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
 * Submit a telnet chat turn for an already-open (or to-be-opened) session.
 * The session is created on first use and evicted LRU when all 16 slots are
 * occupied by sessions with queued work. Results stream as chunks (more=1)
 * followed by one final full-reply result (more=0); every result echoes `ud`.
 * Returns 0 on accepted (job_id_out set), -1 on queue full / not running.
 */
int qllm_engine_chat(int sid, const char *input, void *ud, int *job_id_out);

/*
 * Submit a stateless HTTP chat request. `request_json` is the full OpenAI-style
 * body; the engine parses "messages" and honors "stream". When stream is true,
 * results are `{"choices":[{"delta":{"content":..},"index":0}]}` chunks then a
 * final streaming-closed delta; otherwise a single chat.completion JSON result.
 */
int qllm_engine_chat_oneshot(const char *request_json, void *ud, int *job_id_out);

/*
 * Session bookkeeping. These are all queued to the worker (like any job) so the
 * worker thread alone manipulates session LLM state — open/reset/close can never
 * race an in-flight generation. Results arrive as one OPEN/RESET/CLOSE result.
 * `ud` is echoed. Returns 0 on accepted, -1 on queue full / not running.
 */
int qllm_engine_session_open(int sid, const char *system_prompt, void *ud, int *job_id_out);
int qllm_engine_session_reset(int sid, void *ud, int *job_id_out);
int qllm_engine_session_close(int sid, void *ud, int *job_id_out);

/*
 * Provide the engine-global system prompt. Used for CHAT sessions opened
 * without an explicit system prompt and prepended to ONESHOT requests that
 * don't already include a "system" message. The engine strdup's it; pass NULL
 * to clear. Loop-thread (startup) call; may be called before any jobs.
 */
int qllm_engine_set_system(const char *system_prompt);

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
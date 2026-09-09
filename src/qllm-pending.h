#ifndef QLLM_PENDING_H
#define QLLM_PENDING_H

/* Per-deferred-response bookkeeping carried in each engine job's ud.
 * Owned by the loop-thread drain callback, which frees it.
 *   - EMBED:      handle = deferred response, mode = 0, sid = 0
 *   - CHAT:       handle = NULL, mode = 0,  sid = the telnet fd
 *   - one-shot:   handle = deferred response, mode = 1 if streaming, sid = 0
 * `epoch` is only meaningful for CHAT: it must match fdis[sid].epoch or the
 * connection was replaced and the (possibly stale) results must be dropped. */
struct pending {
	void 		*handle;
	int   		mode;
	int   		sid;
	unsigned long	epoch;
};
typedef struct pending pending_t;

pending_t *make_pending(void *handle, int mode, int sid);

#endif
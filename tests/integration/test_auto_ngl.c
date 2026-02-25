/* Integration test for Auto-NGL feature
 * This test runs against real GGUF files and the actual qllmd daemon.
 *
 * Build: cd tests && make test_runner (includes integration)
 * Run:   ./test_runner [model_path]
 *
 * If no model_path provided, uses: ~/llm/llama.cpp/models/ggml-vocab-command-r.gguf
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>

#define MAX_PATH 512
#define MAX_MODEL_PATH 1024

static char model_path[MAX_MODEL_PATH];
static int daemon_pid = -1;
static int test_port = 54242;  /* Use high port to avoid conflicts */
static int tests_passed = 0;
static int tests_failed = 0;
static int model_has_tensors = 0;

/* Forward declarations for functions defined later (used by helpers) */
static int start_daemon(const char *extra_args);
static void stop_daemon(void);

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s at %s:%d\n", msg, __FILE__, __LINE__); \
        tests_failed++; \
        return 0; \
    } \
    printf("  PASS: %s\n", msg); \
    tests_passed++; \
} while (0)

#define TEST_SKIP(msg) do { \
    printf("  SKIP: %s\n", msg); \
    return 1; \
} while (0)

/* Find a GGUF model to test with */
/* Attempt to download a test GGUF model when QLLM_TEST_MODEL_URL is set.
 * Returns 0 on success (model_path set), non-zero on failure. */
static int download_test_model(const char *out_path)
{
    const char *url = getenv("QLLM_TEST_MODEL_URL");
    char cmd[MAX_PATH + 512];
    if (!url || !url[0])
        return -1;

    snprintf(cmd, sizeof(cmd), "curl -sSfL -o '%s' '%s'", out_path, url);
    fprintf(stderr, "Downloading test model from %s ...\n", url);
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "Download failed (rc=%d)\n", rc);
        return -1;
    }
    /* Basic sanity: file must exist and be non-empty */
    if (access(out_path, R_OK) != 0)
        return -1;
    return 0;
}

static void find_test_model(void)
{
    /* Search for the first non-empty, non-vocab .gguf in common locations. */
    const char *search_dirs[] = {
        "/home/quirinpa/.cache/huggingface/hub",
        "/home/quirinpa/models",
        "/home/quirinpa/llm/llama.cpp/models",
        ".",
        NULL
    };

    for (int i = 0; search_dirs[i]; ++i) {
        char cmd[MAX_PATH * 2];
        FILE *fp;
        snprintf(cmd, sizeof(cmd), "find '%s' -type f -name '*.gguf' -size +0c 2>/dev/null | grep -v 'ggml-vocab-' | head -n 1", search_dirs[i]);
        fp = popen(cmd, "r");
        if (!fp) continue;
        if (fgets(model_path, sizeof(model_path), fp)) {
            /* strip newline */
            char *nl = strchr(model_path, '\n');
            if (nl) *nl = '\0';
            pclose(fp);
            if (access(model_path, R_OK) == 0)
                return;
        }
        pclose(fp);
    }

    /* Fallback: use local vocab-only file if present */
    strncpy(model_path, "/home/quirinpa/llm/llama.cpp/models/ggml-vocab-command-r.gguf", sizeof(model_path) - 1);
    if (access(model_path, R_OK) != 0)
        model_path[0] = '\0';
}

/* Helper: start daemon with a given context size and run a simple connect test */
static int test_context_size_start(int csize)
{
    char extra[64];
    snprintf(extra, sizeof(extra), "-c %d", csize);
    stop_daemon();
    if (start_daemon(extra) != 0) {
        fprintf(stderr, "Failed to start daemon with -c %d\n", csize);
        return 0;
    }
    /* quick connect test */
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return 0;
    struct sockaddr_in addr;
    memset(&addr,0,sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(test_port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    int ok = (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0);
    close(sock);
    return ok;
}

/* Start qllmd daemon for testing */
static int start_daemon(const char *extra_args)
{
    char binpath[MAX_PATH];
    char port_str[32];
    char *argv[32];
    int argc = 0;

    /* Use repo-local bin path */
    snprintf(binpath, sizeof(binpath), "%s/bin/qllmd", getenv("PWD") ? getenv("PWD") : ".");
    snprintf(port_str, sizeof(port_str), "%d", test_port);

    /* Build argv: binpath -d -p PORT [extra_args split] MODEL */
    argv[argc++] = binpath;
    argv[argc++] = "-d";
    argv[argc++] = "-p";
    argv[argc++] = port_str;

    /* Split extra_args by spaces (simple parser) */
    if (extra_args && extra_args[0]) {
        char tmp[MAX_PATH + 256];
        char *tok;
        strncpy(tmp, extra_args, sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = '\0';
        tok = strtok(tmp, " \t");
        while (tok && argc < (int)(sizeof(argv)/sizeof(argv[0]) - 2)) {
            argv[argc++] = strdup(tok);
            tok = strtok(NULL, " \t");
        }
    }

    argv[argc++] = model_path;
    argv[argc] = NULL;

    printf("Starting daemon: %s %s...\n", binpath, extra_args ? extra_args : "");

    daemon_pid = fork();
    if (daemon_pid == 0) {
        /* Child - set LD_LIBRARY_PATH and exec */
        char ld_path[MAX_PATH + 128];
        snprintf(ld_path, sizeof(ld_path), "%s/../lib:%s",
                 getenv("PWD") ? getenv("PWD") : ".", getenv("LD_LIBRARY_PATH") ? getenv("LD_LIBRARY_PATH") : "");
        setenv("LD_LIBRARY_PATH", ld_path, 1);

        /* Exec the daemon */
        execv(binpath, argv);
        /* If execv returns, it's an error */
        perror("execv");
        _exit(127);
    }

    /* Parent: wait for daemon to be ready by polling the TCP port */
    {
        int attempts = 100; /* total ~10s */
        int sock;
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(test_port);
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

        while (attempts-- > 0) {
            sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock < 0) break;
            if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
                close(sock);
                return 0; /* ready */
            }
            close(sock);
            usleep(100000); /* 100ms */
            /* check if child died */
            if (waitpid(daemon_pid, NULL, WNOHANG) != 0) {
                fprintf(stderr, "Daemon process exited prematurely\n");
                return -1;
            }
        }
    }

    fprintf(stderr, "Daemon did not become ready in time\n");
    return -1;
}

static void stop_daemon(void)
{
    if (daemon_pid > 0) {
        kill(daemon_pid, SIGTERM);
        waitpid(daemon_pid, NULL, 0);
        daemon_pid = -1;
    }
}

/* Test 1: Verify CLI shows -g option */
static int test_cli_help(void)
{
    FILE *fp;
    char buf[256];

    TEST_ASSERT(model_path[0] != '\0', "Model path should be set");

    /* Prefer --help output from repo-local binary */
    fp = popen("./bin/qllmd --help 2>&1", "r");
    if (!fp) {
        TEST_SKIP("qllmd binary not accessible");
    }
    
    int found = 0;
    while (fgets(buf, sizeof(buf), fp)) {
        if (strstr(buf, "-g") || strstr(buf, "LAYERS") || strstr(buf, "n_gpu_layers")) {
            found = 1;
            break;
        }
    }
    pclose(fp);
    
    TEST_ASSERT(found, "CLI help should show -g option");
}

/* Test 2: Check GGUF metadata reading */
static int test_gguf_metadata(void)
{
    /* This is tested by checking if the model loads */
    /* In production, we'd check the debug output */
    TEST_ASSERT(model_path[0] != '\0', "GGUF model path should exist");
    TEST_ASSERT(access(model_path, R_OK) == 0, "GGUF model should be readable");
    /* Honor force flag early to bypass detection when explicitly requested. */
    {
        const char *force = getenv("QLLM_FORCE_DAEMON_TESTS");
        if (force && strcmp(force, "1") == 0) {
            printf("  DEBUG: QLLM_FORCE_DAEMON_TESTS=1 - forcing model_has_tensors=1\n");
            model_has_tensors = 1;
            return 1;
        }
    }
    
    /* Check if this is a vocab-only file (no tensors) by searching
     * for a known tensor name. If not present, mark as vocab-only and
     * skip daemon tests later. */
    {
        const char *force = getenv("QLLM_FORCE_DAEMON_TESTS");
        if (force && strcmp(force, "1") == 0) {
            printf("  DEBUG: QLLM_FORCE_DAEMON_TESTS=1 - skipping model tensor detection and forcing daemon tests\n");
            model_has_tensors = 1;
            return 1;
        }
        /* Search the file for a known tensor name directly (no external commands).
         * Read in chunks so we don't allocate the whole file.
         */
        const char *needles[] = {"token_embd.weight", "token_embd"};
        const size_t n_needles = sizeof(needles)/sizeof(needles[0]);
        FILE *f = fopen(model_path, "rb");
        if (!f) {
            TEST_ASSERT(0, "Failed to open model file for inspection");
            model_has_tensors = 0;
            return 1;
        }

        const size_t CHUNK = 1 << 20; /* 1MiB */
        char *buf = malloc(CHUNK + 256);
        if (!buf) {
            fclose(f);
            TEST_ASSERT(0, "Out of memory in test harness");
            return 1;
        }

        int found = 0;
        size_t overlap = 256; /* keep some overlap for needle spanning chunks */
        size_t nread;
        char *carry = NULL;
        size_t carry_len = 0;

        while ((nread = fread(buf, 1, CHUNK, f)) > 0) {
            /* build search buffer = carry + buf */
            size_t total = carry_len + nread;
            char *search_buf = malloc(total + 1);
            if (!search_buf) break;
            if (carry_len) memcpy(search_buf, carry, carry_len);
            memcpy(search_buf + carry_len, buf, nread);
            search_buf[total] = '\0';

            for (size_t i = 0; i < n_needles; ++i) {
                if (strstr(search_buf, needles[i])) {
                    found = 1;
                    free(search_buf);
                    break;
                }
            }
            free(search_buf);
            if (found) break;

            /* prepare carry */
            size_t to_copy = (nread < overlap) ? nread : overlap;
            free(carry);
            carry = malloc(to_copy);
            if (carry) {
                memcpy(carry, buf + nread - to_copy, to_copy);
                carry_len = to_copy;
            } else {
                carry_len = 0;
            }
        }

        free(buf);
        free(carry);
        fclose(f);

        if (found) {
            model_has_tensors = 1;
        } else {
            model_has_tensors = 0;
            printf("  SKIP: Vocab-only GGUF file - not a full model\n");
            return 1;
        }
    }
    
    return 1;
}

/* Test 3: Start daemon with auto-NGL (default) */
static int test_auto_ngl_default(void)
{
    if (!model_has_tensors) {
        TEST_SKIP("Model lacks tensors (vocab-only), skipping daemon-dependent test");
    }

    TEST_ASSERT(start_daemon("") == 0, "Daemon should start with auto-NGL");
    TEST_ASSERT(daemon_pid > 0, "Daemon should be running");

    return 1;
}

/* Test 4: Start daemon with explicit -g 0 (auto) */
static int test_auto_ngl_explicit(void)
{
    if (!model_has_tensors) {
        TEST_SKIP("Model lacks tensors (vocab-only), skipping daemon-dependent test");
    }

    stop_daemon();

    TEST_ASSERT(start_daemon("-g 0") == 0, "Daemon should start with -g 0");
    TEST_ASSERT(daemon_pid > 0, "Daemon should be running");

    return 1;
}

/* Test 5: Start daemon with manual -g */
static int test_manual_gpu_layers(void)
{
    if (!model_has_tensors) {
        TEST_SKIP("Model lacks tensors (vocab-only), skipping daemon-dependent test");
    }

    stop_daemon();

    TEST_ASSERT(start_daemon("-g 32") == 0, "Daemon should start with -g 32");
    TEST_ASSERT(daemon_pid > 0, "Daemon should be running");

    return 1;
}

/* Test 6: Test streaming */
static int test_streaming(void)
{
    if (!model_has_tensors) {
        TEST_SKIP("Model lacks tensors (vocab-only), skipping daemon-dependent test");
    }

    int sock;
    struct sockaddr_in addr;
    char send_buf[256];
    char recv_buf[4096];
    int n;
    
    /* Create socket */
    sock = socket(AF_INET, SOCK_STREAM, 0);
    TEST_ASSERT(sock >= 0, "Should create socket");
    
    /* Connect to daemon */
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(test_port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    
    TEST_ASSERT(connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0,
        "Should connect to daemon");
    
    /* Send streaming request */
    snprintf(send_buf, sizeof(send_buf),
        "messages {\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true}\n");
    
    n = write(sock, send_buf, strlen(send_buf));
    TEST_ASSERT(n > 0, "Should send request");
    
    /* Read response - should get SSE chunks */
    int got_chunk = 0;
    int timeout = 30;  /* 30 seconds timeout */
    while (timeout-- > 0) {
        n = read(sock, recv_buf, sizeof(recv_buf) - 1);
        if (n > 0) {
            recv_buf[n] = '\0';
            if (strstr(recv_buf, "\"type\":\"chunk\"")) {
                got_chunk = 1;
                break;
            }
            if (strstr(recv_buf, "\"type\":\"stop\"")) {
                break;
            }
        }
        usleep(100000);  /* 100ms */
    }
    
    close(sock);
    
    TEST_ASSERT(got_chunk, "Should receive streaming chunk");
    
    return 1;
}

/* Test 7: Test non-streaming (baseline) */
static int test_non_streaming(void)
{
    if (!model_has_tensors) {
        TEST_SKIP("Model lacks tensors (vocab-only), skipping daemon-dependent test");
    }

    int sock;
    struct sockaddr_in addr;
    char send_buf[256];
    char recv_buf[4096];
    int n;
    
    sock = socket(AF_INET, SOCK_STREAM, 0);
    TEST_ASSERT(sock >= 0, "Should create socket");
    
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(test_port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    
    TEST_ASSERT(connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0,
        "Should connect to daemon");
    
    /* Send non-streaming request */
    snprintf(send_buf, sizeof(send_buf),
        "messages {\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}\n");
    
    n = write(sock, send_buf, strlen(send_buf));
    TEST_ASSERT(n > 0, "Should send request");
    
    /* Read response - should get JSON */
    n = read(sock, recv_buf, sizeof(recv_buf) - 1);
    if (n > 0) {
        recv_buf[n] = '\0';
    }
    
    close(sock);
    
    TEST_ASSERT(n > 0, "Should receive response");
    
    return 1;
}

int main(int argc, char *argv[])
{
    printf("=== Auto-NGL Integration Tests ===\n\n");
    
    /* Find model */
    if (argc > 1) {
        strncpy(model_path, argv[1], sizeof(model_path) - 1);
    } else {
        find_test_model();
    }
    
    if (model_path[0] == '\0') {
        fprintf(stderr, "No GGUF model found for testing\n");
        fprintf(stderr, "Usage: %s [path_to_gguf_model]\n", argv[0]);
        return 1;
    }
    
    printf("Using model: %s\n\n", model_path);
    /* Allow forcing daemon-dependent tests when the user requests it. */
    const char *force = getenv("QLLM_FORCE_DAEMON_TESTS");
    if (force && strcmp(force, "1") == 0) {
        printf("  DEBUG: QLLM_FORCE_DAEMON_TESTS=1 - forcing daemon tests to run\n");
        model_has_tensors = 1;
    }
    
    /* Run tests */
    printf("Test 1: CLI help\n");
    test_cli_help();
    
    printf("\nTest 2: GGUF metadata\n");
    test_gguf_metadata();
    
    printf("\nTest 3: Auto-NGL (default)\n");
    test_auto_ngl_default();
    
    printf("\nTest 4: Auto-NGL explicit (-g 0)\n");
    test_auto_ngl_explicit();
    
    printf("\nTest 5: Manual GPU layers (-g 32)\n");
    test_manual_gpu_layers();
    
    printf("\nTest 6: Streaming\n");
    test_streaming();
    
    printf("\nTest 7: Non-streaming\n");
    test_non_streaming();
    
    /* Cleanup */
    stop_daemon();
    
    /* Summary */
    printf("\n=== Results ===\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    
    return tests_failed > 0 ? 1 : 0;
}

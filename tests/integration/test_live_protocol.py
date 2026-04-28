#!/usr/bin/env python3
"""
Live qllmd protocol integration tests.

These tests require a real GGUF model. They start repo-local bin/qllmd on a
test port and validate protocol shape for both the line-based TCP API and the
OpenAI-compatible HTTP API. Generated text is intentionally not compared
verbatim because live model output is nondeterministic.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


EOT = b"\x04"


class LiveProtocolTest:
    def __init__(self, root: Path, model: Path, port: int, timeout: float) -> None:
        self.root = root
        self.model = model
        self.port = port
        self.timeout = timeout
        self.proc: subprocess.Popen[bytes] | None = None
        self.log_path = Path(os.environ.get("QLLMD_LIVE_LOG", "/tmp/qllmd-live-protocol.log"))

    def start(self) -> None:
        qllmd = self.root / "bin" / "qllmd"
        if not qllmd.exists():
            raise AssertionError(f"missing qllmd binary: {qllmd}")

        env = os.environ.copy()
        lib_paths = [
            str(self.root / "lib"),
            str((self.root / ".." / "ndc" / "lib").resolve()),
            str((self.root / ".." / "ndx" / "lib").resolve()),
        ]
        if env.get("LD_LIBRARY_PATH"):
            lib_paths.append(env["LD_LIBRARY_PATH"])
        env["LD_LIBRARY_PATH"] = ":".join(lib_paths)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("wb")
        argv = [
            str(qllmd),
            "-d",
            "-p",
            str(self.port),
            "-c",
            os.environ.get("QLLMD_TEST_CTX", "512"),
            "-g",
            os.environ.get("QLLMD_TEST_NGL", "0"),
            str(self.model),
        ]
        self.proc = subprocess.Popen(argv, cwd=self.root, env=env, stdout=log, stderr=subprocess.STDOUT)
        self._wait_for_tcp()

    def stop(self) -> None:
        if not self.proc:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc = None

    def _wait_for_tcp(self) -> None:
        deadline = time.monotonic() + self.timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise AssertionError(f"qllmd exited early with status {self.proc.returncode}; see {self.log_path}")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                    return
            except OSError as exc:
                last_error = exc
                time.sleep(0.2)
        raise AssertionError(f"qllmd did not accept TCP connections: {last_error}; see {self.log_path}")

    def tcp_command(self, command: str, until_eot: bool = False) -> bytes:
        deadline = time.monotonic() + self.timeout
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            sock.settimeout(1)
            sock.sendall(command.encode("utf-8"))
            data = bytearray()
            while time.monotonic() < deadline:
                try:
                    chunk = sock.recv(8192)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                data.extend(chunk)
                if until_eot and EOT in data:
                    break
                if not until_eot and b"\n" in data:
                    break
            return bytes(data)

    def tcp_chat_messages(self, payload: str) -> bytes:
        deadline = time.monotonic() + self.timeout
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            sock.settimeout(1)
            sock.sendall(b"chat\n")
            time.sleep(0.2)
            sock.sendall(f"messages {payload}\n".encode("utf-8"))
            data = bytearray()
            while time.monotonic() < deadline:
                try:
                    chunk = sock.recv(8192)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                data.extend(chunk)
                if EOT in data:
                    break
            return bytes(data)

    def http_request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        payload = None if body is None else json.dumps(body).encode("utf-8")
        headers = dict(headers or {})
        if payload is not None:
            headers.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=payload, headers=headers)
        res = conn.getresponse()
        data = res.read()
        out_headers = {k.lower(): v for k, v in res.getheaders()}
        conn.close()
        return res.status, out_headers, data

    def http_raw(
        self,
        method: str,
        path: str,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        conn.request(method, path, body=body, headers=headers or {})
        res = conn.getresponse()
        data = res.read()
        out_headers = {k.lower(): v for k, v in res.getheaders()}
        conn.close()
        return res.status, out_headers, data

    def http_stream(self, body: dict, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        req_headers = {"Content-Type": "application/json"}
        if headers:
            req_headers.update(headers)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(body).encode("utf-8"),
            headers=req_headers,
        )
        res = conn.getresponse()
        chunks = bytearray()
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline and b"data: [DONE]" not in chunks:
            chunk = res.read(512)
            if not chunk:
                break
            chunks.extend(chunk)
        headers = {k.lower(): v for k, v in res.getheaders()}
        status = res.status
        conn.close()
        return status, headers, bytes(chunks)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def decode_json(data: bytes) -> object:
    data = data.replace(EOT, b"").strip()
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise AssertionError(f"invalid JSON response: {data[:500]!r}") from exc


def opencode_tools() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file from the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative file path"}
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Apply a small patch to a workspace file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old": {"type": "string"},
                        "new": {"type": "string"},
                    },
                    "required": ["path", "old", "new"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def opencode_chat_body(stream: bool) -> dict[str, object]:
    return {
        "model": "qwen2.5-coder",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are OpenCode running against libqllm. Answer briefly, "
                    "and use tools only when necessary."
                ),
            },
            {
                "role": "tool",
                "tool_call_id": "call_live_protocol_0",
                "content": "read_file returned: libqllm live protocol test fixture",
            },
            {"role": "user", "content": "Reply with exactly one short sentence."},
        ],
        "tools": opencode_tools(),
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "temperature": 0.2,
        "max_tokens": 64,
        "stream": stream,
    }


def sse_payloads(stream: bytes) -> list[object]:
    payloads = []
    for raw_line in stream.splitlines():
        line = raw_line.strip()
        if not line.startswith(b"data: "):
            continue
        payload = line[len(b"data: ") :]
        if payload == b"[DONE]":
            payloads.append("[DONE]")
        else:
            payloads.append(decode_json(payload))
    return payloads


def run_tests(t: LiveProtocolTest) -> None:
    info = t.tcp_command("info\n")
    info_obj = decode_json(info.strip())
    assert_true(isinstance(info_obj, dict), "tcp info returns a JSON object")
    assert_true("model" in info_obj, "tcp info includes model")
    assert_true("template" in info_obj, "tcp info includes template")

    invalid = t.tcp_command("messages not-json\n")
    invalid_obj = decode_json(invalid.strip())
    assert_true(isinstance(invalid_obj, dict), "invalid messages response is JSON")
    assert_true("error" in invalid_obj, "invalid messages response includes error")

    message = json.dumps({"messages": [{"role": "user", "content": "Reply with one short word."}]})
    raw = t.tcp_chat_messages(message)
    assert_true(EOT in raw, "tcp messages response ends with EOT delimiter")
    assert_true(len(raw.split(EOT, 1)[0].strip()) > 0, "tcp messages response has generated content")

    auth_headers = {
        "Authorization": "Bearer dummy",
        "User-Agent": "opencode-live-protocol-test",
    }

    status, headers, body = t.http_request("GET", "/health", headers=auth_headers)
    assert_true(status == 200, f"health status is 200, got {status}")
    assert_true(headers.get("content-type", "").startswith("application/json"), "health content-type is JSON")
    assert_true(decode_json(body) == {"status": "ok"}, "health body is ok")

    status, headers, body = t.http_request("GET", "/v1/models", headers=auth_headers)
    models = decode_json(body)
    assert_true(status == 200, f"models status is 200, got {status}")
    assert_true(isinstance(models, dict), "models response is JSON object")
    assert_true(models.get("object") == "list", "models object is list")
    assert_true(isinstance(models.get("data"), list) and len(models["data"]) >= 1, "models data has at least one entry")

    status, headers, body = t.http_raw(
        "POST",
        "/v1/chat/completions",
        b"{not-json",
        {"Content-Type": "application/json", **auth_headers},
    )
    bad_json = decode_json(body)
    assert_true(status == 400, f"malformed JSON status is 400, got {status}")
    assert_true(isinstance(bad_json, dict) and "error" in bad_json, "malformed JSON returns OpenAI-style error")

    status, headers, body = t.http_request(
        "POST",
        "/v1/chat/completions",
        {"model": "qwen2.5-coder", "stream": False},
        headers=auth_headers,
    )
    missing_messages = decode_json(body)
    assert_true(status == 400, f"missing messages status is 400, got {status}")
    assert_true(isinstance(missing_messages, dict) and "error" in missing_messages, "missing messages returns error")

    chat_body = opencode_chat_body(stream=False)
    status, headers, body = t.http_request("POST", "/v1/chat/completions", chat_body, headers=auth_headers)
    chat = decode_json(body)
    assert_true(status == 200, f"chat status is 200, got {status}: {body[:300]!r}")
    assert_true(isinstance(chat.get("id"), str) and chat["id"].startswith("chatcmpl-"), "chat response has completion id")
    assert_true(chat.get("object") == "chat.completion", "chat response object is chat.completion")
    assert_true(chat.get("model") == "qwen2.5-coder", "chat response echoes requested model")
    assert_true(isinstance(chat.get("choices"), list) and len(chat["choices"]) == 1, "chat response has one choice")
    content = chat["choices"][0].get("message", {}).get("content")
    assert_true(isinstance(content, str), "chat choice has message content")
    assert_true(chat["choices"][0].get("finish_reason") == "stop", "chat finish_reason is stop")
    assert_true(isinstance(chat.get("usage"), dict), "chat response includes usage object")

    status, headers, body = t.http_request(
        "POST",
        "/v1/completions",
        {
            "model": "qwen2.5-coder",
            "prompt": "Complete this identifier: live_protocol_",
            "suffix": " = 1",
            "max_tokens": 16,
            "stop": ["\n"],
        },
        headers=auth_headers,
    )
    completion = decode_json(body)
    assert_true(status == 200, f"completion status is 200, got {status}: {body[:300]!r}")
    assert_true(completion.get("object") == "text_completion", "completion object is text_completion")
    assert_true(isinstance(completion.get("choices"), list) and len(completion["choices"]) == 1, "completion has one choice")
    assert_true(isinstance(completion["choices"][0].get("text"), str), "completion choice has text")
    assert_true(isinstance(completion.get("usage"), dict), "completion response includes usage object")

    stream_body = opencode_chat_body(stream=True)
    stream_body["stream_options"] = {"include_usage": True}
    status, headers, stream = t.http_stream(stream_body, headers=auth_headers)
    assert_true(status == 200, f"stream status is 200, got {status}: {stream[:300]!r}")
    assert_true(headers.get("content-type", "").startswith("text/event-stream"), "stream content-type is SSE")
    assert_true(b"data: " in stream, "stream contains SSE data records")
    assert_true(b"chat.completion.chunk" in stream, "stream contains OpenAI chunk objects")
    assert_true(b"data: [DONE]" in stream, "stream terminates with [DONE]")
    payloads = sse_payloads(stream)
    chunks = [p for p in payloads if isinstance(p, dict)]
    choice_chunks = [
        p for p in chunks
        if isinstance(p.get("choices"), list) and p["choices"]
    ]
    assert_true(len(choice_chunks) >= 2, "stream has at least role and final chunks")
    assert_true(choice_chunks[0]["choices"][0]["delta"].get("role") == "assistant", "first stream chunk declares assistant role")
    tool_chunks = [
        p for p in choice_chunks
        if p["choices"][0].get("delta", {}).get("tool_calls")
    ]
    for chunk in tool_chunks:
        tool_call = chunk["choices"][0]["delta"]["tool_calls"][0]
        assert_true(tool_call.get("type") == "function", "stream tool call type is function")
        assert_true(isinstance(tool_call.get("id"), str), "stream tool call has id")
        assert_true(isinstance(tool_call.get("function", {}).get("name"), str), "stream tool call has function name")
        assert_true(isinstance(tool_call.get("function", {}).get("arguments"), str), "stream tool call has arguments string")
    final_reason = choice_chunks[-1]["choices"][0].get("finish_reason")
    assert_true(final_reason in ("stop", "tool_calls"), "final stream chunk has valid finish_reason")
    if tool_chunks:
        assert_true(final_reason == "tool_calls", "stream tool-call response finishes with tool_calls")


def usable_model(path: Path) -> bool:
    try:
        return (
            path.is_file()
            and path.stat().st_size > 1024 * 1024
            and "ggml-vocab-" not in path.name
        )
    except OSError:
        return False


def qllm_path(root: Path, pattern: str) -> Path | None:
    helper = root / "bin" / "qllm-path"
    if not helper.exists():
        return None
    patterns = [pattern]
    if not any(ch in pattern for ch in "*?[]"):
        patterns.append(f"*{pattern}*")
    for candidate in patterns:
        try:
            proc = subprocess.run(
                [str(helper), candidate],
                cwd=root,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0:
            path = Path(proc.stdout.strip()).expanduser()
            if usable_model(path):
                return path
    return None


def qllm_list_models(root: Path, model_filter: str | None = None) -> Iterable[Path]:
    helper = root / "bin" / "qllm-list"
    if not helper.exists():
        return []
    argv = [str(helper)]
    if model_filter:
        argv.append(model_filter)
    try:
        proc = subprocess.run(
            argv,
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    models = []
    for line in proc.stdout.splitlines():
        name = line.split(None, 1)[0] if line.split(None, 1) else ""
        if not name:
            continue
        resolved = qllm_path(root, name)
        if resolved:
            models.append(resolved)
    return models


def resolve_model(root: Path, value: str) -> Path | None:
    path = Path(value).expanduser()
    if usable_model(path):
        return path
    if not path.is_absolute():
        repo_path = root / path
        if usable_model(repo_path):
            return repo_path
    return qllm_path(root, value)


def filesystem_models(root: Path) -> Iterable[Path]:
    for base in [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / "models",
        Path.home() / "llm" / "llama.cpp" / "models",
        root,
    ]:
        if not base.exists():
            continue
        yield from base.rglob("*.gguf")


def default_model(root: Path) -> Path | None:
    env_model = os.environ.get("MODEL") or os.environ.get("QLLM_TEST_MODEL")
    if env_model:
        resolved = resolve_model(root, env_model)
        if resolved:
            return resolved

    model_filter = os.environ.get("QLLM_TEST_MODEL_FILTER")
    for path in qllm_list_models(root, model_filter):
        if usable_model(path):
            return path

    candidates = []
    for base in [
        "Qwen",
        "Phi-3",
        "Mistral",
        "Gemma",
    ]:
        resolved = qllm_path(root, f"*{base}*.gguf")
        if resolved:
            candidates.append(resolved)

    candidates.extend(filesystem_models(root))
    for path in candidates:
        if usable_model(path):
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live qllmd protocol integration tests")
    parser.add_argument("model", nargs="?", help="path to a real GGUF model")
    parser.add_argument("--port", type=int, default=int(os.environ.get("QLLMD_TEST_PORT", "54243")))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("QLLMD_TEST_TIMEOUT", "180")))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    model = resolve_model(root, args.model) if args.model else default_model(root)
    if not model:
        print(
            "No real GGUF model found. Pass a model path, set MODEL=/path/to/model.gguf, "
            "or set MODEL to a qllm-path pattern.",
            file=sys.stderr,
        )
        return 2
    if not usable_model(model):
        print(f"Model does not exist: {model}", file=sys.stderr)
        return 2

    test = LiveProtocolTest(root, model.resolve(), args.port, args.timeout)
    try:
        print(f"[live-protocol] model: {model}")
        print(f"[live-protocol] port: {args.port}")
        test.start()
        run_tests(test)
        print("[live-protocol] all protocol checks passed")
        return 0
    except AssertionError as exc:
        print(f"[live-protocol] FAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        test.stop()


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    sys.exit(main())

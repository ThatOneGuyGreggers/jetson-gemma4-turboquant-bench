"""Near-capacity prompt and vision benchmark for the Compose stack."""

import base64
import json
import os
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path


ROOT = Path(os.environ.get("BENCH_ROOT", Path(__file__).resolve().parent))
BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:8082")
RESULTS = ROOT / os.environ.get("BENCH_RESULTS", "results/runs/benchmark.jsonl")
CONTEXTS = tuple(int(n) for n in os.environ.get("BENCH_CONTEXTS", "65536,98304,131072").split(","))
MODEL = os.environ.get("BENCH_MODEL", "gemma-4-26b-turboquant")
COMPOSE_OPTIONS = ["--env-file", os.environ["BENCH_COMPOSE_ENV_FILE"]] if os.environ.get("BENCH_COMPOSE_ENV_FILE") else []
CHUNK = " Research notes compare battery chemistry, cost, safety, and charging speed."


def request(path, data=None, timeout=60):
    payload = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(
        BASE + path,
        data=payload,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def token_count(text):
    return len(request("/tokenize", {"content": text, "add_special": False}, 180)["tokens"])


def test_text(ctx):
    # Reserve room for chat-template overhead and the 128 output tokens.
    desired = ctx - 512
    tokens_per_chunk = token_count(CHUNK)
    n = max(1, desired // tokens_per_chunk)
    content = CHUNK * n
    actual = token_count(content)
    if actual > desired:
        n = max(1, n * desired // actual - 1)
        content = CHUNK * n
        actual = token_count(content)

    started = time.monotonic()
    response = request("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": content + "\nSummarize in one sentence."}],
        "max_tokens": 128,
        "temperature": 0,
        "stream": False,
    }, 3600)
    timing = response.get("timings", {})
    usage = response.get("usage", {})
    prompt_n = timing.get("prompt_n") or usage.get("prompt_tokens")
    predicted_n = timing.get("predicted_n") or usage.get("completion_tokens")
    prompt_ms = timing.get("prompt_ms")
    predicted_ms = timing.get("predicted_ms")
    return {
        "tokenizer_prompt_tokens": actual,
        "usage": usage,
        "timings": timing,
        "wall_seconds": round(time.monotonic() - started, 2),
        "prefill_tokens_per_second": round(prompt_n * 1000 / prompt_ms, 2) if prompt_n and prompt_ms else None,
        "decode_tokens_per_second": round(predicted_n * 1000 / predicted_ms, 2) if predicted_n and predicted_ms else None,
        "answer": response["choices"][0]["message"].get("content", "")[:300],
    }


def png_red_square():
    def block(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    width = height = 64
    pixels = b"".join(b"\0" + bytes((255, 0, 0)) * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + block(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + block(b"IDAT", zlib.compress(pixels)) + block(b"IEND", b""))


def test_vision():
    image = base64.b64encode(png_red_square()).decode()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}},
            {"type": "text", "text": "What is the dominant color in the image? One word."},
        ]}],
        "max_tokens": 320,
        "temperature": 0,
        "stream": False,
    }
    if os.environ.get("BENCH_VISION_REASONING_EFFORT"):
        payload["reasoning_effort"] = os.environ["BENCH_VISION_REASONING_EFFORT"]
    response = request("/v1/chat/completions", payload, 180)
    return response["choices"][0]["message"].get("content", "")[:300]


def record(result):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result), flush=True)


def main():
    print(os.environ.get("BENCH_CONFIG_LABEL", f"Model {MODEL}: Q8_0 K / Turbo4 V, Q4_0 MTP, F16 vision"), flush=True)
    for ctx in CONTEXTS:
        env = {**os.environ, "CTX_SIZE": str(ctx)}
        started = time.monotonic()
        print(f"Starting {ctx} context...", flush=True)
        up = subprocess.run(
            ["docker", "compose", *COMPOSE_OPTIONS, "up", "-d", "--no-build", "--force-recreate"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
        )
        result = {"context": ctx, "startup_seconds": None}
        if up.returncode:
            result["error"] = "compose up: " + (up.stderr or up.stdout)[-1500:]
            record(result)
            continue
        while time.monotonic() - started < 420:
            try:
                if request("/health", timeout=5).get("status") == "ok":
                    break
            except (OSError, ValueError):
                pass
            state = subprocess.run(
                ["docker", "compose", *COMPOSE_OPTIONS, "ps", "--status", "exited", "-q"],
                cwd=ROOT, text=True, capture_output=True,
            )
            if state.stdout.strip():
                break
            time.sleep(5)
        result["startup_seconds"] = round(time.monotonic() - started, 2)
        try:
            if request("/health", timeout=5).get("status") != "ok":
                raise RuntimeError("server did not become ready")
            models = request("/v1/models")
            result["model"] = models.get("data", [{}])[0].get("id")
            if ctx == CONTEXTS[0]:
                try:
                    result["vision_answer"] = test_vision()
                except Exception as exc:
                    result["vision_error"] = str(exc)
            result.update(test_text(ctx))
        except Exception as exc:
            result["error"] = str(exc)
            logs = subprocess.run(
                ["docker", "compose", *COMPOSE_OPTIONS, "logs", "--no-color", "--since", "10m"],
                cwd=ROOT, text=True, capture_output=True,
            )
            result["log_excerpt"] = logs.stdout[-2500:]
        record(result)
    print("Results:", RESULTS, flush=True)


if __name__ == "__main__":
    main()

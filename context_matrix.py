"""Measure short-generation health at several allocated llama.cpp context sizes."""

import argparse
import datetime as dt
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CONTEXTS = (32768, 65536, 131072, 262144)


def http(port, route, payload=None, timeout=10):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{route}", data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def compose(model, args, env, timeout=90):
    stack = ROOT / model["stack"] if model["stack"] == "runtime" else Path(model["stack"])
    return subprocess.run(
        ["docker", "compose", *args], cwd=stack, env=env,
        capture_output=True, text=True, timeout=timeout,
    )


def record(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row, sort_keys=True) + "\n")
    print(f'{row["model"]} {row["context"]}: {row["status"]}', flush=True)


def run(model, context, output):
    row = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": model["id"], "context": context, "format": model["format"],
        "method": "configured_context_short_chat", "status": "pending",
    }
    weights = Path(model["weights"])
    if not weights.is_file():
        row.update(status="missing_weights", error=str(weights))
        return record(output, row)
    row["weight_bytes"] = weights.stat().st_size
    if model.get("task_type"):
        row.update(status="not_comparable_task", task_type=model["task_type"],
                   installed_context=model.get("max_native_context"))
        return record(output, row)
    if context > model.get("max_native_context", float("inf")):
        row["status"] = "unsupported_native_context"
        return record(output, row)

    env = {**os.environ, **model.get("env", {}), "CTX_SIZE": str(context)}
    running = compose(model, ["ps", "--status", "running", "-q"], env)
    if running.returncode or running.stdout.strip():
        row.update(status="busy", error=(running.stderr or "stack already running")[-800:])
        return record(output, row)

    started = time.monotonic()
    try:
        up = compose(model, ["up", "-d", "--no-build", "--force-recreate"], env, 120)
        if up.returncode:
            raise RuntimeError("compose up: " + (up.stderr or up.stdout)[-1000:])
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            try:
                if http(model["port"], "/health", timeout=5).get("status") == "ok":
                    break
            except (OSError, ValueError):
                pass
            exited = compose(model, ["ps", "--status", "exited", "-q"], env)
            if exited.stdout.strip():
                raise RuntimeError("server exited during load")
            time.sleep(5)
        else:
            raise TimeoutError("server startup exceeded 300 seconds")

        row["startup_seconds"] = round(time.monotonic() - started, 2)
        props = http(model["port"], "/props")
        row["reported_context"] = props["default_generation_settings"]["n_ctx"]
        row["runtime"] = props.get("build_info")
        row["served_model"] = props.get("model_alias")
        if row["reported_context"] != context:
            row["status"] = "context_mismatch"
        else:
            request_started = time.monotonic()
            response = http(model["port"], "/v1/chat/completions", {
                "model": row["served_model"],
                "messages": [{"role": "user", "content": "Reply with one word: hello"}],
                "max_tokens": 48, "temperature": 0, "stream": False,
            }, 180)
            row["wall_seconds"] = round(time.monotonic() - request_started, 2)
            row["usage"] = response.get("usage", {})
            row["timings"] = response.get("timings", {})
            row["answer"] = (response["choices"][0]["message"].get("content") or "")[:120]
            row["finish_reason"] = response["choices"][0].get("finish_reason")
            row["status"] = "ok"
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        row.update(status="failed", error=str(exc)[-1000:])
        try:
            logs = compose(model, ["logs", "--no-color", "--since", "10m"], env, 20)
            row["log_excerpt"] = logs.stdout[-1800:]
        except (OSError, subprocess.TimeoutExpired):
            pass
    finally:
        compose(model, ["stop", "-t", "15"], env, 60)
    record(output, row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="model IDs from models.json (default: all)")
    parser.add_argument("--contexts", type=int, nargs="+", default=DEFAULT_CONTEXTS)
    parser.add_argument("--output", type=Path, default=ROOT / "results/context-matrix-jetson-orin.jsonl")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "models.json").read_text())
    by_id = {model["id"]: model for model in catalog}
    unknown = set(args.models or []) - by_id.keys()
    if unknown:
        parser.error("unknown model IDs: " + ", ".join(sorted(unknown)))
    selected = [by_id[name] for name in args.models] if args.models else catalog
    completed = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                previous = json.loads(line)
                completed.add((previous["model"], previous["context"]))
    for model in selected:
        for context in args.contexts:
            if (model["id"], context) not in completed:
                run(model, context, args.output)


if __name__ == "__main__":
    main()

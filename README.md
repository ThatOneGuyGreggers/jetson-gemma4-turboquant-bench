# Jetson model test results

Reproducible tests of locally hosted AI models on a Jetson AGX Orin (32 GB shared RAM, JetPack 7.2.1, CUDA 13.2). The original near-full-context Gemma 4 / Qwen3.8 / Bonsai 2 measurements and setup remain in [LEGACY.md](LEGACY.md) and `results/*.jsonl`.

## Context-window test matrix

`context_matrix.py` tests configured server contexts of **32,768 / 65,536 / 131,072 / 262,144 tokens** sequentially, with one model server running at a time. Each attempt records whether the server loaded, its *reported* slot context, and a short chat completion with timings. **A short chat is not proof that the model processed a prompt that fills the context.** Earlier full-length measurements use the separate `benchmark.py`. A failed load, insufficient memory, unsupported native context, or timeout is recorded as such; none is silently treated as a successful result.

The three new Q8 checkpoints have their own Dockge stacks and read-only model mounts:

| Stack | Checkpoint | Size | Endpoint |
| --- | --- | ---: | --- |
| `/opt/stacks/ornith-15-9b-q8` | [Ornith-1.5-9B Q8_0](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF) | 9.79 GB | `127.0.0.1:8091` |
| `/opt/stacks/spark-x25-4b-q8` | [Spark-X2.5-4B Q8_0](https://huggingface.co/XHToken/Spark-X2.5-4B-GGUF) | 4.38 GB | `127.0.0.1:8092` |
| `/opt/stacks/gpt-oss-20b-q8` | [gpt-oss-20b Q8_0](https://huggingface.co/unsloth/gpt-oss-20b-GGUF) | 12.11 GB | `127.0.0.1:8093` |

Copies of the installed Compose configurations are in [`stacks/`](stacks/). Place each stack's `compose.yaml` in its own Dockge directory under `/opt/stacks` with its GGUF in that directory's `models/` folder. The three downloads were verified against their Hugging Face LFS SHA-256 values: Ornith `22086870b009dbe9815ee752c48a82de930118a7c5ce5599590892ae03b8b010`, Spark `5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2`, GPT-OSS `bcd455d4034ec02f71a875b46cb17df44a97911d7258291973be4d21f98329f3`.

GPT-OSS uses native MXFP4 MoE weights even in the Q8_0 GGUF; its *non-MoE* tensors use Q8_0. Its native context limit is 131,072; the 262,144 row is recorded as unsupported. Ornith's native limit is 262,144 and Spark's is 1M. The new stacks share a single SM 87 CUDA image built from llama.cpp commit [`fc07d781`](https://github.com/ggml-org/llama.cpp/commit/fc07d781e61f0d23764394e902b88d26a974e202); Spark requires a newer llama.cpp than the older image on this host. Each stack can be started separately with `docker compose up -d --no-build` in its directory (build the Ornith stack image first if missing).

The manifest `models.json` lists additional existing Docker stacks and the text-generation models tested by the matrix. SmolVLM2 is a vision-language service and EmbeddingGemma is an embedding service; their current deployments have different task types and native contexts, so they are listed separately rather than represented as comparable long-context text runs. ComfyUI, Speaches, and Open Notebook are application stacks rather than comparable chat model checkpoints.

### Observed results (short chat after allocating context)

| Model | 32k | 64k | 128k | 256k |
| --- | --- | --- | --- | --- |
| Spark-X2.5-4B Q8_0 | OK | OK | OK | OK |
| Ornith-1.5-9B Q8_0 | OK | OK | OK | OK |
| GPT-OSS-20B Q8_0 / MXFP4 | OK | OK | OK | native limit 128k |
| Gemma 4 12B Q4_K_XL + MTP | OK | OK | OK | OK |
| Gemma 4 26B A4B Q4_K_XL + MTP | OK | OK | OK | OK |
| Qwen3.8 27B IQ3_S + MTP | OK | OK | OK | OK |
| Bonsai 2 27B PQ2_0 | OK | OK | OK | OK |
| SmolLM3 3B Q4_0 | OK | OK | clamped to 64k | clamped to 64k |
| Granite 4.2 3B Q4_0 | OK | OK | OK | clamped to 128k |
| SmolVLM2 / EmbeddingGemma | different tasks | different tasks | different tasks | different tasks |

`OK` means the server reported the requested slot context and answered a short prompt (roughly 16-254 prompt tokens), **not** a near-full-context validation or proof of recall at that position. Raw measurements, exact token counts, timestamps and timings are in [`results/context-matrix-jetson-orin.jsonl`](results/context-matrix-jetson-orin.jsonl). The earlier near-full-context results in `results/*.jsonl` use different prompts and settings and should not be compared directly with these short-chat timings.

Run from this checkout:

```bash
python3 -u context_matrix.py --models spark-x2.5-4b-q8_0 ornith-1.5-9b-q8_0 gpt-oss-20b-q8_0 \
  --output results/context-matrix-jetson-orin.jsonl
python3 -u context_matrix.py --models gemma-4-12b gemma-4-26b qwen3.8-27b bonsai2-27b smollm3-3b granite-4.2-3b \
  --output results/context-matrix-jetson-orin.jsonl
```

The runner will not take over a stack that is already running. It resumes by skipping `(model, context)` pairs already in the output, and stops each container after its test. `--contexts` restricts the context sizes. The output is append-only JSONL; review `status`, `reported_context`, and `usage.prompt_tokens` before quoting throughput. Only successful generations have decode throughput. `results/context-matrix-jetson-orin.jsonl` is the recorded run, including failures; it is not a simulated table.

Download/reproduce the new GGUF files using `hf download REPO FILE --revision REVISION --local-dir STACK/models` with the exact repo, filename and revision recorded in `models.json`. Model files are excluded from Git. The new image's build recipe is in [`runtime/Dockerfile`](runtime/Dockerfile), and its generic runner stack for the smaller existing models is [`runtime/compose.yaml`](runtime/compose.yaml).

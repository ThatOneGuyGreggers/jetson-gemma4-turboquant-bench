# Gemma 4 TurboQuant on Jetson AGX Orin

Reproducible Docker Compose configuration and measured long-context results for
Gemma 4 12B and 26B A4B QAT GGUF on a Jetson AGX Orin Developer Kit.
Both runs used [TheTom's llama.cpp TurboQuant fork](https://github.com/TheTom/llama-cpp-turboquant)
with **Q8_0 keys / Turbo4 values**, the models' **Q4_0 MTP drafters**, Flash
Attention, and their **F16 vision projectors**. The model weights remain
Unsloth's Q4_K_XL GGUF; Turbo4 applies only to the runtime V cache.

## Test system

| Component | Tested configuration |
| --- | --- |
| Board | NVIDIA Jetson AGX Orin Developer Kit, 32 GB nominal shared memory |
| GPU | Orin (nvgpu), CUDA compute capability 8.7 / SM 87 |
| CPU | 8 x ARM Cortex-A78AE, up to 2.19 GHz |
| OS | Ubuntu 24.04.5 LTS, aarch64; Jetson Linux R39.2.1 |
| Kernel | 6.8.12-1021-tegra |
| CUDA toolkit | 13.2 (nvcc 13.2.86); build/runtime container bases: `nvidia/cuda:13.2.0-*` |
| Docker / Compose | Docker Engine 29.8.0; Compose v5.5.1 |
| TurboQuant source | `TheTom/llama-cpp-turboquant` commit [`a3d5603d110bda29222d2011596cdc84d7fa532d`](https://github.com/TheTom/llama-cpp-turboquant/commit/a3d5603d110bda29222d2011596cdc84d7fa532d) |

For the measurements, other llama.cpp services, Open Notebook, Speaches, and
ComfyUI were stopped to make memory available. Each context used one slot,
all GPU layers, `--fit off`, `--batch-size 1024`, `--ubatch-size 256`,
`--spec-draft-n-max 3`, and eight CPU threads. The F16 projector was loaded
for every run; a red 64 x 64 image was recognized as **Red** for both models.

## Results

The prompt is repeated synthetic research text, tokenized to nearly fill the
configured context, followed by a request to summarize it. The server
reported the following measurements for 128 generated tokens:

| Context | Actual prompt | 12B prefill | 12B decode | 12B MTP accepted | 26B A4B prefill | 26B A4B decode | 26B A4B MTP accepted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 65,039 | 247.41 tok/s | 18.16 tok/s | 91/105 | 356.02 tok/s | 22.57 tok/s | 88/116 |
| 98,304 | 97,813 | 209.70 tok/s | 15.55 tok/s | 91/107 | 293.58 tok/s | 17.59 tok/s | 86/118 |
| 131,072 | 130,573 | 186.08 tok/s | 16.19 tok/s | 91/107 | 255.99 tok/s | 15.15 tok/s | 88/116 |

The 26B A4B model prefilled faster in these tests despite having more total
parameters; it is a mixture-of-experts model. This is an observation from
one run per size, **not** a measured TurboQuant speedup: there is no Q8_0-V
or MTP-off control run. Cold/warm startup times also differ and should not be
compared as model speed. At 131K, the 12B container showed about 6.8 GiB in
`docker stats`; container usage is not a precise measure of GPU KV memory.

**256K (26B A4B):** The server loaded at `--ctx-size 262144` and processed
194,568 prompt tokens (74%). It was stopped at the user's request before the
prompt finished. The last progress log showed a cumulative 205.57 prefill
tok/s; no completed 256K prefill or decode result exists. The 12B 256K
prompt was not run.

The 128 generated tokens at each measured context were spent on model
reasoning, leaving the final-answer fields empty. These figures measure
throughput, **not answer quality**. Full API timings and draft counts are
preserved in [`results/12b.jsonl`](results/12b.jsonl) and
[`results/26b.jsonl`](results/26b.jsonl).

## Download the models

The GGUF files are excluded from Git. Use the exact Unsloth revisions from
these tests so target, MTP drafter, and projector stay paired:

```bash
python3 -m venv .venv
.venv/bin/pip install 'huggingface_hub[hf_xet]'

.venv/bin/hf download unsloth/gemma-4-12B-it-qat-GGUF \
  gemma-4-12B-it-qat-UD-Q4_K_XL.gguf mtp-gemma-4-12B-it.gguf mmproj-F16.gguf \
  --revision 980b060c40a8539ac159e0501a3e0f66a6365af3 --local-dir models/12b

.venv/bin/hf download unsloth/gemma-4-26B-A4B-it-qat-GGUF \
  gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf mtp-gemma-4-26B-A4B-it.gguf mmproj-F16.gguf \
  --revision 7b92b5b28818151e8669af2e45e88d6086f490dd --local-dir models/26b
```

Expected SHA-256 hashes, in download order:

| File | SHA-256 |
| --- | --- |
| 12B target | `90fd44e29e0d7cffeb0fd00dc73cfdab9ed0b0e95306ecf7821ea634c940c370` |
| 12B MTP Q4_0 | `fcb35dea42c71333db904cee11baac525c9ef872818ee3753f6cb156f3c6f4f6` |
| 12B F16 projector | `ecc4e93128da8363b7dbf2193eab98cf1142353f52ceaa0c95c0872997aaadd3` |
| 26B A4B target | `a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891` |
| 26B A4B MTP Q4_0 | `7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a` |
| 26B A4B F16 projector | `d00f211a7d4f7fb19bd9b75d8e9342eccffb5920b08fd9562167560fdfcc5dd1` |

The 12B and 26B projectors have the same filename upstream but are **not
interchangeable**. Keep each in its own directory, as above.

## Build and run

Requirements: ARM64 Jetson with CUDA 13.2, NVIDIA Container Toolkit, Docker
Compose, and enough free shared memory for the selected context. The image
build compiles many CUDA kernels for SM 87 and can take a while; subsequent
context changes only recreate the container. This Compose stack binds its API
to `127.0.0.1:8082` on the host. For Dockge, place the repo in your Dockge
stacks directory and select the desired env file (or set those model variables
in Dockge).

```bash
docker compose --env-file 12b.env build
docker compose --env-file 12b.env up -d --no-build
curl http://127.0.0.1:8082/health
curl http://127.0.0.1:8082/v1/models

# Switch to the 26B A4B, with the same built image:
docker compose --env-file 26b.env up -d --no-build --force-recreate

# Preserve the container and model files when finished:
docker compose --env-file 26b.env stop
```

For a near-capacity 12B run at the three completed sizes (Python standard
library only), use:

```bash
BENCH_MODEL=gemma-4-12b-turboquant \
BENCH_COMPOSE_ENV_FILE=12b.env \
BENCH_CONTEXTS=65536,98304,131072 \
BENCH_RESULTS=results/runs/12b.jsonl \
python3 -u benchmark.py
```

Use `26b.env`, `gemma-4-26b-turboquant`, and a separate output path for 26B.
Add `262144` to `BENCH_CONTEXTS` only if you want a long 256K prompt run.
`benchmark.py` recreates the service for each context; stop competing GPU
workloads before measuring. It **appends** results to the specified JSONL, so
use a fresh output path for each experiment.

## Credits

This repo packages a configuration and original test measurements. The
inference implementation is in [TheTom's TurboQuant fork](https://github.com/TheTom/llama-cpp-turboquant)
(MIT); model files come from [Unsloth 12B](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF)
and [Unsloth 26B A4B](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF)
(Gemma 4 model terms as documented by the model authors). This repository
does not redistribute model files or the upstream llama.cpp source tree.

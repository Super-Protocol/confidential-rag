# Modifications

This repository is Super Protocol's fork of
[NVIDIA-AI-Blueprints/rag](https://github.com/NVIDIA-AI-Blueprints/rag), Apache-2.0,
**forked at release `v2.6.2`** (commit `f20716d73ae69a544ad4a692f38d6178a64e6f36`).

NVIDIA archived the upstream repository on 2026-11-05. It does not move any more, so there is no
upstream to track: **this fork is maintained by Super Protocol** and the branch the images are
built from is `confidential/2.6.2`.

NVIDIA is a trademark of NVIDIA Corporation. This fork is not produced, endorsed or supported by
NVIDIA.

## Why the fork exists

The reference architecture is open, but three of the components it is wired to are not: the
NeMo Retriever embedding and reranking NIMs, and the generation NIM. A NIM needs an NVIDIA AI
Enterprise entitlement for production use and a per-tenant NGC credential to pull, and it is an
opaque container in the middle of a system whose entire claim is that a tenant can verify what ran.

So the substitutions below are the fork's whole reason to exist. They are additive: every upstream
engine is still selectable, and `APP_*_MODELENGINE` chooses.

## What changed

### Retrieval and generation now have an open path

| File | Change |
|---|---|
| `src/nvidia_rag/utils/embedding.py` | New `openai` embedding engine. Sends a plain OpenAI embeddings request — no `input_type`, no `truncate`, no `dimensions`, and no client-side tokenisation — so an open server (Text Embeddings Inference) can serve the retriever. |
| `src/nvidia_rag/utils/tei_reranker.py` | **New file.** `TEIRerank`, a `BaseDocumentCompressor` speaking TEI's `POST /rerank` (`{query, texts}` → `[{index, score}]`) instead of NeMo Retriever's `POST /v1/ranking`. Writes `relevance_score` where the rest of the pipeline reads it, and falls back to the retriever's own order if the endpoint is lost. |
| `src/nvidia_rag/utils/reranker.py` | New `tei` ranking engine, wired to the above. |
| `src/nvidia_rag/utils/llm.py` | New `openai` LLM engine: a plain `ChatOpenAI` against an OpenAI-compatible endpoint (vLLM). Kept separate from the `nvidia-ai-endpoints` branch rather than folded into it — that branch probes the endpoint, carries NVIDIA's default headers and picks generation parameters from a model-name table, which is exactly the dependency this fork removes. |
| `src/nvidia_rag/rag_server/health.py` | Dependency probes ask each engine for the readiness path it serves. `/v1/health/ready` is a NIM route; vLLM and TEI answer `/health`, so before this an entirely healthy open stack reported every component unhealthy. |

### The image

| File | Change |
|---|---|
| `src/nvidia_rag/rag_server/Dockerfile` | Base image is the official `python:3.13-slim-bookworm` from Docker Hub instead of `nvcr.io/nvidia/base/ubuntu` plus the deadsnakes PPA; `--extra elasticsearch` dropped; the legal-compliance apt-source machinery dropped with the vendor base it was written for; `LICENSE`, `LICENSE-3rd-party.txt` and this file copied to `/licenses/`. |
| `.github/workflows/confidential-images.yml` | **New file.** Builds and publishes `ghcr.io/super-protocol/confidential-rag/rag-server`. |

Upstream's `frontend/` is **unchanged and not built into any published image.** Its production
container proxies the collection and document endpoints to `ingestor-server`, which is the service
this port has not substituted yet — so shipping it would publish a console whose document panel
cannot work. The marketplace listing serves its own small interface instead, and upstream's is kept
here intact for the moment the ingestion stage lands.

### Removed

`data/` is deleted. It is **not** Apache-2.0: `data/LICENSE.DATA` is the NVIDIA Asset License —
non-commercial research or evaluation only, no production use, no redistribution. Those files
therefore cannot be in an image, a default dataset, a demo or a screenshot. Nothing in the fork
references them.

## What did not change

* `LICENSE` is upstream's, unmodified, including its copyright line.
* `LICENSE-3rd-party.txt`, `frontend/LICENSE-3rd-party.txt` and
  `deploy/helm/nvidia-blueprint-rag/LICENSE` are upstream's, unmodified.
* Upstream ships no `NOTICE` file anywhere in its tree (checked at fork time, 2026-11-22). If one
  appears it goes into the fork and into every image built from it.
* Every SPDX header is intact. A file this fork changed keeps upstream's header and gains a
  `Modified by Super Protocol, <year>: <what changed>` line beneath it.

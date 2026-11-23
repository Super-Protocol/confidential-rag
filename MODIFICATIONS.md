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
| `src/nvidia_rag/utils/tei_reranker.py` | **New file.** `TEIRerank`, a `BaseDocumentCompressor` speaking TEI's `POST /rerank` (`{query, texts}` → `[{index, score}]`) instead of NeMo Retriever's `POST /v1/ranking`. Writes `relevance_score` where the rest of the pipeline reads it, and falls back to the retriever's own order if the endpoint is lost. **A fallback is reported rather than silent:** it keeps the retriever's own similarity as the score, marks every document `reranker_degraded` with the reason the endpoint gave, and records the same in `reranker_status`. The first version returned those documents with no score at all, so every citation read 0.0 and a reranker refusing every batch with `413` did no work on any query for the whole of the Confidential RAG pilot without anything but a log line saying so (SUP-889). |
| `src/nvidia_rag/utils/reranker_status.py` | **New file.** The last thing the reranker did, as a fact the process can be asked between queries — which is what a status panel polls and what `/v1/health` reports. Per process by default; written to `RERANKER_STATUS_PATH` when set, because the rag-server runs several uvicorn workers and the one answering a health request is not the one that served the query. |
| `src/nvidia_rag/utils/reranker.py` | New `tei` ranking engine, wired to the above. |
| `src/nvidia_rag/utils/llm.py` | New `openai` LLM engine: a plain `ChatOpenAI` against an OpenAI-compatible endpoint (vLLM). Kept separate from the `nvidia-ai-endpoints` branch rather than folded into it — that branch probes the endpoint, carries NVIDIA's default headers and picks generation parameters from a model-name table, which is exactly the dependency this fork removes. |
| `src/nvidia_rag/rag_server/main.py` | `_normalize_relevance_scores` is skipped for the `tei` engine. `sigmoid(0.1·x)` is written for a NeMo Retriever reranker's unbounded logit; a cross-encoder's score is already a probability, and the same transform crushes every score into `[0.500, 0.525]` — measured, 0.989 became 0.5247 and everything below 0.002 became 0.500. The ordering survives, so the answers look right; `RERANKER_SCORE_THRESHOLD` stops working, because every value up to 0.5 filters nothing and every value past 0.526 filters everything. |
| `src/nvidia_rag/rag_server/health.py` | Dependency probes ask each engine for the readiness path it serves. `/v1/health/ready` is a NIM route; vLLM and TEI answer `/health`, so before this an entirely healthy open stack reported every component unhealthy. The `Ranking` entry also reports the last real rerank: TEI answers `/health` with 200 while refusing the batch the pipeline actually sends, so the probe alone called the reranker healthy while it did nothing (SUP-889). |
| `src/nvidia_rag/utils/health_models.py` | `ServiceStatus.DEGRADED` — reachable and not doing its job. Neither `healthy` nor `unhealthy` described a reranker refusing every batch, and the report therefore said `healthy`. |
| `src/nvidia_rag/rag_server/response_generator.py` | `Citations` carries a `reranker` report (`degraded`, `reason`, `score_source`), so an API client sees a degraded reranking in the answer it degraded. And a citation's bounding box is read from `table_metadata.table_location` / `image_metadata.image_location`: upstream reads `content_metadata.location`, which `ContentMetadataSchema` neither defines nor (with `extra="forbid"`) can carry, so it resolved to `None` for every table and image citation ever produced (SUP-874, SUP-890). |
| `src/nvidia_rag/utils/vdb/milvus/milvus_vdb.py` | The dense retrieval path keeps the similarity it already computes, as `retriever_distance` (what Milvus answered — `metric_type: L2`, so smaller is better) and `retriever_score` (the same ranking as a bounded similarity: `1 - d_sq / 4`, which for unit vectors is exactly `(1 + cos) / 2` because `metric_type: L2` answers the SQUARED distance). `similarity_search` is `similarity_search_with_score` with the number dropped, and that number is the only honest thing a reranker fallback can report as a score. |

### Corrections to the above

| When | What was wrong, and what it cost |
|---|---|
| 2026-11-23 (SUP-893, SUP-895) | `_carry_retriever_scores` mapped the distance with `1 - d²/4` and documented the result as `(1 + cos) / 2`. For `metric_type: L2` Milvus returns the SQUARED Euclidean distance — it skips the square root, and langchain-milvus passes `result["distance"]` through unchanged — so the value arriving had already been squared and the code actually computed `1 - (1 - cos)²`. The order it reports survives for `cos ≥ 0`, which is why tests asserting monotonicity and bounds passed either way; what it cost is the degraded path, where every document the query points away from read exactly `0.0` — the value the console and `filter_documents_by_confidence` both treat as "the reranker said nothing". Corrected to `1 - d_sq / 4`, with the cosine identity pinned by a test rather than left to a docstring. Shipped in Confidential RAG listing 0.1.9. |

### The image

| File | Change |
|---|---|
| `src/nvidia_rag/rag_server/Dockerfile` | Base image is the official `python:3.13-slim-bookworm` from Docker Hub instead of `nvcr.io/nvidia/base/ubuntu` plus the deadsnakes PPA; `--extra elasticsearch` dropped; the legal-compliance apt-source machinery dropped with the vendor base it was written for; `LICENSE`, `LICENSE-3rd-party.txt` and this file copied to `/licenses/`. |
| `.github/workflows/confidential-images.yml` | **New file.** Builds and publishes `ghcr.io/super-protocol/confidential-rag/rag-server`. |

Upstream's `frontend/` is **unchanged and not built into any published image.** Its production
container proxies the collection and document endpoints to `ingestor-server`, which this port does
not deploy: `ingestor_server` needs `nv_ingest_client`, and the extraction stage it drives is four
NIM containers. Document ingestion in the listing is `swarm-doc-ingest` (SUP-874) behind the
listing's own console instead, so upstream's frontend is kept here intact and unbuilt.

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

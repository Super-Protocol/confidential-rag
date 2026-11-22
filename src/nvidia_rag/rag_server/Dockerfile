# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Super Protocol, 2026: rebuilt on an open base image and without the
# Elasticsearch extra.
#
# What changed and why:
#
#   * The base was `nvcr.io/nvidia/base/ubuntu`, plus python 3.13 from the
#     deadsnakes PPA on top of it. It is now the official `python` image from
#     Docker Hub. The NGC base pulls anonymously today, but it carries NVIDIA's
#     own container licence into an image whose whole purpose is that a tenant
#     can read what is in it, and three apt pins against a vendor's archive are
#     three things that can stop a rebuild.
#   * `--extra elasticsearch` is gone. The vector store in this port is Milvus,
#     which is Apache-2.0 and is already what the blueprint's own compose
#     profile runs; Elasticsearch's server is under the Elastic License and is
#     not something this listing may deploy. The client library was the only
#     thing the extra added, and nothing left in the image imports it.
#   * The legal-compliance apt/source-download machinery is gone with the
#     vendor base it was written for. The licences this image must carry are
#     copied into /licenses explicitly instead, which is the Apache-2.0 §4
#     obligation this image is redistribution under.
#
# Everything else — the wheel build, the locked sync, the entrypoint — is
# upstream's.

ARG PYTHON_IMAGE=python
ARG PYTHON_TAG=3.13-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.12

# -------- Stage 1: build the wheel --------
FROM ${PYTHON_IMAGE}:${PYTHON_TAG} AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /uvx /bin/

WORKDIR /build

COPY uv.lock pyproject.toml README.md LICENSE ./
COPY ./src ./src

RUN uv build

# -------- Stage 2: runtime --------
FROM ${PYTHON_IMAGE}:${PYTHON_TAG} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# libmagic is what python-magic binds to for content sniffing; libgl1 and
# libglib2.0-0 are pdfplumber's and pillow's transitive native dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libmagic1 \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /bin/uv

WORKDIR /workspace

COPY --from=builder /build/pyproject.toml /build/uv.lock /workspace/

# `--locked` refuses to resolve: the lock is the pin, and a build that quietly
# re-resolved it would produce an image whose dependency set nobody declared.
RUN uv sync --locked --no-install-project --no-dev --extra rag

COPY --from=builder /build/dist/*.whl /workspace/
RUN WHEEL_NAME="$(ls /workspace/nvidia_rag-*.whl)" && \
    uv pip install --no-deps --no-cache-dir "$WHEEL_NAME" && \
    rm -f /workspace/*.whl

# Apache-2.0 §4(a) and §4(b): an image is redistribution in object form, so
# upstream's licence, upstream's third-party licence list and the record of what
# we changed travel inside it.
COPY LICENSE LICENSE-3rd-party.txt MODIFICATIONS.md /licenses/

# The text splitter writes here, and the container runs as a non-root user in
# the cluster space.
RUN mkdir -p /tmp-data && chmod 777 /tmp-data
ENV HF_HOME=/tmp-data

ENV PATH="/workspace/.venv/bin:$PATH"

ENTRYPOINT ["uvicorn", "nvidia_rag.rag_server.server:app"]

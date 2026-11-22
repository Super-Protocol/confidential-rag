# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Super Protocol, 2026: added the `openai` embedding engine, so the
# retriever can be served by an open OpenAI-compatible embedding endpoint
# (Text Embeddings Inference) instead of the NeMo Retriever embedding NIM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The wrapper for interacting with embedding models.
1. get_embedding_model: Get the embedding model. Uses an OpenAI-compatible
   endpoint, the NVIDIA AI Endpoints, or HuggingFace.
"""

import logging
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import OpenAIEmbeddings

from nvidia_rag.utils.common import NVIDIA_API_DEFAULT_HEADERS, sanitize_nim_url
from nvidia_rag.utils.configuration import NvidiaRAGConfig

logger = logging.getLogger(__name__)


def get_embedding_model(
    model: str,
    url: str,
    config: NvidiaRAGConfig | None = None,
    truncate: str | None = "END",
) -> Embeddings:
    """Create the embedding model.

    Args:
        model: Model name
        url: URL endpoint
        config: NvidiaRAGConfig instance. If None, creates a new one.
        truncate: Truncation strategy
    """
    if config is None:
        config = NvidiaRAGConfig()

    # Sanitize the URL
    url = sanitize_nim_url(url, model, "embedding")

    logger.info(
        "Using %s as model engine and %s and model for embeddings",
        config.embeddings.model_engine,
        model,
    )

    # Modified by Super Protocol, 2026: an OpenAI-compatible embedding endpoint.
    #
    # `truncate` and `input_type` are NeMo Retriever extensions, not part of the
    # OpenAI embeddings request, and an open server rejects or ignores them — so
    # neither is sent here. `check_embedding_ctx_length=False` keeps langchain
    # from tokenising with tiktoken and splitting the input itself: the server
    # owns truncation (TEI is started with `--auto-truncate`), and a client-side
    # split would silently change what gets embedded. `dimensions` is likewise
    # omitted: an open checkpoint returns the width it returns, and asking for
    # another one is how a collection ends up with vectors of two widths.
    if config.embeddings.model_engine == "openai":
        api_key = config.embeddings.get_api_key()
        logger.info("Using OpenAI-compatible embedding model %s hosted at %s", model, url)
        return OpenAIEmbeddings(
            base_url=url,
            model=model,
            api_key=api_key or "not-used",
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        )

    if config.embeddings.model_engine == "nvidia-ai-endpoints":
        api_key = config.embeddings.get_api_key()

        if url:
            logger.info("Using embedding model %s hosted at %s", model, url)
            if truncate is not None:
                return NVIDIAEmbeddings(
                    base_url=url,
                    model=model,
                    api_key=api_key,
                    truncate=truncate,
                    dimensions=config.embeddings.dimensions,
                    default_headers=NVIDIA_API_DEFAULT_HEADERS,
                )
            else:
                return NVIDIAEmbeddings(
                    base_url=url,
                    model=model,
                    api_key=api_key,
                    dimensions=config.embeddings.dimensions,
                    default_headers=NVIDIA_API_DEFAULT_HEADERS,
                )

        logger.info("Using embedding model %s hosted at api catalog", model)
        if truncate is not None:
            return NVIDIAEmbeddings(
                model=model,
                api_key=api_key,
                truncate=truncate,
                dimensions=config.embeddings.dimensions,
                default_headers=NVIDIA_API_DEFAULT_HEADERS,
            )
        else:
            return NVIDIAEmbeddings(
                model=model,
                api_key=api_key,
                dimensions=config.embeddings.dimensions,
                default_headers=NVIDIA_API_DEFAULT_HEADERS,
            )

    raise RuntimeError(
        "Unable to find any supported embedding model. Supported engines are "
        f"openai and nvidia-ai-endpoints; APP_EMBEDDINGS_MODELENGINE is "
        f"{config.embeddings.model_engine!r}."
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 Super Protocol. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Added by Super Protocol, 2026: a reranker for an open cross-encoder served by
# Text Embeddings Inference, replacing the NeMo Retriever reranking NIM.
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

"""Reranking against an open cross-encoder served over Text Embeddings Inference.

`NVIDIARerank` speaks the NeMo Retriever reranking API: `POST /v1/ranking` with
`{"model", "query": {"text"}, "passages": [{"text"}]}`, answering `rankings` with
a `logit` per passage. TEI speaks a different, smaller dialect — `POST /rerank`
with `{"query", "texts"}`, answering a list of `{"index", "score"}` — so the two
are not interchangeable behind one base URL, and this class is the adapter.

Two behaviours are deliberately the same as upstream's, because the rest of the
pipeline depends on them:

* the score lands on `doc.metadata["relevance_score"]`, which is what
  `nvidia_rag.utils.common.filter_documents_by_confidence` reads for
  `RERANKER_SCORE_THRESHOLD`;
* at most `top_n` documents come back, highest score first.

One differs, and it matters when reading a score: a cross-encoder reranker
returns a probability in [0, 1] (`raw_scores: false`), where the NIM returns an
unbounded logit. A `RERANKER_SCORE_THRESHOLD` carried over from a NIM deployment
means something else here.
"""

import logging
from typing import Any

import requests
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import ConfigDict

logger = logging.getLogger(__name__)

__all__ = ["TEIRerank"]


class TEIRerank(BaseDocumentCompressor):
    """Rerank documents with a cross-encoder served by Text Embeddings Inference."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_url: str
    """Where the reranker answers, e.g. `http://text-reranker.ns.svc:80`."""

    model: str = ""
    """The checkpoint the endpoint serves. Carried for logs and health reports:
    TEI serves exactly one model and takes no `model` field in the request."""

    api_key: str | None = None
    """Presented as a bearer token. The endpoint authenticates even though it is
    reachable only from inside the namespace."""

    top_n: int = 4
    """How many documents to return."""

    timeout: float = 60.0
    """Seconds to wait for one rerank call."""

    truncate: bool = True
    """Let the server truncate a passage longer than the model's context rather
    than failing the whole batch. The alternative is a 413 for one long chunk
    taking the answer down with it."""

    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        # Tolerate a `/v1` suffix: the embedding endpoint needs one and an
        # operator setting both URLs by hand will copy it across.
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return f"{base}/rerank"

    def _rank(self, query: str, texts: list[str]) -> list[dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "query": query,
            "texts": texts,
            "raw_scores": False,
            "return_text": False,
        }
        if self.truncate:
            payload["truncate"] = True
            payload["truncation_direction"] = "Right"
        response = requests.post(
            self._endpoint(), json=payload, headers=headers, timeout=self.timeout
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, list):
            raise ValueError(
                f"Reranker at {self._endpoint()} answered {type(body).__name__}, "
                "expected a list of {index, score} objects"
            )
        return body

    def compress_documents(
        self,
        documents: list[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> list[Document]:
        """Return at most `top_n` documents, most relevant first."""
        docs = list(documents)
        if not docs:
            return []

        texts = [doc.page_content or "" for doc in docs]
        try:
            rankings = self._rank(query, texts)
        except Exception:
            # Reranking is a refinement of an answer we already have. Losing the
            # endpoint should cost the ordering, not the query — upstream's
            # retriever is already ordered by vector distance.
            logger.exception(
                "Reranking failed at %s; falling back to the retriever's own order",
                self._endpoint(),
            )
            return docs[: self.top_n]

        ranked: list[Document] = []
        for entry in sorted(
            rankings, key=lambda item: item.get("score", 0.0), reverse=True
        ):
            index = entry.get("index")
            if not isinstance(index, int) or not 0 <= index < len(docs):
                logger.warning("Reranker returned out-of-range index %r; skipped", index)
                continue
            doc = docs[index]
            doc.metadata["relevance_score"] = entry.get("score", 0.0)
            ranked.append(doc)
            if len(ranked) >= self.top_n:
                break
        return ranked

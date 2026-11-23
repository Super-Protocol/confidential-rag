# SPDX-FileCopyrightText: Copyright (c) 2026 Super Protocol. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""`TEIRerank`, and in particular what it does when the reranker does not answer.

The adapter's fallback — keep the retriever's order when the endpoint is lost —
is correct and is not what these tests are about. What they are about is that a
fallback must be *legible*: the first version of this class returned the
retriever's documents with no score at all, which made every citation read
`relevance_score` 0.0 with nothing anywhere saying why, and a reranker refusing
every single batch with `413` therefore survived two review rounds of the
Confidential RAG listing (SUP-840, SUP-889).

So the two cases that found that defect in production are the two cases here:
a `413` refusal and a timeout. For each, the result has to carry non-zero scores
and say it is degraded — in the documents, in the API response's citations, and
in the health report an operator or a status panel reads between queries.
"""

from __future__ import annotations

import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest
import requests
from langchain_core.documents import Document

from nvidia_rag.utils import reranker_status
from nvidia_rag.utils.reranker_status import (
    DEGRADED_KEY,
    DEGRADED_REASON_KEY,
)
from nvidia_rag.utils.tei_reranker import RETRIEVER_SCORE_KEY, TEIRerank

BASE_URL = "http://confidential-rag-reranker"


@pytest.fixture(autouse=True)
def _forget_the_last_query(monkeypatch):
    """Each test starts with a reranker nothing has asked anything yet.

    The record is module state on purpose — `/v1/health` has no query to read a
    flag off — so a test that did not clear it would be reading the previous
    test's verdict.
    """
    monkeypatch.delenv(reranker_status.STATUS_PATH_ENV, raising=False)
    reranker_status.reset_for_tests()
    yield
    reranker_status.reset_for_tests()


def retrieved(count: int = 3, *, scored: bool = True) -> list[Document]:
    """What the dense retriever hands the reranker, in the order it found.

    `retriever_score` is what `MilvusVDB.retrieval_langchain` leaves there: the
    similarity the search already computed, highest first.
    """
    return [
        Document(
            page_content=f"passage {index}",
            metadata=(
                {RETRIEVER_SCORE_KEY: round(0.90 - index * 0.11, 4)} if scored else {}
            )
            | {"source": f"corpus:docs.documents#doc-{index}"},
        )
        for index in range(count)
    ]


def refusing(status: int, body: str):
    """A `requests` response that raises for status the way the real one does."""
    response = MagicMock()
    response.status_code = status
    response.text = body
    response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        f"{status} Client Error", response=response
    )
    return response


class TestTheRerankerAnswers:
    """The ordinary path: scores from the cross-encoder, nothing degraded."""

    def test_scores_and_order_come_from_the_endpoint(self):
        reranker = TEIRerank(base_url=BASE_URL, model="bge-reranker-v2-m3", top_n=2)
        answer = MagicMock(status_code=200)
        answer.json.return_value = [
            {"index": 2, "score": 0.98},
            {"index": 0, "score": 0.41},
            {"index": 1, "score": 0.02},
        ]

        with patch("nvidia_rag.utils.tei_reranker.requests.post", return_value=answer):
            out = reranker.compress_documents(retrieved(), "a question")

        assert [doc.page_content for doc in out] == ["passage 2", "passage 0"]
        assert [doc.metadata["relevance_score"] for doc in out] == [0.98, 0.41]
        assert not any(DEGRADED_KEY in doc.metadata for doc in out)
        assert reranker_status.current().degraded is False
        assert reranker_status.current().last_success_at

    def test_a_recovered_reranker_is_not_degraded_any_more(self):
        """`degraded` is about the last attempt, not about ever having failed."""
        reranker = TEIRerank(base_url=BASE_URL, top_n=3)
        with patch(
            "nvidia_rag.utils.tei_reranker.requests.post",
            return_value=refusing(
                413, '{"error":"batch size 40 > maximum allowed batch size 32"}'
            ),
        ):
            reranker.compress_documents(retrieved(), "a question")
        assert reranker_status.current().degraded is True

        answer = MagicMock(status_code=200)
        answer.json.return_value = [{"index": 0, "score": 0.7}]
        with patch("nvidia_rag.utils.tei_reranker.requests.post", return_value=answer):
            out = reranker.compress_documents(retrieved(1), "a question")

        assert reranker_status.current().degraded is False
        assert DEGRADED_KEY not in out[0].metadata


class TestTheRerankerRefusesTheBatch:
    """`413 batch size 40 > maximum allowed batch size 32` — the real defect."""

    @staticmethod
    def _fallback() -> list[Document]:
        reranker = TEIRerank(base_url=BASE_URL, top_n=2)
        with patch(
            "nvidia_rag.utils.tei_reranker.requests.post",
            return_value=refusing(
                413, '{"error":"batch size 40 > maximum allowed batch size 32"}'
            ),
        ):
            return reranker.compress_documents(retrieved(), "a question")

    def test_the_query_still_answers_in_the_retrievers_order(self):
        out = self._fallback()
        assert [doc.page_content for doc in out] == ["passage 0", "passage 1"]

    def test_every_score_is_non_zero_and_is_the_retrievers_own(self):
        """The whole point. Zeroes here are what hid a component doing nothing."""
        out = self._fallback()
        scores = [doc.metadata["relevance_score"] for doc in out]
        assert scores == [0.90, 0.79]
        assert all(score > 0.0 for score in scores)
        # And still descending, because the retriever's order is what they are.
        assert scores == sorted(scores, reverse=True)

    def test_each_document_says_it_was_not_reranked_and_why(self):
        out = self._fallback()
        for doc in out:
            assert doc.metadata[DEGRADED_KEY] is True
            reason = doc.metadata[DEGRADED_REASON_KEY]
            assert "413" in reason
            # The sentence that names the fix has to survive into the reason.
            assert "maximum allowed batch size 32" in reason
            assert BASE_URL in reason

    def test_the_deployment_can_be_asked_between_queries(self):
        self._fallback()
        report = reranker_status.current()
        assert report.degraded is True
        assert "413" in report.reason
        assert report.observed_at

    def test_a_document_the_retriever_left_unscored_gets_no_invented_score(self):
        """A made-up number would be this same failure in a new place."""
        reranker = TEIRerank(base_url=BASE_URL, top_n=2)
        with patch(
            "nvidia_rag.utils.tei_reranker.requests.post",
            return_value=refusing(413, "too big"),
        ):
            out = reranker.compress_documents(retrieved(scored=False), "a question")
        assert all("relevance_score" not in doc.metadata for doc in out)
        assert all(doc.metadata[DEGRADED_KEY] is True for doc in out)


class TestTheRerankerDoesNotAnswerInTime:
    """The second shape of the same production failure: unpinned thread pools."""

    @staticmethod
    def _fallback() -> list[Document]:
        reranker = TEIRerank(base_url=BASE_URL, top_n=3, timeout=60.0)
        with patch(
            "nvidia_rag.utils.tei_reranker.requests.post",
            side_effect=requests.exceptions.ReadTimeout("timed out"),
        ):
            return reranker.compress_documents(retrieved(), "a question")

    def test_scores_survive_the_timeout(self):
        out = self._fallback()
        assert [doc.metadata["relevance_score"] for doc in out] == [0.90, 0.79, 0.68]

    def test_the_reason_names_the_budget_that_was_missed(self):
        out = self._fallback()
        reason = out[0].metadata[DEGRADED_REASON_KEY]
        assert "60s" in reason
        assert reranker_status.current().degraded is True


class TestTheAnswerCarriesTheDegradation:
    """`Citations.reranker` — so an API client sees it without reading a log."""

    def test_citations_report_a_degraded_reranking_and_its_source(self):
        from nvidia_rag.rag_server.response_generator import reranker_report

        out = TestTheRerankerRefusesTheBatch._fallback()
        report = reranker_report(out)
        assert report.degraded is True
        assert report.score_source == "retriever"
        assert "413" in report.reason

    def test_citations_of_a_reranked_answer_say_so(self):
        from nvidia_rag.rag_server.response_generator import reranker_report

        report = reranker_report(retrieved())
        assert report.degraded is False
        assert report.score_source == "reranker"
        assert report.reason == ""


class TestTheRecordIsSharedBetweenWorkers:
    """The rag-server runs several uvicorn workers; `/v1/health` is one of them.

    Without a shared record, the worker answering the status request is usually
    not the worker that served the query, and a degradation would be reported on
    a coin flip.
    """

    def test_a_degradation_in_one_process_is_visible_to_another(
        self, tmp_path: pathlib.Path, monkeypatch
    ):
        path = tmp_path / "reranker-status.json"
        monkeypatch.setenv(reranker_status.STATUS_PATH_ENV, str(path))

        reranker = TEIRerank(base_url=BASE_URL, top_n=1)
        with patch(
            "nvidia_rag.utils.tei_reranker.requests.post",
            return_value=refusing(413, "too big"),
        ):
            reranker.compress_documents(retrieved(1), "a question")

        written = json.loads(path.read_text())
        assert written["degraded"] is True
        assert "413" in written["reason"]

        # A different worker: no in-process record at all, same file.
        reranker_status.reset_for_tests()
        assert reranker_status.current().degraded is True

    def test_an_unwritable_path_does_not_fail_the_query(
        self, tmp_path: pathlib.Path, monkeypatch
    ):
        """The instrument must not be able to break the thing it measures."""
        monkeypatch.setenv(
            reranker_status.STATUS_PATH_ENV, str(tmp_path / "nope" / "x" / "s.json")
        )
        with patch(
            "nvidia_rag.utils.reranker_status.os.makedirs", side_effect=OSError("ro")
        ):
            report = reranker_status.record_degradation("HTTP 413")
        assert report.degraded is True
        assert reranker_status.current().degraded is True

    def test_nothing_is_written_when_no_path_is_configured(self, tmp_path, monkeypatch):
        monkeypatch.delenv(reranker_status.STATUS_PATH_ENV, raising=False)
        reranker_status.record_degradation("HTTP 413")
        assert list(tmp_path.iterdir()) == []


class TestTheHealthReportSaysIt:
    """A health probe answering 200 is not evidence that reranking works."""

    def test_a_reachable_but_degraded_reranker_is_reported_degraded(self):
        from nvidia_rag.rag_server.health import _with_rerank_outcome
        from nvidia_rag.utils.health_models import (
            NIMServiceHealthInfo,
            ServiceStatus,
        )

        reranker_status.record_degradation(
            'HTTP 413 from http://confidential-rag-reranker/rerank: {"error":"batch size 40'
            ' > maximum allowed batch size 32"}'
        )
        probed = NIMServiceHealthInfo(
            service="Ranking",
            url=f"{BASE_URL}/health",
            status=ServiceStatus.HEALTHY,
            latency_ms=3.0,
            http_status=200,
        )
        reported = _with_rerank_outcome(probed)
        assert reported.status == ServiceStatus.DEGRADED
        assert "maximum allowed batch size 32" in reported.message

    def test_a_working_reranker_is_left_as_the_probe_found_it(self):
        from nvidia_rag.rag_server.health import _with_rerank_outcome
        from nvidia_rag.utils.health_models import (
            NIMServiceHealthInfo,
            ServiceStatus,
        )

        reranker_status.record_success()
        probed = NIMServiceHealthInfo(
            service="Ranking",
            url=BASE_URL,
            status=ServiceStatus.HEALTHY,
            latency_ms=3.0,
        )
        assert _with_rerank_outcome(probed).status == ServiceStatus.HEALTHY

    def test_a_reranker_no_query_has_reached_is_left_alone(self):
        """Nothing has been asked of it, so there is nothing to degrade."""
        from nvidia_rag.rag_server.health import _with_rerank_outcome
        from nvidia_rag.utils.health_models import (
            NIMServiceHealthInfo,
            ServiceStatus,
        )

        probed = NIMServiceHealthInfo(
            service="Ranking",
            url=BASE_URL,
            status=ServiceStatus.HEALTHY,
            latency_ms=3.0,
        )
        assert _with_rerank_outcome(probed).status == ServiceStatus.HEALTHY


class TestTheRetrieverKeepsItsOwnScore:
    """The number the fallback reports has to exist, and has to order correctly.

    `metric_type` on this collection is `L2`, so what Milvus answers is a
    DISTANCE and smaller is better. A fallback that wrote it into
    `relevance_score` unconverted would invert the meaning of every citation it
    touched — a new silent wrong in place of the old one.
    """

    def test_a_distance_becomes_a_bounded_similarity_in_the_same_order(self):
        from nvidia_rag.utils.vdb.milvus.milvus_vdb import _carry_retriever_scores

        scored = [
            (Document(page_content=f"d{i}"), d) for i, d in enumerate([0.0, 0.5, 1.4])
        ]
        out = _carry_retriever_scores(scored)

        scores = [doc.metadata[RETRIEVER_SCORE_KEY] for doc in out]
        assert scores[0] == 1.0
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= score <= 1.0 for score in scores)
        # The raw distance is carried too, under a name that cannot be mistaken
        # for a score.
        assert [doc.metadata["retriever_distance"] for doc in out] == [0.0, 0.5, 1.4]

    def test_a_distance_outside_the_unit_sphere_is_clamped_not_negative(self):
        from nvidia_rag.utils.vdb.milvus.milvus_vdb import _carry_retriever_scores

        out = _carry_retriever_scores([(Document(page_content="d"), 9.0)])
        assert out[0].metadata[RETRIEVER_SCORE_KEY] == 0.0

    def test_a_distance_that_is_not_a_number_is_skipped_rather_than_guessed(self):
        from nvidia_rag.utils.vdb.milvus.milvus_vdb import _carry_retriever_scores

        out = _carry_retriever_scores([(Document(page_content="d"), None)])
        assert RETRIEVER_SCORE_KEY not in out[0].metadata

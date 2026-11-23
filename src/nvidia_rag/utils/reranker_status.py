# SPDX-FileCopyrightText: Copyright (c) 2026 Super Protocol. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Added by Super Protocol, 2026: whether the reranker did any work on the last
# query, as a fact the deployment can be asked rather than a line in a log.
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

"""Whether reranking is working, as something the deployment reports about itself.

`TEIRerank` falls back to the retriever's own order when the reranker endpoint
cannot be reached or refuses the batch. That fallback is correct — reranking
refines an answer we already have, and losing it should cost the ordering rather
than the query — but on its own it is also invisible. A reranker that did no work
on a single query survived two review rounds of the Confidential RAG listing
because nothing but a log line said so, while the listing advertised the
component, attested it and reserved two cores and 6Gi for it (SUP-840, SUP-889).

So the fallback is recorded in two places, and both are load-bearing:

* **on the documents**, under `reranker_degraded` and
  `reranker_degradation_reason`, so the degradation travels with the answer it
  degraded and reaches the API response's citations;
* **here**, so `/v1/health` can be asked the question directly — which is what a
  status panel polls and what an operator reads when no query is in flight.

The record is per process, which is not enough on its own: the rag-server runs
several uvicorn workers, and the worker that answers `/v1/health` is usually not
the worker that served the query. So when `RERANKER_STATUS_PATH` names a file,
the record is also written there — one small JSON document, replaced atomically —
and every worker in the pod reads the same latest fact. Unset, this degrades to
the in-process record, which is the right default for a library: nothing is
written anywhere a caller did not ask for.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "DEGRADED_KEY",
    "DEGRADED_REASON_KEY",
    "RerankerReport",
    "current",
    "record_degradation",
    "record_success",
    "status_path",
]

#: Set on every document of a degraded result, and read by the citation builder.
DEGRADED_KEY = "reranker_degraded"
#: Why, in the words the endpoint used where it gave any — an HTTP status and the
#: body it answered with is the single most useful thing to show an operator.
DEGRADED_REASON_KEY = "reranker_degradation_reason"

#: Where the cross-process record lives, when a deployment asks for one.
STATUS_PATH_ENV = "RERANKER_STATUS_PATH"


@dataclass
class RerankerReport:
    """The last thing the reranker did, and when.

    `degraded` describes the most recent attempt, not the whole history: a
    reranker that failed once and has worked since is not degraded now, and
    `last_success_at` is what says the difference.
    """

    degraded: bool = False
    reason: str = ""
    #: When the last attempt — successful or not — was made. Empty means the
    #: reranker has not been asked anything yet, which is a deployment that has
    #: had no query rather than a broken one.
    observed_at: str = ""
    last_success_at: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_LOCK = threading.Lock()
_REPORT = RerankerReport()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def status_path() -> str:
    return os.environ.get(STATUS_PATH_ENV, "").strip()


def _write(report: RerankerReport) -> None:
    """Replace the shared record, or say in the log why it could not be.

    Atomic, because `/v1/health` reads this file while a query is being answered
    and half a JSON document is a status endpoint that raises instead of
    reporting. A failure to write is logged once per occurrence and never
    raised: this is an instrument, and an instrument that can fail a query is
    worse than no instrument.
    """
    path = status_path()
    if not path:
        return
    directory = os.path.dirname(path) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w", dir=directory, prefix=".reranker-status.", delete=False
        )
        try:
            with handle:
                json.dump(report.as_dict(), handle)
            os.replace(handle.name, path)
        except BaseException:
            os.unlink(handle.name)
            raise
    except OSError as error:
        logger.warning("could not write the reranker status to %s: %s", path, error)


def _read() -> RerankerReport | None:
    path = status_path()
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        logger.warning("could not read the reranker status from %s: %s", path, error)
        return None
    if not isinstance(raw, dict):
        return None
    fields = {
        "degraded": bool(raw.get("degraded")),
        "reason": str(raw.get("reason") or ""),
        "observed_at": str(raw.get("observed_at") or ""),
        "last_success_at": str(raw.get("last_success_at") or ""),
    }
    return RerankerReport(**fields)


def _publish(*, degraded: bool, reason: str) -> RerankerReport:
    global _REPORT
    with _LOCK:
        moment = _now()
        _REPORT = RerankerReport(
            degraded=degraded,
            reason=reason,
            observed_at=moment,
            # Timestamps are compared as strings everywhere here, which is sound
            # for this one format and nothing else: UTC, fixed width, no offset.
            last_success_at=_REPORT.last_success_at if degraded else moment,
        )
        report = _REPORT
    _write(report)
    return report


def record_success() -> RerankerReport:
    """The reranker answered and the documents carry its scores."""
    return _publish(degraded=False, reason="")


def record_degradation(reason: str) -> RerankerReport:
    """The reranker did not answer, and the retriever's order was kept instead."""
    return _publish(degraded=True, reason=reason)


def current() -> RerankerReport:
    """The latest record this pod has, from whichever worker produced it.

    Merged rather than preferred one way or the other: the shared file carries
    the newest *attempt* from any worker, while this process may be the only one
    that has seen the newest *success*. Taking the later of each is what keeps a
    report from un-saying something that happened.
    """
    with _LOCK:
        mine = RerankerReport(**_REPORT.as_dict())
    shared = _read()
    if shared is None:
        return mine
    latest = shared if shared.observed_at >= mine.observed_at else mine
    return RerankerReport(
        degraded=latest.degraded,
        reason=latest.reason,
        observed_at=latest.observed_at,
        last_success_at=max(mine.last_success_at, shared.last_success_at),
    )


def reset_for_tests() -> None:
    """Forget everything. Only a test has any business calling this."""
    global _REPORT
    with _LOCK:
        _REPORT = RerankerReport()

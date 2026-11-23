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

"""Where a citation says its content came from.

The field upstream reads for a table or image citation's bounding box,
`content_metadata.location`, is not defined by `ContentMetadataSchema` and the
schema forbids extras — so it resolved to `None` for every such citation ever
produced. Found while reading that schema for SUP-874, fixed on SUP-890, which
is the change that makes tables and charts citable in the Confidential RAG
listing at all.
"""

from __future__ import annotations

from langchain_core.documents import Document

from nvidia_rag.rag_server.response_generator import (
    element_location,
    prepare_citations,
)


class TestACitationKnowsWhereItCameFrom:
    """`table_metadata.table_location`, and not `content_metadata.location`.

    Upstream reads the second, `ContentMetadataSchema` does not define it and
    forbids extras, so the bounding box of every table and image citation was
    `None` — found while reading the schema for SUP-874 and fixed here because
    composing document ingestion is what makes tables and charts citable at all
    (SUP-890).
    """

    def test_a_tables_box_is_read_from_its_own_metadata(self):
        doc = Document(
            page_content="| a | b |",
            metadata={
                "content_metadata": {"type": "structured", "subtype": "table"},
                "table_metadata": {
                    "table_location": [72.0, 108.5, 523.25, 240.0],
                    "table_location_max_dimensions": [612.0, 792.0],
                },
            },
        )
        assert element_location(doc) == ([72.0, 108.5, 523.25, 240.0], [612, 792])

    def test_a_pictures_box_is_read_from_its_own_metadata(self):
        doc = Document(
            page_content="Figure 3: quarterly revenue",
            metadata={
                "content_metadata": {"type": "image"},
                "image_metadata": {
                    "image_location": [10.0, 20.0, 30.0, 40.0],
                    "image_location_max_dimensions": [612.0, 792.0],
                },
            },
        )
        assert element_location(doc) == ([10.0, 20.0, 30.0, 40.0], [612, 792])

    def test_content_metadata_location_is_still_honoured_last(self):
        doc = Document(
            page_content="x",
            metadata={"content_metadata": {"type": "text", "location": [1, 2, 3, 4]}},
        )
        assert element_location(doc) == ([1.0, 2.0, 3.0, 4.0], [])

    def test_a_chunk_with_no_geometry_reports_none_rather_than_a_guess(self):
        # A ClickHouse row, a DOCX paragraph, an HTML page: no page, no box.
        assert element_location(Document(page_content="x")) == ([], [])

    def test_a_box_that_is_not_numbers_is_dropped_not_raised(self):
        doc = Document(
            page_content="x",
            metadata={"table_metadata": {"table_location": ["left", "top"]}},
        )
        assert element_location(doc) == ([], [])

    def test_a_cited_table_carries_its_box_through_prepare_citations(self):
        doc = Document(
            page_content="| region | revenue |\n| --- | --- |\n| EMEA | 4.2 |",
            metadata={
                "source": "upload:report.pdf#4f2a",
                "relevance_score": 0.77,
                "content_metadata": {
                    "type": "text",
                    "page_number": 4,
                    "element_type": "structured",
                    "subtype": "table",
                },
                "table_metadata": {
                    "table_location": [72.0, 108.5, 523.25, 240.0],
                    "table_location_max_dimensions": [612.0, 792.0],
                },
                "image_metadata": {},
            },
        )
        citations = prepare_citations([doc], enable_citations=True)
        assert citations.total_results == 1
        result = citations.results[0]
        assert result.score == 0.77
        assert result.metadata.page_number == 4
        assert result.metadata.location == [72.0, 108.5, 523.25, 240.0]
        assert result.metadata.location_max_dimensions == [612, 792]
        assert citations.reranker.degraded is False

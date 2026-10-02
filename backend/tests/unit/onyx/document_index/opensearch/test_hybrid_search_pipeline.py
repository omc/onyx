import json
from typing import Any

import pytest

from onyx.context.search.models import IndexFilters
from onyx.document_index.interfaces_new import TenantState
from onyx.document_index.opensearch import search
from onyx.document_index.opensearch.constants import (
    HybridSearchSubqueryConfiguration,
    HybridSubqueryOrder,
)
from onyx.document_index.opensearch.search import (
    DocumentQuery,
    get_hybrid_search_pipeline_name,
    get_min_max_normalization_pipeline_name_and_config,
)

_SEARCH_MODULE = "onyx.document_index.opensearch.search"


def _subquery_kinds(query_body: dict[str, Any]) -> list[str]:
    """Returns "knn" or "lexical" for each hybrid sub-query, in order."""
    return [
        "knn" if "knn" in subquery else "lexical"
        for subquery in query_body["query"]["hybrid"]["queries"]
    ]


def _build_hybrid_query() -> dict[str, Any]:
    return DocumentQuery.get_hybrid_search_query(
        query_text="quarterly revenue",
        query_vector=[0.1, 0.2, 0.3],
        num_hits=10,
        tenant_state=TenantState(tenant_id="public", multitenant=False),
        index_filters=IndexFilters(access_control_list=None),
        include_hidden=False,
    )


@pytest.fixture
def two_subqueries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        f"{_SEARCH_MODULE}.HYBRID_SEARCH_SUBQUERY_CONFIGURATION",
        HybridSearchSubqueryConfiguration.CONTENT_VECTOR_TITLE_CONTENT_COMBINED_KEYWORD,
    )


def test_pipeline_name_defaults_to_onyx_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(f"{_SEARCH_MODULE}.OPENSEARCH_SEARCH_PIPELINE", None)
    monkeypatch.setattr(
        f"{_SEARCH_MODULE}.HYBRID_SEARCH_NORMALIZATION_PIPELINE",
        search.HybridSearchNormalizationPipeline.MIN_MAX,
    )

    assert get_hybrid_search_pipeline_name() == "normalization_pipeline_min_max"


def test_pipeline_name_uses_external_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        f"{_SEARCH_MODULE}.OPENSEARCH_SEARCH_PIPELINE",
        "hybrid-pipeline-lexical30-knn70",
    )

    assert get_hybrid_search_pipeline_name() == "hybrid-pipeline-lexical30-knn70"


@pytest.mark.usefixtures("two_subqueries")
@pytest.mark.parametrize(
    ("order", "expected_kinds"),
    [
        (HybridSubqueryOrder.KNN_FIRST, ["knn", "lexical"]),
        (HybridSubqueryOrder.LEXICAL_FIRST, ["lexical", "knn"]),
    ],
)
def test_subquery_order(
    monkeypatch: pytest.MonkeyPatch,
    order: HybridSubqueryOrder,
    expected_kinds: list[str],
) -> None:
    monkeypatch.setattr(f"{_SEARCH_MODULE}.OPENSEARCH_HYBRID_SUBQUERY_ORDER", order)

    assert _subquery_kinds(_build_hybrid_query()) == expected_kinds


def _subquery_weight_pairs() -> list[tuple[str, float]]:
    """Pairs each hybrid sub-query (as JSON) with its Onyx pipeline weight."""
    subqueries = _build_hybrid_query()["query"]["hybrid"]["queries"]
    _, config = get_min_max_normalization_pipeline_name_and_config()
    weights = config["phase_results_processors"][0]["normalization-processor"][
        "combination"
    ]["parameters"]["weights"]
    assert len(subqueries) == len(weights)
    return [
        (json.dumps(subquery, sort_keys=True), weight)
        for subquery, weight in zip(subqueries, weights, strict=True)
    ]


@pytest.mark.parametrize("configuration", list(HybridSearchSubqueryConfiguration))
def test_onyx_pipeline_weights_follow_subquery_order(
    monkeypatch: pytest.MonkeyPatch,
    configuration: HybridSearchSubqueryConfiguration,
) -> None:
    """Each sub-query must keep the same weight in both orders."""
    monkeypatch.setattr(
        f"{_SEARCH_MODULE}.HYBRID_SEARCH_SUBQUERY_CONFIGURATION", configuration
    )
    monkeypatch.setattr(
        f"{_SEARCH_MODULE}.OPENSEARCH_HYBRID_SUBQUERY_ORDER",
        HybridSubqueryOrder.KNN_FIRST,
    )
    knn_first_pairs = _subquery_weight_pairs()

    monkeypatch.setattr(
        f"{_SEARCH_MODULE}.OPENSEARCH_HYBRID_SUBQUERY_ORDER",
        HybridSubqueryOrder.LEXICAL_FIRST,
    )
    lexical_first_pairs = _subquery_weight_pairs()

    assert lexical_first_pairs == list(reversed(knn_first_pairs))

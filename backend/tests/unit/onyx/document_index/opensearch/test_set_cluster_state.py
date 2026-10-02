from unittest.mock import MagicMock

import pytest
from opensearchpy.exceptions import (
    AuthorizationException,
    ConnectionError,
    RequestError,
)

from onyx.document_index.opensearch.client import OpenSearchClient
from onyx.document_index.opensearch.opensearch_document_index import (
    set_cluster_state,
)


@pytest.mark.parametrize(
    "error",
    [
        AuthorizationException(403, "security_exception", {}),
        RequestError(400, "illegal_argument_exception", {}),
    ],
)
def test_rejected_cluster_settings_still_creates_search_pipelines(
    error: Exception,
) -> None:
    """Hosted clusters (e.g. Bonsai) can reject PUT _cluster/settings. Setup
    must continue, because hybrid search needs the search pipelines."""
    client = MagicMock(spec=OpenSearchClient)
    client.put_cluster_settings.side_effect = error

    set_cluster_state(client)

    assert client.create_search_pipeline.call_count == 2


def test_connection_error_on_cluster_settings_propagates() -> None:
    client = MagicMock(spec=OpenSearchClient)
    client.put_cluster_settings.side_effect = ConnectionError(
        "N/A", "connection refused", Exception()
    )

    with pytest.raises(ConnectionError):
        set_cluster_state(client)

    client.create_search_pipeline.assert_not_called()


def test_external_search_pipeline_skips_pipeline_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "onyx.document_index.opensearch.opensearch_document_index.OPENSEARCH_SEARCH_PIPELINE",
        "hybrid-pipeline-lexical30-knn70",
    )
    client = MagicMock(spec=OpenSearchClient)

    set_cluster_state(client)

    client.put_cluster_settings.assert_called_once()
    client.create_search_pipeline.assert_not_called()

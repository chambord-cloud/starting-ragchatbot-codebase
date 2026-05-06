"""Tests for VectorStore.search() — directly exposes the MAX_RESULTS=0 bug."""

from unittest.mock import MagicMock, patch
from vector_store import VectorStore, SearchResults


# ── Helper: build a VectorStore with a fully mocked ChromaDB client ────────

def _make_store(max_results=0, course_content_query_return=None,
                course_catalog_query_return=None):
    """Create a VectorStore with a mocked PersistentClient and collections."""
    mock_col_content = MagicMock()
    if course_content_query_return is not None:
        mock_col_content.query.return_value = course_content_query_return

    mock_col_catalog = MagicMock()
    if course_catalog_query_return is not None:
        mock_col_catalog.query.return_value = course_catalog_query_return

    mock_client = MagicMock()
    # Return different mocks for each collection
    mock_client.get_or_create_collection.side_effect = \
        lambda name, embedding_function: {
            "course_catalog": mock_col_catalog,
            "course_content": mock_col_content,
        }[name]

    with patch("chromadb.PersistentClient", return_value=mock_client):
        with patch("chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
                   return_value=MagicMock()):
            store = VectorStore(
                chroma_path="/fake/path",
                embedding_model="fake-model",
                max_results=max_results
            )
    return store, mock_col_content, mock_col_catalog


def _populated_response():
    return {
        "documents": [["Doc about computer use and Anthropic API"]],
        "metadatas": [[{"course_title": "Building Towards Computer Use", "lesson_number": 1}]],
        "distances": [[0.15]]
    }


def _empty_response():
    return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


# ── Tests ──────────────────────────────────────────────────────────────────

class TestSearchDefaultUsesMaxResults:
    """Prove the MAX_RESULTS=0 bug exists."""

    def test_default_max_results_zero_passes_zero_to_chromadb(self):
        """With max_results=0, n_results=0 is sent to ChromaDB → empty results."""
        store, mock_content, _ = _make_store(max_results=0)
        mock_content.query.return_value = _empty_response()

        results = store.search("What is computer use?")

        call_kwargs = mock_content.query.call_args.kwargs
        assert call_kwargs["n_results"] == 0, \
            f"Expected n_results=0 (the bug), got n_results={call_kwargs['n_results']}"
        assert results.is_empty(), \
            "Results should be empty when n_results=0"

    def test_explicit_limit_overrides_max_results_zero(self):
        """A caller-specified limit should bypass the default max_results=0."""
        store, mock_content, _ = _make_store(max_results=0)
        mock_content.query.return_value = _populated_response()

        results = store.search("query", limit=3)

        call_kwargs = mock_content.query.call_args.kwargs
        assert call_kwargs["n_results"] == 3, \
            f"Expected n_results=3 (explicit limit), got n_results={call_kwargs['n_results']}"
        assert not results.is_empty()

    def test_positive_default_returns_results(self):
        """With max_results=5, n_results=5 → results populated (the fix)."""
        store, mock_content, _ = _make_store(max_results=5)
        mock_content.query.return_value = _populated_response()

        results = store.search("What is computer use?")
        call_kwargs = mock_content.query.call_args.kwargs
        assert call_kwargs["n_results"] == 5, \
            f"Expected n_results=5, got n_results={call_kwargs['n_results']}"
        assert not results.is_empty()
        assert results.documents[0] == "Doc about computer use and Anthropic API"


class TestSearchQueryPropagation:
    """Verify query text and parameters reach ChromaDB correctly."""

    def test_passes_query_text(self):
        store, mock_content, _ = _make_store(max_results=3)
        mock_content.query.return_value = _populated_response()

        store.search("explain MCP in detail")
        call_kwargs = mock_content.query.call_args.kwargs
        assert call_kwargs["query_texts"] == ["explain MCP in detail"]

    def test_no_where_filter_when_no_course_or_lesson(self):
        store, mock_content, _ = _make_store(max_results=3)
        mock_content.query.return_value = _populated_response()

        store.search("topic")
        call_kwargs = mock_content.query.call_args.kwargs
        assert call_kwargs["where"] is None


class TestSearchCourseResolution:
    """Verify course name resolution and filter building."""

    def test_resolves_course_name_and_uses_in_filter(self):
        store, mock_content, mock_catalog = _make_store(max_results=3)
        mock_content.query.return_value = _populated_response()
        mock_catalog.query.return_value = {
            "documents": [["Building Towards Computer Use with Anthropic"]],
            "metadatas": [[{"title": "Building Towards Computer Use with Anthropic"}]],
            "distances": [[0.05]]
        }

        store.search("computer use", course_name="Computer Use")
        call_kwargs = mock_content.query.call_args.kwargs
        assert call_kwargs["where"] == \
            {"course_title": "Building Towards Computer Use with Anthropic"}

    def test_returns_empty_when_course_not_found(self):
        store, mock_content, mock_catalog = _make_store(max_results=3)
        mock_catalog.query.return_value = {
            "documents": [[]], "metadatas": [[]], "distances": [[]]
        }

        results = store.search("topic", course_name="NonexistentCourse")
        assert results.error is not None
        assert "No course found" in results.error
        assert results.is_empty()

    def test_builds_combined_course_and_lesson_filter(self):
        store, mock_content, mock_catalog = _make_store(max_results=3)
        mock_content.query.return_value = _populated_response()
        mock_catalog.query.return_value = {
            "documents": [["Test Course"]],
            "metadatas": [[{"title": "Test Course"}]],
            "distances": [[0.05]]
        }

        store.search("topic", course_name="Test", lesson_number=2)
        call_kwargs = mock_content.query.call_args.kwargs
        assert call_kwargs["where"] == {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 2}
            ]
        }

    def test_builds_lesson_only_filter(self):
        store, mock_content, _ = _make_store(max_results=3)
        mock_content.query.return_value = _populated_response()

        store.search("topic", lesson_number=3)
        call_kwargs = mock_content.query.call_args.kwargs
        assert call_kwargs["where"] == {"lesson_number": 3}


class TestSearchErrorHandling:
    """Verify VectorStore handles ChromaDB errors gracefully."""

    def test_handles_chroma_exception(self):
        store, mock_content, _ = _make_store(max_results=3)
        mock_content.query.side_effect = Exception("ChromaDB connection refused")

        results = store.search("query")
        assert results.error is not None
        assert "Search error" in results.error
        assert "ChromaDB connection refused" in results.error
        assert results.is_empty()


class TestSearchResultsParsing:
    """Verify SearchResults.from_chroma parsing."""

    def test_from_chroma_parses_correctly(self):
        raw = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"course": "A"}, {"course": "B"}]],
            "distances": [[0.1, 0.2]]
        }
        results = SearchResults.from_chroma(raw)
        assert results.documents == ["doc1", "doc2"]
        assert results.metadata == [{"course": "A"}, {"course": "B"}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_handles_empty(self):
        raw = {"documents": [], "metadatas": [], "distances": []}
        results = SearchResults.from_chroma(raw)
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []

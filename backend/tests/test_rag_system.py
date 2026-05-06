"""Integration tests for RAGSystem orchestration with mocked external APIs."""

import tempfile
from unittest.mock import MagicMock, patch
import pytest
from rag_system import RAGSystem


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_mock_chroma_collection():
    """Return a MagicMock collection with standard ChromaDB query/get signatures."""
    col = MagicMock()
    col.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]]
    }
    col.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    return col


def _patch_all(monkeypatch):
    """Patch ChromaDB, embeddings, and OpenAI so nothing hits the network."""
    # ChromaDB
    mock_client = MagicMock()
    mock_catalog = _make_mock_chroma_collection()
    mock_content = _make_mock_chroma_collection()
    mock_client.get_or_create_collection.side_effect = \
        lambda name, **kwargs: {
            "course_catalog": mock_catalog,
            "course_content": mock_content
        }[name]

    def _mock_chroma_client(*args, **kwargs):
        return mock_client

    monkeypatch.setattr("chromadb.PersistentClient", _mock_chroma_client)
    monkeypatch.setattr(
        "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
        lambda **kw: MagicMock()
    )

    # OpenAI — prevent real client creation
    monkeypatch.setattr("openai.OpenAI", MagicMock())

    return mock_catalog, mock_content


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def rag_system(monkeypatch, mock_config):
    """RAGSystem with ChromaDB and OpenAI fully mocked, max_results=5."""
    _patch_all(monkeypatch)
    with patch("rag_system.AIGenerator.__init__", lambda *a, **kw: None):
        rs = RAGSystem(mock_config)
        # Replace ai_generator with a mock that returns canned responses
        rs.ai_generator = MagicMock()
        rs.ai_generator.generate_response.return_value = \
            "This is a test answer about course materials."
    return rs


@pytest.fixture
def rag_with_tool_flow(monkeypatch, mock_config):
    """RAGSystem where the AI generator simulates a tool-call flow."""
    _patch_all(monkeypatch)
    with patch("rag_system.AIGenerator.__init__", lambda *a, **kw: None):
        rs = RAGSystem(mock_config)
        # Swap vector_store with a mock to control search results
        rs.vector_store = MagicMock()
        from vector_store import SearchResults
        rs.vector_store.search.return_value = SearchResults(
            documents=["Course content about MCP architecture..."],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
            distances=[0.1]
        )
        rs.vector_store.get_course_link.return_value = "https://example.com/mcp"
        rs.vector_store.get_lesson_link.return_value = "https://example.com/mcp/lesson1"

        # Re-create search tool with mocked store
        from search_tools import CourseSearchTool
        rs.search_tool = CourseSearchTool(rs.vector_store)
        rs.tool_manager.tools["search_course_content"] = rs.search_tool
    return rs


# ── Tests ──────────────────────────────────────────────────────────────────

class TestRAGSystemInit:
    """Verify RAGSystem wires components together correctly."""

    def test_initialization_creates_all_components(self, rag_system):
        assert rag_system.document_processor is not None
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.session_manager is not None

    def test_tool_manager_has_two_tools_registered(self, rag_system):
        tool_names = list(rag_system.tool_manager.tools.keys())
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() — the main entry point."""

    def test_query_returns_answer_and_sources(self, rag_system):
        rag_system.ai_generator.generate_response.return_value = \
            "This course covers computer use with Anthropic."

        answer, sources = rag_system.query("What is computer use?")

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(sources, list)

    def test_query_creates_prompt_wrapper(self, rag_system):
        rag_system.ai_generator.generate_response.return_value = "Answer"

        rag_system.query("How does MCP work?")

        call_args = rag_system.ai_generator.generate_response.call_args
        kwargs = call_args.kwargs
        assert "Answer this question about course materials" in kwargs["query"]

    def test_query_passes_tools_and_tool_manager(self, rag_system):
        rag_system.ai_generator.generate_response.return_value = "Answer"

        rag_system.query("What is computer use?")

        call_kwargs = rag_system.ai_generator.generate_response.call_args.kwargs
        assert call_kwargs["tools"] is not None
        assert len(call_kwargs["tools"]) == 2
        assert call_kwargs["tool_manager"] is not None

    def test_query_with_session_id_stores_exchange(self, rag_system):
        rag_system.ai_generator.generate_response.return_value = "Session answer."

        session_id = rag_system.session_manager.create_session()
        answer, _ = rag_system.query("Test question?", session_id=session_id)

        history = rag_system.session_manager.get_conversation_history(session_id)
        assert history is not None
        assert "Test question?" in history
        assert "Session answer." in history

    def test_multiple_queries_accumulate_in_session(self, rag_system):
        rag_system.ai_generator.generate_response.return_value = "Response"

        sid = rag_system.session_manager.create_session()
        rag_system.query("First question", session_id=sid)
        rag_system.query("Second question", session_id=sid)

        history = rag_system.session_manager.get_conversation_history(sid)
        assert "First question" in history
        assert "Second question" in history

    def test_different_sessions_are_independent(self, rag_system):
        rag_system.ai_generator.generate_response.return_value = "Answer"

        sid1 = rag_system.session_manager.create_session()
        sid2 = rag_system.session_manager.create_session()

        rag_system.query("Q1", session_id=sid1)
        rag_system.query("Q2", session_id=sid2)

        h1 = rag_system.session_manager.get_conversation_history(sid1)
        h2 = rag_system.session_manager.get_conversation_history(sid2)
        assert "Q1" in h1 and "Q2" not in h1
        assert "Q2" in h2 and "Q1" not in h2

    def test_query_tool_flow_produces_sources(self, rag_with_tool_flow):
        """When the AI calls a tool, sources should be tracked via ToolManager."""
        # Simulate: AI generator uses the tool internally, then returns answer
        # We'll manually call the search tool to populate sources
        rag_with_tool_flow.search_tool.execute(
            query="MCP architecture", course_name="MCP"
        )

        sources = rag_with_tool_flow.tool_manager.get_last_sources()
        assert len(sources) > 0
        assert "MCP Course" in sources[0]["label"]


class TestRAGSystemDocumentProcessing:
    """Tests for document processing and course analytics."""

    def test_add_course_document_from_temp_file(self, rag_system, monkeypatch):
        """Process a real temporary course document and verify parsing."""
        # Build a valid course document
        doc_text = (
            "Course Title: Test Course\n"
            "Course Link: https://example.com/test\n"
            "Course Instructor: Dr. Test\n"
            "\n"
            "Lesson 0: Introduction\n"
            "Lesson Link: https://example.com/test/0\n"
            "This is the introduction text for the test course. "
            "It covers the basics of testing. "
            "Testing is an important part of software development. "
            "Good tests help prevent bugs.\n"
            "Lesson 1: Advanced Topics\n"
            "Lesson Link: https://example.com/test/1\n"
            "This lesson covers advanced testing topics. "
            "We discuss mocking, stubbing, and integration testing. "
            "These techniques help isolate components under test.\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(doc_text)
            f.flush()
            temp_path = f.name

        try:
            course, chunk_count = rag_system.add_course_document(temp_path)
            assert course is not None
            assert course.title == "Test Course"
            assert course.instructor == "Dr. Test"
            assert len(course.lessons) == 2
            assert course.lessons[0].lesson_number == 0
            assert course.lessons[0].title == "Introduction"
            assert course.lessons[1].lesson_number == 1
            assert chunk_count > 0
        finally:
            import os
            os.unlink(temp_path)

    def test_get_course_analytics_empty(self, rag_system):
        analytics = rag_system.get_course_analytics()
        assert analytics["total_courses"] == 0
        assert analytics["course_titles"] == []

"""Tests for CourseSearchTool.execute() and CourseOutlineTool with mocked VectorStore."""

import pytest
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


# ── SimpleMockVectorStore (hand-rolled, for search_tools tests) ────────────

class SimpleMockVectorStore:
    """Records calls and returns configurable SearchResults. Not a MagicMock."""

    def __init__(self):
        self.search_calls = []
        self.search_return = None
        self.lesson_link_return = "https://example.com/lesson/1"
        self.course_link_return = "https://example.com/course"
        self.outline_return = None

    def search(self, query, course_name=None, lesson_number=None, limit=None):
        self.search_calls.append({
            "query": query, "course_name": course_name,
            "lesson_number": lesson_number, "limit": limit
        })
        if self.search_return is not None:
            return self.search_return
        return SearchResults(
            documents=["Sample content about the topic..."],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.15]
        )

    def get_lesson_link(self, course_title, lesson_number):
        return self.lesson_link_return

    def get_course_link(self, course_title):
        return self.course_link_return

    def get_course_outline(self, course_name):
        return self.outline_return


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_store():
    """SimpleMockVectorStore with default populated results."""
    return SimpleMockVectorStore()


@pytest.fixture
def search_tool(mock_store):
    return CourseSearchTool(mock_store)


@pytest.fixture
def outline_tool(mock_store):
    return CourseOutlineTool(mock_store)


@pytest.fixture
def tool_manager(search_tool, outline_tool):
    tm = ToolManager()
    tm.register_tool(search_tool)
    tm.register_tool(outline_tool)
    return tm


# ── CourseSearchTool tests ─────────────────────────────────────────────────

class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() returning formatted results."""

    def test_returns_formatted_results(self, search_tool):
        result = search_tool.execute(query="computer use")
        assert "Sample content about the topic" in result
        assert "[Test Course - Lesson 1]" in result

    def test_markdown_link_included_when_url_available(self, search_tool):
        result = search_tool.execute(query="test")
        assert "[Test Course - Lesson 1](https://example.com/lesson/1)" in result

    def test_multiple_results_separated_by_double_newline(self, search_tool, mock_store):
        mock_store.search_return = SearchResults(
            documents=["First doc", "Second doc"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        result = search_tool.execute(query="test")
        assert "\n\n" in result

    def test_execute_returns_empty_message_no_filters(self, search_tool, mock_store):
        mock_store.search_return = SearchResults(documents=[], metadata=[], distances=[])
        result = search_tool.execute(query="nonexistent")
        assert result == "No relevant content found."

    def test_execute_returns_empty_message_with_course(self, search_tool, mock_store):
        mock_store.search_return = SearchResults(documents=[], metadata=[], distances=[])
        result = search_tool.execute(query="q", course_name="MCP")
        assert "in course 'MCP'" in result

    def test_execute_returns_empty_message_with_lesson(self, search_tool, mock_store):
        mock_store.search_return = SearchResults(documents=[], metadata=[], distances=[])
        result = search_tool.execute(query="q", lesson_number=3)
        assert "in lesson 3" in result

    def test_execute_returns_empty_message_with_both_filters(self, search_tool, mock_store):
        mock_store.search_return = SearchResults(documents=[], metadata=[], distances=[])
        result = search_tool.execute(query="q", course_name="MCP", lesson_number=2)
        assert "in course 'MCP'" in result
        assert "in lesson 2" in result

    def test_execute_returns_error_message(self, search_tool, mock_store):
        mock_store.search_return = SearchResults(
            documents=[], metadata=[], distances=[],
            error="Search error: ChromaDB unavailable"
        )
        result = search_tool.execute(query="q")
        assert result == "Search error: ChromaDB unavailable"

    def test_execute_forwards_parameters_to_store(self, search_tool, mock_store):
        search_tool.execute(query="computer use", course_name="MCP", lesson_number=2)
        call = mock_store.search_calls[0]
        assert call["query"] == "computer use"
        assert call["course_name"] == "MCP"
        assert call["lesson_number"] == 2
        # CourseSearchTool does NOT pass limit, so it should be None
        assert call["limit"] is None


class TestCourseSearchToolSources:
    """Tests for source tracking on the tool."""

    def test_tracks_last_sources_with_label_and_url(self, search_tool):
        search_tool.execute(query="test")
        assert len(search_tool.last_sources) == 1
        source = search_tool.last_sources[0]
        assert source["label"] == "Test Course - Lesson 1"
        assert source["url"] == "https://example.com/lesson/1"

    def test_falls_back_to_course_link_when_lesson_link_none(self, search_tool, mock_store):
        mock_store.lesson_link_return = None
        mock_store.course_link_return = "https://example.com/course-only"
        search_tool.execute(query="test")
        source = search_tool.last_sources[0]
        assert source["url"] == "https://example.com/course-only"

    def test_no_url_when_both_links_none(self, search_tool, mock_store):
        mock_store.lesson_link_return = None
        mock_store.course_link_return = None
        result = search_tool.execute(query="test")
        # Plain text label, no brackets to avoid AI hallucinating a link
        assert result.startswith("Test Course - Lesson 1\n")

    def test_source_label_without_lesson_number(self, search_tool, mock_store):
        mock_store.search_return = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "My Course"}],  # no lesson_number
            distances=[0.1]
        )
        search_tool.execute(query="test")
        assert search_tool.last_sources[0]["label"] == "My Course"

    def test_multiple_sources_tracked(self, search_tool, mock_store):
        mock_store.search_return = SearchResults(
            documents=["Doc 1", "Doc 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 3}
            ],
            distances=[0.1, 0.2]
        )
        search_tool.execute(query="test")
        assert len(search_tool.last_sources) == 2
        assert search_tool.last_sources[0]["label"] == "Course A - Lesson 1"
        assert search_tool.last_sources[1]["label"] == "Course B - Lesson 3"


# ── CourseOutlineTool tests ────────────────────────────────────────────────

class TestCourseOutlineTool:
    """Tests for CourseOutlineTool.execute()."""

    def test_returns_formatted_outline(self, outline_tool, mock_store):
        mock_store.outline_return = {
            "title": "MCP: Build Rich-Context AI Apps",
            "course_link": "https://example.com/mcp",
            "instructor": "Jane Doe",
            "lessons": [
                {"lesson_number": 0, "lesson_title": "Introduction"},
                {"lesson_number": 1, "lesson_title": "Getting Started"}
            ]
        }
        result = outline_tool.execute(course_name="MCP")
        assert "# MCP: Build Rich-Context AI Apps" in result
        assert "[Course Link](https://example.com/mcp)" in result
        assert "## Lessons (2 total)" in result
        assert "- Lesson 0: Introduction" in result
        assert "- Lesson 1: Getting Started" in result

    def test_returns_not_found_when_no_match(self, outline_tool, mock_store):
        mock_store.outline_return = None
        result = outline_tool.execute(course_name="Nonexistent")
        assert "No course found matching 'Nonexistent'" in result

    def test_outline_includes_lesson_links(self, outline_tool, mock_store):
        mock_store.outline_return = {
            "title": "MCP Course",
            "course_link": "https://example.com/mcp",
            "instructor": "Jane Doe",
            "lessons": [
                {"lesson_number": 0, "lesson_title": "Introduction",
                 "lesson_link": "https://example.com/mcp/lesson/0"},
                {"lesson_number": 1, "lesson_title": "Getting Started",
                 "lesson_link": None}
            ]
        }
        result = outline_tool.execute(course_name="MCP")
        assert "[Lesson 0: Introduction](https://example.com/mcp/lesson/0)" in result
        assert "- Lesson 1: Getting Started" in result
        assert "[Course Link](https://example.com/mcp)" in result


# ── ToolManager tests ──────────────────────────────────────────────────────

class TestToolManager:
    """Tests for ToolManager registration, execution, and source tracking."""

    def test_registers_tools(self, tool_manager):
        assert "search_course_content" in tool_manager.tools
        assert "get_course_outline" in tool_manager.tools

    def test_get_tool_definitions_returns_list(self, tool_manager):
        defs = tool_manager.get_tool_definitions()
        assert len(defs) == 2
        names = [d["function"]["name"] for d in defs]
        assert "search_course_content" in names
        assert "get_course_outline" in names

    def test_execute_tool_known(self, tool_manager, mock_store):
        result = tool_manager.execute_tool("search_course_content", query="test")
        assert "Sample content" in result

    def test_execute_tool_unknown(self, tool_manager):
        result = tool_manager.execute_tool("nonexistent_tool")
        assert "not found" in result

    def test_get_last_sources(self, tool_manager, search_tool):
        search_tool.execute(query="test")
        sources = tool_manager.get_last_sources()
        assert len(sources) == 1
        assert sources[0]["label"] == "Test Course - Lesson 1"

    def test_reset_sources_clears_all(self, tool_manager, search_tool):
        search_tool.execute(query="test")
        assert len(search_tool.last_sources) == 1
        tool_manager.reset_sources()
        assert len(search_tool.last_sources) == 0

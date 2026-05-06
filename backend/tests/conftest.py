import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional

# Make bare imports like "from vector_store import VectorStore" work
sys.path.insert(0, str(Path(__file__).parent.parent))

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


# ── ChromaDB mocking ───────────────────────────────────────────────────────

@pytest.fixture
def mock_chroma_collection():
    """A MagicMock ChromaDB collection with configurable query results."""
    col = MagicMock()
    col.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]]
    }
    return col


@pytest.fixture
def mock_chroma_client(mock_chroma_collection):
    """Patch chromadb.PersistentClient to return a mock."""
    with patch("chromadb.PersistentClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_client_cls.return_value = mock_client
        yield mock_client_cls


@pytest.fixture
def populated_chroma_response():
    return {
        "documents": [["Doc text about computer use", "Doc text about MCP"]],
        "metadatas": [[
            {"course_title": "Test Course A", "lesson_number": 1},
            {"course_title": "Test Course B", "lesson_number": 2}
        ]],
        "distances": [[0.1, 0.2]]
    }


@pytest.fixture
def empty_chroma_response():
    return {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]]
    }


# ── SearchResults fixtures ─────────────────────────────────────────────────

@pytest.fixture
def populated_search_results():
    return SearchResults(
        documents=["Doc text about computer use", "Doc text about MCP"],
        metadata=[
            {"course_title": "Test Course A", "lesson_number": 1},
            {"course_title": "Test Course B", "lesson_number": 2}
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def empty_search_results():
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    return SearchResults(documents=[], metadata=[], distances=[],
                         error="Search error: something went wrong")


# ── OpenAI / DeepSeek mocking ──────────────────────────────────────────────

@pytest.fixture
def mock_openai_create():
    """Patch openai.OpenAI.chat.completions.create with a side_effect helper."""
    with patch("openai.OpenAI") as mock_client_cls:
        mock_client = MagicMock()
        mock_create = MagicMock()
        mock_client.return_value.chat.completions.create = mock_create
        # Allow tests to set mock_create.side_effect before calling
        mock_client_cls.return_value = mock_client.return_value
        yield mock_create


def _make_response(content, tool_calls=None):
    """Helper: build a mock OpenAI chat completion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def make_direct_response(content="General knowledge answer."):
    return _make_response(content, tool_calls=None)


def make_tool_call_response(tool_name="search_course_content",
                            arguments='{"query": "computer use"}'):
    tc = MagicMock()
    tc.id = "call_abc123"
    tc.function = MagicMock()
    tc.function.name = tool_name
    tc.function.arguments = arguments
    return _make_response(None, tool_calls=[tc])


def make_follow_up_response(content="The course covers computer use basics..."):
    return _make_response(content, tool_calls=None)


# ── Temp ChromaDB path ─────────────────────────────────────────────────────

@pytest.fixture
def tmp_chroma_path(tmp_path):
    """Ephemeral ChromaDB directory that is cleaned up after the test."""
    path = tmp_path / "chroma_test"
    path.mkdir()
    return str(path)


# ── Config fixture ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_config():
    """Config with safe defaults for testing (no real API calls)."""
    from config import Config
    return Config(
        DEEPSEEK_API_KEY="test-key",
        DEEPSEEK_MODEL="deepseek-v4-flash",
        DEEPSEEK_BASE_URL="https://api.deepseek.com",
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        MAX_RESULTS=5,
        MAX_HISTORY=2,
        CHROMA_PATH="./chroma_test_db"
    )


# ── API test fixtures ─────────────────────────────────────────────────────

def create_test_app(mock_rag):
    """
    Build a FastAPI app with the same API routes as app.py but without
    static file mounting or startup events. This avoids importing app.py
    directly, which would fail when ../frontend doesn't exist in tests.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    test_app = FastAPI(title="Test RAG System")

    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceInfo(BaseModel):
        label: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceInfo]
        session_id: str

    class CourseInfo(BaseModel):
        title: str
        course_link: Optional[str] = None

    class CourseStats(BaseModel):
        total_courses: int
        courses: List[CourseInfo]

    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag.session_manager.create_session()

            answer, source_dicts = mock_rag.query(request.query, session_id)
            sources = [SourceInfo(**s) for s in source_dicts]

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                courses=[CourseInfo(**c) for c in analytics["courses"]],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return test_app


@pytest.fixture
def mock_rag_system():
    """Mock RAGSystem with canned query responses and course analytics."""
    from session_manager import SessionManager

    rag = MagicMock()
    rag.session_manager = SessionManager(max_history=2)
    rag.query.return_value = (
        "This is a test answer about course materials.",
        [{"label": "Test Course - Lesson 1", "url": "https://example.com/lesson1"}],
    )
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "courses": [
            {"title": "Course A", "course_link": "https://example.com/a"},
            {"title": "Course B", "course_link": None},
        ],
    }
    return rag


@pytest.fixture
def client(mock_rag_system):
    """FastAPI TestClient wired to a test app with a mocked RAG system."""
    app = create_test_app(mock_rag_system)
    with TestClient(app) as tc:
        yield tc

# Testing Framework Enhancements

## Changes Made

### 1. pytest Configuration (`pyproject.toml`)
Added `[tool.pytest.ini_options]` section:
- `testpaths = ["backend/tests"]` — pytest discovers tests in the backend tests directory
- `pythonpath = ["backend"]` — adds backend to PYTHONPATH so imports like `from vector_store import VectorStore` work without sys.path hacks
- Custom markers defined: `unit`, `integration`, `api`
- Added `httpx` dev dependency (required by FastAPI TestClient)

### 2. API Test Fixtures (`backend/tests/conftest.py`)
Added three new fixtures and a factory function:
- **`create_test_app(mock_rag)`** — Factory that builds a FastAPI app with the same API routes as `app.py` (`POST /api/query`, `GET /api/courses`) but without static file mounting or startup events. This avoids importing `app.py` directly, which would fail when `../frontend` doesn't exist in the test environment.
- **`mock_rag_system`** — MagicMock RAGSystem with canned `query()` and `get_course_analytics()` return values and a real `SessionManager` instance.
- **`client`** — FastAPI TestClient wired to a test app backed by `mock_rag_system`.

### 3. API Endpoint Tests (`backend/tests/test_api.py`)
17 new tests across 3 test classes:

**TestQueryEndpoint** (9 tests):
- Successful query returns answer, sources, and auto-generated session ID
- Session ID preservation when explicitly provided
- Missing `query` field returns 422
- Invalid JSON returns 422
- Empty query string accepted
- RAG system exception returns 500
- Correct Content-Type header
- Query string passed through to RAG system

**TestCoursesEndpoint** (5 tests):
- Returns course list with counts and titles
- Empty course list
- Exception returns 500
- Correct Content-Type header
- Method validation (POST returns 405)

**TestSessionPersistence** (2 tests):
- Same session ID across multiple sequential queries
- Different sessions receive independent IDs

### 4. Pre-existing Test Fixes
Fixed 4 pre-existing test failures in `test_rag_system.py` and `test_search_tools.py`:
- Updated `test_query_creates_prompt_wrapper` to match current query-passing behavior
- Updated `test_get_course_analytics_empty` to use correct dict key (`courses` instead of `course_titles`)
- Updated `test_returns_formatted_outline` to match current markdown link format
- Updated `test_outline_includes_lesson_links` to match current markdown link format

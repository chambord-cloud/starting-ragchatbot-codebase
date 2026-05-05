# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Install dependencies (Python 3.13 required)
uv sync

# Run the full application (API + static frontend on port 8000)
./run.sh

# Run only the backend server with hot reload
cd backend && uv run uvicorn app:app --reload --port 8000

# The application launches with no test suite.
```

## Architecture

This is a full-stack RAG chatbot that answers questions about course materials using semantic search over ChromaDB and DeepSeek for response generation, served as a single FastAPI process (API + static frontend on port 8000).

### Core RAG Pipeline

1. **Document ingestion** (`backend/document_processor.py`): Parses course scripts from `docs/` with a specific format — `Course Title:`, `Course Link:`, `Course Instructor:` metadata lines, then `Lesson N: Title` markers followed by content. Chunks content into sentence-based overlapping segments.
2. **Vector storage** (`backend/vector_store.py`): ChromaDB with two collections — `course_catalog` (course titles/instructors for fuzzy course name resolution) and `course_content` (content chunks with course+lesson metadata). Embeddings via `all-MiniLM-L6-v2` (sentence-transformers). Data persists in `backend/chroma_db/`.
3. **Query flow** (`backend/rag_system.py`): User query → AI generator creates a prompt → DeepSeek decides whether to invoke the `search_course_content` tool → tool searches ChromaDB with optional course name and lesson filters → results fed back to DeepSeek → final answer with sources.
4. **AI generation** (`backend/ai_generator.py`): Uses DeepSeek API via OpenAI SDK with tool-calling support. System prompt instructs the model to search only for course-specific questions, answer general knowledge directly, and avoid meta-commentary. Temperature 0, max 800 tokens.

### Backend Module Map

- `backend/app.py` — FastAPI entry point. Two endpoints: `POST /api/query` (takes `query` + optional `session_id`) and `GET /api/courses` (returns course count and titles). On startup, loads all documents from `docs/`. Mounts `frontend/` as static files with no-cache headers.
- `backend/config.py` — Single `Config` dataclass with all settings. Loads `.env` via python-dotenv. Key values: `DEEPSEEK_MODEL` (deepseek-v4-flash), `DEEPSEEK_BASE_URL`, `CHUNK_SIZE` (800), `CHUNK_OVERLAP` (100), `MAX_RESULTS` (5), `MAX_HISTORY` (2).
- `backend/models.py` — Pydantic models: `Course`, `Lesson`, `CourseChunk`.
- `backend/search_tools.py` — `Tool` abstract base class with `get_tool_definition()` and `execute()`. `CourseSearchTool` wraps `VectorStore.search()` and formats results. `ToolManager` registers tools, collects sources, and resets state between queries.
- `backend/session_manager.py` — In-memory `Dict[str, List[Message]]`. Sessions are ephemeral (lost on restart). Conversation history is formatted as `"Role: Content"` strings and appended to the system prompt.

### Frontend

Vanilla HTML/JS/CSS. `script.js` talks to `/api/query` and `/api/courses`, renders markdown responses via `marked.js` CDN, and manages a session ID per page load.

### Key Design Decisions

- DeepSeek decides **when** to search (via tool-choice), not the application — the system prompt is the control surface for this behavior.
- Course names in queries are resolved via vector similarity against `course_catalog` before content search.
- Document chunks are re-ingested on startup but existing courses are skipped (checked by title deduplication).
- Conversation history is capped at `MAX_HISTORY` exchanges (2 by default), trimmed oldest-first.

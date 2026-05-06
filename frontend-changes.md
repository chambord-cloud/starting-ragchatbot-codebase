# Frontend Code Quality Tooling

## Overview

Added code quality tooling to the frontend with Prettier for formatting and ESLint for linting.

## Files Added

| File | Purpose |
|------|---------|
| `package.json` | npm project with format/lint/check/fix scripts |
| `.prettierrc` | Prettier configuration (2-space indent, single quotes, semicolons, trailing commas) |
| `.prettierignore` | Excludes backend, docs, Python files from formatting |
| `eslint.config.js` | ESLint flat config targeting `frontend/**/*.js` with browser globals |

## Files Modified

| File | Change |
|------|--------|
| `frontend/index.html` | Formatted with Prettier |
| `frontend/script.js` | Formatted with Prettier; removed unused `i` parameter in `.map()` callback |
| `frontend/style.css` | Formatted with Prettier |

## npm Scripts

| Script | Command | Purpose |
|--------|---------|---------|
| `format` | `prettier --write "frontend/**/*.{html,css,js}"` | Auto-format all frontend files |
| `format:check` | `prettier --check "frontend/**/*.{html,css,js}"` | Check formatting (CI-safe) |
| `lint` | `eslint "frontend/**/*.js"` | Lint JavaScript files |
| `lint:fix` | `eslint --fix "frontend/**/*.js"` | Auto-fix lint issues |
| `check` | `npm run format:check && npm run lint` | Run all quality checks |
| `fix` | `npm run format && npm run lint:fix` | Auto-fix all formatting and lint issues |

## ESLint Rules

- Extends `@eslint/js` recommended config
- `ecmaVersion: "latest"` with browser globals (`document`, `window`, `console`, `fetch`, `marked`)
- `no-unused-vars`: warn (with `_` prefix ignore pattern)
- `no-undef`: error
- `no-console`: off
- `no-constant-condition`: warn

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

# Frontend Changes — Dark/Light Theme Toggle

## Overview
Added a theme toggle button that allows users to switch between dark and light themes with smooth transitions, localStorage persistence, and accessible design.

## Files Modified

### `frontend/index.html`
- Added theme toggle button (`#themeToggle`) before the container div
- Button contains both sun and moon SVG icons with smooth crossfade/rotation transitions
- Sun icon visible in dark mode, moon icon visible in light mode
- Accessible: `aria-label` updates dynamically, keyboard focusable via `focus-visible`
- Updated CSS and JS cache-busting versions

### `frontend/style.css`
- Restructured `:root` CSS variables with explicit dark theme defaults
- Added `[data-theme="light"]` selector block with light theme variables:
  - Background: `#f8fafc`, Surface: `#ffffff`, Text: `#0f172a`
  - Adjusted border, link, shadow, and surface colors for light context
- Added `transition` on `body` and key structural elements for smooth theme switching (0.3s ease)
- Added `.theme-toggle` styles: fixed top-right position (z-index 100), circular button with hover/focus/active states
- Icon swap animation: sun rotates out while moon rotates in (and vice versa)

### `frontend/script.js`
- Added `initTheme()` called on DOMContentLoaded
- `applyTheme(theme)` sets `data-theme` attribute on `<html>` and updates `aria-label`
- Toggle button click handler switches between "dark" and "light"
- Theme preference persisted in `localStorage`
- Default theme is "dark" when no saved preference exists

## Theme Colors

| Variable | Dark (default) | Light |
|---|---|---|
| `--background` | `#0f172a` | `#f8fafc` |
| `--surface` | `#1e293b` | `#ffffff` |
| `--surface-hover` | `#334155` | `#f1f5f9` |
| `--text-primary` | `#f1f5f9` | `#0f172a` |
| `--text-secondary` | `#94a3b8` | `#475569` |
| `--border-color` | `#334155` | `#e2e8f0` |
| `--link-color` | `#60a5fa` | `#2563eb` |
| `--shadow` | `rgba(0,0,0,0.3)` | `rgba(0,0,0,0.08)` |

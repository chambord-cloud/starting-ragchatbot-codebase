"""API endpoint tests for the FastAPI RAG application."""

import pytest


class TestQueryEndpoint:
    """Tests for POST /api/query."""

    def test_successful_query_returns_answer_and_sources(self, client, mock_rag_system):
        mock_rag_system.query.return_value = (
            "Course materials explain MCP architecture.",
            [{"label": "MCP Course - Lesson 2", "url": "https://example.com/mcp/2"}],
        )

        response = client.post("/api/query", json={"query": "What is MCP?"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Course materials explain MCP architecture."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["label"] == "MCP Course - Lesson 2"
        assert data["sources"][0]["url"] == "https://example.com/mcp/2"
        assert data["session_id"].startswith("session_")

    def test_query_with_session_id_preserves_it(self, client, mock_rag_system):
        response = client.post(
            "/api/query",
            json={"query": "Hello", "session_id": "session_42"},
        )

        assert response.status_code == 200
        assert response.json()["session_id"] == "session_42"

    def test_query_without_session_id_creates_one(self, client):
        response = client.post("/api/query", json={"query": "Hello"})

        assert response.status_code == 200
        assert response.json()["session_id"].startswith("session_")

    def test_query_passed_to_rag_system(self, client, mock_rag_system):
        client.post("/api/query", json={"query": "What is computer use?"})

        call_args = mock_rag_system.query.call_args
        assert call_args.args[0] == "What is computer use?"

    def test_missing_query_returns_422(self, client):
        response = client.post("/api/query", json={})

        assert response.status_code == 422

    def test_invalid_json_returns_422(self, client):
        response = client.post("/api/query", content="not json")

        assert response.status_code == 422

    def test_empty_query_string_accepted(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("No question asked.", [])

        response = client.post("/api/query", json={"query": ""})

        assert response.status_code == 200
        assert response.json()["answer"] == "No question asked."

    def test_rag_system_exception_returns_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("Database down")

        response = client.post("/api/query", json={"query": "test"})

        assert response.status_code == 500
        assert "Database down" in response.json()["detail"]

    def test_response_includes_correct_content_type(self, client):
        response = client.post("/api/query", json={"query": "test"})

        assert response.headers["content-type"] == "application/json"


class TestCoursesEndpoint:
    """Tests for GET /api/courses."""

    def test_returns_course_list(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 3,
            "courses": [
                {"title": "Course A", "course_link": "https://a.com"},
                {"title": "Course B", "course_link": None},
                {"title": "Course C", "course_link": "https://c.com"},
            ],
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert len(data["courses"]) == 3
        assert data["courses"][0]["title"] == "Course A"
        assert data["courses"][0]["course_link"] == "https://a.com"
        assert data["courses"][1]["course_link"] is None

    def test_empty_courses(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "courses": [],
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["courses"] == []

    def test_courses_exception_returns_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("ChromaDB error")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "ChromaDB error" in response.json()["detail"]

    def test_courses_response_content_type(self, client):
        response = client.get("/api/courses")

        assert response.headers["content-type"] == "application/json"

    def test_courses_endpoint_not_post(self, client):
        """POST to /api/courses should return 405 Method Not Allowed."""
        response = client.post("/api/courses", json={})
        assert response.status_code == 405

    def test_query_endpoint_not_get(self, client):
        """GET to /api/query should return 405 Method Not Allowed."""
        response = client.get("/api/query")
        assert response.status_code == 405


class TestSessionPersistence:
    """Tests verifying session state is maintained across requests."""

    def test_same_session_across_multiple_queries(self, client, mock_rag_system):
        responses = []
        for i, question in enumerate(["Q1", "Q2", "Q3"]):
            mock_rag_system.query.return_value = (f"Answer {i}", [])
            if i == 0:
                response = client.post("/api/query", json={"query": question})
                sid = response.json()["session_id"]
            else:
                response = client.post(
                    "/api/query",
                    json={"query": question, "session_id": sid},
                )
            responses.append(response)

        # All queries in the same session should succeed
        assert all(r.status_code == 200 for r in responses)

    def test_different_sessions_independent(self, client):
        r1 = client.post("/api/query", json={"query": "Q1"})
        r2 = client.post("/api/query", json={"query": "Q2"})

        assert r1.json()["session_id"] != r2.json()["session_id"]

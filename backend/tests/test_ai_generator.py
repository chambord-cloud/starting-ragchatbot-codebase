"""Tests for AIGenerator.generate_response() with mocked OpenAI client."""

import json
from unittest.mock import MagicMock, patch, ANY
import pytest
from ai_generator import AIGenerator


# ── Helpers ────────────────────────────────────────────────────────────────

def _msg(content=None, tool_calls=None):
    """Build a mock message object."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    return msg


def _response(content=None, tool_calls=None):
    """Build a mock ChatCompletion response with one choice."""
    choice = MagicMock()
    choice.message = _msg(content, tool_calls)
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _tool_call(name="search_course_content", arguments='{"query": "computer use"}'):
    """Build a mock tool_call object."""
    tc = MagicMock()
    tc.id = "call_abc123"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# ── Fixture ────────────────────────────────────────────────────────────────

@pytest.fixture
def ai_gen():
    """AIGenerator with dummy credentials — all API calls will be mocked."""
    return AIGenerator(
        api_key="test-key",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com"
    )


@pytest.fixture
def tool_manager():
    """Mock ToolManager that returns canned search results."""
    tm = MagicMock()
    tm.execute_tool.return_value = "Search results: computer use content..."
    tm.get_last_sources.return_value = []
    return tm


@pytest.fixture
def mock_create():
    """Patch the underlying OpenAI client's create method."""
    with patch.object(AIGenerator, "__init__", lambda self, *a, **kw: None):
        pass  # We'll patch at the instance level instead

    # Actually let's patch differently
    pass


# ── Tests ──────────────────────────────────────────────────────────────────

class TestGenerateResponse:
    """Tests for AIGenerator.generate_response() with mocked API."""

    def test_no_tools_returns_direct_answer(self, ai_gen):
        with patch.object(ai_gen.client.chat.completions, "create") as mock_create:
            mock_create.return_value = _response(content="Paris is the capital of France.")

            result = ai_gen.generate_response(query="What is the capital of France?")

            assert result == "Paris is the capital of France."
            # Verify no tools or tool_choice in the API call
            call_kwargs = mock_create.call_args.kwargs
            assert "tools" not in call_kwargs
            assert "tool_choice" not in call_kwargs

    def test_with_tools_no_tool_call(self, ai_gen):
        """Model receives tools but chooses to answer directly."""
        tools = [{"type": "function", "function": {"name": "search_course_content"}}]

        with patch.object(ai_gen.client.chat.completions, "create") as mock_create:
            mock_create.return_value = _response(content="I'll answer from my knowledge.")

            result = ai_gen.generate_response(query="Hello", tools=tools)

            assert result == "I'll answer from my knowledge."
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["tools"] == tools
            assert call_kwargs["tool_choice"] == "auto"

    def test_tool_call_flow_two_api_calls(self, ai_gen, tool_manager):
        """Model calls a tool → execute → follow-up → final answer."""
        tools = [{"type": "function", "function": {"name": "search_course_content"}}]

        with patch.object(ai_gen.client.chat.completions, "create") as mock_create:
            # First call: model requests tool
            # Second call: model gives final answer
            mock_create.side_effect = [
                _response(tool_calls=[_tool_call()]),
                _response(content="The course covers computer use with Anthropic's API.")
            ]

            result = ai_gen.generate_response(
                query="What is computer use?",
                tools=tools,
                tool_manager=tool_manager
            )

            # Two API calls should have been made
            assert mock_create.call_count == 2
            assert result == "The course covers computer use with Anthropic's API."

            # First call should have tools
            first_call_kwargs = mock_create.call_args_list[0].kwargs
            assert first_call_kwargs["tools"] == tools

    def test_tool_call_executed_with_correct_arguments(self, ai_gen, tool_manager):
        """Tool call arguments are parsed and forwarded correctly."""
        tools = [{"type": "function", "function": {"name": "search_course_content"}}]

        with patch.object(ai_gen.client.chat.completions, "create") as mock_create:
            mock_create.side_effect = [
                _response(tool_calls=[
                    _tool_call(arguments='{"query": "MCP architecture", "course_name": "MCP"}')
                ]),
                _response(content="MCP stands for Model Context Protocol.")
            ]

            ai_gen.generate_response(
                query="What is MCP architecture?",
                tools=tools,
                tool_manager=tool_manager
            )

            tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="MCP architecture",
                course_name="MCP"
            )

    def test_conversation_history_appended_to_system(self, ai_gen):
        """Conversation history is injected into the system prompt."""
        with patch.object(ai_gen.client.chat.completions, "create") as mock_create:
            mock_create.return_value = _response(content="Answer.")

            ai_gen.generate_response(
                query="Follow-up question?",
                conversation_history="User: What is MCP?\nAssistant: MCP is..."
            )

            call_kwargs = mock_create.call_args.kwargs
            system_msg = call_kwargs["messages"][0]
            assert system_msg["role"] == "system"
            assert "Previous conversation:" in system_msg["content"]
            assert "User: What is MCP?" in system_msg["content"]

    def test_tool_result_appended_as_tool_role_message(self, ai_gen, tool_manager):
        """After tool execution, result is added as role='tool' message."""
        tools = [{"type": "function", "function": {"name": "search_course_content"}}]

        with patch.object(ai_gen.client.chat.completions, "create") as mock_create:
            tool_call = _tool_call(arguments='{"query": "test"}')
            mock_create.side_effect = [
                _response(tool_calls=[tool_call]),
                _response(content="Final synthesis.")
            ]

            ai_gen.generate_response(
                query="test query",
                tools=tools,
                tool_manager=tool_manager
            )

            # Check second call's messages include a tool role message
            second_call_messages = mock_create.call_args_list[1].kwargs["messages"]
            tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
            assert len(tool_messages) == 1
            assert tool_messages[0]["tool_call_id"] == tool_call.id
            assert "Search results" in tool_messages[0]["content"]

    def test_multiple_tool_calls_in_one_message(self, ai_gen, tool_manager):
        """Handle multiple tool_calls in a single assistant message."""
        tools = [{"type": "function", "function": {"name": "t1"}},
                 {"type": "function", "function": {"name": "t2"}}]

        with patch.object(ai_gen.client.chat.completions, "create") as mock_create:
            mock_create.side_effect = [
                _response(tool_calls=[
                    _tool_call(name="search_course_content",
                               arguments='{"query": "MCP"}'),
                    _tool_call(name="get_course_outline",
                               arguments='{"course_name": "MCP"}')
                ]),
                _response(content="Combined answer.")
            ]

            result = ai_gen.generate_response(
                query="Tell me about MCP",
                tools=tools,
                tool_manager=tool_manager
            )

            assert result == "Combined answer."
            assert tool_manager.execute_tool.call_count == 2

    def test_empty_content_returns_empty_string(self, ai_gen):
        """When model returns None content (and no tool calls), return ''."""
        with patch.object(ai_gen.client.chat.completions, "create") as mock_create:
            mock_create.return_value = _response(content=None)

            result = ai_gen.generate_response(query="Bad query")
            assert result == ""

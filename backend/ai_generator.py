import json
from typing import List, Optional, Dict, Any
from openai import OpenAI


class AIGenerator:
    """Handles interactions with DeepSeek API for generating responses"""

    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to search tools for course information.

Available Tools:
1. search_course_content — Search course materials for specific topics, concepts, or detailed content.
2. get_course_outline — Get a complete course outline with title, link, and full lesson list.

Tool Usage:
- Use search_course_content **only** for questions about specific course content or detailed educational materials
- Use get_course_outline when the user asks about course structure, lesson list, course overview, or outline
- **Up to 2 sequential tool-calling rounds per query**: In each round you may call one or more tools simultaneously. After receiving results, decide whether you need another round before answering.
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course-specific questions**: Use the appropriate tool, then answer
- **Outline queries**: When returning a course outline, include the course title, course link, the number of lessons, and each lesson's number and title
- **Course mentions**: When mentioning a course by name in your answer, always hyperlink it in markdown if you know the course link (e.g., [Course Title](https://...))
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional sequential tool usage.

        Supports up to 2 rounds of tool calling. Each round the model sees
        previous tool results and can request additional tools before answering.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use (OpenAI format)
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]

        api_params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 800
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**api_params)
        message = response.choices[0].message

        if not message.tool_calls or not tool_manager:
            return message.content or ""

        max_rounds = 2
        for _ in range(max_rounds):
            messages.append(message)

            for tool_call in message.tool_calls:
                try:
                    result = tool_manager.execute_tool(
                        tool_call.function.name,
                        **json.loads(tool_call.function.arguments)
                    )
                except Exception as e:
                    result = f"Error executing {tool_call.function.name}: {str(e)}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

            api_params = {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 800,
                "tools": tools,
                "tool_choice": "auto"
            }

            response = self.client.chat.completions.create(**api_params)
            message = response.choices[0].message

            if not message.tool_calls:
                return message.content or ""

        # Max rounds reached — force final answer without tools
        final_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=800
        )

        return final_response.choices[0].message.content or ""

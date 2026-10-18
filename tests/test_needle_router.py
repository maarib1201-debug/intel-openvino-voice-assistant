from __future__ import annotations

import unittest

from v2v import tools
from v2v.needle_router import NeedleRouter


class FakeNeedleAgent:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def complete(self, _text: str) -> dict:
        return self.responses.pop(0)


def call(name: str, arguments: dict, *, ungrounded: list[str] | None = None) -> dict:
    return {
        "function_calls": [{"name": name, "arguments": arguments}],
        "validation": {"ungrounded": ungrounded or []},
    }


class NeedleRouterTests(unittest.TestCase):
    def test_messy_time_utterance_uses_deterministic_router(self) -> None:
        decision = tools.deterministic_tool_decision(
            "Um no what's the time in Dubai right now"
        )
        self.assertEqual(
            decision,
            tools.ToolDecision(
                tool="get_current_time", arguments={"timezone": "Dubai"}
            ),
        )

    def test_contextual_followup_stays_on_previous_tool(self) -> None:
        decision = tools.deterministic_tool_decision(
            "What about in Dubai?", previous_tool="get_current_time"
        )
        self.assertEqual(
            decision,
            tools.ToolDecision(
                tool="get_current_time", arguments={"timezone": "Dubai"}
            ),
        )

    def test_valid_tools_preserve_semantically_valid_arguments(self) -> None:
        agent = FakeNeedleAgent(
            [
                call("get_weather", {"location": "Dubai"}),
                call("get_current_time", {"timezone": "London"}),
                call("calculator", {"expression": "45 times 19"}),
            ]
        )
        router = NeedleRouter(agent=agent)

        self.assertEqual(
            router.route("What's the weather in Dubai?").model_dump(),
            {"tool": "get_weather", "arguments": {"location": "Dubai"}},
        )
        self.assertEqual(
            router.route("What time is it in London?").model_dump(),
            {"tool": "get_current_time", "arguments": {"timezone": "London"}},
        )
        self.assertEqual(
            router.route("Calculate 45 times 19").model_dump(),
            {"tool": "calculator", "arguments": {"expression": "45 times 19"}},
        )
        self.assertEqual(agent.reset_count, 3)

    def test_non_math_calculator_calls_become_no_tool(self) -> None:
        agent = FakeNeedleAgent(
            [
                call("calculator", {"expression": "tell me a joke"}),
                call("calculator", {"expression": "help me find a job"}),
                call("calculator", {"expression": "give me a good job"}),
            ]
        )
        router = NeedleRouter(agent=agent)

        for text in ("Tell me a joke", "Help me find a job", "Give me a good job"):
            with self.subTest(text=text):
                self.assertEqual(
                    router.route(text).model_dump(),
                    {"tool": None, "arguments": {}},
                )

    def test_ungrounded_call_becomes_no_tool(self) -> None:
        agent = FakeNeedleAgent(
            [call("get_weather", {"location": "Dubai"}, ungrounded=["get_weather.location"])]
        )
        router = NeedleRouter(agent=agent)

        self.assertEqual(
            router.route("What's the weather in Dubai?").model_dump(),
            {"tool": None, "arguments": {}},
        )


if __name__ == "__main__":
    unittest.main()

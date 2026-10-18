from __future__ import annotations

import json
import unittest
from datetime import timezone
from typing import Any
from unittest.mock import patch

import requests

from v2v import tools


class FakeResponse:
    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
        content_type: str = "application/json",
    ) -> None:
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self._body = json.dumps(payload).encode()
        self.closed = False

    def iter_content(self, chunk_size: int):
        for offset in range(0, len(self._body), chunk_size):
            yield self._body[offset : offset + chunk_size]

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def sample_geocode() -> dict[str, Any]:
    return {
        "results": [
            {
                "name": "London",
                "country": "United Kingdom",
                "latitude": 51.5085,
                "longitude": -0.1257,
                "timezone": "Europe/London",
            }
        ]
    }


def sample_dubai_geocode() -> dict[str, Any]:
    return {
        "results": [
            {
                "name": "Dubai",
                "country": "United Arab Emirates",
                "latitude": 25.2048,
                "longitude": 55.2708,
                "timezone": "Asia/Dubai",
            }
        ]
    }


def sample_forecast() -> dict[str, Any]:
    return {
        "timezone": "Europe/London",
        "current": {
            "temperature_2m": 18.4,
            "apparent_temperature": 17.2,
            "relative_humidity_2m": 64,
            "weather_code": 2,
            "wind_speed_10m": 12.5,
        },
        "daily": {
            "temperature_2m_max": [21.0],
            "temperature_2m_min": [12.6],
            "precipitation_probability_max": [30],
        },
    }


class ToolTests(unittest.TestCase):
    def test_deterministic_fast_router_acceptance_cases(self) -> None:
        cases = {
            "What is Python?": {"tool": None, "arguments": {}},
            "What is 2026?": {"tool": None, "arguments": {}},
            "What is the temperature of the sun?": {"tool": None, "arguments": {}},
            "What time is it?": {"tool": "get_current_time", "arguments": {"timezone": None}},
            "What is 239 times 47?": {"tool": "calculator", "arguments": {"expression": "239*47"}},
            "What is forty five times nineteen?": {
                "tool": "calculator",
                "arguments": {"expression": "45*19"},
            },
            "What's the weather in London?": {"tool": "get_weather", "arguments": {"location": "London"}},
            "What's the weather in London and should I wear a jacket?": {
                "tool": "get_weather",
                "arguments": {"location": "London"},
            },
            "What's the weather?": {"tool": "get_weather", "arguments": {"location": None}},
            "Hey, what's the weather in Dubai?": {
                "tool": "get_weather",
                "arguments": {"location": "Dubai"},
            },
            "Um, what is the weather in Hyderabad?": {
                "tool": "get_weather",
                "arguments": {"location": "Hyderabad"},
            },
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                decision = tools.deterministic_tool_decision(prompt)
                self.assertIsNotNone(decision)
                self.assertEqual(decision.model_dump(), expected)

    def test_deterministic_time_router_extracts_places(self) -> None:
        cases = {
            "Okay, what is the time in Dubai right now?": "Dubai",
            "Alright, what time is it in London?": "London",
            "Um no was the time in Dubai Right Now.": "Dubai",
            "Okay no what is the time in London now?": "London",
            "Uh what's the time in Barcelona right now?": "Barcelona",
            "Well um what time is it in Tokyo?": "Tokyo",
            "Oh, what's the time in Dubai right now?": "Dubai",
            "Oh um what's the time in Dubai right now?": "Dubai",
            "Okay oh no what's the time in Dubai?": "Dubai",
            "In Dubai, what time it is.": "Dubai",
            "In Dubai, what time is it?": "Dubai",
            "Dubai, what time is it?": "Dubai",
            "What time in Dubai?": "Dubai",
            "What's the time in Dubai right now?": "Dubai",
            "Can you tell me the time in London?": "London",
            "Could you tell me what time it is in New York right now?": "New York",
            "Tell me the current time in Tokyo": "Tokyo",
            "Dubai time": "Dubai",
            "What is Dubai time?": "Dubai",
            "What time is it in Asia/Dubai?": "Asia/Dubai",
        }
        for prompt, location in cases.items():
            with self.subTest(prompt=prompt):
                decision = tools.deterministic_tool_decision(prompt)
                self.assertEqual(
                    decision,
                    tools.ToolDecision(
                        tool="get_current_time", arguments={"timezone": location}
                    ),
                )

    def test_ordinary_time_sentence_does_not_route_to_time_tool(self) -> None:
        decision = tools.deterministic_tool_decision(
            "I have time in Dubai after work."
        )
        self.assertIsNone(decision)

    def test_context_aware_followups_reuse_previous_tool(self) -> None:
        cases = [
            ("get_weather", "What about in Kuala Lumpur?", {"location": "Kuala Lumpur"}),
            ("get_current_time", "What about in London?", {"timezone": "London"}),
            ("get_weather", "No, I meant Hyderabad.", {"location": "Hyderabad"}),
        ]
        for previous_tool, prompt, arguments in cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    tools.deterministic_tool_decision(prompt, previous_tool),
                    tools.ToolDecision(tool=previous_tool, arguments=arguments),
                )

    def test_location_followup_without_previous_tool_context_is_not_a_tool(self) -> None:
        decision = tools.deterministic_tool_decision("What about Dubai?")
        self.assertEqual(decision, tools.ToolDecision(tool=None, arguments={}))

    def test_current_time_resolves_city_and_accepts_iana_timezone(self) -> None:
        with (
            patch.object(tools._weather_client, "get_timezone", return_value="Asia/Dubai") as get_timezone,
            patch.object(tools, "ZoneInfo", return_value=timezone.utc),
        ):
            city_result = tools.execute_tool("get_current_time", {"timezone": "Dubai"})
        get_timezone.assert_called_once_with("Dubai", tools._DEFAULT_WEATHER_TIMEOUT_S)
        self.assertEqual(city_result["timezone"], "Asia/Dubai")

        with patch.object(tools, "ZoneInfo", return_value=timezone.utc):
            iana_result = tools.execute_tool(
                "get_current_time", {"timezone": "Asia/Dubai"}
            )
        self.assertEqual(iana_result["timezone"], "Asia/Dubai")

    def test_registry_has_exactly_the_three_approved_tools(self) -> None:
        self.assertEqual(
            set(tools.TOOL_REGISTRY),
            {"get_current_time", "calculator", "get_weather"},
        )
        self.assertTrue(all(spec.description for spec in tools.TOOL_REGISTRY.values()))
        self.assertEqual(len(tools.TOOL_DEFINITIONS), 3)

    def test_calculator_exact_value_and_natural_formatter(self) -> None:
        result = tools.execute_tool("calculator", {"expression": "239*47"})
        self.assertEqual(result, {"expression": "239*47", "value": "11233"})
        self.assertEqual(
            tools.format_tool_result("calculator", {"expression": "239*47"}, result),
            "239 times 47 is 11,233.",
        )

    def test_calculator_rejects_malicious_or_unbounded_input(self) -> None:
        expressions = [
            "__import__('os').system('whoami')",
            "open('secret')",
            "(1).__class__",
            "[1, 2][0]",
            "1 // 2",
            "1 % 2",
            "1 / 0",
            "2 ** 13",
            "2 ** 0.5",
            "9" * 65,
            "+".join(["1"] * 40),
        ]
        for expression in expressions:
            with self.subTest(expression=expression):
                with self.assertRaises(tools.ToolError) as raised:
                    tools.execute_tool("calculator", {"expression": expression})
                self.assertEqual(
                    raised.exception.user_message,
                    "I couldn't calculate that expression.",
                )
                self.assertNotIn(expression, raised.exception.user_message)

    def test_spoken_math_normalizes_only_calculator_intents(self) -> None:
        cases = {
            "What is 45 times 19": {"tool": "calculator", "arguments": {"expression": "45*19"}},
            "What is forty five times nineteen?": {
                "tool": "calculator",
                "arguments": {"expression": "45*19"},
            },
            "what is forty-five times nineteen": {
                "tool": "calculator",
                "arguments": {"expression": "45*19"},
            },
            "calculate twenty plus seven": {
                "tool": "calculator",
                "arguments": {"expression": "20+7"},
            },
            "What is one hundred divided by five": {
                "tool": "calculator",
                "arguments": {"expression": "100/5"},
            },
            "What is twelve minus four?": {
                "tool": "calculator",
                "arguments": {"expression": "12-4"},
            },
            "What is six point five times two": {
                "tool": "calculator",
                "arguments": {"expression": "6.5*2"},
            },
            "Compute twenty plus seven": {
                "tool": "calculator",
                "arguments": {"expression": "20+7"},
            },
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                decision = tools.deterministic_tool_decision(prompt)
                self.assertEqual(decision.model_dump(), expected)
                if expected["tool"] == "calculator":
                    tools.calculate(expected["arguments"]["expression"])

    def test_spoken_number_words_in_non_math_do_not_route_to_calculator(self) -> None:
        prompts = [
            "What is Python?",
            "What is forty five?",
            "Tell me a joke about twenty people",
            "I need a good job",
            "Give me a good job",
            "What is twenty questions",
            "What is the temperature of the sun?",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                decision = tools.deterministic_tool_decision(prompt)
                self.assertNotEqual(getattr(decision, "tool", None), "calculator")

    def test_normalize_spoken_math_expression_direct(self) -> None:
        self.assertEqual(tools.normalize_spoken_math_expression("forty five times nineteen"), "45*19")
        self.assertEqual(tools.normalize_spoken_math_expression("twenty plus seven"), "20+7")
        self.assertEqual(tools.normalize_spoken_math_expression("six point five times two"), "6.5*2")
        self.assertIsNone(tools.normalize_spoken_math_expression("forty five"))
        self.assertIsNone(tools.normalize_spoken_math_expression("twenty plus people"))
        self.assertIsNone(tools.normalize_spoken_math_expression("Python"))

    def test_calculator_decimal_arithmetic_avoids_float_artifacts(self) -> None:
        self.assertEqual(tools.calculate("0.1 + 0.2"), "0.3")
        self.assertEqual(tools.calculate("-(2 + 3) * 4"), "-20")

    def test_time_formatter_is_deterministic(self) -> None:
        formatted = tools.format_tool_result(
            "get_current_time",
            {"timezone": "Europe/London"},
            {"timezone": "Europe/London", "datetime": "2026-09-18T14:05:00+01:00"},
        )
        self.assertEqual(
            formatted,
            "In Europe/London, it's 2:05 PM.",
        )

    def test_weather_success_uses_hard_coded_https_endpoints(self) -> None:
        client = tools.WeatherClient()
        session = FakeSession(
            [FakeResponse(sample_geocode()), FakeResponse(sample_forecast())]
        )
        with patch.object(client, "_session", return_value=session):
            result = client.get_weather("London", 3.0)

        self.assertEqual(result.location, "London")
        self.assertEqual(result.condition, "partly cloudy skies")
        self.assertEqual(result.temperature_c, 18.4)
        self.assertEqual(result.today_high_c, 21.0)
        self.assertEqual(
            [call["url"] for call in session.calls],
            [
                "https://geocoding-api.open-meteo.com/v1/search",
                "https://api.open-meteo.com/v1/forecast",
            ],
        )
        for call in session.calls:
            self.assertFalse(call["allow_redirects"])
            self.assertTrue(call["stream"])
            self.assertTrue(call["verify"])
            self.assertIsInstance(call["timeout"], tuple)

    def test_get_timezone_uses_geocoder_only(self) -> None:
        client = tools.WeatherClient()
        session = FakeSession([FakeResponse(sample_dubai_geocode())])
        with patch.object(client, "_session", return_value=session):
            timezone_name = client.get_timezone("Dubai", 3.0)

        self.assertEqual(timezone_name, "Asia/Dubai")
        self.assertEqual(
            [call["url"] for call in session.calls],
            ["https://geocoding-api.open-meteo.com/v1/search"],
        )

    def test_weather_formatter_uses_validated_values(self) -> None:
        result = tools.WeatherResult(
            location="London",
            country="United Kingdom",
            latitude=51.5,
            longitude=-0.1,
            timezone="Europe/London",
            temperature_c=18.4,
            apparent_temperature_c=17.2,
            relative_humidity_percent=64,
            weather_code=2,
            condition="partly cloudy skies",
            wind_speed_kmh=12.5,
            today_high_c=21.0,
            today_low_c=12.6,
            precipitation_probability_percent=30,
        ).model_dump(mode="json")
        self.assertEqual(
            tools.format_tool_result("get_weather", {"location": "London"}, result),
            "It's 18.4 degrees Celsius with partly cloudy skies in London right now. "
            "Today's high is 21 degrees, with a 30 percent chance of rain.",
        )

    def test_weather_sanitizes_failures(self) -> None:
        cases = [
            ([FakeResponse({}, status_code=500)], "weather_provider_error"),
            ([FakeResponse({"results": []})], "location_not_found"),
            (
                [
                    FakeResponse(sample_geocode()),
                    FakeResponse({"current": {}, "daily": {}}),
                ],
                "weather_provider_error",
            ),
        ]
        for responses, expected_code in cases:
            with self.subTest(code=expected_code):
                client = tools.WeatherClient()
                session = FakeSession(responses)
                with patch.object(client, "_session", return_value=session):
                    with self.assertRaises(tools.ToolError) as raised:
                        client.get_weather("private-location", 3.0)
                self.assertEqual(raised.exception.code, expected_code)
                self.assertNotIn("private-location", raised.exception.user_message)
                self.assertNotIn("http", raised.exception.user_message.lower())

    def test_weather_timeout_is_sanitized(self) -> None:
        client = tools.WeatherClient()
        session = FakeSession([requests.Timeout("provider secret")])
        with patch.object(client, "_session", return_value=session):
            with self.assertRaises(tools.ToolError) as raised:
                client.get_weather("London", 3.0)
        self.assertEqual(raised.exception.code, "weather_timeout")
        self.assertEqual(
            raised.exception.user_message, "I couldn't get the weather in time."
        )

    def test_missing_weather_location_never_calls_provider(self) -> None:
        with patch.object(
            tools._weather_client,
            "get_weather",
            side_effect=AssertionError("provider must not be called"),
        ):
            with self.assertRaises(tools.MissingLocationError) as raised:
                tools.execute_tool("get_weather", {"location": "  "})
        self.assertEqual(raised.exception.user_message, "Which city should I check?")

    def test_acknowledgement_is_weather_only(self) -> None:
        self.assertEqual(
            tools.acknowledgement_for("get_weather"), "Checking the weather."
        )
        self.assertIsNone(tools.acknowledgement_for("calculator"))
        self.assertIsNone(tools.acknowledgement_for("get_current_time"))
        self.assertIsNone(tools.acknowledgement_for("not_a_tool"))

    def test_validate_arguments_returns_normalized_plain_dict(self) -> None:
        validated = tools.validate_tool_arguments(
            "get_weather", {"location": "  New   York "}
        )
        self.assertIs(type(validated), dict)
        self.assertEqual(validated, {"location": "New York"})

    def test_router_malformed_or_invalid_decisions_fail_closed(self) -> None:
        inputs = [
            "not json",
            "{}",
            '{"tool":"unknown","arguments":{}}',
            '{"tool":"calculator","arguments":{"expression":4}}',
            '{"tool":"calculator","arguments":{"expression":"2+2"},"extra":true}',
            '{"tool":null,"arguments":{"unexpected":true}}',
            '{"tool":"calculator","arguments":{"expression":"2+2"}} trailing',
        ]
        for raw in inputs:
            with self.subTest(raw=raw):
                self.assertEqual(
                    tools.parse_tool_decision(raw),
                    tools.ToolDecision(tool=None, arguments={}),
                )

    def test_router_valid_decision_and_schema(self) -> None:
        decision = tools.parse_tool_decision(
            ' {"tool":"calculator","arguments":{"expression":" 239*47 "}} '
        )
        self.assertEqual(
            decision,
            tools.ToolDecision(tool="calculator", arguments={"expression": "239*47"}),
        )
        schema = json.loads(tools.router_schema_json())
        self.assertEqual(schema, tools.ROUTER_JSON_SCHEMA)
        self.assertFalse(schema["additionalProperties"])

    def test_requires_synthesis_is_tool_agnostic(self) -> None:
        simple = "Calculate 239*47."
        compound = "Calculate 239*47 and explain the method."
        self.assertFalse(tools.requires_synthesis("calculator", simple))
        self.assertTrue(tools.requires_synthesis("calculator", compound))
        self.assertTrue(tools.requires_synthesis("get_weather", compound))
        self.assertTrue(
            tools.requires_synthesis(
                "get_current_time", "What time is it—am I too late to call?"
            )
        )
        self.assertTrue(
            tools.requires_synthesis("get_weather", "Will I need a jacket in London?")
        )


if __name__ == "__main__":
    unittest.main()

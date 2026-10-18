from __future__ import annotations

import re
import threading
import time
from typing import Any

try:
    import needle
except ImportError:  # Optional fallback; deterministic tools and Qwen still work.
    needle = None

from .tools import ToolDecision


NEEDLE_TOOLS = [
    {
        "name": "get_current_time",
        "description": (
            "Use ONLY when the user asks for the current local time or clock time "
            "in a specific city, place, or timezone. Do NOT use for weather, "
            "temperature, forecast, jobs, jokes, or general conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": (
                        "The specific city, place, or IANA timezone whose current "
                        "local clock time the user requested."
                    ),
                }
            },
            "required": ["timezone"],
        },
    },
    {
        "name": "get_weather",
        "description": (
            "Use ONLY when the user asks about weather, temperature, heat, rain, "
            "conditions, or forecast for a place. Do NOT use for clock time, "
            "jobs, jokes, calculations, or general conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "The specific city or place for which the user asks about "
                        "weather, temperature, heat, rain, conditions, or forecast."
                    ),
                }
            },
            "required": ["location"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Use ONLY for explicit arithmetic or mathematical calculations. "
            "Preserve the complete mathematical expression from the user's request."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "The complete mathematical expression from the user, "
                        "including every number and operation. Do not omit operands."
                    ),
                }
            },
            "required": ["expression"],
        },
    },
]

_MATH_OPERATOR = re.compile(
    r"[+\-*/^]|\b(?:plus|minus|times|multiplied|divided|over)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _call_value(call: Any, key: str) -> Any:
    if isinstance(call, dict):
        return call.get(key)
    return getattr(call, key, None)


def _needle_validation_passes(result: dict[str, Any]) -> bool:
    validation = result.get("validation") or {}
    if not isinstance(validation, dict):
        return True
    return not (
        validation.get("ungrounded")
        or validation.get("invalid")
        or validation.get("errors")
    )


def _validate_arguments(tool: str, arguments: Any) -> tuple[bool, str]:
    argument_name = {
        "get_current_time": "timezone",
        "get_weather": "location",
        "calculator": "expression",
    }.get(tool)
    if argument_name is None:
        return False, f"unknown Needle tool {tool!r}"
    value = arguments.get(argument_name) if isinstance(arguments, dict) else None
    if not isinstance(value, str) or not value.strip():
        return False, f"missing non-empty {argument_name} argument"
    if tool == "calculator":
        if len(_NUMBER.findall(value)) < 2 or _MATH_OPERATOR.search(value) is None:
            return False, "calculator expression is not real arithmetic"
    return True, ""


class NeedleRouter:
    """Single-session Needle fallback router with no conversation ownership."""

    def __init__(self, agent: Any | None = None) -> None:
        if agent is not None:
            self._agent = agent
        elif needle is None:
            raise RuntimeError("Needle fallback is not installed; install its optional runtime to enable it")
        else:
            self._agent = needle.Needle(tools=NEEDLE_TOOLS)
        self._lock = threading.Lock()

    def route(self, user_text: str, cancel: Any | None = None) -> ToolDecision:
        with self._lock:
            started = time.perf_counter()
            raw_calls: list[Any] = []
            effective = "no-tool"
            rejection_reason = ""
            try:
                if cancel is not None and cancel.is_set():
                    rejection_reason = "cancelled before Needle routing"
                else:
                    self._agent.reset()
                    result = self._agent.complete(user_text)
                    raw_calls = result.get("function_calls", []) or []
                    if not raw_calls:
                        rejection_reason = "Needle returned no function calls"
                    elif not _needle_validation_passes(result):
                        rejection_reason = f"Needle validation rejected call: {result.get('validation')}"
                    elif len(raw_calls) != 1:
                        rejection_reason = "Needle returned more than one function call"
                    else:
                        call = raw_calls[0]
                        tool = _call_value(call, "name")
                        arguments = _call_value(call, "arguments") or {}
                        valid, rejection_reason = _validate_arguments(tool, arguments)
                        if valid:
                            effective = tool
            except Exception as exc:
                rejection_reason = f"Needle routing error: {type(exc).__name__}: {exc}"

            latency_ms = (time.perf_counter() - started) * 1000.0
            decision = (
                ToolDecision(tool=effective, arguments=_call_value(raw_calls[0], "arguments") or {})
                if effective != "no-tool" and raw_calls
                else ToolDecision(tool=None, arguments={})
            )
            print(f"needle_router_used=true raw_function_calls={raw_calls!r}", flush=True)
            print(f"needle_router_effective_decision={decision.model_dump()!r}", flush=True)
            print(f"needle_router_latency_ms={latency_ms:.2f}", flush=True)
            if rejection_reason:
                print(f"needle_router_rejection_reason={rejection_reason}", flush=True)
            return decision

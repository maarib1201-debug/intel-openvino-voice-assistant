"""Validated, deterministic tools used by the voice assistant.

This module deliberately contains no model or session orchestration.  It owns the
public schemas, validation, execution, deterministic formatting, and the small
amount of provider I/O needed by the three registered tools.
"""
from __future__ import annotations

import ast
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import (
    Decimal,
    DecimalException,
    DivisionByZero,
    InvalidOperation,
    localcontext,
)
from typing import Any, Callable, Literal, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from requests.adapters import HTTPAdapter


ToolName = Literal["get_current_time", "calculator", "get_weather"]
_IANA_TIMEZONE = re.compile(
    r"^(?:UTC|[A-Za-z][A-Za-z0-9._+-]*(?:/[A-Za-z0-9][A-Za-z0-9._+-]*)+)$"
)


class ToolError(RuntimeError):
    """A user-safe tool failure; provider details must never enter the message."""

    def __init__(self, user_message: str, *, code: str = "tool_error") -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.code = code


class MissingLocationError(ToolError):
    def __init__(self) -> None:
        super().__init__("Which city should I check?", code="missing_location")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CurrentTimeArguments(_StrictModel):
    timezone: str | None = Field(
        default=None,
        max_length=64,
        description="Optional IANA timezone or city/place name, for example Europe/London or Dubai.",
    )

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()

        if not value or any(ord(char) < 32 for char in value):
            raise ValueError("timezone or location must not be empty")

        if _IANA_TIMEZONE.fullmatch(value) is not None and ".." not in value:
            return value

        if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{0,63}", value):
            return value

        raise ValueError("timezone must be an IANA timezone or a valid place name")


class CurrentTimeResult(_StrictModel):
    timezone: str = Field(min_length=1, max_length=64)
    datetime: str = Field(min_length=1, max_length=64)

    @field_validator("datetime")
    @classmethod
    def validate_datetime(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("datetime must include an offset")
        return value


class CalculatorArguments(_StrictModel):
    expression: str = Field(
        min_length=1,
        max_length=128,
        description="Arithmetic using numbers, parentheses, +, -, *, /, and **.",
    )

    @field_validator("expression")
    @classmethod
    def normalize_expression(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("expression is empty")
        return value


class CalculatorResult(_StrictModel):
    expression: str = Field(min_length=1, max_length=128)
    value: str = Field(min_length=1, max_length=256)

    @field_validator("value")
    @classmethod
    def validate_decimal(cls, value: str) -> str:
        number = Decimal(value)
        if not number.is_finite():
            raise ValueError("value must be finite")
        return value


class WeatherArguments(_StrictModel):
    location: str | None = Field(
        default=None,
        max_length=100,
        description="City or place name. Omit only when a configured default may be used.",
    )

    @field_validator("location")
    @classmethod
    def normalize_location(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        if not value:
            return None
        if any(ord(char) < 32 for char in value):
            raise ValueError("location contains control characters")
        return value


class WeatherResult(_StrictModel):
    location: str = Field(min_length=1, max_length=160)
    country: str | None = Field(default=None, max_length=100)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str = Field(min_length=1, max_length=64)
    temperature_c: float = Field(ge=-100, le=100)
    apparent_temperature_c: float = Field(ge=-120, le=120)
    relative_humidity_percent: int = Field(ge=0, le=100)
    weather_code: int = Field(ge=0, le=99)
    condition: str = Field(min_length=1, max_length=80)
    wind_speed_kmh: float = Field(ge=0, le=500)
    today_high_c: float = Field(ge=-100, le=100)
    today_low_c: float = Field(ge=-100, le=100)
    precipitation_probability_percent: int = Field(ge=0, le=100)


class ToolDecision(_StrictModel):
    tool: ToolName | None
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    name: ToolName
    description: str
    arguments_model: type[_StrictModel]
    result_model: type[_StrictModel]
    handler: Callable[[BaseModel, float | None], BaseModel]
    formatter: Callable[[BaseModel, BaseModel], str]
    acknowledgement: str | None = None

    def router_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.arguments_model.model_json_schema(),
        }


_MAX_EXPRESSION_LENGTH = 128
_MAX_SOURCE_DIGITS = 64
_MAX_AST_NODES = 64
_MAX_AST_DEPTH = 12
_MAX_POWER = 12
_MAX_ABS_VALUE = Decimal("1e100")
_MIN_NONZERO_ABS_VALUE = Decimal("1e-100")
_CALCULATOR_CHARS = re.compile(r"^[0-9eE+\-*/().\s]+$")
_NUMBER_CHARS = re.compile(r"[0-9]")
_SPOKEN_ONES = {
    "zero": 0,
    "nought": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_SPOKEN_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SPOKEN_OPERATORS = {"+", "-", "*", "/", "**", "(", ")"}


def _spoken_number_token(token: str) -> bool:
    lowered = token.lower()
    return lowered in _SPOKEN_ONES or lowered in _SPOKEN_TENS or lowered in {"hundred", "thousand", "point"}


def _take_spoken_number(tokens: list[str], index: int) -> tuple[str, int] | None:
    """Parse one spoken or digit number starting at index. Conservative: 0–9999 and simple decimals."""

    if index >= len(tokens):
        return None
    first = tokens[index]
    if re.fullmatch(r"\d+(?:\.\d+)?", first):
        return first, index + 1

    value = 0
    current = 0
    started = False
    decimal_digits: list[str] = []
    in_fraction = False
    i = index
    while i < len(tokens):
        token = tokens[i].lower()
        if token == "and" and started and not in_fraction:
            i += 1
            continue
        if token == "a" and not started and i + 1 < len(tokens) and tokens[i + 1].lower() in {"hundred", "thousand"}:
            current = 1
            started = True
            i += 1
            continue
        if token == "point" and started and not in_fraction:
            in_fraction = True
            i += 1
            continue
        if in_fraction:
            if token in _SPOKEN_ONES and _SPOKEN_ONES[token] <= 9:
                decimal_digits.append(str(_SPOKEN_ONES[token]))
                i += 1
                continue
            if re.fullmatch(r"\d", token):
                decimal_digits.append(token)
                i += 1
                continue
            break
        if token in _SPOKEN_ONES:
            ones = _SPOKEN_ONES[token]
            if ones >= 10 and current % 10:
                break
            current += ones
            started = True
            i += 1
            continue
        if token in _SPOKEN_TENS:
            if current % 100:
                break
            current += _SPOKEN_TENS[token]
            started = True
            i += 1
            continue
        if token == "hundred":
            if current == 0:
                current = 1
            if current > 19:
                break
            current *= 100
            started = True
            i += 1
            continue
        if token == "thousand":
            chunk = current if current else 1
            value += chunk * 1000
            current = 0
            started = True
            i += 1
            continue
        break
    if not started:
        return None
    value += current
    if in_fraction and not decimal_digits:
        return None
    text = str(value)
    if decimal_digits:
        text = f"{value}.{''.join(decimal_digits)}"
    return text, i


def normalize_spoken_math_expression(raw: str) -> str | None:
    """Convert a calculator-intent fragment of spoken arithmetic into digits and symbols.

    Returns None when the fragment is not a complete number-operator expression.
    Does not rewrite ordinary conversation.
    """

    text = " ".join((raw or "").strip().split())
    if not text:
        return None
    text = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", text)
    text = re.sub(r"\bmultiplied\s+by\b", "*", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdivided\s+by\b", "/", text, flags=re.IGNORECASE)
    text = re.sub(r"\bto\s+the\s+power\s+of\b", "**", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:times|x)\b", "*", text, flags=re.IGNORECASE)
    text = re.sub(r"\bplus\b", "+", text, flags=re.IGNORECASE)
    text = re.sub(r"\bminus\b", "-", text, flags=re.IGNORECASE)
    tokens = [token for token in re.split(r"\s+", text) if token]
    if not tokens:
        return None
    parts: list[str] = []
    expect_number = True
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if expect_number:
            if token == "-" and index + 1 < len(tokens):
                parsed = _take_spoken_number(tokens, index + 1)
                if parsed is None:
                    return None
                number, index = parsed
                parts.append("-" + number)
                expect_number = False
                continue
            parsed = _take_spoken_number(tokens, index)
            if parsed is None:
                return None
            number, index = parsed
            parts.append(number)
            expect_number = False
            continue
        if token in _SPOKEN_OPERATORS:
            parts.append(token)
            expect_number = token != ")"
            if token == ")":
                expect_number = False
            elif token == "(":
                expect_number = True
            else:
                expect_number = True
            index += 1
            continue
        return None
    if expect_number or not parts:
        return None
    expression = "".join(parts)
    if not any(char.isdigit() for char in expression):
        return None
    if not any(operator in expression for operator in "+-*/"):
        return None
    if _CALCULATOR_CHARS.fullmatch(expression) is None:
        return None
    return expression


def _check_decimal(value: Decimal) -> Decimal:
    if not value.is_finite():
        raise ToolError("I couldn't calculate that expression.", code="invalid_calculation")
    absolute = abs(value)
    if absolute > _MAX_ABS_VALUE:
        raise ToolError("I couldn't calculate that expression.", code="calculation_too_large")
    if value and absolute < _MIN_NONZERO_ABS_VALUE:
        raise ToolError("I couldn't calculate that expression.", code="calculation_too_small")
    return value


def _decimal_literal(node: ast.Constant, source: str) -> Decimal:
    if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
        raise ToolError("I couldn't calculate that expression.", code="invalid_expression")
    segment = ast.get_source_segment(source, node)
    if segment is None or "_" in segment:
        raise ToolError("I couldn't calculate that expression.", code="invalid_expression")
    try:
        return _check_decimal(Decimal(segment))
    except (DecimalException, ValueError) as exc:
        raise ToolError(
            "I couldn't calculate that expression.", code="invalid_expression"
        ) from exc


def _calculate_node(node: ast.AST, source: str) -> Decimal:
    if isinstance(node, ast.Expression):
        return _calculate_node(node.body, source)
    if isinstance(node, ast.Constant):
        return _decimal_literal(node, source)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _calculate_node(node.operand, source)
        return _check_decimal(value if isinstance(node.op, ast.UAdd) else -value)
    if not isinstance(node, ast.BinOp) or not isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
    ):
        raise ToolError("I couldn't calculate that expression.", code="invalid_expression")

    left = _calculate_node(node.left, source)
    right = _calculate_node(node.right, source)
    try:
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        elif isinstance(node.op, ast.Div):
            if not right:
                raise DivisionByZero
            value = left / right
        else:
            if right != right.to_integral_value() or abs(right) > _MAX_POWER:
                raise ToolError(
                    "I couldn't calculate that expression.", code="invalid_exponent"
                )
            value = left ** int(right)
    except ToolError:
        raise
    except (DecimalException, DivisionByZero, InvalidOperation, OverflowError) as exc:
        raise ToolError(
            "I couldn't calculate that expression.", code="invalid_calculation"
        ) from exc
    return _check_decimal(value)


def _ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 if not children else 1 + max(_ast_depth(child) for child in children)


def _decimal_text(value: Decimal) -> str:
    if not value:
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def calculate(expression: str) -> str:
    """Evaluate the deliberately small arithmetic language and return exact text."""

    expression = expression.strip()
    if (
        not expression
        or len(expression) > _MAX_EXPRESSION_LENGTH
        or len(_NUMBER_CHARS.findall(expression)) > _MAX_SOURCE_DIGITS
        or _CALCULATOR_CHARS.fullmatch(expression) is None
    ):
        raise ToolError("I couldn't calculate that expression.", code="invalid_expression")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, MemoryError) as exc:
        raise ToolError(
            "I couldn't calculate that expression.", code="invalid_expression"
        ) from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_AST_NODES or _ast_depth(tree) > _MAX_AST_DEPTH:
        raise ToolError("I couldn't calculate that expression.", code="expression_too_complex")
    try:
        with localcontext() as context:
            context.prec = 80
            context.Emax = 120
            context.Emin = -120
            return _decimal_text(_calculate_node(tree, expression))
    except ToolError:
        raise
    except (DecimalException, ArithmeticError, MemoryError) as exc:
        raise ToolError(
            "I couldn't calculate that expression.", code="invalid_calculation"
        ) from exc


def _execute_calculator(args: BaseModel, timeout_s: float | None) -> CalculatorResult:
    del timeout_s
    assert isinstance(args, CalculatorArguments)
    return CalculatorResult(expression=args.expression, value=calculate(args.expression))


def _execute_current_time(args: BaseModel, timeout_s: float | None) -> CurrentTimeResult:
    assert isinstance(args, CurrentTimeArguments)
    if args.timezone is not None:
        zone_name = args.timezone
        if _IANA_TIMEZONE.fullmatch(zone_name) is None:
            zone_name = _weather_client.get_timezone(
                zone_name,
                _DEFAULT_WEATHER_TIMEOUT_S if timeout_s is None else timeout_s,
            )
        try:
            zone = ZoneInfo(zone_name)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ToolError(
                "I couldn't get the time for that timezone.", code="timezone_unavailable"
            ) from exc
        now = datetime.now(zone)
    else:
        now = datetime.now().astimezone()
        zone_name = getattr(now.tzinfo, "key", None) or now.tzname() or "local"
    return CurrentTimeResult(timezone=zone_name, datetime=now.isoformat(timespec="seconds"))


_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_MAX_RESPONSE_BYTES = 256 * 1024
_DEFAULT_WEATHER_TIMEOUT_S = 6.0


WMO_CONDITIONS: dict[int, str] = {
    0: "clear skies",
    1: "mostly clear skies",
    2: "partly cloudy skies",
    3: "overcast skies",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "light freezing drizzle",
    57: "heavy freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "thunderstorms with light hail",
    99: "thunderstorms with heavy hail",
}


def _finite_number(value: Any, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError("number is out of range")
    return result


def _bounded_integer(value: Any, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected an integer")
    result = int(value)
    if result != value or not minimum <= result <= maximum:
        raise ValueError("integer is out of range")
    return result


class WeatherClient:
    """Thread-local pooled Open-Meteo client with one total call deadline."""

    def __init__(self) -> None:
        self._local = threading.local()
        self._lock = threading.Lock()
        self._sessions: set[requests.Session] = set()
        self._closed = False

    def _session(self) -> requests.Session:
        with self._lock:
            if self._closed:
                raise ToolError(
                    "I couldn't get the weather right now.", code="weather_unavailable"
                )
            session = getattr(self._local, "session", None)
            if session is not None:
                return session
            session = requests.Session()
            adapter = HTTPAdapter(
                pool_connections=1, pool_maxsize=2, max_retries=0, pool_block=True
            )
            session.mount("https://", adapter)
            self._sessions.add(session)
            self._local.session = session
        return session

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ToolError(
                "I couldn't get the weather in time.", code="weather_timeout"
            )
        return remaining

    def _get_json(
        self, url: str, params: Mapping[str, Any], deadline: float
    ) -> Mapping[str, Any]:
        if url not in {_GEOCODING_URL, _FORECAST_URL}:
            raise ToolError(
                "I couldn't get the weather right now.", code="weather_unavailable"
            )
        remaining = self._remaining(deadline)
        timeout = (min(2.0, remaining), min(2.0, remaining))
        response: requests.Response | None = None
        try:
            response = self._session().get(
                url,
                params=dict(params),
                timeout=timeout,
                allow_redirects=False,
                stream=True,
                verify=True,
            )
            if response.status_code != 200:
                raise ToolError(
                    "I couldn't get the weather right now.", code="weather_provider_error"
                )
            content_type = response.headers.get("Content-Type", "")
            if content_type and "json" not in content_type.lower():
                raise ToolError(
                    "I couldn't get the weather right now.", code="weather_provider_error"
                )
            body = bytearray()
            for chunk in response.iter_content(chunk_size=16 * 1024):
                self._remaining(deadline)
                if not chunk:
                    continue
                if len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
                    raise ToolError(
                        "I couldn't get the weather right now.",
                        code="weather_response_too_large",
                    )
                body.extend(chunk)
            self._remaining(deadline)
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("provider payload is not an object")
            return payload
        except ToolError:
            raise
        except requests.Timeout as exc:
            raise ToolError(
                "I couldn't get the weather in time.", code="weather_timeout"
            ) from exc
        except (requests.RequestException, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ToolError(
                "I couldn't get the weather right now.", code="weather_provider_error"
            ) from exc
        finally:
            if response is not None:
                response.close()

    def get_timezone(self, location: str, timeout_s: float) -> str:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ToolError("I couldn't get the time in time.", code="timezone_timeout")
        timeout_s = min(timeout_s, 15.0)
        deadline = time.monotonic() + timeout_s
        geocoding = self._get_json(
            _GEOCODING_URL,
            {"name": location, "count": 1, "language": "en", "format": "json"},
            deadline,
        )
        results = geocoding.get("results")
        if not isinstance(results, list) or not results or not isinstance(results[0], dict):
            raise ToolError("I couldn't find that place.", code="location_not_found")
        try:
            timezone_name = str(results[0]["timezone"]).strip()
            if (
                not timezone_name
                or len(timezone_name) > 64
                or _IANA_TIMEZONE.fullmatch(timezone_name) is None
            ):
                raise ValueError("invalid timezone metadata")
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolError(
                "I couldn't get the time right now.", code="timezone_provider_error"
            ) from exc
        return timezone_name


    def get_weather(self, location: str, timeout_s: float) -> WeatherResult:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ToolError("I couldn't get the weather in time.", code="weather_timeout")
        timeout_s = min(timeout_s, 15.0)
        deadline = time.monotonic() + timeout_s
        geocoding = self._get_json(
            _GEOCODING_URL,
            {"name": location, "count": 1, "language": "en", "format": "json"},
            deadline,
        )
        results = geocoding.get("results")
        if not isinstance(results, list) or not results or not isinstance(results[0], dict):
            raise ToolError("I couldn't find that city.", code="location_not_found")
        place = results[0]
        try:
            latitude = _finite_number(place["latitude"], minimum=-90, maximum=90)
            longitude = _finite_number(place["longitude"], minimum=-180, maximum=180)
            place_name = str(place["name"]).strip()
            country_raw = place.get("country")
            country = str(country_raw).strip() if country_raw is not None else None
            timezone_name = str(place["timezone"]).strip()
            if not place_name or not timezone_name:
                raise ValueError("missing place metadata")
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolError(
                "I couldn't get the weather right now.", code="weather_provider_error"
            ) from exc

        forecast = self._get_json(
            _FORECAST_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "weather_code,wind_speed_10m"
                ),
                "daily": (
                    "temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "forecast_days": 1,
                "timezone": "auto",
            },
            deadline,
        )
        try:
            current = forecast["current"]
            daily = forecast["daily"]
            if not isinstance(current, dict) or not isinstance(daily, dict):
                raise ValueError("missing forecast sections")
            code = _bounded_integer(current["weather_code"], minimum=0, maximum=99)
            high_values = daily["temperature_2m_max"]
            low_values = daily["temperature_2m_min"]
            precip_values = daily["precipitation_probability_max"]
            if not all(isinstance(values, list) and values for values in (high_values, low_values, precip_values)):
                raise ValueError("missing daily values")
            result = WeatherResult(
                location=place_name,
                country=country or None,
                latitude=latitude,
                longitude=longitude,
                timezone=str(forecast.get("timezone") or timezone_name),
                temperature_c=_finite_number(
                    current["temperature_2m"], minimum=-100, maximum=100
                ),
                apparent_temperature_c=_finite_number(
                    current["apparent_temperature"], minimum=-120, maximum=120
                ),
                relative_humidity_percent=_bounded_integer(
                    current["relative_humidity_2m"], minimum=0, maximum=100
                ),
                weather_code=code,
                condition=WMO_CONDITIONS.get(code, "unknown conditions"),
                wind_speed_kmh=_finite_number(
                    current["wind_speed_10m"], minimum=0, maximum=500
                ),
                today_high_c=_finite_number(high_values[0], minimum=-100, maximum=100),
                today_low_c=_finite_number(low_values[0], minimum=-100, maximum=100),
                precipitation_probability_percent=_bounded_integer(
                    precip_values[0], minimum=0, maximum=100
                ),
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise ToolError(
                "I couldn't get the weather right now.", code="weather_provider_error"
            ) from exc
        return result

    def close(self) -> None:
        with self._lock:
            self._closed = True
            sessions = tuple(self._sessions)
            self._sessions.clear()
        for session in sessions:
            try:
                session.close()
            except requests.RequestException:
                pass


_weather_client = WeatherClient()


def _execute_weather(args: BaseModel, timeout_s: float | None) -> WeatherResult:
    assert isinstance(args, WeatherArguments)
    if not args.location:
        raise MissingLocationError()
    return _weather_client.get_weather(
        args.location,
        _DEFAULT_WEATHER_TIMEOUT_S if timeout_s is None else timeout_s,
    )


def _format_number(value: float) -> str:
    rounded = round(value, 1)
    return str(int(rounded)) if rounded.is_integer() else f"{rounded:.1f}"


def _format_decimal_for_speech(value: str) -> str:
    number = Decimal(value)
    if number == number.to_integral_value():
        return format(number, ",f").split(".", 1)[0]
    integer, fractional = _decimal_text(number).split(".", 1)
    integer_text = format(Decimal(integer), ",f").split(".", 1)[0]
    return f"{integer_text}.{fractional}"


def _spoken_expression(expression: str) -> str:
    spoken = expression
    spoken = spoken.replace("**", " to the power of ")
    spoken = spoken.replace("*", " times ")
    spoken = spoken.replace("/", " divided by ")
    spoken = spoken.replace("+", " plus ")
    spoken = spoken.replace("-", " minus ")
    spoken = spoken.replace("(", " open parenthesis ")
    spoken = spoken.replace(")", " close parenthesis ")
    return " ".join(spoken.split())


def _format_calculator(args: BaseModel, result: BaseModel) -> str:
    del args
    assert isinstance(result, CalculatorResult)
    return (
        f"{_spoken_expression(result.expression)} is "
        f"{_format_decimal_for_speech(result.value)}."
    )


def _format_current_time(args: BaseModel, result: BaseModel) -> str:
    assert isinstance(args, CurrentTimeArguments)
    assert isinstance(result, CurrentTimeResult)
    value = datetime.fromisoformat(result.datetime)
    clock = f"{value.hour % 12 or 12}:{value.minute:02d} {value.strftime('%p')}"
    prefix = f"In {result.timezone}, it's" if args.timezone else "It's"
    return f"{prefix} {clock}."


def _format_weather(args: BaseModel, result: BaseModel) -> str:
    del args
    assert isinstance(result, WeatherResult)
    temperature = _format_number(result.temperature_c)
    high = _format_number(result.today_high_c)
    return (
        f"It's {temperature} degrees Celsius with {result.condition} in {result.location} right now. "
        f"Today's high is {high} degrees, with a {result.precipitation_probability_percent} percent "
        "chance of rain."
    )


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "get_current_time": ToolSpec(
        name="get_current_time",
        description=(
            "Get the current local date and time, optionally in an IANA timezone or city/place name."
        ),
        arguments_model=CurrentTimeArguments,
        result_model=CurrentTimeResult,
        handler=_execute_current_time,
        formatter=_format_current_time,
    ),
    "calculator": ToolSpec(
        name="calculator",
        description=(
            "Calculate an arithmetic expression using numbers, parentheses, addition, "
            "subtraction, multiplication, division, and bounded powers."
        ),
        arguments_model=CalculatorArguments,
        result_model=CalculatorResult,
        handler=_execute_calculator,
        formatter=_format_calculator,
    ),
    "get_weather": ToolSpec(
        name="get_weather",
        description="Get the current weather and today's forecast for a city or place.",
        arguments_model=WeatherArguments,
        result_model=WeatherResult,
        handler=_execute_weather,
        formatter=_format_weather,
        acknowledgement="Checking the weather.",
    ),
}

TOOLS = TOOL_REGISTRY
TOOL_DEFINITIONS = tuple(spec.router_definition() for spec in TOOL_REGISTRY.values())
ROUTER_JSON_SCHEMA: dict[str, Any] = ToolDecision.model_json_schema()
TOOL_ROUTER_PROMPT = (
    "Select at most one tool for the user's latest request. Return exactly one JSON object "
    "and no other text. Use {\"tool\":null,\"arguments\":{}} when no tool applies. "
    "Never invent arguments. Registered tools:\n"
    + json.dumps(TOOL_DEFINITIONS, ensure_ascii=True, separators=(",", ":"))
)


_TIME_INTENT = re.compile(
    r"^(?:please\s+)?(?:"
    r"what(?:'s| is)\s+(?:the\s+)?(?:current\s+)?time(?:\s+is\s+it)?|"
    r"what\s+time(?:\s+is\s+it)?|"
    r"(?:can|could)\s+you\s+tell\s+me\s+(?:what\s+time\s+it\s+is|the\s+time)|"
    r"tell\s+me\s+(?:the\s+)?(?:current\s+)?time"
    r")(?:\s+(?:in|for)\s+[A-Za-z0-9_./+ -]{1,64})?"
    r"(?:\s+right\s+now)?[?.!]*$",
    re.IGNORECASE,
)
_TIME_PLACE_FIRST = re.compile(
    r"^(?:please\s+)?(?:in\s+)?(?P<location>[A-Za-z0-9_./+ -]{1,64}?)"
    r"[, ]+what\s+time\s+(?:is\s+it|it\s+is)"
    r"(?:\s+right\s+now)?[?.!]*$",
    re.IGNORECASE,
)
_TIME_PLACE_COMPACT = re.compile(
    r"^(?:please\s+)?(?P<location>[A-Za-z0-9_./+ -]{1,64}?)\s+time"
    r"(?:\s+right\s+now)?[?.!]*$",
    re.IGNORECASE,
)
_TIME_WHAT_IS_PLACE = re.compile(
    r"^(?:please\s+)?what\s+is\s+(?P<location>[A-Za-z0-9_./+ -]{1,64}?)\s+time"
    r"(?:\s+right\s+now)?[?.!]*$",
    re.IGNORECASE,
)
_TIME_LOCATION = re.compile(
    r"\bin\s+(?P<location>[A-Za-z][A-Za-z .'-]{0,63}?)(?="
    r"\s+(?:right\s+now|now)\b|[?.!]*$)",
    re.IGNORECASE,
)
_TIME_CONTEXT = re.compile(
    r"\b(?:what|what's|was|right\s+now|now|current)\b",
    re.IGNORECASE,
)
_FOLLOWUP_LOCATION = re.compile(
    r"^(?:please\s+)?(?:what|how)\s+about(?:\s+in)?\s+"
    r"(?P<location>[A-Za-z][A-Za-z .'-]{0,63})[?.!]*$",
    re.IGNORECASE,
)
_CORRECTION_LOCATION = re.compile(
    r"^(?:no,\s*)?i\s+meant\s+(?:the\s+time\s+in\s+)?"
    r"(?P<location>[A-Za-z][A-Za-z '-]{0,63}?)[?.!]*$",
    re.IGNORECASE,
)
_CORRECTION_TIME_LOCATION = re.compile(
    r"^meant\s+was\s+(?:the\s+)?time\s+in\s+"
    r"(?P<location>[A-Za-z][A-Za-z '-]{0,63}?)[?.!]*$",
    re.IGNORECASE,
)
_WEATHER_INTENT = re.compile(
    r"\b(?:weather|forecast|temperature|chance of rain|is it raining)\b",
    re.IGNORECASE,
)
_WEATHER_LOCATION = re.compile(
    r"\b(?:weather|forecast|temperature)(?:\s+like)?\s+(?:in|for|at)\s+"
    r"([A-Za-z][A-Za-z .'-]{0,80})",
    re.IGNORECASE,
)
_CALCULATOR_LEAD = re.compile(
    r"^(?:please\s+)?(?:what(?:'s| is)|calculate|compute|work out|evaluate)\s+(.+?)[?.!]*$",
    re.IGNORECASE,
)
_LEADING_TOOL_FILLER = re.compile(
    r"^(?:oh|um|uh|okay|ok|hey|alright|well|no),?\s+",
    re.IGNORECASE,
)


def normalize_tool_router_text(user_text: str) -> str:
    """Apply the exact bounded normalization used by deterministic routing."""

    text = " ".join((user_text or "").strip().split())
    for _ in range(4):
        normalized = _LEADING_TOOL_FILLER.sub("", text, count=1)
        if normalized == text:
            break
        text = normalized
    return text


def deterministic_tool_decision(
    user_text: str, previous_tool: ToolName | None = None
) -> ToolDecision | None:
    """Resolve clear low-latency intents; return ``None`` only when genuinely ambiguous.

    This narrow gate prevents a small model from inventing arguments. The existing
    Qwen router remains the constrained fallback for phrasing that mentions a tool
    domain but cannot be classified safely here.
    """

    text = normalize_tool_router_text(user_text)
    if not text:
        return ToolDecision(tool=None, arguments={})

    location_match = _TIME_LOCATION.search(text)
    if (
        location_match is not None
        and re.search(r"\btime\b", text, re.IGNORECASE)
        and _TIME_CONTEXT.search(text) is not None
    ):
        try:
            return ToolDecision(
                tool="get_current_time",
                arguments=validate_tool_arguments(
                    "get_current_time",
                    {"timezone": location_match.group("location").strip()},
                ),
            )
        except ToolError:
            return ToolDecision(tool=None, arguments={})

    time_match = _TIME_INTENT.fullmatch(text)
    place_first_match = _TIME_PLACE_FIRST.fullmatch(text)
    compact_match = _TIME_PLACE_COMPACT.fullmatch(text)
    what_is_place_match = _TIME_WHAT_IS_PLACE.fullmatch(text)
    if time_match or place_first_match or compact_match or what_is_place_match:
        location_match = re.search(
            r"\b(?:in|for)\s+([A-Za-z0-9_./+ -]{1,64}?)(?:\s+right\s+now)?[?.!]*$",
            text,
            re.IGNORECASE,
        )
        location = (
            location_match.group(1).strip()
            if location_match
            else next(
                (
                    match.group("location").strip()
                    for match in (
                        place_first_match,
                        what_is_place_match,
                        compact_match,
                    )
                    if match is not None
                ),
                None,
            )
        )
        arguments = {"timezone": location}
        try:
            return ToolDecision(
                tool="get_current_time",
                arguments=validate_tool_arguments("get_current_time", arguments),
            )
        except ToolError:
            return ToolDecision(tool=None, arguments={})

    if previous_tool in {"get_weather", "get_current_time"}:
        followup_match = _FOLLOWUP_LOCATION.fullmatch(text)
        correction_match = _CORRECTION_LOCATION.fullmatch(text)
        correction_time_match = _CORRECTION_TIME_LOCATION.fullmatch(text)
        location_match = followup_match or correction_match or correction_time_match
        if location_match is not None:
            location = location_match.group("location").strip()
            argument_name = "location" if previous_tool == "get_weather" else "timezone"
            try:
                return ToolDecision(
                    tool=previous_tool,
                    arguments=validate_tool_arguments(
                        previous_tool, {argument_name: location}
                    ),
                )
            except ToolError:
                return ToolDecision(tool=None, arguments={})

    if _WEATHER_INTENT.search(text):
        # Knowledge questions about weather itself are ordinary conversation.
        if re.search(r"\b(?:what is weather|how does weather|explain weather|temperature of)\b", text, re.IGNORECASE):
            return ToolDecision(tool=None, arguments={})
        location_match = _WEATHER_LOCATION.search(text)
        arguments: dict[str, Any] = {}
        if location_match:
            location = re.split(
                r"\s+(?:and|today|tomorrow|right now|currently|this (?:morning|afternoon|evening))\b",
                location_match.group(1),
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" .,?!")
            if location:
                arguments["location"] = location
        return ToolDecision(tool="get_weather", arguments=validate_tool_arguments("get_weather", arguments))

    lead = _CALCULATOR_LEAD.match(text)
    if lead:
        expression = re.split(
            r"\s+(?:and\s+)?(?:explain|show|tell|why|then)\b",
            lead.group(1),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        spoken = normalize_spoken_math_expression(expression)
        if spoken:
            expression = spoken
        else:
            expression = re.sub(r"\bmultiplied\s+by\b", "*", expression, flags=re.IGNORECASE)
            expression = re.sub(r"\bdivided\s+by\b", "/", expression, flags=re.IGNORECASE)
            expression = re.sub(r"\bto\s+the\s+power\s+of\b", "**", expression, flags=re.IGNORECASE)
            expression = re.sub(r"\b(?:times|x)\b", "*", expression, flags=re.IGNORECASE)
            expression = re.sub(r"\bplus\b", "+", expression, flags=re.IGNORECASE)
            expression = re.sub(r"\bminus\b", "-", expression, flags=re.IGNORECASE)
            expression = "".join(expression.split())
        explicit_calculate = re.match(
            r"^(?:please\s+)?(?:calculate|compute|work out|evaluate)\b", text, re.IGNORECASE
        )
        has_operator = any(operator in expression for operator in ("+", "-", "*", "/"))
        if (
            any(char.isdigit() for char in expression)
            and (explicit_calculate or has_operator)
            and _CALCULATOR_CHARS.fullmatch(expression)
        ):
            try:
                arguments = validate_tool_arguments("calculator", {"expression": expression})
                # Parse now as a semantic guard, but execution still happens in the tool worker.
                calculate(arguments["expression"])
                return ToolDecision(tool="calculator", arguments=arguments)
            except ToolError:
                return ToolDecision(tool=None, arguments={})

    tool_domain_words = re.search(
        r"\b(?:time|clock|weather|forecast|temperature|calculate|compute|arithmetic)\b",
        text,
        re.IGNORECASE,
    )
    if tool_domain_words:
        return None
    return ToolDecision(tool=None, arguments={})


def router_schema_json() -> str:
    return json.dumps(ROUTER_JSON_SCHEMA, ensure_ascii=True, separators=(",", ":"))


def _no_tool_decision() -> ToolDecision:
    return ToolDecision(tool=None, arguments={})


def _validate_arguments_model(name: str, arguments: Mapping[str, Any]) -> BaseModel:
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        raise ToolError("That tool is unavailable.", code="unknown_tool")
    if not isinstance(arguments, Mapping):
        raise ToolError("I couldn't use those tool arguments.", code="invalid_arguments")
    try:
        return spec.arguments_model.model_validate(dict(arguments))
    except ValidationError as exc:
        raise ToolError(
            "I couldn't use those tool arguments.", code="invalid_arguments"
        ) from exc


def validate_tool_arguments(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Validate tool arguments and return a normalized JSON-compatible mapping."""

    return _validate_arguments_model(name, arguments).model_dump(mode="json")


def parse_tool_decision(raw: str | bytes | Mapping[str, Any] | ToolDecision) -> ToolDecision:
    """Strictly parse one router object, failing closed on every invalid decision."""

    try:
        if isinstance(raw, ToolDecision):
            decision = raw
        elif isinstance(raw, Mapping):
            decision = ToolDecision.model_validate(dict(raw))
        else:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if not isinstance(raw, str) or len(raw) > 4096:
                return _no_tool_decision()
            payload = json.loads(raw.strip())
            decision = ToolDecision.model_validate(payload)
        if decision.tool is None:
            return decision if not decision.arguments else _no_tool_decision()
        validated = validate_tool_arguments(decision.tool, decision.arguments)
        return ToolDecision(tool=decision.tool, arguments=validated)
    except (
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValidationError,
        ToolError,
        RecursionError,
        MemoryError,
    ):
        return _no_tool_decision()


def execute_tool(
    name: str, arguments: Mapping[str, Any], *, timeout_s: float | None = None
) -> dict[str, Any]:
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        raise ToolError("That tool is unavailable.", code="unknown_tool")
    validated = _validate_arguments_model(name, arguments)
    try:
        result = spec.handler(validated, timeout_s)
        result = spec.result_model.model_validate(result)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError("That tool failed.", code="tool_failed") from exc
    return result.model_dump(mode="json")


def format_tool_result(
    name: str, arguments: Mapping[str, Any], result: Mapping[str, Any]
) -> str:
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        raise ToolError("That tool is unavailable.", code="unknown_tool")
    validated_args = _validate_arguments_model(name, arguments)
    try:
        validated_result = spec.result_model.model_validate(dict(result))
        return spec.formatter(validated_args, validated_result)
    except ToolError:
        raise
    except (TypeError, ValueError, ValidationError) as exc:
        raise ToolError("That tool returned an invalid result.", code="invalid_result") from exc


def acknowledgement_for(name: str) -> str | None:
    spec = TOOL_REGISTRY.get(name)
    return spec.acknowledgement if spec is not None else None


_SYNTHESIS_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bshould\b",
        r"\bwhy\b",
        r"\bexplain(?:ed|ing|s)?\b",
        r"\bcompar(?:e|ed|ing|ison)\b",
        r"\bwear(?:ing)?\b",
        r"\bneed(?:ed|ing|s)?\b",
        r"\brecommend(?:ation|ations|ed|ing|s)?\b",
        r"\bsafe(?:r|st|ty)?\b",
        r"\bumbrella\b",
        r"\bjacket\b",
        r"\badvice\b",
        r"\bwhich (?:is|would be) better\b",
        r"\bhow (?:did|do|does|would|can)\b",
        r"\btoo late\b",
    )
)


def requires_synthesis(name: str, user_text: str) -> bool:
    """Return whether a request needs reasoning beyond a direct tool formatter."""

    del name  # Intentionally tool-agnostic: only the user's intent controls this.
    return any(pattern.search(user_text) for pattern in _SYNTHESIS_PATTERNS)


def close_tool_clients() -> None:
    _weather_client.close()


__all__ = [
    "CalculatorArguments",
    "CalculatorResult",
    "CurrentTimeArguments",
    "CurrentTimeResult",
    "MissingLocationError",
    "ROUTER_JSON_SCHEMA",
    "TOOLS",
    "TOOL_DEFINITIONS",
    "TOOL_REGISTRY",
    "TOOL_ROUTER_PROMPT",
    "ToolDecision",
    "ToolError",
    "ToolSpec",
    "WeatherArguments",
    "WeatherClient",
    "WeatherResult",
    "WMO_CONDITIONS",
    "acknowledgement_for",
    "calculate",
    "close_tool_clients",
    "deterministic_tool_decision",
    "execute_tool",
    "format_tool_result",
    "normalize_tool_router_text",
    "parse_tool_decision",
    "requires_synthesis",
    "router_schema_json",
    "validate_tool_arguments",
]

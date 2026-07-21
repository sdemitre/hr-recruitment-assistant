"""Unified LLM client supporting OpenAI and Anthropic structured outputs."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.config import LLMConfig, get_llm_config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Anthropic output token ceiling for Sonnet 4.x models.
DEFAULT_ANTHROPIC_MAX_TOKENS = 16_384
ANTHROPIC_MAX_OUTPUT_TOKENS = 16_384
MAX_JSON_PARSE_ATTEMPTS = 3


class LLMError(Exception):
    """Raised when an LLM API call fails."""


class LLMClient:
    """Provider-agnostic client for structured JSON responses."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or get_llm_config()
        self._openai_client = None
        self._anthropic_client = None

    @property
    def openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=self.config.openai_api_key)
        return self._openai_client

    @property
    def anthropic_client(self):
        if self._anthropic_client is None:
            from anthropic import Anthropic

            self._anthropic_client = Anthropic(api_key=self.config.anthropic_api_key)
        return self._anthropic_client

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float = 0.3,
    ) -> T:
        """Generate a structured response parsed into a Pydantic model."""
        try:
            if self.config.provider == "openai":
                return self._generate_openai(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=response_model,
                    temperature=temperature,
                )
            return self._generate_anthropic(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
                temperature=temperature,
            )
        except LLMError:
            raise
        except Exception as exc:
            logger.exception("Unexpected LLM error")
            raise LLMError(f"LLM request failed: {exc}") from exc

    def _generate_openai(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float,
    ) -> T:
        try:
            response = self.openai_client.beta.chat.completions.parse(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_model,
                temperature=temperature,
            )
        except Exception as exc:
            raise LLMError(f"OpenAI API error: {exc}") from exc

        message = response.choices[0].message
        if message.refusal:
            raise LLMError(f"OpenAI refused the request: {message.refusal}")
        if message.parsed is None:
            raise LLMError("OpenAI returned no parsed structured output.")
        return message.parsed

    def _anthropic_max_tokens(self) -> int:
        configured = self.config.anthropic_max_tokens
        return min(max(configured, 8192), ANTHROPIC_MAX_OUTPUT_TOKENS)

    def _build_schema_instruction(self, response_model: type[T]) -> str:
        schema = response_model.model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", list(properties.keys()))

        field_lines: list[str] = []
        for name in required:
            prop = properties.get(name, {})
            type_hint = prop.get("type", "string")
            if type_hint == "array":
                item_type = prop.get("items", {}).get("type", "string")
                type_hint = f"array of {item_type}"
            description = prop.get("description", "")
            suffix = f" — {description}" if description else ""
            field_lines.append(f'- "{name}" ({type_hint}){suffix}')

        return (
            "\n\nReturn ONLY a single JSON object with the fields below. "
            "Do NOT return JSON Schema metadata (no \"properties\", \"required\", "
            "or \"$defs\" keys). Do NOT use markdown fences.\n"
            + "\n".join(field_lines)
        )

    @staticmethod
    def _clean_schema_for_anthropic(schema: dict[str, Any]) -> dict[str, Any]:
        """Strip JSON Schema metadata that Anthropic tool input_schema rejects."""
        cleaned = {k: v for k, v in schema.items() if k not in ("$defs", "title")}
        properties = cleaned.get("properties", {})
        if isinstance(properties, dict):
            for name, prop in properties.items():
                if isinstance(prop, dict):
                    properties[name] = {
                        k: v
                        for k, v in prop.items()
                        if k not in ("title", "description")
                    }
        return cleaned

    def _generate_anthropic(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float,
    ) -> T:
        tool_schema = self._clean_schema_for_anthropic(
            response_model.model_json_schema()
        )
        max_tokens = self._anthropic_max_tokens()

        try:
            response = self.anthropic_client.messages.create(
                model=self.config.anthropic_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[
                    {
                        "name": "submit_response",
                        "description": (
                            f"Submit the structured {response_model.__name__} result."
                        ),
                        "input_schema": tool_schema,
                    }
                ],
                tool_choice={"type": "tool", "name": "submit_response"},
            )
        except Exception as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc

        for block in response.content:
            if block.type == "tool_use" and block.input:
                try:
                    payload = self._normalize_tool_payload(block.input, response_model)
                    return response_model.model_validate(payload)
                except ValidationError as exc:
                    raise LLMError(
                        f"Anthropic tool output failed validation: {exc}"
                    ) from exc

        # Fallback to legacy JSON-in-text parsing if no tool block returned.
        return self._generate_anthropic_json_fallback(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            temperature=temperature,
        )

    @staticmethod
    def _normalize_tool_payload(data: Any, response_model: type[BaseModel]) -> Any:
        """Normalize common Anthropic tool output quirks before validation."""
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return data

        if not isinstance(data, dict):
            return data

        normalized = dict(data)

        def normalize_email_fields(item: Any) -> Any:
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except json.JSONDecodeError:
                    return item
            if isinstance(item, dict):
                item = dict(item)
                if "candidate_name" not in item and "recipient_name" in item:
                    item["candidate_name"] = item.pop("recipient_name")
            return item

        if response_model.__name__ == "CommunicationBundle":
            if isinstance(normalized.get("rejection_emails"), list):
                normalized["rejection_emails"] = [
                    normalize_email_fields(item)
                    for item in normalized["rejection_emails"]
                ]
            if "advancement_email" in normalized:
                normalized["advancement_email"] = normalize_email_fields(
                    normalized["advancement_email"]
                )
            if "manager_notification" in normalized:
                mn = normalized["manager_notification"]
                if isinstance(mn, str):
                    try:
                        mn = json.loads(mn)
                    except json.JSONDecodeError:
                        pass
                normalized["manager_notification"] = mn

        return normalized

    def _generate_anthropic_json_fallback(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float,
    ) -> T:
        schema_instruction = self._build_schema_instruction(response_model)
        max_tokens = self._anthropic_max_tokens()
        concise_suffix = ""

        last_error: Exception | None = None
        last_raw_text = ""

        for attempt in range(1, MAX_JSON_PARSE_ATTEMPTS + 1):
            effective_user_prompt = user_prompt + concise_suffix
            try:
                response = self.anthropic_client.messages.create(
                    model=self.config.anthropic_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt + schema_instruction,
                    messages=[{"role": "user", "content": effective_user_prompt}],
                )
            except Exception as exc:
                raise LLMError(f"Anthropic API error: {exc}") from exc

            raw_text = self._extract_response_text(response)
            last_raw_text = raw_text
            stop_reason = getattr(response, "stop_reason", None)
            truncated = stop_reason == "max_tokens"

            if truncated:
                logger.warning(
                    "Anthropic response hit max_tokens=%s (attempt %s/%s)",
                    max_tokens,
                    attempt,
                    MAX_JSON_PARSE_ATTEMPTS,
                )

            try:
                return self._parse_structured_json(raw_text, response_model)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "Anthropic JSON parse failed on attempt %s/%s: %s",
                    attempt,
                    MAX_JSON_PARSE_ATTEMPTS,
                    exc,
                )

                repaired = self._repair_truncated_json(raw_text)
                if repaired is not None:
                    try:
                        return self._parse_structured_json(repaired, response_model)
                    except (json.JSONDecodeError, ValidationError) as repair_exc:
                        last_error = repair_exc
                        logger.warning("Repaired JSON still invalid: %s", repair_exc)

                if attempt < MAX_JSON_PARSE_ATTEMPTS:
                    completed = self._complete_truncated_json(
                        system_prompt=system_prompt + schema_instruction,
                        user_prompt=effective_user_prompt,
                        partial_text=raw_text,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    if completed != raw_text:
                        last_raw_text = completed
                        try:
                            return self._parse_structured_json(
                                completed, response_model
                            )
                        except (json.JSONDecodeError, ValidationError) as completion_exc:
                            last_error = completion_exc
                            repaired_completed = self._repair_truncated_json(completed)
                            if repaired_completed is not None:
                                try:
                                    return self._parse_structured_json(
                                        repaired_completed, response_model
                                    )
                                except (json.JSONDecodeError, ValidationError):
                                    pass

                concise_suffix = (
                    "\n\nIMPORTANT: Keep the response concise so it fits in one JSON "
                    "object. Limit list items to 8 entries with brief phrases. "
                    "Keep full_text under 1200 words."
                )

        raise LLMError(
            "Failed to parse Anthropic JSON response after "
            f"{MAX_JSON_PARSE_ATTEMPTS} attempts: {last_error}\n"
            f"Raw: {last_raw_text[:500]}"
        ) from last_error

    def _complete_truncated_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        partial_text: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Ask the model to continue a truncated JSON payload."""
        completion_prompt = (
            "Your previous JSON response was cut off before it finished. "
            "Continue exactly where you stopped and output ONLY the remaining "
            "characters needed to complete the JSON object. Do not repeat any "
            "content already written. Do not wrap the continuation in markdown."
        )
        try:
            response = self.anthropic_client.messages.create(
                model=self.config.anthropic_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": partial_text},
                    {"role": "user", "content": completion_prompt},
                ],
            )
        except Exception as exc:
            logger.warning("Anthropic JSON completion request failed: %s", exc)
            return partial_text

        continuation = self._extract_response_text(response)
        if not continuation:
            return partial_text

        merged = partial_text + continuation
        if continuation.lstrip().startswith("{"):
            # Model restarted the object; prefer the fresh response.
            return continuation

        return merged

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        text_blocks = [
            block.text for block in response.content if block.type == "text"
        ]
        if not text_blocks:
            raise LLMError("Anthropic returned no text content.")
        return LLMClient._normalize_json_text("".join(text_blocks))

    @staticmethod
    def _normalize_json_text(raw_text: str) -> str:
        text = raw_text.strip()
        if not text.startswith("```"):
            return text

        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    @staticmethod
    def _coerce_payload(data: Any, response_model: type[T]) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise TypeError("Structured response must be a JSON object.")

        if "properties" in data and isinstance(data["properties"], dict):
            logger.warning("Anthropic returned JSON Schema wrapper; unwrapping properties.")
            data = data["properties"]

        schema = response_model.model_json_schema()
        properties = schema.get("properties", {})
        coerced: dict[str, Any] = {}

        for name, prop in properties.items():
            if name not in data:
                continue
            value = data[name]
            if prop.get("type") == "array" and isinstance(value, str):
                value = [line.strip("- ").strip() for line in value.splitlines() if line.strip()]
            coerced[name] = value

        for name, value in data.items():
            coerced.setdefault(name, value)

        return coerced

    @staticmethod
    def _parse_structured_json(raw_text: str, response_model: type[T]) -> T:
        data = json.loads(raw_text)
        payload = LLMClient._coerce_payload(data, response_model)
        return response_model.model_validate(payload)

    @staticmethod
    def _repair_truncated_json(raw_text: str) -> str | None:
        """Best-effort repair for JSON truncated mid-stream."""
        text = raw_text.strip()
        if not text:
            return None

        if text.startswith("```"):
            text = LLMClient._normalize_json_text(text)

        in_string = False
        escaped = False
        stack: list[str] = []

        for char in text:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append("}" if char == "{" else "]")
            elif char in "}]" and stack and stack[-1] == char:
                stack.pop()

        suffix = ""
        if in_string:
            suffix += '"'
        suffix += "".join(reversed(stack))

        candidate = text + suffix
        candidate = re.sub(r",(\s*[\]}])", r"\1", candidate)

        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            trimmed = text.rstrip(" \t\n\r,:")
            if trimmed != text:
                return LLMClient._repair_truncated_json(trimmed)
            return None

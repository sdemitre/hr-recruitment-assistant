"""Unified LLM client supporting OpenAI and Anthropic structured outputs."""

from __future__ import annotations

import json
import logging
from typing import TypeVar

from pydantic import BaseModel

from src.config import LLMConfig, get_llm_config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


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

    def _generate_anthropic(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float,
    ) -> T:
        schema = response_model.model_json_schema()
        schema_instruction = (
            f"\n\nRespond with valid JSON matching this schema exactly:\n"
            f"{json.dumps(schema, indent=2)}"
        )

        try:
            response = self.anthropic_client.messages.create(
                model=self.config.anthropic_model,
                max_tokens=8192,
                temperature=temperature,
                system=system_prompt + schema_instruction,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc

        text_blocks = [
            block.text for block in response.content if block.type == "text"
        ]
        if not text_blocks:
            raise LLMError("Anthropic returned no text content.")

        raw_text = text_blocks[0].strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1]
            raw_text = raw_text.rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw_text)
            return response_model.model_validate(data)
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(
                f"Failed to parse Anthropic JSON response: {exc}\nRaw: {raw_text[:500]}"
            ) from exc

"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

NGO_NAME = "Global Rescue Health Initiative (GRHI)"
NGO_MISSION = (
    "Deploying public health research scientists and field workers to remote, "
    "high-risk global regions facing disease outbreaks, malnutrition, and "
    "humanitarian crises."
)


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    openai_api_key: str | None
    openai_model: str
    anthropic_api_key: str | None
    anthropic_model: str

    @classmethod
    def from_env(cls) -> LLMConfig:
        provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()
        if provider not in {"openai", "anthropic"}:
            raise ValueError(
                f"Invalid LLM_PROVIDER '{provider}'. Must be 'openai' or 'anthropic'."
            )
        return cls(
            provider=provider,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            anthropic_model=os.getenv(
                "ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"
            ),
        )

    def validate(self) -> None:
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
                "Copy .env.example to .env and set your key."
            )
        if self.provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic. "
                "Copy .env.example to .env and set your key."
            )


def get_llm_config() -> LLMConfig:
    config = LLMConfig.from_env()
    config.validate()
    return config

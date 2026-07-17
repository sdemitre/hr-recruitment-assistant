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


def _get_setting(name: str, default: str | None = None) -> str | None:
    """Read config from environment variables or Streamlit Cloud secrets."""
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass

    return default


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    openai_api_key: str | None
    openai_model: str
    anthropic_api_key: str | None
    anthropic_model: str

    @classmethod
    def from_env(cls) -> LLMConfig:
        provider = (_get_setting("LLM_PROVIDER", "openai") or "openai").lower().strip()
        if provider not in {"openai", "anthropic"}:
            raise ValueError(
                f"Invalid LLM_PROVIDER '{provider}'. Must be 'openai' or 'anthropic'."
            )
        return cls(
            provider=provider,
            openai_api_key=_get_setting("OPENAI_API_KEY"),
            openai_model=_get_setting("OPENAI_MODEL", "gpt-4o") or "gpt-4o",
            anthropic_api_key=_get_setting("ANTHROPIC_API_KEY"),
            anthropic_model=_get_setting(
                "ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"
            )
            or "claude-3-5-sonnet-20241022",
        )

    def validate(self) -> None:
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
                "Set it in .env locally or in Streamlit Cloud secrets."
            )
        if self.provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic. "
                "Set it in .env locally or in Streamlit Cloud secrets."
            )


def get_llm_config() -> LLMConfig:
    config = LLMConfig.from_env()
    config.validate()
    return config

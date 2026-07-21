"""Job description generation for the NGO Public Health Research Scientist role."""

from __future__ import annotations

from src.config import NGO_MISSION, NGO_NAME
from src.llm_client import LLMClient
from src.models import JobDescription

SYSTEM_PROMPT = f"""You are an expert HR specialist for {NGO_NAME}, an international
humanitarian NGO focused on rescue missions in remote, high-risk global regions.

Organization mission: {NGO_MISSION}

Generate highly detailed, realistic job descriptions for field-deployable public health
research roles. Emphasize:
- Field adaptability in remote/harsh environments (conflict zones, disaster areas)
- Epidemiological research skills and outbreak investigation
- Crisis management under resource constraints
- Cultural competency working with diverse communities
- Resilience and psychological readiness for high-stress deployments

Write in a professional NGO tone. Be specific about deployment contexts (e.g., sub-Saharan
Africa, Southeast Asia, post-disaster zones)."""


def generate_job_description(client: LLMClient | None = None) -> JobDescription:
    """Generate a detailed job description for a Public Health Research Scientist."""
    llm = client or LLMClient()

    user_prompt = f"""Create a comprehensive job description for a "Public Health Research
Scientist" position at {NGO_NAME}.

Include:
1. A compelling summary of the role's impact on humanitarian rescue missions
2. 8 specific responsibilities covering field research, data collection, community
   engagement, outbreak response, and reporting to international health bodies
3. Required qualifications (MD/PhD in public health/epidemiology, field deployment experience, etc.)
4. Preferred qualifications (humanitarian NGO experience, languages, GIS/biostatistics)
5. Critical criteria explicitly covering: field adaptability, epidemiological research,
   crisis management, cultural competency, and resilience
6. A full_text field with the complete formatted job posting ready to share with candidates

The role involves 60-70% field deployment to remote regions with limited infrastructure.

Keep each list item concise (one sentence). Keep full_text under 1200 words."""

    return llm.generate_structured(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=JobDescription,
        temperature=0.4,
    )

"""Resume analysis against job description."""

from __future__ import annotations

from pydantic import BaseModel

from src.config import NGO_NAME
from src.llm_client import LLMClient
from src.models import CandidateEvaluation, JobDescription


class BatchEvaluationResult(BaseModel):
    """Wrapper for batch evaluation LLM response."""

    evaluations: list[CandidateEvaluation]


SYSTEM_PROMPT = f"""You are a senior recruitment analyst for {NGO_NAME}, an international
humanitarian NGO deploying public health professionals to remote, high-risk regions.

Evaluate candidate resumes against job descriptions with rigor and fairness. Focus on:
- Alignment with remote fieldwork in harsh environments
- Epidemiological and public health research expertise
- Crisis management experience
- Cultural competency and language skills
- Resilience indicators (extended deployments, high-stress contexts)

Provide honest, specific assessments. Match scores should reflect genuine fit (0-100%).
Do not inflate scores. Cite concrete evidence from each resume."""


def analyze_resumes(
    job_description: JobDescription,
    resumes: dict[str, str],
    client: LLMClient | None = None,
) -> list[CandidateEvaluation]:
    """
    Analyze all candidate resumes against the job description.

    Args:
        job_description: Generated job posting.
        resumes: Mapping of candidate label -> resume text.
        client: Optional LLM client instance.

    Returns:
        List of CandidateEvaluation objects, one per resume.
    """
    if len(resumes) != 3:
        raise ValueError(f"Expected exactly 3 resumes, got {len(resumes)}")

    llm = client or LLMClient()

    resume_blocks = []
    for label, text in resumes.items():
        resume_blocks.append(f"=== {label} ===\n{text.strip()}\n")

    user_prompt = f"""Analyze the following three candidate resumes against this job description.

JOB DESCRIPTION:
{job_description.full_text}

CANDIDATE RESUMES:
{"".join(resume_blocks)}

For each candidate, provide:
- candidate_name (extract from resume or use the label if name not found)
- match_score (0-100, quantitative fit percentage)
- strengths (3-5 specific points highlighting remote fieldwork and public health alignment)
- weaknesses (2-4 specific gaps such as missing field experience or technical skills)
- fieldwork_alignment (paragraph assessing remote/harsh environment readiness)
- public_health_expertise (paragraph assessing epidemiological/research skills)
- summary (2-3 sentence overall evaluation)

Return exactly 3 evaluations in the 'evaluations' list."""

    result = llm.generate_structured(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=BatchEvaluationResult,
        temperature=0.2,
    )

    if len(result.evaluations) != 3:
        raise ValueError(
            f"Expected 3 evaluations from LLM, got {len(result.evaluations)}"
        )

    return result.evaluations

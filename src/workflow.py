"""Orchestrates the full recruitment screening workflow."""

from __future__ import annotations

from src.analyzer import analyze_resumes
from src.communications import generate_communications
from src.job_description import generate_job_description
from src.llm_client import LLMClient
from src.models import ScreeningResult
from src.ranker import rank_candidates


def run_screening_workflow(
    resumes: dict[str, str],
    client: LLMClient | None = None,
) -> ScreeningResult:
    """
    Execute the complete sequential recruitment workflow.

    Steps:
    1. Generate job description
    2. Analyze all three resumes
    3. Rank candidates 1st through 3rd
    4. Generate rejection, advancement, and manager emails

    Args:
        resumes: Mapping of candidate label -> resume text (exactly 3).
        client: Optional LLM client instance.

    Returns:
        ScreeningResult containing all workflow outputs.
    """
    if len(resumes) != 3:
        raise ValueError("Workflow requires exactly 3 candidate resumes.")

    llm = client or LLMClient()

    job_description = generate_job_description(client=llm)
    evaluations = analyze_resumes(job_description, resumes, client=llm)
    ranking = rank_candidates(job_description, evaluations, client=llm)
    communications = generate_communications(job_description, ranking, client=llm)

    return ScreeningResult(
        job_description=job_description,
        evaluations=evaluations,
        ranking=ranking,
        communications=communications,
    )

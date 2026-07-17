"""Candidate ranking logic."""

from __future__ import annotations

from src.config import NGO_MISSION, NGO_NAME
from src.llm_client import LLMClient
from src.models import (
    CandidateEvaluation,
    CandidateRanking,
    JobDescription,
    RankingLLMResult,
    RankingResult,
)


SYSTEM_PROMPT = f"""You are the lead talent acquisition strategist for {NGO_NAME}.

Organization mission: {NGO_MISSION}

Rank candidates for a Public Health Research Scientist role considering BOTH:
1. Technical/scientific fit for epidemiological field research
2. Cultural and mission alignment with NGO humanitarian rescue values

Ranking criteria (in priority order):
- Demonstrated field deployment in remote/high-risk environments
- Epidemiological research and outbreak response capability
- Crisis management under resource constraints
- Cultural competency and community engagement
- Resilience and adaptability
- Overall match score (use as input, not sole determinant)

Provide clear rationale for each rank. The human hiring manager makes the final decision;
your ranking is advisory only."""


def _match_evaluation(
    name: str, evaluations: list[CandidateEvaluation]
) -> CandidateEvaluation:
    """Find evaluation by candidate name with fuzzy fallback."""
    normalized = name.strip().lower()
    for ev in evaluations:
        if ev.candidate_name.strip().lower() == normalized:
            return ev

    for ev in evaluations:
        if normalized in ev.candidate_name.lower() or ev.candidate_name.lower() in normalized:
            return ev

    raise ValueError(f"No evaluation found for ranked candidate: {name}")


def rank_candidates(
    job_description: JobDescription,
    evaluations: list[CandidateEvaluation],
    client: LLMClient | None = None,
) -> RankingResult:
    """
    Rank three candidates from 1st to 3rd based on overall NGO and role fit.

    Args:
        job_description: The job posting used for screening.
        evaluations: Individual candidate evaluations.
        client: Optional LLM client instance.

    Returns:
        RankingResult with ordered rankings and summary.
    """
    if len(evaluations) != 3:
        raise ValueError(f"Expected exactly 3 evaluations, got {len(evaluations)}")

    llm = client or LLMClient()

    eval_summaries = []
    for ev in evaluations:
        eval_summaries.append(
            f"Candidate: {ev.candidate_name}\n"
            f"Match Score: {ev.match_score}%\n"
            f"Summary: {ev.summary}\n"
            f"Strengths: {'; '.join(ev.strengths)}\n"
            f"Weaknesses: {'; '.join(ev.weaknesses)}\n"
            f"Fieldwork: {ev.fieldwork_alignment}\n"
            f"Public Health: {ev.public_health_expertise}\n"
        )

    evaluations_text = "".join(
        f"--- Evaluation {i + 1} ---\n{s}" for i, s in enumerate(eval_summaries)
    )

    user_prompt = f"""Rank these three candidates from 1st to 3rd place for the following role.

JOB DESCRIPTION SUMMARY:
Title: {job_description.title}
Organization: {job_description.organization}
Critical Criteria: {', '.join(job_description.critical_criteria)}

CANDIDATE EVALUATIONS:
{evaluations_text}

Return:
- rankings: list of 3 entries with rank (1, 2, 3), candidate_name (exact name from
  evaluations), and rationale for that rank
- ranking_summary: overall comparison narrative

Ensure each candidate appears exactly once. Rank 1 = best fit."""

    llm_result = llm.generate_structured(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=RankingLLMResult,
        temperature=0.2,
    )

    merged_rankings: list[CandidateRanking] = []
    for entry in llm_result.rankings:
        evaluation = _match_evaluation(entry.candidate_name, evaluations)
        merged_rankings.append(
            CandidateRanking(
                rank=entry.rank,
                candidate_name=evaluation.candidate_name,
                match_score=evaluation.match_score,
                evaluation=evaluation,
                rationale=entry.rationale,
            )
        )

    return RankingResult(
        rankings=sorted(merged_rankings, key=lambda r: r.rank),
        ranking_summary=llm_result.ranking_summary,
    )

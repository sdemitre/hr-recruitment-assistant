"""Automated communication generation for recruitment workflow."""

from __future__ import annotations

from src.config import NGO_NAME
from src.llm_client import LLMClient
from src.models import CommunicationBundle, JobDescription, RankingResult


SYSTEM_PROMPT = f"""You are a compassionate HR communications specialist for {NGO_NAME},
an international humanitarian NGO.

Write professional, warm, and respectful emails. For rejections:
- Be empathetic and encouraging
- Acknowledge specific strengths from their application
- Leave the door open for future opportunities
- Never be generic or cold

For interview invitations:
- Express genuine enthusiasm for their qualifications
- Provide clear next steps for a virtual interview with an agency manager
- Reflect the organization's humanitarian mission

For internal manager notifications:
- Present rankings clearly (1st, 2nd, 3rd)
- Include summaries, strengths, and weaknesses for each candidate
- Explicitly state that the AI agent performed the evaluation
- Emphasize that the FINAL hiring decision rests with the human manager

Sign emails appropriately for an NGO HR team."""


def generate_communications(
    job_description: JobDescription,
    ranking: RankingResult,
    client: LLMClient | None = None,
) -> CommunicationBundle:
    """
    Generate rejection emails, advancement email, and manager notification.

    Args:
        job_description: The job posting.
        ranking: Final candidate rankings.
        client: Optional LLM client instance.

    Returns:
        CommunicationBundle with all four emails.
    """
    llm = client or LLMClient()

    sorted_rankings = sorted(ranking.rankings, key=lambda r: r.rank)
    top = sorted_rankings[0]
    second = sorted_rankings[1]
    third = sorted_rankings[2]

    ranking_detail = []
    for r in sorted_rankings:
        ev = r.evaluation
        ranking_detail.append(
            f"Rank {r.rank}: {r.candidate_name} ({r.match_score}%)\n"
            f"  Rationale: {r.rationale}\n"
            f"  Summary: {ev.summary}\n"
            f"  Strengths: {'; '.join(ev.strengths)}\n"
            f"  Weaknesses: {'; '.join(ev.weaknesses)}\n"
        )

    user_prompt = f"""Generate recruitment communications for the {job_description.title} role
at {NGO_NAME}.

RANKING RESULTS:
{ranking.ranking_summary}

DETAILED RANKINGS:
{"".join(ranking_detail)}

Generate:
1. rejection_emails (2): One for rank 2 candidate ({second.candidate_name}) and one for
   rank 3 candidate ({third.candidate_name}). Each needs subject and body.
2. advancement_email (1): Interview invitation for rank 1 candidate ({top.candidate_name}).
   Invite them to a virtual interview with an agency manager. Include subject and body.
3. manager_notification (1): Internal email to the hiring manager stating:
   - The AI recruitment agent evaluated 3 resumes for {job_description.title}
   - Clear 1st, 2nd, 3rd ranking with names
   - Summary, strengths, and weaknesses for each candidate
   - Explicit statement that the final hiring decision rests with the human manager

Use professional NGO tone throughout."""

    return llm.generate_structured(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=CommunicationBundle,
        temperature=0.5,
    )

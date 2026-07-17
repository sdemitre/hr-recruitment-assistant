"""Pydantic models for structured LLM outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class JobDescription(BaseModel):
    """Generated job description for the NGO role."""

    title: str
    organization: str
    location: str
    employment_type: str
    summary: str
    responsibilities: list[str]
    required_qualifications: list[str]
    preferred_qualifications: list[str]
    critical_criteria: list[str] = Field(
        description=(
            "Key criteria: field adaptability, epidemiological skills, "
            "crisis management, cultural competency, resilience"
        )
    )
    full_text: str = Field(description="Complete formatted job description")


class CandidateEvaluation(BaseModel):
    """Evaluation of a single candidate against the job description."""

    candidate_name: str
    match_score: float = Field(ge=0, le=100, description="Match score 0-100%")
    strengths: list[str]
    weaknesses: list[str]
    fieldwork_alignment: str = Field(
        description="Assessment of remote/harsh environment fieldwork fit"
    )
    public_health_expertise: str = Field(
        description="Assessment of epidemiological and public health skills"
    )
    summary: str = Field(description="Brief overall evaluation summary")


class RankingEntry(BaseModel):
    """Single ranking entry returned by the LLM (no nested evaluation)."""

    rank: int = Field(ge=1, le=3)
    candidate_name: str
    rationale: str = Field(description="Why this candidate received this rank")


class RankingLLMResult(BaseModel):
    """LLM ranking output before merging with evaluations."""

    rankings: list[RankingEntry] = Field(min_length=3, max_length=3)
    ranking_summary: str


class CandidateRanking(BaseModel):
    """Ranked candidate with evaluation details."""

    rank: int = Field(ge=1, le=3)
    candidate_name: str
    match_score: float
    evaluation: CandidateEvaluation
    rationale: str = Field(description="Why this candidate received this rank")


class RankingResult(BaseModel):
    """Full ranking of all three candidates."""

    rankings: list[CandidateRanking] = Field(min_length=3, max_length=3)
    ranking_summary: str = Field(
        description="Overall summary of how candidates compare for NGO culture and role fit"
    )


class RejectionEmail(BaseModel):
    """Rejection email for a candidate."""

    candidate_name: str
    subject: str
    body: str


class AdvancementEmail(BaseModel):
    """Interview invitation for top candidate."""

    candidate_name: str
    subject: str
    body: str


class ManagerNotification(BaseModel):
    """Internal email to hiring manager."""

    subject: str
    body: str


class CommunicationBundle(BaseModel):
    """All generated communications for the recruitment workflow."""

    rejection_emails: list[RejectionEmail] = Field(min_length=2, max_length=2)
    advancement_email: AdvancementEmail
    manager_notification: ManagerNotification


class ScreeningResult(BaseModel):
    """Complete output of the recruitment screening workflow."""

    job_description: JobDescription
    evaluations: list[CandidateEvaluation]
    ranking: RankingResult
    communications: CommunicationBundle

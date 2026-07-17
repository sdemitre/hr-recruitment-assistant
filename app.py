"""Streamlit web UI for the HR Recruitment Assistant Agent."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.config import NGO_MISSION, NGO_NAME, LLMConfig
from src.llm_client import LLMError
from src.resume_parser import extract_text_from_bytes
from src.workflow import run_screening_workflow

SAMPLE_DIR = Path(__file__).parent / "data" / "sample_resumes"

st.set_page_config(
    page_title="GRHI Recruitment Assistant",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 HR Recruitment Assistant Agent")
st.subheader(f"{NGO_NAME}")
st.caption(NGO_MISSION)

st.info(
    "**Human-in-the-loop:** This AI agent screens, ranks, and drafts communications. "
    "Final hiring decisions remain strictly with human recruiters."
)


def _load_sample(name: str) -> str:
    return (SAMPLE_DIR / name).read_text(encoding="utf-8")


def _check_api_config() -> LLMConfig | None:
    try:
        config = LLMConfig.from_env()
        config.validate()
        return config
    except ValueError as exc:
        st.error(str(exc))
        st.markdown(
            "Copy `.env.example` to `.env` and set your API key:\n"
            "```bash\ncp .env.example .env\n```"
        )
        return None


def _render_job_description(result) -> None:
    jd = result.job_description
    st.header("📋 Generated Job Description")
    st.markdown(f"**{jd.title}** — {jd.organization}")
    st.markdown(f"📍 {jd.location} | {jd.employment_type}")
    st.markdown(jd.full_text)
    with st.expander("Structured Details"):
        st.markdown("**Responsibilities**")
        for item in jd.responsibilities:
            st.markdown(f"- {item}")
        st.markdown("**Required Qualifications**")
        for item in jd.required_qualifications:
            st.markdown(f"- {item}")
        st.markdown("**Critical Criteria**")
        for item in jd.critical_criteria:
            st.markdown(f"- {item}")


def _render_evaluations(result) -> None:
    st.header("🔍 Candidate Evaluations")
    cols = st.columns(3)
    for idx, ev in enumerate(result.evaluations):
        with cols[idx]:
            st.subheader(ev.candidate_name)
            st.metric("Match Score", f"{ev.match_score:.0f}%")
            st.markdown(f"**Summary:** {ev.summary}")
            st.markdown("**Strengths**")
            for s in ev.strengths:
                st.markdown(f"- {s}")
            st.markdown("**Weaknesses / Gaps**")
            for w in ev.weaknesses:
                st.markdown(f"- {w}")
            with st.expander("Detailed Assessment"):
                st.markdown(f"**Fieldwork:** {ev.fieldwork_alignment}")
                st.markdown(f"**Public Health:** {ev.public_health_expertise}")


def _render_ranking(result) -> None:
    st.header("🏆 Candidate Ranking")
    st.markdown(result.ranking.ranking_summary)

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    for r in sorted(result.ranking.rankings, key=lambda x: x.rank):
        st.markdown(
            f"### {medals.get(r.rank, '')} #{r.rank} — {r.candidate_name} "
            f"({r.match_score:.0f}%)"
        )
        st.markdown(f"**Rationale:** {r.rationale}")


def _render_communications(result) -> None:
    st.header("✉️ Generated Communications")
    comms = result.communications

    tab_reject, tab_advance, tab_manager = st.tabs(
        ["Rejection Emails (×2)", "Interview Invitation", "Manager Notification"]
    )

    with tab_reject:
        for i, email in enumerate(comms.rejection_emails, 1):
            st.subheader(f"Rejection — {email.candidate_name}")
            st.markdown(f"**Subject:** {email.subject}")
            st.text_area(
                f"Body ({email.candidate_name})",
                email.body,
                height=250,
                key=f"reject_{i}",
            )

    with tab_advance:
        adv = comms.advancement_email
        st.subheader(f"Interview Invitation — {adv.candidate_name}")
        st.markdown(f"**Subject:** {adv.subject}")
        st.text_area("Body", adv.body, height=250, key="advance")

    with tab_manager:
        mgr = comms.manager_notification
        st.markdown(f"**Subject:** {mgr.subject}")
        st.text_area("Body", mgr.body, height=400, key="manager")


def _render_export(result) -> None:
    st.header("💾 Export Results")
    export_data = result.model_dump()
    st.download_button(
        label="Download Full Report (JSON)",
        data=json.dumps(export_data, indent=2),
        file_name="screening_results.json",
        mime="application/json",
    )


# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    config = _check_api_config()
    if config:
        st.success(f"LLM Provider: **{config.provider}**")
        model = (
            config.openai_model
            if config.provider == "openai"
            else config.anthropic_model
        )
        st.caption(f"Model: {model}")

    st.divider()
    st.header("Load Sample Resumes")
    if st.button("Load All Sample Candidates"):
        st.session_state["resume_1"] = _load_sample("candidate_1_amara_okonkwo.txt")
        st.session_state["resume_2"] = _load_sample("candidate_2_james_whitfield.txt")
        st.session_state["resume_3"] = _load_sample("candidate_3_sofia_reyes.txt")
        st.session_state["name_1"] = "Amara Okonkwo"
        st.session_state["name_2"] = "James Whitfield"
        st.session_state["name_3"] = "Sofia Reyes"
        st.success("Sample resumes loaded!")

# --- Main input area ---
st.header("👥 Candidate Resumes")
st.markdown("Provide three candidate resumes via text paste or file upload.")

candidates = []
for i in range(1, 4):
    with st.expander(f"Candidate {i}", expanded=(i == 1)):
        name = st.text_input(
            "Candidate Name",
            value=st.session_state.get(f"name_{i}", f"Candidate {i}"),
            key=f"name_{i}",
        )
        uploaded = st.file_uploader(
            f"Upload resume (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            key=f"upload_{i}",
        )
        default_text = st.session_state.get(f"resume_{i}", "")
        if uploaded is not None:
            try:
                default_text = extract_text_from_bytes(
                    uploaded.read(), uploaded.name
                )
                st.success(f"Extracted text from {uploaded.name}")
            except ValueError as exc:
                st.error(str(exc))

        resume_text = st.text_area(
            "Resume Text",
            value=default_text,
            height=200,
            key=f"resume_{i}",
            placeholder="Paste resume content here...",
        )
        candidates.append({"name": name, "text": resume_text})

st.divider()

run_disabled = not config or any(not c["text"].strip() for c in candidates)

if st.button(
    "🚀 Run Recruitment Screening Workflow",
    type="primary",
    disabled=run_disabled,
):
    if not config:
        st.stop()

    resumes = {
        c["name"]: c["text"] for c in candidates if c["text"].strip()
    }

    if len(resumes) != 3:
        st.error("Please provide exactly 3 non-empty resumes.")
        st.stop()

    progress = st.progress(0, text="Starting workflow...")
    try:
        progress.progress(10, text="Generating job description...")
        result = run_screening_workflow(resumes)
        progress.progress(100, text="Complete!")
        st.session_state["screening_result"] = result
        st.success("Screening workflow completed successfully!")
    except LLMError as exc:
        st.error(f"LLM API error: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Workflow failed: {exc}")
        st.stop()

if "screening_result" in st.session_state:
    result = st.session_state["screening_result"]
    st.divider()
    _render_job_description(result)
    st.divider()
    _render_evaluations(result)
    st.divider()
    _render_ranking(result)
    st.divider()
    _render_communications(result)
    st.divider()
    _render_export(result)

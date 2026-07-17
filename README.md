# HR Recruitment Assistant Agent

An AI-powered recruitment screening system for **Global Rescue Health Initiative (GRHI)**, an international humanitarian NGO deploying public health research scientists to remote, high-risk global regions.

The agent automates resume screening, candidate ranking, and communication drafting while keeping **final hiring decisions strictly with human recruiters**.

## Features

- **Job Description Generation** — Creates detailed, realistic postings for Public Health Research Scientist roles
- **Resume Analysis** — Parses text, PDF, or DOCX resumes against the job description
- **Match Matrix** — Strengths, weaknesses, and 0–100% match scores per candidate
- **Ranking** — Ranks three candidates by NGO mission fit and role requirements
- **Communication Drafts** — Rejection emails (×2), interview invitation (×1), and internal manager notification (×1)

## Requirements

- Python 3.10+
- OpenAI API key (`gpt-4o`) **or** Anthropic API key (`claude-3-5-sonnet`)

## Installation

```bash
cd hr-recruitment-assistant
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and add your API key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Choose provider: openai or anthropic
LLM_PROVIDER=openai

# OpenAI
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o

# Anthropic (alternative)
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

## Usage

### Streamlit Web UI (recommended)

```bash
streamlit run app.py
```

1. Click **Load All Sample Candidates** in the sidebar, or paste/upload three resumes
2. Click **Run Recruitment Screening Workflow**
3. Review job description, evaluations, rankings, and generated emails
4. Download the full JSON report if needed

### CLI

Run with bundled sample resumes:

```bash
python cli.py --samples
```

Run with your own resume files:

```bash
python cli.py --resume path/to/resume1.txt --resume path/to/resume2.pdf --resume path/to/resume3.docx --output results.json
```

## Workflow

```
┌─────────────────────────┐
│ 1. Generate Job Desc    │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 2. Analyze 3 Resumes    │  → Strengths, Weaknesses, Match Score
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 3. Rank Candidates      │  → 1st, 2nd, 3rd by NGO + role fit
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 4. Generate Emails      │  → 2 rejections, 1 interview invite, 1 manager note
└─────────────────────────┘
```

## Project Structure

```
hr-recruitment-assistant/
├── app.py                  # Streamlit web UI
├── cli.py                  # Command-line interface
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py           # Environment and NGO constants
│   ├── llm_client.py       # OpenAI / Anthropic structured output client
│   ├── models.py           # Pydantic schemas
│   ├── job_description.py  # Step 1: JD generation
│   ├── resume_parser.py    # PDF/DOCX/TXT extraction
│   ├── analyzer.py         # Step 2: Resume evaluation
│   ├── ranker.py           # Step 3: Candidate ranking
│   ├── communications.py   # Step 4: Email generation
│   └── workflow.py         # End-to-end orchestration
└── data/
    └── sample_resumes/     # Three example candidates
```

## Sample Candidates

| Candidate | Profile |
|-----------|---------|
| **Amara Okonkwo** | Senior field epidemiologist, MSF deployments, PhD |
| **James Whitfield** | Academic modeler, strong publications, no field experience |
| **Sofia Reyes** | IRC field manager, disaster response, MPH only |

Expected ranking: Amara (1st) > Sofia (2nd) > James (3rd), though the LLM may vary slightly based on weighting.

## Human-in-the-Loop

The manager notification email explicitly states:

> The AI recruitment agent evaluated three resumes and provides this advisory ranking. **The final hiring decision rests with the human manager.**

No emails are sent automatically — all outputs are drafts for recruiter review.

## Error Handling

- Missing API keys raise clear configuration errors at startup
- LLM API failures are caught and surfaced with descriptive messages
- Unsupported file types and empty PDFs are rejected with actionable errors

## License

MIT

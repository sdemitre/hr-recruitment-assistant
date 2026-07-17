#!/usr/bin/env python3
"""CLI entry point for the HR Recruitment Assistant Agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import NGO_NAME, get_llm_config
from src.llm_client import LLMError
from src.resume_parser import extract_text_from_file
from src.workflow import run_screening_workflow

SAMPLE_DIR = Path(__file__).parent / "data" / "sample_resumes"


def load_sample_resumes() -> dict[str, str]:
    """Load the three bundled sample resumes."""
    files = sorted(SAMPLE_DIR.glob("candidate_*.txt"))
    if len(files) < 3:
        raise FileNotFoundError(f"Expected 3 sample resumes in {SAMPLE_DIR}")

    resumes = {}
    for path in files[:3]:
        text = path.read_text(encoding="utf-8")
        name_line = text.strip().split("\n")[0]
        resumes[name_line] = text
    return resumes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"{NGO_NAME} — HR Recruitment Assistant Agent (CLI)"
    )
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Run workflow with bundled sample resumes",
    )
    parser.add_argument(
        "--resume",
        action="append",
        metavar="PATH",
        help="Path to a resume file (.txt, .pdf, .docx). Provide exactly 3.",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="Write full JSON results to this file",
    )
    args = parser.parse_args()

    try:
        get_llm_config()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        print("Copy .env.example to .env and set your API key.", file=sys.stderr)
        return 1

    try:
        if args.samples:
            resumes = load_sample_resumes()
        elif args.resume:
            if len(args.resume) != 3:
                print("Error: Provide exactly 3 --resume paths.", file=sys.stderr)
                return 1
            resumes = {}
            for path in args.resume:
                text = extract_text_from_file(path)
                name = Path(path).stem.replace("_", " ").title()
                resumes[name] = text
        else:
            print("Error: Use --samples or provide 3 --resume paths.", file=sys.stderr)
            parser.print_help()
            return 1

        print(f"Running screening workflow for {len(resumes)} candidates...")
        result = run_screening_workflow(resumes)

        print("\n" + "=" * 60)
        print("RANKING RESULTS")
        print("=" * 60)
        for r in sorted(result.ranking.rankings, key=lambda x: x.rank):
            print(f"  #{r.rank} {r.candidate_name} — {r.match_score:.0f}%")

        print("\n" + "=" * 60)
        print("TOP CANDIDATE — INTERVIEW INVITATION")
        print("=" * 60)
        adv = result.communications.advancement_email
        print(f"Subject: {adv.subject}\n{adv.body}")

        output_json = json.dumps(result.model_dump(), indent=2)
        if args.output:
            Path(args.output).write_text(output_json, encoding="utf-8")
            print(f"\nFull results written to {args.output}")
        else:
            print("\nUse --output to save the full JSON report.")

        return 0

    except LLMError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

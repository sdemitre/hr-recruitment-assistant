"""Resume text extraction from plain text, PDF, and DOCX sources."""

from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader
from docx import Document


def extract_text_from_bytes(content: bytes, filename: str) -> str:
    """Extract resume text from uploaded file bytes."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(content)
    if suffix in {".docx", ".doc"}:
        return _extract_docx(content)
    if suffix in {".txt", ".md"}:
        return content.decode("utf-8", errors="replace").strip()

    raise ValueError(
        f"Unsupported file type '{suffix}'. Use .txt, .pdf, or .docx."
    )


def extract_text_from_file(path: str | Path) -> str:
    """Extract resume text from a file on disk."""
    file_path = Path(path)
    content = file_path.read_bytes()
    return extract_text_from_bytes(content, file_path.name)


def _extract_pdf(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError("Could not extract text from PDF. The file may be scanned/image-only.")
    return text


def _extract_docx(content: bytes) -> str:
    document = Document(io.BytesIO(content))
    paragraphs = [para.text for para in document.paragraphs if para.text.strip()]
    text = "\n".join(paragraphs).strip()
    if not text:
        raise ValueError("Could not extract text from DOCX file.")
    return text

"""PDF classification and text extraction utilities using pdf_oxide."""

import math
from typing import List

from pdf_oxide import PdfDocument

from .config import (
    get_pdf_sample_min,
    get_pdf_sample_max,
    get_pdf_word_threshold,
    get_pdf_text_ratio,
)


def calculate_sample_size(
    total_pages: int,
    min_sample: int,
    max_sample: int,
) -> int:
    """Determine how many pages to sample for text-vs-scanned detection.

    Uses sqrt(total_pages) for adaptive coverage: small PDFs get high
    coverage, large PDFs get efficient lower coverage.  The result is
    clamped between *min_sample* and *max_sample*.
    """
    if total_pages <= min_sample:
        return total_pages
    sample = max(min_sample, math.ceil(math.sqrt(total_pages)))
    return min(sample, max_sample)


def get_sample_page_indices(total_pages: int, sample_size: int) -> List[int]:
    """Return evenly-spaced page indices across the document."""
    if sample_size >= total_pages:
        return list(range(total_pages))
    step = total_pages / sample_size
    return [int(i * step) for i in range(sample_size)]


def is_text_based_pdf(file_path: str) -> bool:
    """Classify a PDF as text-based or scanned.

    Adaptively samples pages (sqrt scaling) and counts words on each
    sampled page.  If enough pages exceed the word threshold the PDF
    is considered text-based.
    """
    doc = PdfDocument(file_path)
    total_pages = doc.page_count()

    if total_pages == 0:
        return False

    min_sample = get_pdf_sample_min()
    max_sample = get_pdf_sample_max()
    word_threshold = get_pdf_word_threshold()
    text_ratio = get_pdf_text_ratio()

    sample_size = calculate_sample_size(total_pages, min_sample, max_sample)
    indices = get_sample_page_indices(total_pages, sample_size)

    text_page_count = 0
    for idx in indices:
        text = doc.extract_text(idx)
        if len(text.split()) >= word_threshold:
            text_page_count += 1

    return text_page_count >= len(indices) * text_ratio


def extract_markdown_with_pdf_oxide(file_path: str) -> List[str]:
    """Convert every page of a PDF to Markdown using pdf_oxide."""
    doc = PdfDocument(file_path)
    total_pages = doc.page_count()
    return [doc.to_markdown(i, detect_headings=True) for i in range(total_pages)]

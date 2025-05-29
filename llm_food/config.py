import os
from typing import Optional, Union


def get_pdf_backend():
    return os.getenv("PDF_BACKEND", "gemini")


"""Configuration and environment variables"""

DEFAULT_GEMINI_OCR_PROMPT = """OCR this document to Markdown with text formatting such as bold, italic, headings, tables, numbered and bulleted lists properly rendered in Markdown.

Do not surround the output with Markdown fences.

Preserve as much content as possible, such as headings, tables, lists. etc.

Do not add any preamble or additional explanation of any other kind --simply output the well-formatted text output in Markdown.

If the text contains complex equations, wrap them in $$...$$ for LaTeX rendering.
"""


def get_gemini_prompt():
    return os.getenv("GEMINI_OCR_PROMPT", DEFAULT_GEMINI_OCR_PROMPT)


def get_api_auth_token() -> Optional[str]:
    return os.getenv("API_AUTH_TOKEN")


# other configuration
def get_gcs_project_id():
    return os.getenv("GOOGLE_CLOUD_PROJECT")


def get_max_file_size_bytes() -> Union[int, None]:
    max_size_mb_str = os.getenv("MAX_FILE_SIZE_MB")
    if max_size_mb_str:
        try:
            max_size_mb = int(max_size_mb_str)
            if max_size_mb > 0:
                return max_size_mb * 1024 * 1024  # Convert MB to Bytes
            else:
                # Log this invalid configuration? For now, treat as no limit.
                pass
        except ValueError:
            # Log this invalid configuration? For now, treat as no limit.
            pass
    return None  # No limit or invalid value treated as no limit


SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".rtf", ".pptx", ".html", ".htm"]

# --- DuckDB Setup ---
DUCKDB_FILE = os.getenv("DUCKDB_FILE", "batch_tasks.duckdb")

# --- Environment Variables for Batch Processing ---
GCS_BATCH_BUCKET = os.getenv("GCS_BUCKET")
GEMINI_MODEL_FOR_VISION = os.getenv(
    "GEMINI_MODEL_FOR_VISION", "gemini-2.0-flash-001"
)  # Or other vision model

if not GCS_BATCH_BUCKET:
    print(
        "Warning: GCS_BUCKET environment variable is not set. PDF batch processing will fail."
    )

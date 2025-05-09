from fastapi import (
    FastAPI,
    UploadFile,
    File,
    BackgroundTasks,
    HTTPException,
    Query,
    Depends,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import asyncio
import base64
from typing import List, Union, Optional
import uuid
import os
import hashlib
import httpx
import mammoth

# Imports for GCS
from google.cloud import storage
from google.oauth2 import service_account  # For local testing with service account
from markdownify import markdownify

from io import BytesIO
from striprtf.striprtf import rtf_to_text
from pptx import Presentation
import trafilatura


# --- Conditional imports based on the PDF backend ---
def get_pdf_backend():
    return os.getenv("PDF_BACKEND", "gemini")


match get_pdf_backend():
    case "pymupdf4llm":
        from pymupdf4llm import to_markdown
        import pymupdf
    case "pypdf2":
        from pypdf import PdfReader
    case "gemini":
        from pdf2image import convert_from_bytes
        from google import genai
        from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions

        prompt = """OCR this document to Markdown with text formatting such as bold, italic, headings, tables, numbered and bulleted lists properly rendered in Markdown Do not suround the out with Markdown fences. Preserve as much content as possible, such as headings, tables, lists. etc. Do not add any preamble or additional explanation of any other kind --simply output the well-formatted text output in Markdown."""
    case invalid_backend:
        raise ValueError(f"Invalid PDF backend: {invalid_backend}")

app = FastAPI()

# --- Security ---
bearer_scheme = HTTPBearer(
    auto_error=False
)  # auto_error=False to handle optional token & custom errors


def get_api_auth_token() -> Optional[str]:
    return os.getenv("API_AUTH_TOKEN")


async def authenticate_request(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> None:
    configured_token = get_api_auth_token()
    if configured_token:  # Only enforce auth if a token is configured server-side
        if credentials is None:
            raise HTTPException(
                status_code=401,
                detail="Not authenticated. Authorization header is missing.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if credentials.scheme != "Bearer":
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication scheme. Only Bearer is supported.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if credentials.token != configured_token:
            raise HTTPException(
                status_code=403,
                detail="Invalid token.",
            )
    # If no token is configured server-side, or if authentication passes, do nothing.
    return


# other configuration
def get_gcs_project_id():
    return os.getenv("GOOGLE_CLOUD_PROJECT")


# For local GCS testing with a service account JSON file
def get_gcs_credentials():
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        return service_account.Credentials.from_service_account_file(credentials_path)
    return None  # Fallback to default environment auth if not set


def get_gemini_client():
    project = get_gcs_project_id()
    location = "us-central1"
    api_key = os.getenv("ALTAI_GEMINI_API_KEY")
    client = (
        genai.Client(vertexai=False, api_key=api_key) if api_key else genai.Client()
    )
    return client


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


class ConversionResponse(BaseModel):
    filename: str
    content_hash: str
    texts: List[str]


class BatchRequest(BaseModel):
    input_paths: Union[str, List[str]]
    output_path: str


TASKS = {}


def _process_docx_sync(content_bytes: bytes) -> List[str]:
    try:
        doc = BytesIO(content_bytes)
        doc_html = mammoth.convert_to_html(doc).value
        doc_md = markdownify(doc_html).strip()
        return [doc_md]
    except Exception as e:
        return [f"Error processing DOCX: {str(e)}"]


def _process_rtf_sync(content_bytes: bytes) -> List[str]:
    try:
        return [rtf_to_text(content_bytes.decode("utf-8", errors="ignore"))]
    except Exception as e:
        return [f"Error processing RTF: {str(e)}"]


def _process_pptx_sync(content_bytes: bytes) -> List[str]:
    try:
        prs = Presentation(BytesIO(content_bytes))
        # Corrected list comprehension for PPTX to build a single string per slide, then list of slide texts
        slide_texts = []
        for slide in prs.slides:
            text_on_slide = "\n".join(
                shape.text
                for shape in slide.shapes
                if hasattr(shape, "text") and shape.text
            )
            if text_on_slide:  # Only add if there's text
                slide_texts.append(text_on_slide)
        return (
            slide_texts if slide_texts else [""]
        )  # Return list of slide texts, or list with empty string if no text
    except Exception as e:
        return [f"Error processing PPTX: {str(e)}"]


def _process_html_sync(content_bytes: bytes) -> List[str]:
    try:
        extracted_text = trafilatura.extract(
            content_bytes.decode("utf-8", errors="ignore"), output_format="markdown"
        )
        return [extracted_text if extracted_text is not None else ""]
    except Exception as e:
        return [f"Error processing HTML: {str(e)}"]


def _process_pdf_pymupdf4llm_sync(content_bytes: bytes) -> List[str]:
    try:
        pymupdf_doc = pymupdf.Document(stream=content_bytes, filetype="pdf")
        page_data_list = to_markdown(pymupdf_doc, page_chunks=True)
        return [page_dict.get("text", "") for page_dict in page_data_list]
    except Exception as e:
        return [f"Error processing PDF with pymupdf4llm: {str(e)}"]


def _process_pdf_pypdf2_sync(content_bytes: bytes) -> List[str]:
    try:
        reader = PdfReader(BytesIO(content_bytes))
        return [p.extract_text() or "" for p in reader.pages]
    except Exception as e:
        return [f"Error processing PDF with pypdf: {str(e)}"]


async def _process_file_content(
    ext: str, content: bytes, pdf_backend_choice: str
) -> List[str]:
    texts_list: List[str] = []
    if ext == ".pdf":
        if pdf_backend_choice == "pymupdf4llm":
            texts_list = await asyncio.to_thread(_process_pdf_pymupdf4llm_sync, content)
        elif pdf_backend_choice == "pypdf2":
            texts_list = await asyncio.to_thread(_process_pdf_pypdf2_sync, content)
        elif pdf_backend_choice == "gemini":
            pages = convert_from_bytes(content)
            images_b64 = []
            for page in pages:
                buffer = BytesIO()
                page.save(buffer, format="PNG")
                image_data = buffer.getvalue()
                b64_str = base64.b64encode(image_data).decode("utf-8")
                images_b64.append(b64_str)
            client = get_gemini_client()
            payloads = [
                [
                    {"inline_data": {"data": b64_str, "mime_type": "image/png"}},
                    {"text": prompt},
                ]
                for b64_str in images_b64
            ]
            results = await asyncio.gather(
                *[
                    client.aio.models.generate_content(
                        model="gemini-2.0-flash", contents=payload
                    )
                    for payload in payloads
                ]
            )
            texts_list = [result.text for result in results]
        else:
            texts_list = ["Invalid PDF backend specified."]
    elif ext in [".docx"]:
        texts_list = await asyncio.to_thread(_process_docx_sync, content)
    elif ext in [".rtf"]:
        texts_list = await asyncio.to_thread(_process_rtf_sync, content)
    elif ext in [".pptx"]:
        texts_list = await asyncio.to_thread(_process_pptx_sync, content)
    elif ext in [".html", ".htm"]:
        texts_list = await asyncio.to_thread(_process_html_sync, content)
    else:
        texts_list = ["Unsupported file type encountered in _process_file_content."]
    return texts_list


@app.post(
    "/convert",
    response_model=ConversionResponse,
    dependencies=[Depends(authenticate_request)],
)
async def convert_file_upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    content = await file.read()

    max_size = get_max_file_size_bytes()
    if max_size is not None and len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File size {len(content) / (1024 * 1024):.2f}MB exceeds maximum allowed size of {max_size / (1024 * 1024):.2f}MB.",
        )

    content_hash = hashlib.sha256(content).hexdigest()
    pdf_backend_choice = get_pdf_backend()

    texts_list = await _process_file_content(ext, content, pdf_backend_choice)

    if texts_list and (
        texts_list[0].startswith("Error processing")
        or texts_list[0].startswith("Invalid PDF backend")
        or texts_list[0].startswith("Unsupported file type")
    ):
        raise HTTPException(status_code=400, detail=texts_list[0])

    return ConversionResponse(
        filename=file.filename, content_hash=content_hash, texts=texts_list
    )


@app.get(
    "/convert",
    response_model=ConversionResponse,
    dependencies=[Depends(authenticate_request)],
)
async def convert_url(
    url: str = Query(..., description="URL of the webpage to convert to Markdown"),
):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            html_content = response.text
            content_bytes = html_content.encode("utf-8")
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Error fetching URL: {str(e)}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Error fetching URL: {e.response.reason_phrase}",
        )

    if not html_content:
        raise HTTPException(status_code=400, detail="Fetched content is empty.")

    content_hash = hashlib.sha256(content_bytes).hexdigest()

    extracted_text = trafilatura.extract(html_content, output_format="markdown")
    texts_list = [extracted_text if extracted_text is not None else ""]

    filename = os.path.basename(url) or url

    return ConversionResponse(
        filename=filename, content_hash=content_hash, texts=texts_list
    )


@app.get("/status/{task_id}", dependencies=[Depends(authenticate_request)])
def status(task_id: str):
    return TASKS.get(task_id, {"status": "not_found"})


@app.post("/batch", dependencies=[Depends(authenticate_request)])
def batch(request: BatchRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "status": "pending",
        "processed_files": 0,
        "failed_files": 0,
        "details": [],
    }
    background_tasks.add_task(
        run_batch_task, request.input_paths, request.output_path, task_id
    )
    return {"task_id": task_id}


# Actual batch processing logic
async def run_batch_task(
    input_paths: Union[str, List[str]], output_gcs_path_str: str, task_id: str
):
    TASKS[task_id]["status"] = "initializing"
    project_id = get_gcs_project_id()
    credentials = get_gcs_credentials()

    if (
        not project_id and not credentials
    ):  # Basic check if running outside GCP and no local creds
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["details"].append(
            "GCS_PROJECT_ID not set or GOOGLE_APPLICATION_CREDENTIALS not found for GCS access."
        )
        return

    try:
        storage_client = storage.Client(project=project_id, credentials=credentials)
    except Exception as e:
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["details"].append(f"Failed to initialize GCS client: {str(e)}")
        return

    TASKS[task_id]["status"] = "running"
    files_to_process = []

    pdf_backend_choice = get_pdf_backend()

    try:
        if isinstance(input_paths, str):  # It's a GCS directory path
            TASKS[task_id]["details"].append(f"Processing directory: {input_paths}")
            bucket_name, prefix = input_paths.replace("gs://", "").split("/", 1)
            bucket = storage_client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix))
            for blob in blobs:
                if any(
                    blob.name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS
                ) and not blob.name.endswith("/"):  # check if not a directory itself
                    files_to_process.append((bucket_name, blob.name))
        else:  # It's a list of GCS file paths
            TASKS[task_id]["details"].append(
                f"Processing list of files: {len(input_paths)} files"
            )
            for path_str in input_paths:
                if not path_str.startswith("gs://"):
                    TASKS[task_id]["details"].append(
                        f"Skipping invalid GCS path: {path_str}"
                    )
                    continue
                bucket_name, blob_name = path_str.replace("gs://", "").split("/", 1)
                if any(blob_name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    files_to_process.append((bucket_name, blob_name))
                else:
                    TASKS[task_id]["details"].append(
                        f"Skipping unsupported file type: {blob_name}"
                    )

        if not files_to_process:
            TASKS[task_id]["details"].append("No supported files found to process.")
            TASKS[task_id]["status"] = "completed_with_no_files"
            return

        TASKS[task_id]["total_files"] = len(files_to_process)
        output_bucket_name, output_prefix = output_gcs_path_str.replace(
            "gs://", ""
        ).split("/", 1)
        output_bucket = storage_client.bucket(output_bucket_name)

        for i, (bucket_name, blob_name) in enumerate(files_to_process):
            TASKS[task_id]["current_file_processing"] = (
                f"{i + 1}/{len(files_to_process)}: {blob_name}"
            )
            try:
                input_bucket_obj = storage_client.bucket(bucket_name)
                blob_obj = input_bucket_obj.blob(blob_name)
                file_content_bytes = blob_obj.download_as_bytes()

                file_ext = os.path.splitext(blob_name)[1].lower()

                markdown_texts = await _process_file_content(
                    file_ext, file_content_bytes, pdf_backend_choice
                )

                # Combine all pages/sections into a single markdown string for the output file
                full_markdown_output = "\n\n---\n\n".join(markdown_texts)

                output_blob_name = (
                    output_prefix
                    + "/"
                    + os.path.basename(blob_name).rsplit(".", 1)[0]
                    + ".md"
                )

                output_blob_obj = output_bucket.blob(output_blob_name)

                output_blob_obj.upload_from_string(
                    full_markdown_output, content_type="text/markdown"
                )
                TASKS[task_id]["processed_files"] += 1
                TASKS[task_id]["details"].append(
                    f"Successfully processed and uploaded: {blob_name} to {output_blob_obj.public_url if hasattr(output_blob_obj, 'public_url') else output_blob_obj.name}"
                )
            except Exception as e:
                TASKS[task_id]["failed_files"] += 1
                TASKS[task_id]["details"].append(
                    f"Failed to process file {blob_name}: {str(e)}"
                )

        TASKS[task_id]["status"] = "completed"

    except Exception as e:
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["details"].append(
            f"An unexpected error occurred during batch processing: {str(e)}"
        )
    finally:
        if "current_file_processing" in TASKS[task_id]:
            del TASKS[task_id]["current_file_processing"]  # Clean up

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel
from typing import List, Literal, Union
import uuid
import os
import shutil
import hashlib
import httpx
import time  # Keep for now, for placeholder task simulation if GCS fails

# Imports for GCS
from google.cloud import storage
from google.oauth2 import service_account  # For local testing with service account

from io import BytesIO
from pymupdf4llm import to_markdown
import pymupdf
from pypdf import PdfReader
from docx import Document
from striprtf.striprtf import rtf_to_text
from pptx import Presentation
import trafilatura

app = FastAPI()


# --- Configuration ---
def get_pdf_backend():
    return os.getenv("PDF_BACKEND", "pymupdf4llm")


def get_gcs_project_id():
    return os.getenv("GOOGLE_CLOUD_PROJECT")


# For local GCS testing with a service account JSON file
def get_gcs_credentials():
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        return service_account.Credentials.from_service_account_file(credentials_path)
    return None  # Fallback to default environment auth if not set


SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".rtf", ".pptx", ".html", ".htm"]


class ConversionResponse(BaseModel):
    filename: str
    content_hash: str
    texts: List[str]


class BatchRequest(BaseModel):
    input_paths: Union[str, List[str]]
    output_path: str


TASKS = {}


def _process_file_content(
    ext: str, content: bytes, pdf_backend_choice: str
) -> List[str]:
    texts_list: List[str] = []
    if ext == ".pdf":
        if pdf_backend_choice == "pymupdf4llm":
            try:
                pymupdf_doc = pymupdf.Document(stream=content, filetype="pdf")
                page_data_list = to_markdown(pymupdf_doc, page_chunks=True)
                texts_list = [page_dict.get("text", "") for page_dict in page_data_list]
            except Exception as e:
                texts_list = [f"Error processing PDF with pymupdf4llm: {str(e)}"]
        elif pdf_backend_choice == "pypdf2":
            try:
                reader = PdfReader(BytesIO(content))
                texts_list = [p.extract_text() or "" for p in reader.pages]
            except Exception as e:
                texts_list = [f"Error processing PDF with pypdf: {str(e)}"]
        else:
            texts_list = ["Invalid PDF backend specified."]
    elif ext in [".docx"]:
        try:
            doc = Document(BytesIO(content))
            texts_list = ["\n".join(p.text for p in doc.paragraphs)]
        except Exception as e:
            texts_list = [f"Error processing DOCX: {str(e)}"]
    elif ext in [".rtf"]:
        try:
            texts_list = [rtf_to_text(content.decode("utf-8", errors="ignore"))]
        except Exception as e:
            texts_list = [f"Error processing RTF: {str(e)}"]
    elif ext in [".pptx"]:
        try:
            prs = Presentation(BytesIO(content))
            texts_list = [
                "\n".join(
                    shape.text
                    for slide in prs.slides
                    for shape in slide.shapes
                    if hasattr(shape, "text") and shape.text
                )
                for slide in prs.slides
            ]
        except Exception as e:
            texts_list = [f"Error processing PPTX: {str(e)}"]
    elif ext in [".html", ".htm"]:
        try:
            extracted_text = trafilatura.extract(
                content.decode("utf-8", errors="ignore"), output_format="markdown"
            )
            texts_list = [extracted_text if extracted_text is not None else ""]
        except Exception as e:
            texts_list = [f"Error processing HTML: {str(e)}"]
    else:
        texts_list = ["Unsupported file type encountered in _process_file_content."]
    return texts_list


@app.post("/convert", response_model=ConversionResponse)
async def convert_file_upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    content = await file.read()

    content_hash = hashlib.sha256(content).hexdigest()
    pdf_backend_choice = get_pdf_backend()

    texts_list = _process_file_content(ext, content, pdf_backend_choice)

    if texts_list and (
        texts_list[0].startswith("Error processing")
        or texts_list[0].startswith("Invalid PDF backend")
        or texts_list[0].startswith("Unsupported file type")
    ):
        raise HTTPException(status_code=400, detail=texts_list[0])

    return ConversionResponse(
        filename=file.filename, content_hash=content_hash, texts=texts_list
    )


@app.get("/convert", response_model=ConversionResponse)
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


@app.get("/status/{task_id}")
def status(task_id: str):
    return TASKS.get(task_id, {"status": "not_found"})


@app.post("/batch")
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
def run_batch_task(
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

                markdown_texts = _process_file_content(
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

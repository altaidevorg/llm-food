from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import uuid
import os
import shutil
import hashlib

from io import BytesIO
from pymupdf4llm import to_markdown
import pymupdf
from pypdf import PdfReader
from docx import Document
from striprtf.striprtf import rtf_to_text
from pptx import Presentation
import trafilatura

app = FastAPI()


# Config placeholder
def get_pdf_backend():
    return os.getenv("PDF_BACKEND", "pymupdf4llm")


class ConversionResponse(BaseModel):
    filename: str
    content_hash: str
    texts: List[str]


class BatchRequest(BaseModel):
    input_gcs_path: str
    output_gcs_path: str


# Placeholder for task store
TASKS = {}


@app.post("/convert", response_model=ConversionResponse)
async def convert(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    content = await file.read()

    # Calculate SHA256 hash of the content
    content_hash = hashlib.sha256(content).hexdigest()

    texts_list: List[str] = []

    if ext == ".pdf":
        backend = get_pdf_backend()
        if backend == "pymupdf4llm":
            pymupdf_doc = pymupdf.Document(stream=content)
            page_data_list = to_markdown(pymupdf_doc, page_chunks=True)
            texts_list = [page_dict.get("text", "") for page_dict in page_data_list]
        elif backend == "pypdf2":
            reader = PdfReader(BytesIO(content))
            texts_list = [p.extract_text() or "" for p in reader.pages]
        else:
            raise HTTPException(400, "Invalid PDF backend")

    elif ext in [".docx"]:
        doc = Document(BytesIO(content))
        texts_list = ["\\n".join(p.text for p in doc.paragraphs)]

    elif ext in [".rtf"]:
        texts_list = [rtf_to_text(content.decode("utf-8"))]

    elif ext in [".pptx"]:
        prs = Presentation(BytesIO(content))
        texts_list = [
            "\\n".join(
                shape.text
                for shape in slide.shapes
                if hasattr(shape, "text") and shape.text
            )
            for slide in prs.slides
        ]

    elif ext in [".html", ".htm"]:
        extracted_text = trafilatura.extract(
            content.decode("utf-8"), output_format="markdown"
        )
        texts_list = [extracted_text if extracted_text is not None else ""]

    else:
        raise HTTPException(400, "Unsupported file type")

    return ConversionResponse(
        filename=file.filename, content_hash=content_hash, texts=texts_list
    )


@app.post("/batch")
def batch(request: BatchRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"status": "pending"}
    background_tasks.add_task(
        run_batch_task, request.input_gcs_path, request.output_gcs_path, task_id
    )
    return {"task_id": task_id}


@app.get("/status/{task_id}")
def status(task_id: str):
    return TASKS.get(task_id, {"status": "not_found"})


# Dummy runner for async task
def run_batch_task(input_path, output_path, task_id):
    import time

    TASKS[task_id] = {"status": "running"}
    time.sleep(2)  # simulate work
    # logic for pulling from GCS, converting, and saving goes here
    TASKS[task_id] = {"status": "done"}

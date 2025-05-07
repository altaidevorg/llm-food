# Product Requirements Document (PRD)

**Title:** Any-to-Markdown Conversion API Service for LLM Text Extraction
**Owner:** Yusuf Sarıgöz
**Date:** 2025-05-07
**Version:** 1.0

---

### 1. **Overview**

A FastAPI-based microservice to convert supported input formats (PDF, DOC/DOCX, RTF, PPTX, HTML/webpages) into clean Markdown text for downstream LLM pipelines. It supports:

* **Single file sync processing** (upload + direct response)
* **Batch async processing** via Google Cloud Storage (GCS) input/output directories

---

### 2. **Use Cases**

* Preprocessing documents for fine-tuning or RAG ingestion
* Automated dataset cleaning for instruction generation
* Lightweight service usable in local dev or production cloud workflows

---

### 3. **Supported Formats**

| Format    | Extractor                                         |
| --------- | ------------------------------------------------- |
| PDF       | `pymupdf4llm` (default) / `PyPDF2` (configurable) |
| DOC/DOCX  | `python-docx`                                     |
| RTF       | `pyrtf` or `striprtf`                             |
| PPTX      | `python-pptx`                                     |
| HTML/URLs | `prafilatura`                                     |

---

### 4. **Architecture**

#### API Layer

* **Framework:** FastAPI
* **Endpoints:**

  * `POST /convert` – Sync, single file
  * `POST /batch` – Async, with GCS input/output locations
  * `GET /status/{task_id}` – Async task status

#### Extraction Logic

* Pluggable backend extractors (PDF mode config: `pdf_backend = pymupdf4llm|pypdf2`)
* Normalization pipeline to Markdown via `html2text` or equivalent
* Unified output for all the supported file types

#### Storage / Task Handling

* **Async Mode:** Pulls batch file paths from input GCS bucket
* Outputs saved in GCS output bucket (same filename, `.md` extension)

---

### 5. **Configuration**

| Option           | Description                              |
| ---------------- | ---------------------------------------- |
| `pdf_backend`    | `'pymupdf4llm'` (default) or `'pypdf2'`  |
| `max_file_size`  | Optional size cap in MB                  |
| `gcs_project_id` | GCP Project ID for bucket access         |
| `auth_token`     | Optional Bearer token for endpoint usage |

---

### 6. **Non-Functional Requirements**

* Stateless and deployable via Docker
* Unit and integration tested
* Supports logging with structured logs
* Optional auth via token

---

### 7. **License Considerations**

* `pymupdf4llm` (AGPL): default in dev, **must be swappable** for commercial use
* Toggle via env var: `PDF_BACKEND=pypdf2`

---

### 8. **Stretch Goals**

* OCR fallback for scanned PDFs (`tesserocr` or `pytesseract`)
* Language detection
* Basic metadata extraction

## Starter code

```python
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import uuid, os
import shutil

app = FastAPI()

# Config placeholder
def get_pdf_backend():
    return os.getenv("PDF_BACKEND", "pymupdf4llm")

class BatchRequest(BaseModel):
    input_gcs_path: str
    output_gcs_path: str

# Placeholder for task store
TASKS = {}

@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    content = await file.read()
    
    if ext == ".pdf":
        backend = get_pdf_backend()
        if backend == "pymupdf4llm":
            from pymupdf4llm import to_markdown
            text = to_markdown(content)
        elif backend == "pypdf2":
            from PyPDF2 import PdfReader
            from io import BytesIO
            reader = PdfReader(BytesIO(content))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        else:
            raise HTTPException(400, "Invalid PDF backend")

    elif ext in [".docx"]:
        from docx import Document
        from io import BytesIO
        doc = Document(BytesIO(content))
        text = "\n".join(p.text for p in doc.paragraphs)

    elif ext in [".rtf"]:
        from striprtf.striprtf import rtf_to_text
        text = rtf_to_text(content.decode("utf-8"))

    elif ext in [".pptx"]:
        from pptx import Presentation
        from io import BytesIO
        prs = Presentation(BytesIO(content))
        text = "\n".join([shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")])

    elif ext in [".html", ".htm"]:
        import prafilatura
        text = prafilatura.extract(content.decode("utf-8"))

    else:
        raise HTTPException(400, "Unsupported file type")

    return {"markdown": text}

@app.post("/batch")
def batch(request: BatchRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"status": "pending"}
    background_tasks.add_task(run_batch_task, request.input_gcs_path, request.output_gcs_path, task_id)
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
```

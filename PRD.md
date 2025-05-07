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
| `GOOGLE_CLOUD_PROJECT` | GCP Project ID for bucket access         |
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

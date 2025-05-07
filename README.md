# llm-food

*Serving files for hungry LLMs*

---

## Overview

`llm-food` is a FastAPI-based microservice designed to convert various input document formats into clean Markdown text. This output is optimized for downstream Large Language Model (LLM) pipelines, such as those used for Retrieval Augmented Generation (RAG) or fine-tuning.

The service supports synchronous single file processing, asynchronous batch processing via Google Cloud Storage (GCS), and direct URL-to-Markdown conversion.

## Features

* **Multiple Format Support:** Convert PDF, DOC/DOCX, RTF, PPTX, and HTML/webpages to Markdown.
* **Flexible PDF Backend:** Choose between `pymupdf4llm` (default, AGPL) and `pypdf2` (for commercial-friendly use) for PDF extraction.
* **Synchronous Conversion:** Upload a single file and receive its Markdown content directly.
* **URL Conversion:** Provide a URL and get its main content as Markdown.
* **Asynchronous Batch Processing:** Process multiple files from a Google Cloud Storage (GCS) bucket and save Markdown outputs to another GCS location.
* **Configurable File Size Limit:** Set a maximum size for uploaded files.
* **Optional Authentication:** Secure your endpoints with a Bearer token.
* **Dockerized:** Ready for containerized deployment.

## Supported Formats

| Format    | Extractor Library Used                      |
| :-------- | :------------------------------------------ |
| PDF       | `pymupdf4llm` (default) / `pypdf`           |
| DOC/DOCX  | `python-docx`                               |
| RTF       | `striprtf`                                  |
| PPTX      | `python-pptx`                               |
| HTML/URLs | `trafilatura`                               |

## API Endpoints

* `POST /convert` (File Upload):
  * Synchronously converts an uploaded file to Markdown.
  * Request: `multipart/form-data` with a `file` field.
  * Response: JSON with `filename`, `content_hash`, and `texts` (list of Markdown strings, one per page/section).
* `GET /convert` (URL Conversion):
  * Synchronously converts the content of a given URL to Markdown.
  * Request: Query parameter `url=your_url_here`.
  * Response: JSON with `filename` (derived from URL), `content_hash`, and `texts`.
* `POST /batch`:
  * Asynchronously processes files from GCS.
  * Request: JSON body with `input_paths` (a GCS directory URI `gs://bucket/prefix/` or a list of GCS file URIs `["gs://bucket/file1.pdf", ...]`) and `output_path` (GCS directory URI `gs://bucket/output_prefix/`).
  * Response: JSON with `task_id`.
* `GET /status/{task_id}`:
  * Checks the status of an asynchronous batch task.
  * Response: JSON with task status, processed/failed file counts, and details.

## Configuration

The service is configured using environment variables. Create a `.env` file in the project root (you can copy `.env.sample`) or set these variables in your deployment environment.

```env
# .env.sample content:

# Backend for PDF processing: 'pymupdf4llm' (default) or 'pypdf2'
PDF_BACKEND=pymupdf4llm

# Google Cloud Project ID (Required for GCS batch operations if not running in GCP with default creds)
GOOGLE_CLOUD_PROJECT=

# Path to Google Cloud service account JSON file (Optional, for local GCS access)
# Example: GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-file.json
GOOGLE_APPLICATION_CREDENTIALS=

# Maximum file size for uploads in Megabytes (Optional)
# Example: MAX_FILE_SIZE_MB=50
MAX_FILE_SIZE_MB=

# API Authentication Bearer Token (Optional. If set, all endpoints will require this token)
# Example: API_AUTH_TOKEN=your-secret-bearer-token
API_AUTH_TOKEN=
```

**Key Variables:**

* `PDF_BACKEND`: Choose PDF processing library. `pypdf2` (via `pypdf` package) is recommended if AGPL license of `pymupdf4llm` is a concern.
* `GOOGLE_CLOUD_PROJECT`: Your GCP Project ID, necessary for batch GCS operations.
* `GOOGLE_APPLICATION_CREDENTIALS`: Path to your service account key file for local GCS access.
* `MAX_FILE_SIZE_MB`: Optional limit for uploaded file sizes in the synchronous `/convert` endpoint.
* `API_AUTH_TOKEN`: If set, all API endpoints will require this token in the `Authorization: Bearer <token>` header.

## Setup and Running

### 1. Local Development (without Docker)

   **Prerequisites:**

* Python 3.10+
* Pip

   **Steps:**

   1. Clone the repository:

       ```bash
       git clone <your-repo-url>
       cd llm-food
       ```

   2. Create and activate a virtual environment:

       ```bash
       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate
       ```

   3. Install dependencies:

       ```bash
       pip install -r requirements.txt
       ```

   4. Set up your environment variables (e.g., by creating a `.env` file from `.env.sample`).
   5. Run the FastAPI application using Uvicorn:

       ```bash
       uvicorn main:app --reload
       ```

       The API will be available at `http://127.0.0.1:8000`.

### 2. Using Docker

   **Prerequisites:**

* Docker installed and running.

   **Steps:**

   1. Clone the repository (if not already done).
   2. Ensure you have a `.env` file configured in the project root if you need to pass environment variables to the container (e.g., for GCS credentials or auth tokens). Alternatively, you can pass them directly with `docker run -e VAR=value ...`.
   3. Build the Docker image:

       ```bash
       docker build -t llm-food .
       ```

   4. Run the Docker container:

       ```bash
       docker run -d -p 8000:8000 --name llm-food-container --env-file .env llm-food
       ```

       * `-d`: Run in detached mode.
       * `-p 8000:8000`: Map port 8000 on the host to port 8000 in the container.
       * `--env-file .env`: Load environment variables from your `.env` file.

       The API will be available at `http://localhost:8000`.

## Batch Processing with Google Cloud Storage (GCS)

The `/batch` endpoint allows for processing multiple files stored in GCS.

1. **Input:**
    * `input_paths`: Can be a GCS directory URI (e.g., `gs://your-input-bucket/path/to/docs/`) or a list of individual GCS file URIs (e.g., `["gs://your-bucket/doc1.pdf", "gs://your-bucket/doc2.docx"]`).
    * `output_path`: A GCS directory URI where the Markdown files will be saved (e.g., `gs://your-output-bucket/markdown_output/`).
2. **Authentication for GCS:**
    * If running outside GCP (e.g., locally or in a non-GCP CI/CD), ensure `GOOGLE_APPLICATION_CREDENTIALS` environment variable points to your service account key JSON file, and `GOOGLE_CLOUD_PROJECT` is set.
    * If running within a GCP environment (like GCE, GKE, Cloud Run) with appropriate service account permissions, the client library should pick up credentials automatically.
3. **Output:**
    * Markdown files will be created in the specified `output_path` GCS directory. Each output file will have the same name as the input file but with a `.md` extension (e.g., `document.pdf` becomes `document.md`).

## Authentication

If the `API_AUTH_TOKEN` environment variable is set, all API endpoints will be protected. Clients must include an `Authorization` header with a Bearer token:

`Authorization: Bearer your-secret-bearer-token`

If the token is not set in the environment, the API will be accessible without authentication.

## License Considerations

Note that `pymupdf4llm`, the default PDF extraction backend, is licensed under AGPL. If this is a concern for your use case, you can switch to the `pypdf2` backend (which uses the `pypdf` library, typically under a more permissive license like MIT or BSD) by setting the `PDF_BACKEND=pypdf2` environment variable.

## Future Enhancements (Stretch Goals from PRD)

* Gemini backend for PDFs
* OCR fallback for scanned PDFs (e.g., using `tesserocr` or `pytesseract`).
* Language detection.
* More metadata extraction from documents.

# llm-food

*Serving files for hungry LLMs*

![LOGO](./assets//logo.png)

---

## Overview

`llm-food` is a FastAPI-based microservice designed to convert various input document formats into clean Markdown text. This output is optimized for downstream Large Language Model (LLM) pipelines, such as those used for Retrieval Augmented Generation (RAG) or fine-tuning.

The service supports synchronous single file processing, asynchronous batch processing via Google Cloud Storage (GCS), and direct URL-to-Markdown conversion.
All conversion tasks are handled asynchronously to ensure server responsiveness.

## Features

* **Multiple Format Support:** Convert PDF, DOC/DOCX, RTF, PPTX, and HTML/webpages to Markdown.
* **Advanced PDF Processing:** Utilizes Google's Gemini model by default for high-quality OCR and Markdown conversion of PDFs. Alternative backends (`pymupdf4llm`, `pypdf2`) are also available.
* **Synchronous Conversion:** Upload a single file and receive its Markdown content directly.
* **URL Conversion:** Provide a URL and get its main content as Markdown.
* **Asynchronous Batch Processing:** Process multiple files from a Google Cloud Storage (GCS) bucket and save Markdown outputs to another GCS location.
* **Configurable File Size Limit:** Set a maximum size for uploaded files.
* **Optional Authentication:** Secure your endpoints with a Bearer token.
* **Dockerized:** Ready for containerized deployment.

## Supported Formats

| Format    | Extractor Library Used                      |
| :-------- | :------------------------------------------ |
| PDF       | `google-generativeai` (Gemini - default) / `pymupdf4llm` / `pypdf` |
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

# Backend for PDF processing: 'gemini' (default), 'pymupdf4llm', or 'pypdf2'
PDF_BACKEND=gemini

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

# Gemini API Key (Required for 'gemini' PDF_BACKEND if not using Application Default Credentials)
# Example: ALTAI_GEMINI_API_KEY=your-gemini-api-key
ALTAI_GEMINI_API_KEY=
```

**Key Variables:**

* `PDF_BACKEND`: Choose the PDF processing backend.
  * `gemini` (default): Uses Google's Gemini model for advanced OCR and Markdown conversion. Requires `ALTAI_GEMINI_API_KEY` or appropriate Application Default Credentials.
  * `pymupdf4llm`: Uses the PyMuPDF library (AGPLv3 licensed). Often provides good quality extraction.
  * `pypdf2`: Uses the `pypdf` library (typically MIT/BSD licensed). A good option if AGPL is a concern and Gemini is not used.
* `GOOGLE_CLOUD_PROJECT`: Your GCP Project ID, necessary for batch GCS operations and potentially for Gemini if using Vertex AI authentication.
* `GOOGLE_APPLICATION_CREDENTIALS`: Path to your service account key file for local GCS access and can also be used by Gemini if `ALTAI_GEMINI_API_KEY` is not set and ADC are configured via this file.
* `MAX_FILE_SIZE_MB`: Optional limit for uploaded file sizes in the synchronous `/convert` endpoint.
* `API_AUTH_TOKEN`: If set, all API endpoints will require this token in the `Authorization: Bearer <token>` header.
* `ALTAI_GEMINI_API_KEY`: Your API key for Google Gemini. If this is set, it will be used for the `gemini` PDF backend. If not set and `PDF_BACKEND="gemini"`, the service will attempt to use Application Default Credentials (e.g., when running in a GCP environment or if `gcloud auth application-default login` has been run).

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

The default PDF processing backend is `gemini`, which uses Google's Generative AI SDK. Please review Google's terms of service for Gemini.
Alternative PDF backends are available:

* `pymupdf4llm`: This library is licensed under AGPLv3. Ensure compliance if you choose to use this backend.
* `pypdf2`: This library (via `pypdf`) typically uses more permissive licenses like MIT or BSD.

## Future Enhancements

* Support Batch Prediction with Gemini for cost-friendly inference
* Integrate Duckdb for task tracking and caching
* Advanced OCR capabilities are now primarily handled by the Gemini backend. Alternative OCR solutions (e.g., `tesserocr`, `pytesseract`) could be considered for non-Gemini PDF backends or offline support.
* Language detection.
* More metadata extraction from documents.

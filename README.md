# llm-food

*Serving files for hungry LLMs*

![LOGO](./assets//logo.png)

[Blog Post](https://altai.dev/blog/introducing-llm-food)

[PyPi](https://pypi.org/project/llm-food)

---

## Overview

`llm-food` is a Python package that provides a FastAPI-based microservice (the **server**) for converting various input document formats into clean Markdown text. This output is optimized for downstream Large Language Model (LLM) pipelines, such as those used for Retrieval Augmented Generation (RAG) or fine-tuning.

The package also includes a convenient Python **client** library and an ergonomic command-line interface (**CLI**) for easy interaction with the server.

The server supports:

* Synchronous single file processing via file upload (`/convert`).
* Synchronous URL-to-Markdown conversion (`/convert?url=...`).
* Asynchronous batch processing of multiple uploaded files (`/batch`), with PDFs leveraging Google's Gemini Batch API for efficient, scalable OCR and conversion. Other file types in the batch are processed individually.
* Task status tracking and result retrieval for batch jobs using a local DuckDB database.

## Motivation

Extracting clean text from PDFs is still a mess. Tools like `dockling` and `marker` do a decent job—but they’re slow and resource-hungry. `pymupdf4llm` is fast, but it’s AGPL-licensed, which means you'd need to open-source everything that talks to it—even over the network.

[**Gemini Batch Prediction**](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini) gives you blazing throughput and unbeatable pricing—[**\$1 for 6,000 pages**](https://www.sergey.fyi/articles/gemini-flash-2).
The catch? It’s a pain to use.

That is, until now. We wrapped it up in a few friendly CLI commands—simple enough for your grandparents to enjoy.

## Features

* **Multiple Format Support:** Convert PDF, DOC/DOCX, RTF, PPTX, and HTML/webpages to Markdown.
* **Advanced PDF Processing (Synchronous Server):** The server's `/convert` endpoint can use Google's Gemini model for high-quality OCR of single PDFs, with alternative backends (`pymupdf4llm`, `pypdf2`) available via server configuration.
* **Scalable Batch PDF Processing (Server):** The server's `/batch` endpoint uses Google's Gemini Batch Prediction API for high-throughput and extremely cost-friendly conversion of multiple PDFs.
* **Batch Processing for Other Formats (Server):** Non-PDF files uploaded to `/batch` (DOCX, RTF, PPTX, HTML) are processed individually as background tasks on the server.
* **Asynchronous Operations (Server):** All batch processing tasks are handled asynchronously by the server.
* **Task Management with DuckDB (Server):** Batch job progress, individual file statuses, and GCS output locations are tracked in a local DuckDB database on the server.
* **Status & Result Retrieval (Server):** API endpoints to check job status and retrieve results.
* **Python Client & CLI:**
  * Programmatic access to all server endpoints via an `async` Python client.
  * Command-line interface for easy interaction with the server (file conversion, batch jobs, status checks).
* **Configurable File Size Limit (Server):** Set a maximum size for uploaded files.
* **Optional Authentication (Server):** Secure all server endpoints with a Bearer token.
* **Dockerized Server:** Ready for containerized deployment.

## Supported Formats & Processing (Server-Side)

| Format    | Extractor Library/Method Used                                  | `/convert` (Single File) | `/batch` (Multiple Files)                                     |
| :-------- | :------------------------------------------------------------- | :----------------------- | :------------------------------------------------------------ |
| PDF       | `google-genai` (Gemini - default) / `pymupdf4llm` / `pypdf`    | Yes                      | Yes (via Gemini Batch API, temporary page images stored in GCS) |
| DOC/DOCX  | `mammoth`                                                      | Yes                      | Yes (individual background task)                                |
| RTF       | `striprtf`                                                     | Yes                      | Yes (individual background task)                                |
| PPTX      | `python-pptx`                                                  | Yes                      | Yes (individual background task)                                |
| HTML/URLs | `trafilatura`                                                  | Yes (file or URL)        | Yes (HTML files, individual background task)                  |

## Installation & Usage

This project is a Python package and can be installed using pip.

### Dependencies & Extras

The package defines several dependency groups (extras):

* **`server`**: Installs all dependencies required to run the FastAPI server (FastAPI, Uvicorn, Google GenAI SDK, document processing libraries, DuckDB, etc.).
* **`pymupdf`**: Installs `pymupdf4llm` if you wish to use it as a PDF backend for the server's synchronous `/convert` endpoint. This is optional and an alternative to the default Gemini backend.
* The **client** part of the package has minimal dependencies (`httpx`, `pydantic`), which are always installed.

### 1. Server Setup

**Prerequisites:**

* Python 3.10+
* Pip

**Steps:**

1. Clone the repository:

   ```bash
   git clone https://github.com/altaidevorg/llm-food.git
   cd llm-food
   ```

**Love yourself**: `uv sync`

  **Masochist?** Here's the the pip version:

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package with server dependencies:

   ```bash
   pip install .[server]
   ```

   If you also want to use `pymupdf4llm` as a PDF backend option for the `/convert` endpoint:

   ```bash
   pip install .[server,pymupdf]
   ```

4. **Configure Server:** Set up your environment variables by creating a `.env` file in the project root (you can copy `.env.sample` provided in the repository and fill it). See the "Server Configuration" section below for details on variables.
   * **Crucial for `/batch`**: Ensure `GCS_BUCKET`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` are correctly set. If running locally and using a service account, also set `GOOGLE_APPLICATION_CREDENTIALS` pointing to a valid service account key JSON file.

5. Run the FastAPI server:

   ```bash
   llm-food-serve
   ```

   The server will start, typically on `http://0.0.0.0:8000`. The API documentation (Swagger UI) will be available at `/docs`.
   You can also configure host, port, and reload options via environment variables:
   * `LLM_FOOD_HOST` (default: `0.0.0.0`)
   * `LLM_FOOD_PORT` (default: `8000`)
   * `LLM_FOOD_RELOAD` (default: `false`, set to `true` for development)

### 2. Client & CLI Usage

**Prerequisites:**

* Python 3.10+
* Pip
* A running `llm-food` server.

**Installation:**

1. If you've already installed the server in an environment, the client and CLI are also available.
2. To install only the client and CLI (e.g., on a different machine):

   ```bash
   uv add llm-food
   # or
   pip install llm-food
   # Or, from a cloned repository:
   # pip install .
   ```

**CLI Usage:**

The CLI interacts with a running `llm-food` server.

* **Configure Server URL and Token (Optional):**
  * Set environment variables:
    * `LLM_FOOD_SERVER_URL`: URL of the llm-food server (default: `http://localhost:8000`).
    * `LLM_FOOD_API_TOKEN`: API token if the server requires authentication.
  * Or use CLI options: `--server-url` and `--token`.

* **Commands:**

  ```bash
  # Convert a local file
  llm-food convert-file /path/to/your/document.pdf

  # Convert content from a URL
  llm-food convert-url "http://example.com/article.html"

  # Create a batch job (upload multiple files)
  llm-food batch-create /path/to/file1.docx /path/to/file2.pdf gs://your-bucket/outputs/

  # Get the status of a batch job
  llm-food batch-status <your_task_id>

  # get the results in Markdown
  llm-food batch-results <your_task_id>

  # Get help
  llm-food --help
  llm-food convert-file --help
  ```

**Python Client Usage (Programmatic):**

```python
import asyncio
from llm_food.client import LLMFoodClient, LLMFoodClientError

async def main():
    client = LLMFoodClient(base_url="http://localhost:8000", api_token="your-optional-token")

    try:
        # Convert a local file
        conversion_response = await client.convert_file("path/to/your/file.docx")
        print("Converted File:")
        print(f"  Filename: {conversion_response.filename}")
        print(f"  Content Hash: {conversion_response.content_hash}")
        # print(f"  Texts: {conversion_response.texts}") # Can be long

        # Convert from URL
        url_response = await client.convert_url("https://www.example.com")
        print("\nConverted URL:")
        print(f"  Filename: {url_response.filename}")

        # Create a batch job
        batch_job = await client.create_batch_job(
            file_paths=["path/to/report.pdf", "path/to/notes.docx"],
            output_gcs_path="gs://your-gcs-bucket/batch_outputs/"
        )
        task_id = batch_job["task_id"]
        print(f"\nBatch job created with Task ID: {task_id}")

        # Poll for batch job status (example, implement more robust polling)
        # await asyncio.sleep(10) # Give server time to process
        # batch_status = await client.get_batch_job_status(task_id)
        # print(f"\nBatch Job Status ({task_id}): {batch_status.status}")
        # if batch_status.outputs:
        #     print("Outputs:")
        #     for item in batch_status.outputs:
        #         print(f"  - {item.original_filename}: {item.gcs_output_uri}")

    except LLMFoodClientError as e:
        print(f"Client Error: {e}")
    except FileNotFoundError as e:
        print(f"File Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Using Docker (Server Deployment)

**Prerequisites:**

* Docker installed and running.

**Steps:**

1. Clone the repository.
2. **Configure Server:** Create a `.env` file in the project root with your server configuration (see "Server Configuration" section). This file will be used by the Docker container.
   * **Important for GCS/Gemini:** If `GOOGLE_APPLICATION_CREDENTIALS` is used in your `.env` file and points to a local path, you'll need to mount this file into the container and ensure the path in `.env` matches the path *inside* the container. For cloud deployments, prefer service accounts or workload identity.
3. Build the Docker image:

   ```bash
   docker build -t llm-food .
   ```

4. Run the Docker container:

   ```bash
   # Example: Mount a local directory for DuckDB persistence and provide .env file
   docker run -d -p 8000:8000 \
     --name llm-food-container \
     --env-file .env \
     # Example for DuckDB persistence (ensure DUCKDB_FILE in .env is like ./data/batch_tasks.duckdb)
     # -v $(pwd)/data:/app/data \
     # Example for mounting service account key if GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/gcp-key.json in .env
     # -v /path/to/your/local/gcp-key.json:/app/secrets/gcp-key.json:ro \
     llm-food
   ```

   * The server will be available on port 8000 of your Docker host.

## Server API Endpoints

* `POST /convert` (File Upload):
  * Synchronously converts an uploaded file. The server uses its configured `PDF_BACKEND` for PDFs.
  * Request: `multipart/form-data` with a `file` field.
  * Response: JSON with `filename`, `content_hash`, and `texts` (list of Markdown strings, one per page/section).
* `GET /convert` (URL Conversion):
  * Synchronously converts the content of a given URL to Markdown.
  * Request: Query parameter `url=your_url_here`.
  * Response: JSON with `filename` (derived from URL), `content_hash`, and `texts`.
* `POST /batch`:
  * Asynchronously processes multiple uploaded files. PDF files are processed using the Gemini Batch API; other supported formats are processed as individual background tasks.
  * Request: `multipart/form-data` with:
    * `files`: One or more files.
    * `output_gcs_path`: A GCS directory URI (e.g., `gs://your-output-bucket/markdown_output/`) where the final Markdown files will be saved.
  * Response: JSON with `task_id` for the main batch job.
* `GET /status/{task_id}`:
  * Checks the status of an asynchronous batch job created via `/batch`.
  * Response: JSON with detailed job status, including overall progress, Gemini PDF batch sub-job status (if any), and individual file processing statuses stored in DuckDB.
* `GET /batch/{task_id}`:
  * Retrieves the Markdown output for successfully processed files from a completed batch job.
  * Response: JSON containing the job status, a list of successfully converted files (with their original filename, GCS output URI, and Markdown content), and a list of any errors encountered for specific files.

## Server Configuration (Environment Variables)

The server is configured using environment variables. Create a `.env` file in the project root (you can copy `.env.sample` from the repository) or set these variables in your deployment environment.

```env
# .env.sample content (illustrative, refer to .env.sample in repo for full list)

# --- General Server Configuration ---
# API Authentication Bearer Token (Optional. If set, all server endpoints will require this token)
API_AUTH_TOKEN=
# Maximum file size for uploads in Megabytes (Optional, for POST /convert)
MAX_FILE_SIZE_MB=50

# --- PDF Processing Configuration (for POST /convert endpoint on the server) ---
# Backend for PDF processing: 'gemini' (default), 'pymupdf4llm', or 'pypdf2'
PDF_BACKEND=gemini

# --- Google Cloud & Gemini Configuration (Required for /batch PDF processing and /convert with PDF_BACKEND='gemini') ---
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1 # e.g., us-central1, europe-west1
GCS_BUCKET=your-llm-food-bucket # For temporary files and batch outputs
GOOGLE_APPLICATION_CREDENTIALS= # Path to service account JSON for local/non-GCP environments
GEMINI_MODEL_FOR_VISION=gemini-2.0-flash-001
# GEMINI_OCR_PROMPT="Your custom OCR prompt here..."

# --- DuckDB Configuration (Server-side) ---
DUCKDB_FILE=batch_tasks.duckdb # Path to the DuckDB database file

# --- Server Uvicorn Configuration (Optional) ---
# LLM_FOOD_HOST=0.0.0.0
# LLM_FOOD_PORT=8000
# LLM_FOOD_RELOAD=false # Set to true for development (server auto-restarts on code changes)
```

**Key Variables Explained:**

* `API_AUTH_TOKEN`: If set, secures all server API endpoints.
* `MAX_FILE_SIZE_MB`: Limit for single file uploads to the server's `/convert` endpoint.
* `PDF_BACKEND`: For the server's synchronous `/convert` endpoint when processing PDFs. Does not affect `/batch` PDF processing, which always uses Gemini Batch API.
* `GOOGLE_CLOUD_PROJECT`: Essential for all GCS operations and Gemini Vertex AI.
* `GOOGLE_CLOUD_LOCATION`: Region for Gemini Vertex AI client.
* `GCS_BUCKET`: **Crucial for `/batch` operations.** This single bucket is used by the server for:
  * Storing temporary intermediate files for PDF batch processing.
  * The `output_gcs_path` provided in the `/batch` request will also typically be a path *within* this bucket.
* `GOOGLE_APPLICATION_CREDENTIALS`: For local development or non-GCP environments to authenticate GCS and Gemini calls.
* `GEMINI_MODEL_FOR_VISION`: Gemini model used for OCR.
* `GEMINI_OCR_PROMPT`: Allows customization of the prompt sent to Gemini for OCR tasks.
* `DUCKDB_FILE`: Path where the DuckDB database file for task tracking will be stored by the server.

## Authentication

If the `API_AUTH_TOKEN` environment variable is set on the server, all its API endpoints will be protected.
Clients (Python client or CLI) must then provide this token.

* **CLI:** Use the `--token` option or `LLM_FOOD_API_TOKEN` environment variable.
* **Python Client:** Pass the `api_token` argument to the `LLMFoodClient` constructor.

If the token is not set on the server, the API is accessible without authentication.

## License Considerations

* **Core Package:** MIT License.
* **Gemini:** PDF processing via Gemini uses Google's Generative AI SDK. Review Google's terms of service.
* **Alternative PDF Backends (for server's `/convert`):**
  * `pymupdf4llm`: Licensed under AGPLv3. (Optional dependency)
  * `pypdf2` (via `pypdf`): Typically uses permissive licenses (MIT/BSD).
* **DuckDB:** MIT licensed.

Ensure compliance with all relevant licenses for the components you use.

## Notes on Batch PDF Processing with Gemini (Server Logic)

* The server's `/batch` endpoint, when processing PDFs, converts each page to a PNG image.
* These images are temporarily uploaded by the server to a folder within your `GCS_BUCKET`.
* A `payload.jsonl` file referencing these GCS image URIs is created and also uploaded to `GCS_BUCKET`.
* A Gemini Batch Prediction job is then submitted by the server.
* The server polls this job, and upon success, parses results, aggregates Markdown, and saves final `.md` files to the `output_gcs_path` specified in the `/batch` request.
* Consider GCS lifecycle policies for temporary file prefixes in your `GCS_BUCKET` to manage costs.

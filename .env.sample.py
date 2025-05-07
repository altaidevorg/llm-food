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

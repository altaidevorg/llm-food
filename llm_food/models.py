"""Pydantic models"""

from datetime import datetime
from typing import Union, List, Optional
from pydantic import BaseModel


class ConversionResponse(BaseModel):
    filename: str
    content_hash: str
    texts: List[str]


class BatchRequest(BaseModel):
    input_paths: Union[str, List[str]]
    output_path: str


# Pydantic models for GET /batch/{task_id} response
class BatchOutputItem(BaseModel):
    original_filename: str
    markdown_content: str
    gcs_output_uri: str


class BatchJobFailedFileOutput(BaseModel):
    original_filename: str
    file_type: str
    page_number: Optional[int] = None
    error_message: Optional[str] = None
    status: str


class BatchJobOutputResponse(BaseModel):
    job_id: str
    status: str
    outputs: List[BatchOutputItem] = []
    errors: List[BatchJobFailedFileOutput] = []
    message: Optional[str] = None


# Pydantic models for GET /status/{task_id} response
class FileTaskDetail(BaseModel):
    original_filename: str
    file_type: str
    status: str
    gcs_output_markdown_uri: Optional[str] = None
    error_message: Optional[str] = None
    page_number: Optional[int] = None


class GeminiPDFSubJobDetail(BaseModel):
    gemini_batch_sub_job_id: str
    batch_job_id: str  # References main batch_jobs.job_id
    gemini_api_job_name: Optional[str] = None
    status: str
    payload_gcs_uri: Optional[str] = None
    gemini_output_gcs_uri_prefix: Optional[str] = None
    total_pdf_pages_for_batch: Optional[int] = 0
    processed_pdf_pages_count: Optional[int] = 0
    failed_pdf_pages_count: Optional[int] = 0
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class BatchJobStatusResponse(BaseModel):
    job_id: str
    output_gcs_path: str
    status: str
    submitted_at: datetime
    total_input_files: int
    overall_processed_count: Optional[int] = 0
    overall_failed_count: Optional[int] = 0
    last_updated_at: Optional[datetime] = None
    gemini_pdf_processing_details: List[GeminiPDFSubJobDetail] = []
    file_processing_details: List[FileTaskDetail] = []


# New models for text batch processing
from enum import Enum

class ChatMessageInput(BaseModel):
    role: str  # e.g., "user", "assistant"
    content: str


class TextBatchTaskItemInput(BaseModel):
    task_id: Optional[str] = None  # can be client-provided or generated
    system_instruction: Optional[str] = None
    history: List[ChatMessageInput]


class TextBatchTaskItemOutput(BaseModel):
    task_id: str
    original_input: TextBatchTaskItemInput
    generated_text: Optional[str] = None
    error: Optional[str] = None
    status: str  # e.g., "completed", "failed"


class TextBatchJobCreateRequest(BaseModel):
    file_gcs_path: Optional[str] = None  # if providing input via GCS
    job_name: Optional[str] = None
    # Actual file upload will be handled by FastAPI's UploadFile


class TextBatchJobCreateResponse(BaseModel):
    job_id: str
    message: str


class TextBatchJobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"


class TextBatchIndividualTaskStatusInfo(BaseModel):
    task_id: str
    status: str
    error: Optional[str] = None


class TextBatchJobStatusResponse(BaseModel):
    job_id: str
    job_name: Optional[str] = None
    status: TextBatchJobStatus
    submitted_at: datetime
    last_updated_at: datetime
    total_tasks: int
    processed_tasks: int
    failed_tasks: int
    tasks: Optional[List[TextBatchIndividualTaskStatusInfo]] = None  # for detailed view


class TextBatchJobResultItem(BaseModel):
    task_id: str
    original_input: TextBatchTaskItemInput
    generated_text: Optional[str] = None
    error: Optional[str] = None
    status: str # e.g., "completed", "failed"


class TextBatchJobResultsResponse(BaseModel):
    job_id: str
    status: TextBatchJobStatus
    results: List[TextBatchJobResultItem]
    # Consider if 'errors' list is needed if failed items are in 'results'
    # For now, assuming failed items are included in results with status "failed"

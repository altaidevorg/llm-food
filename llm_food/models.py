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

"""Pydantic models"""

from datetime import datetime
from typing import Literal, Union, List, Optional
from pydantic import BaseModel, Field, model_validator


ChunkStrategy = Literal["token", "sentence", "recursive"]
DEFAULT_CHUNK_STRATEGY: ChunkStrategy = "token"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 128


class ConversionResponse(BaseModel):
    filename: str
    content_hash: str
    texts: List[str]


class BatchRequest(BaseModel):
    input_paths: Union[str, List[str]]
    output_path: str


class ChunkParams(BaseModel):
    strategy: ChunkStrategy = DEFAULT_CHUNK_STRATEGY
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, gt=0)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP, ge=0)

    @model_validator(mode="after")
    def validate_overlap_smaller_than_size(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        return self


class ChunkRequest(ChunkParams):
    text: str


class ChunkResponse(BaseModel):
    chunks: List[str]
    total_chunks: int
    strategy: str
    chunk_size: int
    chunk_overlap: int


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

"""Pydantic models"""

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

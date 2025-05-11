"""client to a running llm-food server"""

import httpx
from typing import List, Optional, Dict, Any
from io import BytesIO
import os

# Assuming models.py is in the same package and accessible
from .models import (
    ConversionResponse,
    BatchJobOutputResponse,
    BatchJobStatusResponse,
)


class LLMFoodClientError(Exception):
    """Custom exception for client-side errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text

    def __str__(self):
        return f"{super().__str__()} (Status: {self.status_code}, Response: {self.response_text})"


class LLMFoodClient:
    def __init__(self, base_url: str, api_token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.headers = {"Accept": "application/json"}  # Ensure client prefers JSON
        if self.api_token:
            self.headers["Authorization"] = f"Bearer {self.api_token}"

    async def _request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        url = f"{self.base_url}{endpoint}"
        # Ensure headers passed to httpx.AsyncClient are merged with instance headers
        request_headers = self.headers.copy()
        if "headers" in kwargs:
            request_headers.update(kwargs.pop("headers"))

        async with httpx.AsyncClient() as client:
            try:
                response = await client.request(
                    method, url, headers=request_headers, **kwargs
                )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                error_detail = e.response.text
                try:
                    error_json = e.response.json()
                    if "detail" in error_json:
                        if isinstance(error_json["detail"], str):
                            error_detail = error_json["detail"]
                        else:
                            error_detail = str(
                                error_json["detail"]
                            )  # handle if detail is not a string
                except Exception:
                    pass
                raise LLMFoodClientError(
                    f"HTTP error {e.response.status_code} for {e.request.url}",
                    status_code=e.response.status_code,
                    response_text=error_detail,
                ) from e
            except httpx.RequestError as e:
                raise LLMFoodClientError(
                    f"Request error for {e.request.url}: {str(e)}"
                ) from e

    async def convert_file(
        self,
        file_path: str,
    ) -> ConversionResponse:
        """Converts a local file."""
        endpoint = "/convert"
        try:
            # Robust filename extraction
            file_name = os.path.basename(file_path)

            with open(file_path, "rb") as f:
                # Pass BytesIO for content to ensure it's handled as a stream by httpx
                files_payload = {
                    "file": (file_name, BytesIO(f.read()), "application/octet-stream")
                }
                response = await self._request("POST", endpoint, files=files_payload)

            return ConversionResponse(**response.json())

        except FileNotFoundError:
            raise LLMFoodClientError(f"File not found: {file_path}")
        except LLMFoodClientError:  # Re-raise client errors
            raise
        except Exception as e:
            raise LLMFoodClientError(
                f"Error during file conversion ({file_path}): {str(e)}"
            )

    async def convert_url(
        self,
        url_to_convert: str,
    ) -> ConversionResponse:
        """Converts content from a URL."""
        endpoint = "/convert"
        params = {"url": url_to_convert}

        try:
            response = await self._request("GET", endpoint, params=params)
            return ConversionResponse(**response.json())
        except LLMFoodClientError:  # Re-raise client errors
            raise
        except Exception as e:
            raise LLMFoodClientError(
                f"Error during URL conversion ({url_to_convert}): {str(e)}"
            )

    async def create_batch_job(
        self,
        file_paths: List[str],
        output_gcs_path: str,
    ) -> Dict[str, Any]:
        """Submits multiple local files for batch processing."""
        endpoint = "/batch"

        opened_files_objects = []
        try:
            # Prepare file payload list for httpx
            # Each item in 'files_payload' should be ('files', (filename, file_object, content_type))
            # as the server expects a list of files under the key 'files'.
            files_payload_for_httpx = []
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                file_name = os.path.basename(file_path)
                f_obj = open(file_path, "rb")
                opened_files_objects.append(f_obj)  # Keep track to close later
                files_payload_for_httpx.append(
                    ("files", (file_name, f_obj, "application/octet-stream"))
                )

            data_payload = {"output_gcs_path": output_gcs_path}

            response = await self._request(
                "POST", endpoint, files=files_payload_for_httpx, data=data_payload
            )
            return response.json()

        except FileNotFoundError as e:
            # Ensure any opened files are closed even if one is not found before request
            for f_obj in opened_files_objects:
                f_obj.close()
            raise LLMFoodClientError(
                f"File not found: {e.filename if hasattr(e, 'filename') else str(e)}"
            )  # Ensure str for e
        except LLMFoodClientError:  # Re-raise client errors
            for f_obj in opened_files_objects:  # Ensure files are closed
                f_obj.close()
            raise
        except Exception as e:
            raise LLMFoodClientError(f"Error during batch job creation: {str(e)}")
        finally:
            # Ensure all file objects are closed after the request or if an error occurs
            for f_obj in opened_files_objects:
                if not f_obj.closed:
                    f_obj.close()

    async def get_batch_job_results(self, task_id: str) -> BatchJobOutputResponse:
        """Retrieves the results (outputs and errors) of a batch job from /batch/{task_id}."""
        endpoint = f"/batch/{task_id}"
        try:
            response = await self._request("GET", endpoint)
            return BatchJobOutputResponse(**response.json())
        except LLMFoodClientError:  # Re-raise client errors
            raise
        except Exception as e:
            raise LLMFoodClientError(
                f"Error retrieving batch job results ({task_id}): {str(e)}"
            )

    async def get_detailed_batch_job_status(
        self, task_id: str
    ) -> BatchJobStatusResponse:
        """Retrieves the detailed status of a batch job from /status/{task_id}."""
        endpoint = f"/status/{task_id}"
        try:
            response = await self._request("GET", endpoint)
            return BatchJobStatusResponse(**response.json())
        except LLMFoodClientError:  # Re-raise client errors
            raise
        except Exception as e:
            raise LLMFoodClientError(
                f"Error retrieving detailed batch job status ({task_id}): {str(e)}"
            )

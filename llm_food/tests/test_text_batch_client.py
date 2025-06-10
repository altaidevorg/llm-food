# Unit tests for text batch processing client methods.

import unittest
from unittest.mock import AsyncMock, patch, MagicMock, mock_open
import httpx
import json
import os
import tempfile
import shutil

from llm_food.client import LLMFoodClient, LLMFoodClientError
# Models for type hinting or asserting response structure if needed, though mocks return dicts
# from llm_food.models import TextBatchJobCreateResponse, TextBatchJobStatusResponse, TextBatchJobResultsResponse


class TestTextBatchClient(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.base_url = "http://testserver.com/api"
        self.api_token = "test_token"
        self.client = LLMFoodClient(base_url=self.base_url, api_token=self.api_token)
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("httpx.AsyncClient.request", new_callable=AsyncMock)
    async def test_create_text_batch_job_success(self, mock_request):
        mock_response_data = {"job_id": "job_123", "message": "Job created successfully"}
        mock_httpx_response = httpx.Response(
            200, json=mock_response_data, request=MagicMock()
        )
        mock_request.return_value = mock_httpx_response

        temp_file_path = os.path.join(self.test_dir, "test_input.jsonl")
        with open(temp_file_path, "w") as f:
            f.write('{"history": [{"role": "user", "content": "Hello"}]}\n')
            f.write('{"task_id": "task_2", "history": [{"role": "user", "content": "World"}]}\n')

        job_name = "My Test Job"
        response_data = await self.client.create_text_batch_job(temp_file_path, job_name=job_name)

        self.assertEqual(response_data, mock_response_data)
        
        # Check call arguments
        called_args, called_kwargs = mock_request.call_args
        self.assertEqual(called_args[0], "POST") # method
        self.assertEqual(called_args[1], f"{self.base_url}/text-batch") # url

        # Check files payload
        self.assertIn("files", called_kwargs)
        files_payload = called_kwargs["files"]
        self.assertIn("file", files_payload)
        file_tuple = files_payload["file"]
        self.assertEqual(file_tuple[0], "test_input.jsonl") # filename
        # file_tuple[1] is the file object, check its content
        file_content_bytes = file_tuple[1].read()
        self.assertEqual(file_content_bytes.decode(), '{"history": [{"role": "user", "content": "Hello"}]}\n{"task_id": "task_2", "history": [{"role": "user", "content": "World"}]}\n')
        file_tuple[1].close() # Important to close the object passed to the mock
        self.assertEqual(file_tuple[2], "application/jsonl") # content_type
        
        # Check data payload
        self.assertIn("data", called_kwargs)
        data_payload = called_kwargs["data"]
        self.assertEqual(data_payload, {"job_name": job_name})
        
        self.assertIn("headers", called_kwargs)
        self.assertEqual(called_kwargs["headers"]["Authorization"], f"Bearer {self.api_token}")


    async def test_create_text_batch_job_file_not_found(self):
        with self.assertRaisesRegex(LLMFoodClientError, "File not found: non_existent_file.jsonl"):
            await self.client.create_text_batch_job("non_existent_file.jsonl")

    @patch("httpx.AsyncClient.request", new_callable=AsyncMock)
    async def test_create_text_batch_job_server_error(self, mock_request):
        mock_request.side_effect = httpx.HTTPStatusError(
            message="Server error",
            request=MagicMock(),
            response=httpx.Response(500, text="Internal Server Error", request=MagicMock()),
        )
        temp_file_path = os.path.join(self.test_dir, "test_input.jsonl")
        with open(temp_file_path, "w") as f:
            f.write('{"history": [{"role": "user", "content": "Hello"}]}\n')

        with self.assertRaises(LLMFoodClientError) as cm:
            await self.client.create_text_batch_job(temp_file_path, job_name="Test Job")
        self.assertEqual(cm.exception.status_code, 500)
        self.assertIn("Internal Server Error", str(cm.exception))


    @patch("httpx.AsyncClient.request", new_callable=AsyncMock)
    async def test_get_text_batch_job_status_success(self, mock_request):
        job_id = "job_xyz"
        mock_response_data = {
            "job_id": job_id, "status": "completed", "total_tasks": 10, 
            "processed_tasks": 10, "failed_tasks": 0, 
            "submitted_at": "2023-01-01T12:00:00Z", "last_updated_at": "2023-01-01T12:05:00Z"
        }
        mock_httpx_response = httpx.Response(
            200, json=mock_response_data, request=MagicMock()
        )
        mock_request.return_value = mock_httpx_response

        response_data = await self.client.get_text_batch_job_status(job_id)

        self.assertEqual(response_data, mock_response_data)
        mock_request.assert_called_once_with(
            "GET", f"{self.base_url}/text-batch/{job_id}/status", headers=self.client.headers
        )

    @patch("httpx.AsyncClient.request", new_callable=AsyncMock)
    async def test_get_text_batch_job_status_not_found(self, mock_request):
        job_id = "job_not_found"
        mock_request.side_effect = httpx.HTTPStatusError(
            message="Not Found",
            request=MagicMock(),
            response=httpx.Response(404, json={"detail":"Job not found"}, request=MagicMock()),
        )

        with self.assertRaises(LLMFoodClientError) as cm:
            await self.client.get_text_batch_job_status(job_id)
        self.assertEqual(cm.exception.status_code, 404)
        self.assertIn("Job not found", str(cm.exception.response_text))

    @patch("httpx.AsyncClient.request", new_callable=AsyncMock)
    async def test_get_text_batch_job_results_success(self, mock_request):
        job_id = "job_abc"
        mock_response_data = {
            "job_id": job_id, "status": "completed", 
            "results": [{"task_id": "t1", "original_input": {}, "generated_text": "Result 1", "status": "completed"}]
        }
        mock_httpx_response = httpx.Response(
            200, json=mock_response_data, request=MagicMock()
        )
        mock_request.return_value = mock_httpx_response

        response_data = await self.client.get_text_batch_job_results(job_id)
        self.assertEqual(response_data, mock_response_data)
        mock_request.assert_called_once_with(
            "GET", f"{self.base_url}/text-batch/{job_id}/results", headers=self.client.headers
        )

    @patch("httpx.AsyncClient.request", new_callable=AsyncMock)
    async def test_get_text_batch_job_results_server_error(self, mock_request):
        job_id = "job_error_results"
        mock_request.side_effect = httpx.HTTPStatusError(
            message="Server Error",
            request=MagicMock(),
            response=httpx.Response(500, text="Internal Server Error Processing Results", request=MagicMock()),
        )
        with self.assertRaises(LLMFoodClientError) as cm:
            await self.client.get_text_batch_job_results(job_id)
        self.assertEqual(cm.exception.status_code, 500)
        self.assertIn("Internal Server Error Processing Results", str(cm.exception.response_text))

    @patch("httpx.AsyncClient.request", new_callable=AsyncMock)
    async def test_create_text_batch_job_no_job_name(self, mock_request):
        mock_response_data = {"job_id": "job_456", "message": "Job created"}
        mock_httpx_response = httpx.Response(200, json=mock_response_data, request=MagicMock())
        mock_request.return_value = mock_httpx_response

        temp_file_path = os.path.join(self.test_dir, "test_input_no_name.jsonl")
        with open(temp_file_path, "w") as f:
            f.write('{"history": [{"role": "user", "content": "Another test"}]}\n')

        response_data = await self.client.create_text_batch_job(temp_file_path, job_name=None)
        self.assertEqual(response_data, mock_response_data)
        
        _called_args, called_kwargs = mock_request.call_args
        self.assertIn("data", called_kwargs)
        # Ensure data payload is empty if job_name is None, as server endpoint has Optional[str] = Form(None)
        self.assertEqual(called_kwargs["data"], {}) 


if __name__ == "__main__":
    unittest.main()

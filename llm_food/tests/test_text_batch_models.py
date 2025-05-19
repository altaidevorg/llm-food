# Unit tests for text batch processing Pydantic models.

import unittest
from pydantic import ValidationError
from datetime import datetime
from typing import List, Optional, Dict, Any # Required for model definitions if not directly used here

from llm_food.models import (
    ChatMessageInput,
    TextBatchTaskItemInput,
    TextBatchTaskItemOutput,
    TextBatchJobCreateRequest,
    TextBatchJobCreateResponse,
    TextBatchJobStatus,
    TextBatchIndividualTaskStatusInfo,
    TextBatchJobStatusResponse,
    TextBatchJobResultItem,
    TextBatchJobResultsResponse,
)


class TestTextBatchModels(unittest.TestCase):

    def test_chat_message_input_valid(self):
        data = {"role": "user", "content": "Hello!"}
        msg = ChatMessageInput(**data)
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello!")

    def test_chat_message_input_invalid(self):
        with self.assertRaises(ValidationError):
            ChatMessageInput(role="user")  # Missing content
        with self.assertRaises(ValidationError):
            ChatMessageInput(content="Hello")  # Missing role
        with self.assertRaises(ValidationError):
            ChatMessageInput(role=123, content="Hello")  # Invalid role type

    def test_text_batch_task_item_input_valid(self):
        chat_history = [ChatMessageInput(role="user", content="Hi")]
        data = {
            "task_id": "task1",
            "system_instruction": "Be helpful.",
            "history": chat_history,
        }
        item = TextBatchTaskItemInput(**data)
        self.assertEqual(item.task_id, "task1")
        self.assertEqual(item.system_instruction, "Be helpful.")
        self.assertEqual(len(item.history), 1)
        self.assertEqual(item.history[0].content, "Hi")

        # Test with optional fields missing
        data_minimal = {"history": chat_history}
        item_minimal = TextBatchTaskItemInput(**data_minimal)
        self.assertIsNone(item_minimal.task_id)
        self.assertIsNone(item_minimal.system_instruction)
        self.assertEqual(len(item_minimal.history), 1)

    def test_text_batch_task_item_input_invalid(self):
        with self.assertRaises(ValidationError):
            TextBatchTaskItemInput(task_id="task1")  # Missing history
        with self.assertRaises(ValidationError):
            TextBatchTaskItemInput(history="not a list") # Invalid history type
        with self.assertRaises(ValidationError):
            TextBatchTaskItemInput(history=[{"role": "user"}]) # Invalid item in history

    def test_text_batch_task_item_output_valid(self):
        original_input_data = {"history": [ChatMessageInput(role="user", content="Test")]}
        original_input = TextBatchTaskItemInput(**original_input_data)
        data = {
            "task_id": "task1_out",
            "original_input": original_input,
            "generated_text": "This is the output.",
            "error": None,
            "status": "completed",
        }
        item = TextBatchTaskItemOutput(**data)
        self.assertEqual(item.task_id, "task1_out")
        self.assertEqual(item.status, "completed")
        self.assertEqual(item.generated_text, "This is the output.")

        # Test with optional fields missing
        data_minimal = {
            "task_id": "task2_out",
            "original_input": original_input,
            "status": "failed",
            "error": "Something went wrong"
        }
        item_minimal = TextBatchTaskItemOutput(**data_minimal)
        self.assertIsNone(item_minimal.generated_text)
        self.assertEqual(item_minimal.error, "Something went wrong")


    def test_text_batch_task_item_output_invalid(self):
        original_input_data = {"history": [ChatMessageInput(role="user", content="Test")]}
        original_input = TextBatchTaskItemInput(**original_input_data)
        with self.assertRaises(ValidationError):
            TextBatchTaskItemOutput(original_input=original_input, status="completed") # Missing task_id
        with self.assertRaises(ValidationError):
            TextBatchTaskItemOutput(task_id="t1", status="completed") # Missing original_input
        with self.assertRaises(ValidationError):
            TextBatchTaskItemOutput(task_id="t1", original_input=original_input) # Missing status

    def test_text_batch_job_create_request_valid(self):
        req = TextBatchJobCreateRequest(file_gcs_path="gs://bucket/file.jsonl", job_name="My Job")
        self.assertEqual(req.file_gcs_path, "gs://bucket/file.jsonl")
        self.assertEqual(req.job_name, "My Job")
        
        req_minimal = TextBatchJobCreateRequest()
        self.assertIsNone(req_minimal.file_gcs_path)
        self.assertIsNone(req_minimal.job_name)

    def test_text_batch_job_create_response_valid(self):
        resp = TextBatchJobCreateResponse(job_id="job_abc", message="Created")
        self.assertEqual(resp.job_id, "job_abc")
        self.assertEqual(resp.message, "Created")

    def test_text_batch_job_create_response_invalid(self):
        with self.assertRaises(ValidationError):
            TextBatchJobCreateResponse(job_id="job_abc") # Missing message
        with self.assertRaises(ValidationError):
            TextBatchJobCreateResponse(message="Created") # Missing job_id
            
    def test_text_batch_job_status_enum(self):
        self.assertEqual(TextBatchJobStatus.PENDING, "pending")
        self.assertEqual(TextBatchJobStatus.PROCESSING, "processing")
        self.assertEqual(TextBatchJobStatus.COMPLETED, "completed")
        self.assertEqual(TextBatchJobStatus.FAILED, "failed")
        self.assertEqual(TextBatchJobStatus.COMPLETED_WITH_ERRORS, "completed_with_errors")

    def test_text_batch_individual_task_status_info_valid(self):
        info = TextBatchIndividualTaskStatusInfo(task_id="task1", status="completed", error=None)
        self.assertEqual(info.task_id, "task1")
        self.assertEqual(info.status, "completed")
        self.assertIsNone(info.error)

        info_with_error = TextBatchIndividualTaskStatusInfo(task_id="task2", status="failed", error="Big error")
        self.assertEqual(info_with_error.error, "Big error")

    def test_text_batch_individual_task_status_info_invalid(self):
        with self.assertRaises(ValidationError):
            TextBatchIndividualTaskStatusInfo(task_id="task1") # Missing status
        with self.assertRaises(ValidationError):
            TextBatchIndividualTaskStatusInfo(status="completed") # Missing task_id

    def test_text_batch_job_status_response_valid(self):
        now = datetime.utcnow()
        task_info = [TextBatchIndividualTaskStatusInfo(task_id="t1", status="completed")]
        data = {
            "job_id": "job123",
            "job_name": "My Test Job",
            "status": TextBatchJobStatus.COMPLETED_WITH_ERRORS,
            "submitted_at": now,
            "last_updated_at": now,
            "total_tasks": 10,
            "processed_tasks": 8,
            "failed_tasks": 2,
            "tasks": task_info
        }
        resp = TextBatchJobStatusResponse(**data)
        self.assertEqual(resp.job_id, "job123")
        self.assertEqual(resp.status, TextBatchJobStatus.COMPLETED_WITH_ERRORS)
        self.assertEqual(len(resp.tasks), 1)

        # Test with optional fields missing
        data_minimal = {
            "job_id": "job124",
            "status": TextBatchJobStatus.PENDING,
            "submitted_at": now,
            "last_updated_at": now,
            "total_tasks": 5,
            "processed_tasks": 0,
            "failed_tasks": 0,
        }
        resp_minimal = TextBatchJobStatusResponse(**data_minimal)
        self.assertIsNone(resp_minimal.job_name)
        self.assertIsNone(resp_minimal.tasks) # tasks is Optional, defaults to None if not provided

    def test_text_batch_job_status_response_invalid(self):
        now = datetime.utcnow()
        with self.assertRaises(ValidationError): # Missing job_id
            TextBatchJobStatusResponse(status=TextBatchJobStatus.PENDING, submitted_at=now, last_updated_at=now, total_tasks=1, processed_tasks=0, failed_tasks=0)
        with self.assertRaises(ValidationError): # Invalid status type
            TextBatchJobStatusResponse(job_id="j1", status="unknown_status", submitted_at=now, last_updated_at=now, total_tasks=1, processed_tasks=0, failed_tasks=0)

    def test_text_batch_job_result_item_valid(self):
        original_input_data = {"history": [ChatMessageInput(role="user", content="Test result")]}
        original_input = TextBatchTaskItemInput(**original_input_data)
        data = {
            "task_id": "task_res_1",
            "original_input": original_input,
            "generated_text": "Result text",
            "error": None,
            "status": "completed"
        }
        item = TextBatchJobResultItem(**data)
        self.assertEqual(item.task_id, "task_res_1")
        self.assertEqual(item.generated_text, "Result text")
        self.assertEqual(item.status, "completed")

    def test_text_batch_job_result_item_invalid(self):
        original_input_data = {"history": [ChatMessageInput(role="user", content="Test result")]}
        original_input = TextBatchTaskItemInput(**original_input_data)
        with self.assertRaises(ValidationError): # Missing task_id
            TextBatchJobResultItem(original_input=original_input, status="completed")

    def test_text_batch_job_results_response_valid(self):
        original_input_data = {"history": [ChatMessageInput(role="user", content="Test full result")]}
        original_input = TextBatchTaskItemInput(**original_input_data)
        result_item = TextBatchJobResultItem(task_id="tr1", original_input=original_input, status="completed", generated_text="Full result")
        
        data = {
            "job_id": "job_res_abc",
            "status": TextBatchJobStatus.COMPLETED,
            "results": [result_item]
        }
        resp = TextBatchJobResultsResponse(**data)
        self.assertEqual(resp.job_id, "job_res_abc")
        self.assertEqual(resp.status, TextBatchJobStatus.COMPLETED)
        self.assertEqual(len(resp.results), 1)
        self.assertEqual(resp.results[0].task_id, "tr1")

    def test_text_batch_job_results_response_invalid(self):
         with self.assertRaises(ValidationError): # Missing job_id
            TextBatchJobResultsResponse(status=TextBatchJobStatus.COMPLETED, results=[])
         with self.assertRaises(ValidationError): # Missing results
            TextBatchJobResultsResponse(job_id="j1", status=TextBatchJobStatus.COMPLETED)


if __name__ == '__main__':
    unittest.main()

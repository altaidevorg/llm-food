"""CLI interface to the client"""

import asyncio
import argparse
import os
import sys
import json
from typing import Optional
from pathlib import Path

from .client import LLMFoodClient, LLMFoodClientError
from .models import BatchJobStatusResponse

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".rtf", ".pptx", ".html", ".htm"]

# Default server URL, can be overridden by env var or CLI arg
DEFAULT_SERVER_URL = "http://localhost:8000"


def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)


async def main_async():
    parser = argparse.ArgumentParser(description="CLI for LLM Food Service")
    parser.add_argument(
        "--server-url",
        type=str,
        default=get_env_var("LLM_FOOD_SERVER_URL", DEFAULT_SERVER_URL),
        help=f"Base URL of the LLM Food server (env: LLM_FOOD_SERVER_URL, default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=get_env_var("LLM_FOOD_API_TOKEN"),
        help="API token for authentication (env: LLM_FOOD_API_TOKEN)",
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Command to execute"
    )

    # Convert file command
    parser_convert_file = subparsers.add_parser(
        "convert-file", help="Convert a local file"
    )
    parser_convert_file.add_argument(
        "file_path", type=str, help="Path to the local file"
    )
    parser_convert_file.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="Path to save the converted text. If not provided, prints JSON to stdout.",
    )

    # Convert URL command
    parser_convert_url = subparsers.add_parser(
        "convert-url", help="Convert content from a URL"
    )
    parser_convert_url.add_argument("url", type=str, help="URL to convert")
    parser_convert_url.add_argument(
        "--output-file",
        type=str,
        help="Path to save the converted text. If not provided, prints JSON to stdout.",
    )

    # Batch create command
    parser_batch_create = subparsers.add_parser(
        "batch-create", help="Create a new batch processing job"
    )
    parser_batch_create.add_argument(
        "file_paths", nargs="+", type=str, help="List of local file paths to process"
    )
    parser_batch_create.add_argument(
        "output_gcs_path",
        type=str,
        help="GCS path for storing batch outputs (e.g., gs://bucket/path/)",
    )

    # Batch status command
    parser_batch_status = subparsers.add_parser(
        "batch-status",
        help="Get the detailed status of a batch job from the server.",
    )
    parser_batch_status.add_argument("task_id", type=str, help="ID of the batch task")
    parser_batch_status.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed sub-job and individual file task statuses.",
    )

    # Batch results command - NEW
    parser_batch_results = subparsers.add_parser(
        "batch-results", help="Retrieve and save results of a completed batch job."
    )
    parser_batch_results.add_argument("task_id", type=str, help="ID of the batch task")
    parser_batch_results.add_argument(
        "--save-dir",
        type=str,
        help="Directory to save successful Markdown outputs. Files will be named based on original filenames. Defaults to a folder named after the task ID in the current directory if not provided and job is successful.",
        default=None,  # Will be handled dynamically later
    )

    args = parser.parse_args()

    client = LLMFoodClient(base_url=args.server_url, api_token=args.token)

    try:
        if args.command == "convert-file":
            result = await client.convert_file(args.file_path)
            if args.output_file:
                content_to_save = "\n\n".join(result.texts)
                try:
                    with open(args.output_file, "w", encoding="utf-8") as f:
                        f.write(content_to_save)
                    print(f"Converted content saved to: {args.output_file}")
                except IOError as e:
                    print(
                        f"Error saving file {args.output_file}: {e}",
                        file=sys.stderr if "sys" in globals() else os.sys.stderr,
                    )
                    exit(1)  # Exit if saving failed
            else:
                print(json.dumps(result.model_dump(), indent=2))
        elif args.command == "convert-url":
            result = await client.convert_url(args.url)
            if args.output_file:
                content_to_save = "\n\n".join(result.texts)
                try:
                    with open(args.output_file, "w", encoding="utf-8") as f:
                        f.write(content_to_save)
                    print(f"Converted content saved to: {args.output_file}")
                except IOError as e:
                    print(
                        f"Error saving file {args.output_file}: {e}",
                        file=sys.stderr if "sys" in globals() else os.sys.stderr,
                    )
                    exit(1)  # Exit if saving failed
            else:
                print(json.dumps(result.model_dump(), indent=2))
        elif args.command == "batch-create":
            files_to_process = []
            has_errors = False

            for path_str in args.file_paths:
                input_path = Path(path_str)
                if not input_path.exists():
                    print(
                        f"Error: Path does not exist: {path_str}",
                        file=sys.stderr if "sys" in globals() else os.sys.stderr,
                    )
                    has_errors = True
                    continue  # Continue to check other paths but mark that an error occurred

                if input_path.is_file():
                    if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        files_to_process.append(
                            str(input_path.resolve())
                        )  # Use absolute path
                    else:
                        print(
                            f"Warning: File skipped (unsupported extension): {path_str}",
                            file=sys.stderr if "sys" in globals() else os.sys.stderr,
                        )
                elif input_path.is_dir():
                    print(f"Processing directory: {path_str}")
                    found_in_dir = False
                    for ext in SUPPORTED_EXTENSIONS:
                        for found_file in input_path.rglob(f"*{ext}"):
                            if (
                                found_file.is_file()
                            ):  # Ensure it's a file, not a dir ending with .ext
                                files_to_process.append(
                                    str(found_file.resolve())
                                )  # Use absolute path
                                found_in_dir = True
                    if not found_in_dir:
                        print(
                            f"Warning: No supported files found in directory: {path_str}",
                            file=sys.stderr if "sys" in globals() else os.sys.stderr,
                        )
                else:
                    # This case should ideally not be reached if exists() and is_file()/is_dir() are comprehensive
                    print(
                        f"Error: Path is not a valid file or directory: {path_str}",
                        file=sys.stderr if "sys" in globals() else os.sys.stderr,
                    )
                    has_errors = True

            if has_errors:
                print(
                    "Errors occurred while validating input paths. Aborting batch creation.",
                    file=sys.stderr if "sys" in globals() else os.sys.stderr,
                )
                exit(1)

            if not files_to_process:
                print(
                    "Error: No supported files found to process. Batch creation aborted.",
                    file=sys.stderr if "sys" in globals() else os.sys.stderr,
                )
                exit(1)

            # Deduplicate list while preserving order (Python 3.7+ for dict.fromkeys)
            # For broader compatibility, set then list might reorder, but for file paths order might not be critical here.
            # Using resolve() for absolute paths should help in deduplication if relative paths point to same file.
            unique_files_to_process = sorted(
                list(set(files_to_process))
            )  # Sort for consistent ordering if duplicates existed

            if not unique_files_to_process:
                print(
                    "Error: No supported files remained after deduplication. Batch creation aborted.",
                    file=sys.stderr if "sys" in globals() else os.sys.stderr,
                )
                exit(1)

            print(
                f"Found {len(unique_files_to_process)} unique supported file(s) to process for the batch job."
            )
            # Optionally print the list of files if verbose or for debugging
            # for f_path in unique_files_to_process:
            #     print(f"  - {f_path}")

            result = await client.create_batch_job(
                unique_files_to_process, args.output_gcs_path
            )
            print(json.dumps(result, indent=2))  # server returns a dict directly
        elif args.command == "batch-status":
            # This command now shows DETAILED status from /status/{task_id}
            detailed_status: BatchJobStatusResponse = (
                await client.get_detailed_batch_job_status(args.task_id)
            )

            print(f"Detailed Status for Batch Job ID: {detailed_status.job_id}")
            print(f"  Output GCS Path: {detailed_status.output_gcs_path}")
            print(f"  Overall Status: {detailed_status.status}")
            print(f"  Submitted At: {detailed_status.submitted_at}")
            print(f"  Total Input Files: {detailed_status.total_input_files}")
            print(f"  Processed Count: {detailed_status.overall_processed_count}")
            print(f"  Failed Count: {detailed_status.overall_failed_count}")
            print(f"  Last Updated At: {detailed_status.last_updated_at}")

            if args.verbose:
                print("---")
                if detailed_status.gemini_pdf_processing_details:
                    print(
                        f"Gemini PDF Processing Details ({len(detailed_status.gemini_pdf_processing_details)} sub-job(s)):"
                    )
                    for gemini_job in detailed_status.gemini_pdf_processing_details:
                        print(
                            f"  - Gemini Sub Job ID: {gemini_job.gemini_batch_sub_job_id}"
                        )
                        print(
                            f"    Gemini API Job Name: {gemini_job.gemini_api_job_name or 'N/A'}"
                        )
                        print(f"    Status: {gemini_job.status}")
                        print(
                            f"    Payload GCS URI: {gemini_job.payload_gcs_uri or 'N/A'}"
                        )
                        print(
                            f"    Gemini Output GCS Prefix: {gemini_job.gemini_output_gcs_uri_prefix or 'N/A'}"
                        )
                        print(
                            f"    Total PDF Pages: {gemini_job.total_pdf_pages_for_batch}"
                        )
                        print(
                            f"    Processed PDF Pages: {gemini_job.processed_pdf_pages_count}"
                        )
                        print(
                            f"    Failed PDF Pages: {gemini_job.failed_pdf_pages_count}"
                        )
                        if gemini_job.error_message:
                            print(f"    Error Message: {gemini_job.error_message}")
                else:
                    print("No Gemini PDF processing sub-jobs found.")
                print("---")

                if detailed_status.file_processing_details:
                    print(
                        f"Individual File Processing Details ({len(detailed_status.file_processing_details)} tasks):"
                    )
                    for file_task in detailed_status.file_processing_details:
                        page_info = (
                            f" (Page: {file_task.page_number})"
                            if file_task.page_number is not None
                            else ""
                        )
                        print(
                            f"  - Original Filename: {file_task.original_filename}{page_info}"
                        )
                        print(f"    File Type: {file_task.file_type}")
                        print(f"    Status: {file_task.status}")
                        if file_task.gcs_output_markdown_uri:
                            print(
                                f"    GCS Output URI (final/aggregated): {file_task.gcs_output_markdown_uri}"
                            )
                        if file_task.error_message:
                            print(f"    Error: {file_task.error_message}")
                else:
                    print("No individual file processing tasks found.")
            else:
                print("---")
            print(
                "Run with --verbose to see detailed sub-job and individual file task statuses."
            )

        elif args.command == "batch-results":
            # This command fetches and optionally saves results from /batch/{task_id}
            results_response = await client.get_batch_job_results(args.task_id)

            print(f"Results for Batch Job ID: {results_response.job_id}")
            print(f"Overall Status: {results_response.status}")
            if results_response.message:
                print(f"Message: {results_response.message}")
            print("---")

            save_directory = args.save_dir
            if (
                not save_directory and results_response.outputs
            ):  # Default save_dir if outputs exist and not specified
                save_directory = os.path.join(
                    os.getcwd(), f"llm_food_outputs_{results_response.job_id}"
                )

            if results_response.outputs:
                print(f"Successful Outputs ({len(results_response.outputs)}):")
                if save_directory:
                    if not os.path.exists(save_directory):
                        os.makedirs(save_directory)
                        print(f"  Created directory for results: {save_directory}")
                else:
                    print("  (Run with --save-dir to download content)")

                for item in results_response.outputs:
                    print(f"  - Original Filename: {item.original_filename}")
                    print(f"    GCS Output URI: {item.gcs_output_uri}")
                    if save_directory:
                        base_name = os.path.basename(item.original_filename)
                        file_name_stem = os.path.splitext(base_name)[0]
                        output_filename = f"{file_name_stem}.md"
                        output_path = os.path.join(save_directory, output_filename)
                        try:
                            with open(output_path, "w", encoding="utf-8") as f:
                                f.write(item.markdown_content)
                            print(f"    Saved content to: {output_path}")
                        except IOError as e:
                            print(f"    Error saving file {output_path}: {e}")
            else:
                print("No successful outputs available for this job (yet or at all).")
            print("---")

            if results_response.errors:
                print(
                    f"Reported Errors for files in this batch ({len(results_response.errors)}):"
                )
                for error_item in results_response.errors:
                    page_info = (
                        f" (Page: {error_item.page_number})"
                        if error_item.page_number is not None
                        else ""
                    )
                    print(f"  - Original Filename: {error_item.original_filename}")
                    print(f"    File Type: {error_item.file_type}{page_info}")
                    print(f"    Status: {error_item.status}")
                    if error_item.error_message:
                        print(f"    Error: {error_item.error_message}")
            else:
                print("No errors reported for this job.")

    except LLMFoodClientError as e:
        print(
            f"Client Error: {e}",
            file=sys.stderr,
        )  # Print to stderr
        if e.response_text:
            print(
                f"Server Response: {e.response_text}",
                file=sys.stderr,
            )
        exit(1)
    except Exception as e:
        print(
            f"An unexpected error occurred: {e}",
            file=sys.stderr,
        )
        exit(1)


def main():
    # Wrapper to handle async execution for console_scripts

    if hasattr(asyncio, "run"):  # Python 3.7+
        asyncio.run(main_async())
    else:  # Fallback for older Python (though project requires >=3.10)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main_async())


if __name__ == "__main__":
    # This allows running `python -m llm_food.cli ...` for testing
    main()

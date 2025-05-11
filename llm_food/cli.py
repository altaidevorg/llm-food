"""CLI interface to the client"""

import asyncio
import argparse
import os
import sys
import json
from typing import Optional

from .client import LLMFoodClient, LLMFoodClientError

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

    # Convert URL command
    parser_convert_url = subparsers.add_parser(
        "convert-url", help="Convert content from a URL"
    )
    parser_convert_url.add_argument("url", type=str, help="URL to convert")

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
        "batch-status", help="Get the status of a batch job and optionally save results"
    )
    parser_batch_status.add_argument("task_id", type=str, help="ID of the batch task")
    parser_batch_status.add_argument(
        "--save-dir",
        type=str,
        help="Directory to save successful Markdown outputs. Files will be named based on original filenames.",
        default=None,
    )

    args = parser.parse_args()

    client = LLMFoodClient(base_url=args.server_url, api_token=args.token)

    try:
        if args.command == "convert-file":
            result = await client.convert_file(args.file_path)
            print(json.dumps(result.model_dump(), indent=2))
        elif args.command == "convert-url":
            result = await client.convert_url(args.url)
            print(json.dumps(result.model_dump(), indent=2))
        elif args.command == "batch-create":
            result = await client.create_batch_job(
                args.file_paths, args.output_gcs_path
            )
            print(json.dumps(result, indent=2))  # server returns a dict directly
        elif args.command == "batch-status":
            result = await client.get_batch_job_status(args.task_id)

            print(f"Batch Job Status for Task ID: {result.job_id}")
            print(f"Overall Status: {result.status}")
            if result.message:
                print(f"Message: {result.message}")
            print("---")

            if result.outputs:
                print(f"Successful Outputs ({len(result.outputs)}):")
                for item in result.outputs:
                    print(f"  - Original Filename: {item.original_filename}")
                    print(f"    GCS Output URI: {item.gcs_output_uri}")
                    if args.save_dir:
                        if not os.path.exists(args.save_dir):
                            os.makedirs(args.save_dir)
                            print(f"    Created directory: {args.save_dir}")

                        # Sanitize filename slightly - replace potential path separators from original_filename
                        # and ensure it ends with .md
                        base_name = os.path.basename(item.original_filename)
                        file_name_stem = os.path.splitext(base_name)[0]
                        output_filename = f"{file_name_stem}.md"
                        output_path = os.path.join(args.save_dir, output_filename)

                        try:
                            with open(output_path, "w", encoding="utf-8") as f:
                                f.write(item.markdown_content)
                            print(f"    Saved content to: {output_path}")
                        except IOError as e:
                            print(f"    Error saving file {output_path}: {e}")
            else:
                print(
                    "No successful outputs found for this job yet (or job did not produce any)."
                )

            print("---")
            if result.errors:
                print(f"Errors Encountered ({len(result.errors)}):")
                for error_item in result.errors:
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

            # Original JSON dump can be useful for debugging, optionally keep it or remove
            # print(json.dumps(result.model_dump(), indent=2))
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

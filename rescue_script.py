"""
Intron Voice API Rescue Script - Call Center Audio Transcription

This script automates the workflow of downloading call center audio files from AWS S3,
uploading them to the Intron Voice API for transcription and analysis, and collecting
comprehensive results into a CSV file.

Features:
- Supports multiple input formats: TXT, CSV, and XLSX files
- Processes ALL audio files from input (no sampling)
- Automatically applies call center analysis parameters
- Flattens nested API responses into CSV columns
- Includes: agent scoring, sentiment analysis, compliance checks, product insights, etc.

Author: Intron Health Integration Team
"""

import argparse
import csv
import logging
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import boto3
import pandas as pd
import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Disable SSL warnings when verify=False is used
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================================
# SECTION 1: CONSTANTS & CONFIGURATION
# ============================================================================

# Intron Voice API endpoints
INTRON_UPLOAD_URL = "https://infer.voice.intron.io/file/v1/upload"
INTRON_STATUS_URL_TEMPLATE = "https://infer.voice.intron.io/file/v1/status/{file_id}"

# Polling configuration
DEFAULT_POLL_INTERVAL_SECONDS = 20
DEFAULT_MAX_WAIT_SECONDS = 600

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("intron_rescue")


# ============================================================================
# SECTION 2: HTTP SESSION MANAGEMENT
# ============================================================================


def create_requests_session() -> requests.Session:
    """
    Create a requests session with automatic retry logic for resilient HTTP operations.

    Returns:
        requests.Session: Configured session with retry adapter for common failure scenarios
    """
    retry_strategy = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


# ============================================================================
# SECTION 3: FILE DOWNLOAD OPERATIONS
# ============================================================================


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """
    Parse an S3 URI into bucket and key components.

    Args:
        uri: S3 URI in format 's3://bucket-name/path/to/object'

    Returns:
        Tuple of (bucket_name, object_key)

    Raises:
        ValueError: If URI is not a valid S3 URI
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI format: {uri}")
    return uri[5:].split("/", 1)


def download_from_s3(uri: str, destination_path: Path) -> Path:
    """
    Download a file from AWS S3.

    Args:
        uri: S3 URI of the file to download
        destination_path: Local path where file should be saved

    Returns:
        Path to the downloaded file

    Raises:
        ValueError: If S3 URI is invalid
        botocore.exceptions.ClientError: If S3 download fails
    """
    bucket, key = parse_s3_uri(uri)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    s3_client = boto3.client("s3")
    s3_client.download_file(bucket, key, str(destination_path))

    return destination_path


def download_from_http(
    url: str, destination_path: Path, session: requests.Session
) -> Path:
    """
    Download a file from an HTTP/HTTPS URL.

    Args:
        url: HTTP(S) URL of the file to download
        destination_path: Local path where file should be saved
        session: Requests session with retry logic

    Returns:
        Path to the downloaded file

    Raises:
        requests.exceptions.RequestException: If HTTP download fails
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with session.get(url, stream=True, timeout=300, verify=False) as response:
        response.raise_for_status()
        with open(destination_path, "wb") as file_handle:
            for chunk in response.iter_content(chunk_size=8192):
                file_handle.write(chunk)

    return destination_path


def _download_worker(
    url: str, destination_path: Path, session: requests.Session
) -> Path:
    """
    Worker function that routes downloads to appropriate handler (S3 or HTTP).

    Args:
        url: URL to download (S3 URI or HTTP URL)
        destination_path: Local path where file should be saved
        session: Requests session for HTTP downloads

    Returns:
        Path to the downloaded file
    """
    if url.startswith("s3://"):
        return download_from_s3(url, destination_path)
    else:
        return download_from_http(url, destination_path, session)


def download_files(
    urls: List[str], output_directory: Path, max_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    Download multiple files concurrently from S3 or HTTP sources.

    Args:
        urls: List of file URLs (S3 URIs or HTTP URLs)
        output_directory: Directory where files will be saved
        max_workers: Maximum number of concurrent downloads

    Returns:
        List of dictionaries containing download results with keys:
            - uuid: Unique identifier assigned to the file
            - original_url: Source URL
            - local_path: Path where file was saved
            - error: Error message if download failed, None otherwise
    """
    session = create_requests_session()
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_metadata = {}

        for url in urls:
            file_uuid = str(uuid.uuid4())
            filename = Path(url.split("/")[-1].split("?")[0] or file_uuid)
            destination_path = output_directory / f"{file_uuid}_{filename}"

            future = executor.submit(_download_worker, url, destination_path, session)
            future_to_metadata[future] = (url, destination_path, file_uuid)

        for future in as_completed(future_to_metadata):
            url, destination_path, file_uuid = future_to_metadata[future]

            result_entry = {
                "uuid": file_uuid,
                "original_url": url,
                "local_path": str(destination_path),
                "error": None,
            }

            try:
                future.result()
                logger.info(f"Successfully downloaded: {url}")
            except Exception as exc:
                logger.error(f"Failed to download {url}: {exc}")
                result_entry["error"] = str(exc)

            results.append(result_entry)

    return results


# ============================================================================
# SECTION 4: INTRON VOICE API INTEGRATION
# ============================================================================


def build_upload_payload(file_path: str, template_id: str) -> Dict[str, str]:
    """
    Build the payload for Intron API upload request with call center parameters.

    Args:
        file_path: Path to the file being uploaded
        template_id: Template ID for Intron API processing

    Returns:
        Dictionary containing payload fields for the API request with all
        call center analysis parameters enabled
    """
    payload = {
        "audio_file_name": os.path.basename(file_path),
        "use_category": "file_category_call_center",
        "use_template_id": template_id,
        "get_summary": "TRUE",
        "get_call_center_results": "TRUE",
        "get_call_center_agent_score": "TRUE",
        "get_call_center_agent_score_category": "TRUE",
        "get_call_center_product_info": "TRUE",
        "get_call_center_product_insights": "TRUE",
        "get_call_center_compliance": "TRUE",
        "get_call_center_feedback": "TRUE",
        "get_call_center_sentiment": "TRUE",
    }

    return payload


def upload_to_intron(
    file_path: str, api_key: str, payload: Dict[str, str], session: requests.Session
) -> Dict[str, Any]:
    """
    Upload an audio file to the Intron Voice API for transcription.

    Args:
        file_path: Path to the audio file to upload
        api_key: Intron API authentication key
        payload: Dictionary containing upload parameters
        session: Requests session with retry logic

    Returns:
        API response as a dictionary containing file_id and status

    Raises:
        requests.exceptions.HTTPError: If upload fails
        FileNotFoundError: If file does not exist
    """
    with open(file_path, "rb") as audio_file:
        response = session.post(
            INTRON_UPLOAD_URL,
            data=payload,
            files={"audio_file_blob": (os.path.basename(file_path), audio_file)},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=300,
            verify=False,
        )
        response.raise_for_status()

        return response.json()


def poll_transcription_status(
    file_id: str,
    api_key: str,
    session: requests.Session,
    poll_interval: int = DEFAULT_POLL_INTERVAL_SECONDS,
    max_wait: int = DEFAULT_MAX_WAIT_SECONDS,
) -> Dict[str, Any]:
    """
    Poll the Intron API to check transcription status until completion or timeout.

    Args:
        file_id: Unique identifier returned from upload
        api_key: Intron API authentication key
        session: Requests session with retry logic
        poll_interval: Seconds to wait between status checks
        max_wait: Maximum seconds to wait before timing out

    Returns:
        Final API response containing transcription results and status

    Raises:
        TimeoutError: If transcription doesn't complete within max_wait seconds
    """
    status_url = INTRON_STATUS_URL_TEMPLATE.format(file_id=file_id)
    headers = {"Authorization": f"Bearer {api_key}"}
    start_time = time.time()

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > max_wait:
            raise TimeoutError(
                f"Transcription polling timed out after {max_wait}s for file_id: {file_id}"
            )

        try:
            response = session.get(
                status_url, headers=headers, timeout=30, verify=False
            )
            response.raise_for_status()

            json_response = response.json()

            data = json_response.get("data", {})
            processing_status = str(data.get("processing_status", "")).upper()

            logger.info(f"[{file_id}] Status: {processing_status}")

            # Check for completion statuses
            completion_indicators = ["DONE", "COMPLETE", "SUCCESS", "FILE_TRANSCRIBED"]
            if any(
                indicator in processing_status for indicator in completion_indicators
            ):
                return json_response

            # Check for failure statuses
            failure_indicators = ["FAILED", "ERROR"]
            if any(indicator in processing_status for indicator in failure_indicators):
                return json_response

        except Exception as exc:
            logger.warning(f"Polling error for file_id {file_id}: {exc}")

        time.sleep(poll_interval)


# ============================================================================
# SECTION 5: DATA MANAGEMENT & CSV OUTPUT
# ============================================================================


def flatten_json(data: Any, parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """
    Recursively flatten nested JSON/dict structures using underscore separator.

    Args:
        data: Dictionary or other data structure to flatten
        parent_key: Key prefix for nested fields
        sep: Separator to use between nested keys (default: "_")

    Returns:
        Flattened dictionary with underscore-separated keys

    Examples:
        >>> flatten_json({"a": {"b": 1, "c": 2}})
        {"a_b": 1, "a_c": 2}

        >>> flatten_json({"data": {"status": "done", "results": {"count": 5}}})
        {"data_status": "done", "data_results_count": 5}
    """
    items = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                items.extend(flatten_json(value, new_key, sep=sep).items())
            elif isinstance(value, list):
                # Convert lists to JSON string for CSV compatibility
                items.append((new_key, str(value)))
            else:
                items.append((new_key, value))
    else:
        # Non-dict values are returned as-is with parent key
        items.append((parent_key, data))

    return dict(items)


def write_results_to_csv(output_path: Path, results: List[Dict[str, Any]]) -> None:
    """
    Write transcription results to a CSV file with dynamic columns.

    This function flattens all nested JSON fields from API responses and creates
    a CSV with columns for every unique field found across all results.

    Args:
        output_path: Path where the CSV file will be created
        results: List of result dictionaries to write

    The CSV will contain:
        - Base columns: uuid, original_url, local_path, file_id, error
        - All flattened API response fields (dynamically discovered)
        - Missing values are filled with "N/A"
    """
    if not results:
        logger.warning("No results to write to CSV")
        return

    # Flatten all results and collect unique column names
    flattened_results = []
    all_columns = set()

    for result in results:
        # Create a copy to avoid modifying the original
        flattened_result = {}

        # Preserve base columns
        for base_key in ["uuid", "original_url", "local_path", "file_id", "error"]:
            if base_key in result:
                flattened_result[base_key] = result.get(base_key, "N/A")

        # Flatten the entire result to capture all nested fields
        flat_data = flatten_json(result)
        flattened_result.update(flat_data)

        # Track all unique columns
        all_columns.update(flattened_result.keys())
        flattened_results.append(flattened_result)

    # Define column order: base columns first, then alphabetically sorted API fields
    base_columns = ["uuid", "original_url", "local_path", "file_id", "error"]
    api_columns = sorted([col for col in all_columns if col not in base_columns])
    fieldnames = base_columns + api_columns

    # Write to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for result in flattened_results:
            # Fill missing columns with "N/A"
            row = {col: result.get(col, "N/A") for col in fieldnames}
            writer.writerow(row)

    logger.info(f"Wrote {len(flattened_results)} results to {output_path}")
    logger.info(f"CSV contains {len(fieldnames)} columns")


# ============================================================================
# SECTION 6: WORKFLOW ORCHESTRATION
# ============================================================================


def auto_detect_recordings_file() -> str:
    """
    Auto-detect recordings file in the current directory.

    Searches for files named 'recordings' with extensions in priority order:
    1. recordings.txt
    2. recordings.csv
    3. recordings.xlsx

    Returns:
        Path to the first found recordings file

    Raises:
        FileNotFoundError: If no recordings file is found
    """
    file_priorities = ["recordings.txt", "recordings.csv", "recordings.xlsx"]

    for filename in file_priorities:
        file_path = Path(filename)
        if file_path.exists():
            logger.info(f"Auto-detected recordings file: {filename}")
            return str(file_path)

    raise FileNotFoundError(
        "No recordings file found. Expected one of: recordings.txt, recordings.csv, or recordings.xlsx"
    )


def load_all_urls(url_list_path: str) -> List[str]:
    """
    Load all URLs from a file (supports TXT, CSV, XLSX formats).

    For CSV and XLSX files, reads URLs from the first column.
    For TXT files, reads one URL per line.

    Args:
        url_list_path: Path to file containing URLs

    Returns:
        List of all URLs found in the file

    Raises:
        FileNotFoundError: If URL list file doesn't exist
        ValueError: If file is empty or unsupported format
    """
    file_path = Path(url_list_path)

    if not file_path.exists():
        raise FileNotFoundError(f"URL list file not found: {url_list_path}")

    file_extension = file_path.suffix.lower()
    urls = []

    if file_extension == ".txt":
        # Read text file line by line
        with open(url_list_path, "r") as file:
            urls = [line.strip() for line in file if line.strip()]

    elif file_extension == ".csv":
        # Read CSV file, extract first column
        try:
            df = pd.read_csv(url_list_path)
            if df.empty:
                raise ValueError(f"CSV file is empty: {url_list_path}")
            # Get first column values and convert to list
            urls = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        except Exception as exc:
            raise ValueError(f"Failed to read CSV file {url_list_path}: {exc}")

    elif file_extension in [".xlsx", ".xls"]:
        # Read Excel file, extract first column from first sheet
        try:
            df = pd.read_excel(url_list_path, engine="openpyxl")
            if df.empty:
                raise ValueError(f"Excel file is empty: {url_list_path}")
            # Get first column values and convert to list
            urls = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        except Exception as exc:
            raise ValueError(f"Failed to read Excel file {url_list_path}: {exc}")

    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}. "
            f"Supported formats: .txt, .csv, .xlsx, .xls"
        )

    if not urls:
        raise ValueError(f"No URLs found in file: {url_list_path}")

    logger.info(f"Loaded {len(urls)} URLs from {url_list_path}")
    return urls


def upload_files_to_intron(
    downloaded_files: List[Dict[str, Any]],
    api_key: str,
    template_id: str,
    max_workers: int,
) -> List[Dict[str, Any]]:
    """
    Upload successfully downloaded files to Intron API concurrently.

    Args:
        downloaded_files: List of download results from download_files()
        api_key: Intron API authentication key
        template_id: Template ID for Intron API processing
        max_workers: Maximum number of concurrent uploads

    Returns:
        List of upload results with file_id and status
    """
    session = create_requests_session()
    upload_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {}

        for file_info in downloaded_files:
            if file_info.get("error"):
                logger.warning(
                    f"Skipping upload for {file_info['uuid']} due to download error"
                )
                continue

            payload = build_upload_payload(file_info["local_path"], template_id)
            future = executor.submit(
                upload_to_intron,
                file_info["local_path"],
                api_key,
                payload,
                session,
            )
            future_to_file[future] = file_info

        for future in as_completed(future_to_file):
            file_info = future_to_file[future]

            try:
                file_id = future.result().get("data", {}).get("file_id")

                file_info["file_id"] = file_id
                file_info["status"] = "UPLOADED"
                upload_results.append(file_info)

                logger.info(
                    f"Successfully uploaded {file_info['uuid']} -> file_id: {file_id}"
                )

            except Exception as exc:
                logger.error(f"Upload failed for {file_info['uuid']}: {exc}")
                file_info["error"] = str(exc)
                upload_results.append(file_info)

    successful_uploads = sum(1 for r in upload_results if not r.get("error"))
    logger.info(
        f"Upload complete: {successful_uploads}/{len(upload_results)} succeeded"
    )

    return upload_results


def poll_transcription_results(
    uploaded_files: List[Dict[str, Any]], api_key: str, max_workers: int
) -> List[Dict[str, Any]]:
    """
    Poll Intron API for transcription results concurrently.

    Args:
        uploaded_files: List of upload results with file_id
        api_key: Intron API authentication key
        max_workers: Maximum number of concurrent polling operations

    Returns:
        List of final results with transcripts and status
    """
    session = create_requests_session()
    final_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(
                poll_transcription_status, file_info["file_id"], api_key, session
            ): file_info
            for file_info in uploaded_files
            if not file_info.get("error")
        }

        for future in as_completed(future_to_file):
            file_info = future_to_file[future]

            try:
                data = future.result().get("data", {})
                file_info.update(data)

                logger.info(f"Transcription complete for {file_info['uuid']}")

            except Exception as exc:
                logger.error(f"Polling failed for {file_info['uuid']}: {exc}")
                file_info["error"] = str(exc)

            final_results.append(file_info)

    successful_transcriptions = sum(
        1 for r in final_results if r.get("audio_transcript") and not r.get("error")
    )
    logger.info(
        f"Transcription complete: {successful_transcriptions}/{len(final_results)} succeeded"
    )

    return final_results


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Configure and return the command-line argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Intron Voice API Rescue Script - Call Center Audio Transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Using defaults (auto-detects recordings.txt/csv/xlsx, uses today's date)
  python3 rescue_script.py --template-id YOUR_TEMPLATE_ID

  # Specifying custom file and date
  python3 rescue_script.py --template-id YOUR_TEMPLATE_ID --url-list s3_urls.txt --date 2025-10-14

  # Using auto-detected file with custom date
  python3 rescue_script.py --template-id YOUR_TEMPLATE_ID --date 2025-10-14

  # Dry run to preview files
  python3 rescue_script.py --template-id YOUR_TEMPLATE_ID --dry-run

Supported input formats:
  - TXT: One S3 URL per line
  - CSV: S3 URLs in first column
  - XLSX: S3 URLs in first column

Auto-detection: If --url-list is not specified, the script will look for:
  1. recordings.txt (first priority)
  2. recordings.csv (second priority)
  3. recordings.xlsx (third priority)

Note: All files in the input list will be processed (no sampling).
Call center analysis parameters are automatically enabled.
        """,
    )

    # Optional arguments (both now optional with smart defaults)
    parser.add_argument(
        "--url-list",
        required=False,
        help="Path to file containing audio file URLs (supports .txt, .csv, .xlsx). "
        "If not specified, auto-detects recordings.txt/csv/xlsx",
    )
    parser.add_argument(
        "--date",
        required=False,
        help="Date string for output file naming (e.g., 2025-10-14). "
        "If not specified, uses today's date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--template-id",
        required=False,
        help="Template ID for Intron API processing (can also use TEMPLATE_ID environment variable)",
    )

    # Optional arguments
    parser.add_argument(
        "--api-key",
        help="Intron API key (can also use INTRON_API_KEY environment variable)",
    )
    parser.add_argument(
        "--out-dir",
        default="downloads",
        help="Directory for downloaded files (default: downloads)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent workers for downloads/uploads (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview URLs to be processed without downloading or uploading",
    )

    return parser


# ============================================================================
# SECTION 7: MAIN ENTRY POINT
# ============================================================================


def main() -> None:
    """
    Main entry point for the Intron Voice API rescue script.

    This orchestrates the complete workflow:
    1. Parse command-line arguments
    2. Load and sample URLs
    3. Download audio files
    4. Upload to Intron API
    5. Poll for transcription results
    6. Save results to CSV
    """
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Validate API key
    api_key = args.api_key or os.getenv("INTRON_API_KEY")
    if not api_key:
        raise SystemExit(
            "ERROR: API key required. Provide via --api-key argument or INTRON_API_KEY environment variable"
        )

    # Validate template ID
    template_id = args.template_id or os.getenv("TEMPLATE_ID")
    if not template_id:
        raise SystemExit(
            "ERROR: Template ID required. Provide via --template-id argument or TEMPLATE_ID environment variable"
        )

    # Auto-detect url-list if not specified
    url_list_path = args.url_list
    if not url_list_path:
        try:
            url_list_path = auto_detect_recordings_file()
        except FileNotFoundError as error:
            raise SystemExit(f"ERROR: {error}")

    # Use today's date if not specified
    date_string = args.date
    if not date_string:
        date_string = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info(f"Using today's date: {date_string}")

    # Load all URLs from input file
    try:
        all_urls = load_all_urls(url_list_path)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}")

    # Dry run mode - preview URLs and exit
    if args.dry_run:
        logger.info("DRY RUN MODE - Previewing URLs to be processed:")
        for idx, url in enumerate(all_urls, 1):
            print(f"  {idx}. {url}")
        logger.info(f"Total: {len(all_urls)} URLs will be processed")
        return

    # Clean and recreate downloads folder
    output_directory = Path(args.out_dir)
    if output_directory.exists():
        logger.info(f"Cleaning existing downloads folder: {output_directory}")
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created fresh downloads folder: {output_directory}")

    # Step 1: Download files
    logger.info("=" * 70)
    logger.info("STEP 1: Downloading audio files")
    logger.info("=" * 70)
    downloaded_files = download_files(all_urls, output_directory, args.workers)

    # Step 2: Upload to Intron API
    logger.info("=" * 70)
    logger.info("STEP 2: Uploading files to Intron Voice API")
    logger.info("=" * 70)
    uploaded_files = upload_files_to_intron(
        downloaded_files, api_key, template_id, args.workers
    )

    # Step 3: Poll for transcription results
    logger.info("=" * 70)
    logger.info("STEP 3: Polling for transcription results")
    logger.info("=" * 70)
    final_results = poll_transcription_results(uploaded_files, api_key, args.workers)

    # Step 4: Save results to CSV
    logger.info("=" * 70)
    logger.info("STEP 4: Saving results to CSV")
    logger.info("=" * 70)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_csv_path = Path(f"results_{date_string}_{timestamp}.csv")
    write_results_to_csv(output_csv_path, final_results)

    # Summary
    logger.info("=" * 70)
    logger.info("WORKFLOW COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_csv_path}")
    logger.info(f"Total files processed: {len(final_results)}")
    logger.info(
        f"Successful transcriptions: {sum(1 for r in final_results if r.get('audio_transcript'))}"
    )
    logger.info(f"Failed operations: {sum(1 for r in final_results if r.get('error'))}")


if __name__ == "__main__":
    main()

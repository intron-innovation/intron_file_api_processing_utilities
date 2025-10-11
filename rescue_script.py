"""
Intron Voice API Rescue Script

This script automates the workflow of downloading audio files from AWS S3 or HTTP URLs,
uploading them to the Intron Voice API for transcription, and collecting results into a CSV file.

Author: Intron Health Integration Team
Last Updated: 2025-10-11
"""

import argparse
import csv
import logging
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import boto3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================================
# SECTION 1: CONSTANTS & CONFIGURATION
# ============================================================================

# Intron Voice API endpoints
INTRON_UPLOAD_URL = "https://infer.voice.intron.io/file/v1/upload"
INTRON_STATUS_URL_TEMPLATE = "https://infer.voice.intron.io/file/v1/status/{file_id}"

# API constraints
MAX_FILE_SIZE_MB = 100
MAX_AUDIO_DURATION_SECONDS = 600  # 10 minutes
API_RATE_LIMIT_PER_MINUTE = 30

# Polling configuration
DEFAULT_POLL_INTERVAL_SECONDS = 5
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

    uri_without_protocol = uri[5:]
    bucket, key = uri_without_protocol.split("/", 1)

    return bucket, key


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

    with session.get(url, stream=True, timeout=300) as response:
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
            except Exception as error:
                logger.error(f"Failed to download {url}: {error}")
                result_entry["error"] = str(error)

            results.append(result_entry)

    return results


# ============================================================================
# SECTION 4: INTRON VOICE API INTEGRATION
# ============================================================================


def build_upload_payload(args: argparse.Namespace, file_path: str) -> Dict[str, str]:
    """
    Build the payload for Intron API upload request from CLI arguments.

    Args:
        args: Parsed command-line arguments containing optional API parameters
        file_path: Path to the file being uploaded

    Returns:
        Dictionary containing payload fields for the API request
    """
    payload = {"audio_file_name": os.path.basename(file_path)}

    # Include optional API parameters from CLI flags
    optional_params = [
        "use_category",
        "use_diarization",
        "use_prompt_id",
        "get_summary",
        "get_participants",
        "get_decisions",
        "get_action_items",
        "get_key_topics",
        "get_next_steps",
    ]

    for param in optional_params:
        value = getattr(args, param, None)
        if value is not None:
            api_key = param.replace("_", "-")
            payload[api_key] = value

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
        files = {"audio_file_blob": (os.path.basename(file_path), audio_file)}

        headers = {"Authorization": f"Bearer {api_key}"}

        response = session.post(
            INTRON_UPLOAD_URL, data=payload, files=files, headers=headers, timeout=300
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
            response = session.get(status_url, headers=headers, timeout=30)
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

        except Exception as error:
            logger.warning(f"Polling error for file_id {file_id}: {error}")

        time.sleep(poll_interval)


# ============================================================================
# SECTION 5: DATA MANAGEMENT & CSV OUTPUT
# ============================================================================


def write_results_to_csv(output_path: Path, results: List[Dict[str, Any]]) -> None:
    """
    Write transcription results to a CSV file.

    Args:
        output_path: Path where the CSV file will be created
        results: List of result dictionaries to write

    The CSV will contain the following columns:
        - uuid: Unique identifier assigned to the file
        - original_url: Source URL of the audio file
        - local_path: Path where file was downloaded
        - file_id: Intron API file identifier
        - status: Processing status from API
        - transcript: Transcribed text (if available)
        - error: Error message (if any)
    """
    fieldnames = [
        "uuid",
        "original_url",
        "local_path",
        "file_id",
        "status",
        "transcript",
        "error",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result)


# ============================================================================
# SECTION 6: WORKFLOW ORCHESTRATION
# ============================================================================


def load_and_sample_urls(url_list_path: str, sample_size: int) -> List[str]:
    """
    Load URLs from a file and randomly sample a subset.

    Args:
        url_list_path: Path to text file containing URLs (one per line)
        sample_size: Number of URLs to randomly select

    Returns:
        List of sampled URLs

    Raises:
        FileNotFoundError: If URL list file doesn't exist
        ValueError: If file is empty
    """
    if not Path(url_list_path).exists():
        raise FileNotFoundError(f"URL list file not found: {url_list_path}")

    with open(url_list_path, "r") as file:
        urls = [line.strip() for line in file if line.strip()]

    if not urls:
        raise ValueError(f"URL list file is empty: {url_list_path}")

    sampled_urls = random.sample(urls, min(sample_size, len(urls)))
    logger.info(f"Loaded {len(urls)} URLs, sampled {len(sampled_urls)} for processing")

    return sampled_urls


def upload_files_to_intron(
    downloaded_files: List[Dict[str, Any]],
    api_key: str,
    args: argparse.Namespace,
    max_workers: int,
) -> List[Dict[str, Any]]:
    """
    Upload successfully downloaded files to Intron API concurrently.

    Args:
        downloaded_files: List of download results from download_files()
        api_key: Intron API authentication key
        args: CLI arguments containing optional API parameters
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

            payload = build_upload_payload(args, file_info["local_path"])
            future = executor.submit(
                upload_to_intron, file_info["local_path"], api_key, payload, session
            )
            future_to_file[future] = file_info

        for future in as_completed(future_to_file):
            file_info = future_to_file[future]

            try:
                api_response = future.result()
                file_id = api_response.get("data", {}).get("file_id")

                file_info["file_id"] = file_id
                file_info["status"] = "UPLOADED"
                upload_results.append(file_info)

                logger.info(
                    f"Successfully uploaded {file_info['uuid']} -> file_id: {file_id}"
                )

            except Exception as error:
                logger.error(f"Upload failed for {file_info['uuid']}: {error}")
                file_info["error"] = str(error)
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
                api_response = future.result()
                data = api_response.get("data", {})

                file_info["status"] = data.get("processing_status")
                file_info["transcript"] = data.get("audio_transcript")

                logger.info(f"Transcription complete for {file_info['uuid']}")

            except Exception as error:
                logger.error(f"Polling failed for {file_info['uuid']}: {error}")
                file_info["error"] = str(error)

            final_results.append(file_info)

    successful_transcriptions = sum(
        1 for r in final_results if r.get("transcript") and not r.get("error")
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
        description="Intron Voice API Rescue Script - Automated audio transcription workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python rescue_script.py --url-list s3_urls.txt --date 2025-10-11 --n 10 --api-key $INTRON_API_KEY
  python rescue_script.py --url-list urls.txt --date test --n 3 --dry-run
        """,
    )

    # Required arguments
    parser.add_argument(
        "--url-list",
        required=True,
        help="Path to text file containing audio file URLs (one per line)",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Date string for output file naming (e.g., 2025-10-11)",
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of files to randomly sample from the URL list",
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
        help="Preview selected URLs without downloading or uploading",
    )

    # Intron API optional parameters
    parser.add_argument(
        "--use-category", help="File processing category (e.g., telehealth, legal)"
    )
    parser.add_argument(
        "--use-diarization", help="Enable speaker diarization (TRUE/FALSE)"
    )
    parser.add_argument("--use-prompt-id", help="Custom prompt identifier")
    parser.add_argument("--get-summary", help="Request summary generation (TRUE/FALSE)")
    parser.add_argument(
        "--get-participants", help="Request participant identification (TRUE/FALSE)"
    )
    parser.add_argument(
        "--get-decisions", help="Request decision extraction (TRUE/FALSE)"
    )
    parser.add_argument(
        "--get-action-items", help="Request action item extraction (TRUE/FALSE)"
    )
    parser.add_argument(
        "--get-key-topics", help="Request key topic extraction (TRUE/FALSE)"
    )
    parser.add_argument(
        "--get-next-steps", help="Request next steps extraction (TRUE/FALSE)"
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

    # Load and sample URLs
    try:
        sampled_urls = load_and_sample_urls(args.url_list, args.n)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}")

    # Dry run mode - preview URLs and exit
    if args.dry_run:
        logger.info("DRY RUN MODE - Previewing selected URLs:")
        for url in sampled_urls:
            print(f"  • {url}")
        logger.info(f"Total: {len(sampled_urls)} URLs selected")
        return

    # Step 1: Download files
    logger.info("=" * 70)
    logger.info("STEP 1: Downloading audio files")
    logger.info("=" * 70)
    output_directory = Path(args.out_dir)
    downloaded_files = download_files(sampled_urls, output_directory, args.workers)

    # Step 2: Upload to Intron API
    logger.info("=" * 70)
    logger.info("STEP 2: Uploading files to Intron Voice API")
    logger.info("=" * 70)
    uploaded_files = upload_files_to_intron(
        downloaded_files, api_key, args, args.workers
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
    output_csv_path = Path(f"results_{args.date}_{timestamp}.csv")
    write_results_to_csv(output_csv_path, final_results)

    # Summary
    logger.info("=" * 70)
    logger.info("WORKFLOW COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {output_csv_path}")
    logger.info(f"Total files processed: {len(final_results)}")
    logger.info(
        f"Successful transcriptions: {sum(1 for r in final_results if r.get('transcript'))}"
    )
    logger.info(f"Failed operations: {sum(1 for r in final_results if r.get('error'))}")


if __name__ == "__main__":
    main()

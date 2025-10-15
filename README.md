# Call Center Audio Transcription - Quick Reference Guide

## Overview

This script automates the complete workflow for transcribing and analyzing call center audio recordings using the Intron Voice API. It downloads audio files from AWS S3, processes them with comprehensive call center analysis, and outputs detailed results to CSV.

**Key Features:**
- ✓ Multiple input formats: TXT, CSV, XLSX
- ✓ Auto-detects recordings file (no need to specify --url-list)
- ✓ Uses today's date by default (no need to specify --date)
- ✓ Processes ALL files in input (no sampling required)
- ✓ Automatic call center analysis (agent scoring, sentiment, compliance, etc.)
- ✓ Dynamic CSV output with ALL API fields (19+ columns)
- ✓ Concurrent processing for optimal performance

---

## Installation

```bash
conda create -n rescue_script python=3.12
conda activate rescue_script
pip install -r requirements.txt
```

---

## Using the Runner Script

The repository includes `runner.sh`, a convenient wrapper script that handles conda environment setup and automatically loads environment variables from `.env`.

### Setup

1. **Create a `.env` file** in the project root with your credentials:
```bash
INTRON_API_KEY=your-api-key-here
AWS_DEFAULT_REGION=eu-west-2
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
```

2. **Make the runner script executable**:
```bash
chmod +x runner.sh
```

### Usage

The runner script accepts all the same arguments as the Python script and passes them through:

```bash
# Simplest usage - auto-detects recordings file and uses today's date
./runner.sh --prompt-id YOUR_PROMPT_ID

# With custom date
./runner.sh --prompt-id YOUR_PROMPT_ID --date 2025-10-15

# With specific file
./runner.sh --prompt-id YOUR_PROMPT_ID --url-list my_recordings.txt

# With additional options
./runner.sh --prompt-id YOUR_PROMPT_ID --url-list recordings.csv --date 2025-10-15 --workers 8

# Dry run to preview files (with auto-detection)
./runner.sh --prompt-id YOUR_PROMPT_ID --dry-run

# With custom output directory
./runner.sh --prompt-id YOUR_PROMPT_ID --out-dir /path/to/output
```

**Note:** If you don't specify `--url-list`, the script automatically looks for `recordings.txt`, `recordings.csv`, or `recordings.xlsx` (in that order).

### What the runner script does:
- ✓ Automatically loads environment variables from `.env` file
- ✓ Initializes conda and activates the `rescue_script` environment
- ✓ Passes all command-line arguments to the Python script
- ✓ No need to manually activate conda or export environment variables

### Alternative: Direct Python execution

If you prefer to run the script directly without the wrapper:

```bash
conda activate rescue_script

# Simplest - uses auto-detection and today's date
python rescue_script.py --prompt-id YOUR_PROMPT_ID

# With specific parameters
python rescue_script.py --prompt-id YOUR_PROMPT_ID --url-list recordings.txt --date 2025-10-15 --api-key $INTRON_API_KEY
```

---

## Command-Line Arguments

### Required Arguments

- `--prompt-id`: Prompt ID for Intron API processing **(REQUIRED)**
  - This is a required parameter that must be provided
  - Contact Intron Health to get your prompt ID

### Optional Arguments

- `--url-list`: Path to input file (TXT/CSV/XLSX)
  - **Default:** Auto-detects `recordings.txt`, `recordings.csv`, or `recordings.xlsx` (in priority order)

- `--date`: Date string for output filename (e.g., 2025-10-15)
  - **Default:** Today's date in YYYY-MM-DD format

- `--api-key`: Intron API key
  - **Alternative:** Set `INTRON_API_KEY` environment variable in `.env` file

- `--out-dir`: Download directory
  - **Default:** `downloads`

- `--workers`: Concurrent workers for parallel processing
  - **Default:** 4

- `--dry-run`: Preview files without processing
  - **Default:** Disabled

**Quick help:**
```bash
./runner.sh --help
# or
python rescue_script.py --help
```

---

## Quick Start Examples

```bash
# Simplest usage - uses auto-detected recordings file and today's date
./runner.sh --prompt-id YOUR_PROMPT_ID

# Dry run to preview what will be processed
./runner.sh --prompt-id YOUR_PROMPT_ID --dry-run

# With specific date
./runner.sh --prompt-id YOUR_PROMPT_ID --date 2025-10-15

# With specific file
./runner.sh --prompt-id YOUR_PROMPT_ID --url-list my_files.txt

# With high concurrency
./runner.sh --prompt-id YOUR_PROMPT_ID --workers 8

# Full specification
./runner.sh --prompt-id YOUR_PROMPT_ID --url-list recordings.csv --date 2025-10-15 --workers 8
```

**Notes:**
- All call center analysis parameters (agent scoring, sentiment, compliance, etc.) are automatically enabled
- CSV output includes ALL API response fields (19+ columns)

---

## Input File Formats

The script supports three input formats. **S3 URLs must always be in the first column** for CSV and XLSX files.

### Auto-Detection Priority

If you don't specify `--url-list`, the script automatically searches for recordings files in this order:
1. **recordings.txt** (first priority)
2. **recordings.csv** (second priority)
3. **recordings.xlsx** (third priority)

### 1. TXT File (One URL per line)
```text
s3://my-bucket/call-center/recording1.wav
s3://my-bucket/call-center/recording2.mp3
s3://my-bucket/call-center/recording3.wav
```

### 2. CSV File (URLs in first column)
```csv
s3_url
s3://my-bucket/call1.wav
s3://my-bucket/call2.mp3
s3://my-bucket/call3.wav
```

### 3. XLSX File (URLs in first column)
| Column A (S3 URLs) |
|-------------------|
| s3://my-bucket/call1.wav |
| s3://my-bucket/call2.mp3 |
| s3://my-bucket/call3.wav |

**Supported URL formats:**
- S3 URIs: `s3://bucket-name/path/to/file` (recommended)
- HTTP/HTTPS URLs: Any publicly accessible or signed URL

**Important:** All files in the input list will be processed (no sampling)

---

## Output Structure

### CSV File: `results_{date}_{timestamp}.csv`

The output CSV contains **dynamic columns** that capture ALL fields from the API response. Nested JSON structures are flattened using underscores.

**Base Columns (always present):**

| Column | Description | Example |
|--------|-------------|---------|
| uuid | Unique file identifier | `a1b2c3d4-5678-90ab-cdef-1234567890ab` |
| original_url | Source URL | `s3://bucket/audio.wav` |
| local_path | Downloaded file path | `downloads/a1b2c3d4_audio.wav` |
| file_id | Intron API identifier | `file_abc123xyz` |
| error | Error message (if any) | `None` or error description |

**Dynamic API Response Columns (19+ columns):**

Call center analysis results are automatically flattened and included:

| Column Example | Description |
|---------------|-------------|
| `audio_file_name` | Original audio filename |
| `audio_transcript` | Full transcript text |
| `processed_audio_duration_in_seconds` | Audio duration |
| `processing_status` | Processing status |
| `transcript_call_center_agent_score` | Overall agent performance score (e.g., "55/73") |
| `transcript_call_center_agent_score_category` | Score category (green/yellow/red) |
| `transcript_call_center_sentiment` | Call sentiment analysis |
| `transcript_call_center_compliance` | Detailed compliance check results |
| `transcript_call_center_product_info` | Product information extracted |
| `transcript_call_center_product_insights` | Product-related insights |
| `transcript_call_center_results` | Overall call results summary |
| `transcript_call_center_feedback` | Agent feedback and recommendations |
| `transcript_summary` | Call summary |
| ... | (additional fields based on API configuration) |

**Note:**
- Missing values are filled with `"N/A"`
- Nested structures are flattened using underscores
- All API response fields are automatically captured and included

---

## Environment Variables

```bash
# API Key (alternative to --api-key)
export INTRON_API_KEY="your-api-key-here"

# AWS Credentials (if using S3)
export AWS_ACCESS_KEY_ID="your-aws-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret"
export AWS_DEFAULT_REGION="eu-west-2"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | `pip install -r requirements.txt` |
| NoCredentialsError | Set AWS credentials in `.env` or run `aws configure` |
| API key required | Set `INTRON_API_KEY` in `.env` file |
| Prompt ID required error | `--prompt-id` is required - contact Intron Health to get your prompt ID |
| No recordings file found | Create `recordings.txt`, `recordings.csv`, or `recordings.xlsx` in the current directory |
| File not found | Check file path and ensure it exists, or use auto-detection |
| Unsupported format | Use .txt, .csv, or .xlsx files only |
| CSV is empty | Ensure URLs are in the first column with proper headers |
| Slow downloads | Increase workers: `--workers 10` |
| SSL certificate errors | Script automatically disables SSL verification for the Intron API |
| Missing CSV columns | All API fields are automatically captured - if missing, check API response |

---

## Performance Tips

- **Increase workers** for faster processing: `--workers 8` or `--workers 10`
- **Process large datasets in batches** (split files into chunks of 100-500 URLs)
- **Monitor progress**: `./runner.sh ... 2>&1 | tee process.log`


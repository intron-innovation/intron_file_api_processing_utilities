# Call Center Audio Transcription - Quick Reference Guide

## Overview

This script automates the complete workflow for transcribing and analyzing call center audio recordings using the Intron Voice API. It downloads audio files from AWS S3, processes them with comprehensive call center analysis, and outputs detailed results to CSV.

**Key Features:**
- ✓ Multiple input formats: TXT, CSV, XLSX
- ✓ Processes ALL files in input (no sampling required)
- ✓ Automatic call center analysis (agent scoring, sentiment, compliance, etc.)
- ✓ Dynamic CSV output with flattened API response fields
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
# Basic usage
./runner.sh --url-list recordings.txt --date 2025-10-15

# With additional options
./runner.sh --url-list recordings.csv --date 2025-10-15 --workers 8

# Dry run to preview files
./runner.sh --url-list recordings.txt --date test --dry-run

# With custom output directory
./runner.sh --url-list recordings.xlsx --date 2025-10-15 --out-dir /path/to/output
```

### What the runner script does:
- ✓ Automatically loads environment variables from `.env` file
- ✓ Initializes conda and activates the `rescue_script` environment
- ✓ Passes all command-line arguments to the Python script
- ✓ No need to manually activate conda or export environment variables

### Alternative: Direct Python execution

If you prefer to run the script directly without the wrapper:

```bash
conda activate rescue_script
python rescue_script.py --url-list recordings.txt --date 2025-10-15 --api-key $INTRON_API_KEY
```

---

## Command-Line Arguments

### Required Arguments
- `--url-list`: Path to input file (TXT/CSV/XLSX)
- `--date`: Date string for output filename (e.g., 2025-10-15)

### Optional Arguments
- `--api-key`: Intron API key (or set `INTRON_API_KEY` env var)
- `--out-dir`: Download directory (default: `downloads`)
- `--workers`: Concurrent workers (default: 4)
- `--dry-run`: Preview files without processing

**Quick help:**
```bash
python rescue_script.py --help
```

---

## Quick Start Examples

```bash
# Basic usage with TXT file
./runner.sh --url-list recordings.txt --date 2025-10-15

# With CSV file and high concurrency
./runner.sh --url-list recordings.csv --date 2025-10-15 --workers 8

# Preview files without processing (dry run)
./runner.sh --url-list recordings.txt --date test --dry-run
```

**Note:** All call center analysis parameters (agent scoring, sentiment, compliance, etc.) are automatically enabled

---

## Input File Formats

The script supports three input formats. **S3 URLs must always be in the first column** for CSV and XLSX files.

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

**Dynamic API Response Columns (examples):**

Call center analysis results are automatically flattened and included:

| Column Example | Description |
|---------------|-------------|
| `data_processing_status` | Processing status |
| `data_audio_transcript` | Full transcript text |
| `data_call_center_results_agent_score` | Overall agent performance score |
| `data_call_center_results_sentiment` | Call sentiment analysis |
| `data_call_center_compliance` | Compliance check results |
| `data_call_center_product_info` | Product information extracted |
| `data_call_center_agent_score_category` | Categorized agent scores |
| `data_call_center_feedback` | Customer feedback analysis |
| `data_summary` | Call summary |
| ... | (and many more depending on API response) |

**Note:**
- Missing values are filled with `"N/A"`
- Nested structures like `data.call_center_results.agent_score` become `data_call_center_results_agent_score`
- Column count varies based on API responses (typically 30-50+ columns)

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
| File not found | Check file path and ensure it exists |
| Unsupported format | Use .txt, .csv, or .xlsx files only |
| CSV is empty | Ensure URLs are in the first column |
| Slow downloads | Increase workers: `--workers 10` |
| Missing CSV columns | Columns are dynamic based on API responses |

---

## Performance Tips

- **Increase workers** for faster processing: `--workers 8` or `--workers 10`
- **Process large datasets in batches** (split files into chunks of 100-500 URLs)
- **Monitor progress**: `./runner.sh ... 2>&1 | tee process.log`


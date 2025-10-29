# Call Center Audio Transcription & Agent Scoring

Automates call center audio transcription using Intron Voice API. Downloads audio from AWS S3, processes with AI-powered analysis, and outputs agent performance scores, sentiment, compliance, and insights to CSV.

---

## Installation

```bash
conda create -n agent_scoring python=3.12
conda activate agent_scoring
pip install -r requirements.txt
```

---

## Python Execution

**Configure credentials first** - Edit `agent_scoring.py` lines 45-51:

```python
INTRON_API_KEY = "your-actual-api-key-here"
TEMPLATE_ID = "your-actual-template-id-here"
AWS_ACCESS_KEY_ID = "your-aws-access-key-id"
AWS_SECRET_ACCESS_KEY = "your-aws-secret-access-key"
AWS_DEFAULT_REGION = "eu-west-2"
```

**Run the script:**

```bash
# Activate environment
conda activate agent_scoring

# Process all files
python3 agent_scoring.py --url-list recordings.txt --date 2025-10-15

# Process first 10 files (sampling)
python3 agent_scoring.py --url-list recordings.txt --date 2025-10-15 --sample 10

# Preview without processing (dry run)
python3 agent_scoring.py --url-list recordings.txt --date 2025-10-15 --dry-run

# High performance (8 workers)
python3 agent_scoring.py --url-list recordings.txt --date 2025-10-15 --workers 8
```

**Advanced: Using .env file with runner.sh**

Create `.env` file instead of editing script:

```bash
INTRON_API_KEY=your-api-key-here
TEMPLATE_ID=your-template-id-here
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_DEFAULT_REGION=eu-west-2
```

Then use `./runner.sh` (auto-uses recordings.txt and today's date):

```bash
./runner.sh
./runner.sh --sample 10
./runner.sh --date 2025-10-20 --workers 8
```

**Get credentials:** Intron Health (support@intron.io) | AWS IAM Console

---

## Command-Line Arguments

**Required:**
- `--url-list` - Path to file with audio URLs
- `--date` - Date for output file (YYYY-MM-DD)

**Optional:**
- `--sample` / `-n` - Process first N files (default: all)
- `--workers` - Concurrent workers (default: 4)
- `--out-dir` - Download directory (default: downloads)
- `--dry-run` - Preview without processing

---

## Quick Start Examples

```bash
# Setup
conda create -n agent_scoring python=3.12
conda activate agent_scoring
pip install -r requirements.txt

# Configure: Edit agent_scoring.py lines 45-51 with your credentials

# Prepare: Edit recordings.txt with your S3 URLs

# Test
python3 agent_scoring.py --url-list recordings.txt --date 2025-10-15 --dry-run
python3 agent_scoring.py --url-list recordings.txt --date 2025-10-15 --sample 3

# Run
python3 agent_scoring.py --url-list recordings.txt --date 2025-10-15
```

---

## Input File Format

Edit `recordings.txt` with your S3 URLs (one per line):

```text
s3://support-file-uploads/support-calls.wav
s3://support-file-uploads/support-calls.mp3
```

**Supported formats:**
- TXT: One URL per line
- CSV/XLSX: URLs in first column

**Supported URLs:**
- S3: `s3://bucket/path/file.wav`
- HTTP/HTTPS: Any accessible URL

---

## Output Structure

CSV file: `results_{date}_{timestamp}.csv`

**Key columns:**
- `uuid`, `original_url`, `local_path`, `file_id`, `error`
- `audio_file_name`, `audio_transcript`, `processed_audio_duration_in_seconds`
- `transcript_call_center_agent_score` - e.g., "65/73"
- `transcript_call_center_agent_score_category` - green/yellow/red
- `transcript_call_center_sentiment` - Call sentiment
- `transcript_call_center_compliance` - Compliance results
- `transcript_call_center_feedback` - Coaching recommendations
- `transcript_summary` - Call summary

All API fields automatically included. Missing values filled with "N/A".

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | `pip install -r requirements.txt` |
| Configuration error | Update constants in `agent_scoring.py` (lines 45-51) |
| AWS credentials error | Update AWS constants in script |
| File not found | Check `--url-list` path exists |
| S3 download fails | Verify AWS permissions |
| Slow processing | Increase `--workers` (8-12) |

**Pre-flight checklist:**
- [ ] Credentials configured in `agent_scoring.py`
- [ ] Conda environment activated
- [ ] Dependencies installed
- [ ] Input file exists with valid URLs
- [ ] AWS credentials have S3 read access

---

## Performance Tips

**Batch sizes:**
- Small (1-10 files): `--workers 4` (default)
- Medium (10-100): `--workers 8`
- Large (100+): `--workers 10-12`

**Cost control:**
```bash
# Test first
python3 agent_scoring.py --url-list recordings.txt --date 2025-10-15 --sample 5

# Then run full batch
python3 agent_scoring.py --url-list recordings.txt --date 2025-10-15
```

**Monitor long jobs:**
```bash
python3 agent_scoring.py --url-list recordings.txt --date 2025-10-15 2>&1 | tee process.log
tail -f process.log  # In another terminal
```

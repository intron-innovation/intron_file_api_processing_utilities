# Quick Reference Guide

## Installation

```bash
conda create -n rescue_script python=3.12
conda activate rescue_script

pip install -r requirements.txt
```

Configure AWS credentials:
```bash
aws configure
# OR set environment variables
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```

---

## Common Use Cases

### 1. Basic Transcription (10 files)
```bash
python rescue_script.py \
  --url-list audio_urls.txt \
  --date 2025-10-11 \
  --n 10 \
  --api-key $INTRON_API_KEY
```

### 2. Telehealth with Summary (20 files, 8 workers)
```bash
python rescue_script.py \
  --url-list medical_recordings.txt \
  --date 2025-10-11 \
  --n 20 \
  --api-key $INTRON_API_KEY \
  --use-category telehealth \
  --get-summary TRUE \
  --workers 8
```

### 3. Legal Transcription with Action Items
```bash
python rescue_script.py \
  --url-list legal_recordings.txt \
  --date 2025-10-11 \
  --n 15 \
  --api-key $INTRON_API_KEY \
  --use-category legal \
  --use-diarization TRUE \
  --get-action-items TRUE \
  --get-decisions TRUE
```

### 4. Preview Mode (Dry Run)
```bash
python rescue_script.py \
  --url-list test_urls.txt \
  --date test \
  --n 5 \
  --dry-run
```

---

## URL List Format

Create a text file with one URL per line:

```text
s3://my-bucket/audio/recording1.wav
s3://my-bucket/audio/recording2.mp3
https://example.com/audio/recording3.wav
https://signed-url.s3.amazonaws.com/recording4.mp3?AWSAccessKeyId=...
```

**Supported formats:**
- S3 URIs: `s3://bucket-name/path/to/file`
- HTTP/HTTPS URLs: Any publicly accessible or signed URL

---

## Output Structure

### CSV File: `results_{date}_{timestamp}.csv`

| Column | Description | Example |
|--------|-------------|---------|
| uuid | Unique file identifier | `a1b2c3d4-5678-90ab-cdef-1234567890ab` |
| original_url | Source URL | `s3://bucket/audio.wav` |
| local_path | Downloaded file path | `downloads/a1b2c3d4_audio.wav` |
| file_id | Intron API identifier | `file_abc123xyz` |
| status | Processing status | `FILE_TRANSCRIBED` |
| transcript | Transcription text | `Hello, this is a test...` |
| error | Error message (if any) | `None` or error description |

---

## Workflow Steps

```
1. LOAD & SAMPLE
   └─ Read URL list → Randomly select N files

2. DOWNLOAD
   └─ Concurrent downloads from S3/HTTP → Local storage

3. UPLOAD
   └─ Submit files to Intron API → Receive file_id

4. POLL
   └─ Check transcription status → Wait for completion

5. SAVE
   └─ Write results to CSV → Summary statistics
```

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

### Issue: "ModuleNotFoundError: No module named 'boto3'"
**Solution:**
```bash
pip install boto3 requests
```

### Issue: "NoCredentialsError: Unable to locate credentials"
**Solution:**
```bash
aws configure
# OR
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```

### Issue: "ERROR: API key required"
**Solution:**
```bash
export INTRON_API_KEY="your-api-key"
# OR
python rescue_script.py --api-key "your-api-key" ...
```

### Issue: "FileNotFoundError: URL list file not found"
**Solution:** Check the file path is correct
```bash
ls -la your-url-list.txt
python rescue_script.py --url-list /full/path/to/urls.txt ...
```

### Issue: Downloads are slow
**Solution:** Increase workers (default: 4)
```bash
python rescue_script.py --workers 10 ...
```

### Issue: Need to check what will be processed
**Solution:** Use dry-run mode
```bash
python rescue_script.py --dry-run ...
```

---

## Performance Tips

1. **Optimize worker count** based on your network and CPU:
   - Fast network: `--workers 10` or more
   - Slow network: `--workers 4` (default)
   - CPU bound: Match your CPU core count

2. **Process in batches** for large datasets:
   ```bash
   # Split large URL list into chunks
   split -l 100 all_urls.txt batch_

   # Process each batch
   for batch in batch_*; do
     python rescue_script.py --url-list $batch --n 100 ...
   done
   ```

3. **Monitor progress** with logs:
   ```bash
   python rescue_script.py ... 2>&1 | tee process.log
   ```

---

## API Rate Limits

The Intron Voice API has these constraints:

- **File size**: Max 100 MB per file
- **Duration**: Max 10 minutes per audio
- **Rate limit**: 30 requests per minute

The script handles retries automatically but respects these limits.

---

## Code Structure Quick Reference

```
rescue_script.py
├─ Section 1: Constants & Configuration
├─ Section 2: HTTP Session Management
├─ Section 3: File Download Operations
│  ├─ parse_s3_uri()
│  ├─ download_from_s3()
│  ├─ download_from_http()
│  └─ download_files()
├─ Section 4: Intron Voice API Integration
│  ├─ build_upload_payload()
│  ├─ upload_to_intron()
│  └─ poll_transcription_status()
├─ Section 5: Data Management & CSV Output
│  └─ write_results_to_csv()
├─ Section 6: Workflow Orchestration
│  ├─ load_and_sample_urls()
│  ├─ upload_files_to_intron()
│  └─ poll_transcription_results()
└─ Section 7: Main Entry Point
   ├─ setup_argument_parser()
   └─ main()
```

---

## Support

- **Script issues**: Check README.md and REFACTORING_SUMMARY.md
- **API questions**: voice@intron.io
- **API docs**: https://transcribe.intron.health/docs/

#!/bin/bash

# Automatically export all variables
set -a
source .env
set +a

source ~/anaconda3/etc/profile.d/conda.sh
conda activate agent_scoring

# Set defaults if not provided as arguments
DEFAULT_URL_LIST="recordings.txt"
DEFAULT_DATE=$(date +%Y-%m-%d)

# Parse arguments to check if --url-list and --date are provided
HAS_URL_LIST=false
HAS_DATE=false

for arg in "$@"; do
    if [[ "$arg" == "--url-list" ]]; then
        HAS_URL_LIST=true
    fi
    if [[ "$arg" == "--date" ]]; then
        HAS_DATE=true
    fi
done

# Build command with defaults if needed
ARGS=("$@")
if [[ "$HAS_URL_LIST" == false ]]; then
    ARGS=("--url-list" "$DEFAULT_URL_LIST" "${ARGS[@]}")
fi
if [[ "$HAS_DATE" == false ]]; then
    ARGS=("--date" "$DEFAULT_DATE" "${ARGS[@]}")
fi

python3 agent_scoring.py "${ARGS[@]}"

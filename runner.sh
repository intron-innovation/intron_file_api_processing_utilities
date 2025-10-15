#!/bin/bash

# Automatically export all variables
set -a
source .env
set +a

# Initialize conda
# TODO: fix the users per PC
source ~/anaconda3/etc/profile.d/conda.sh

# Activate the rescue_script environment
conda activate rescue_script

# Run the Python script with all passed arguments
python3 rescue_script.py "$@"

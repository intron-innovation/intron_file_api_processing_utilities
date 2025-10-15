#!/bin/bash

# Automatically export all variables
set -a
source .env
set +a

source ~/anaconda3/etc/profile.d/conda.sh
conda activate rescue_script

python3 rescue_script.py "$@"

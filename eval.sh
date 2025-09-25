#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu-prio,csedu
#SBATCH --qos=csedu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=eval.out
#SBATCH --error=eval.err

# Set environment variables for Hugging Face cache
export HF_HOME="/vol/csedu-nobackup/project/prooijendijk/huggingface/cache"
export HUGGINGFACE_HUB_CACHE="/vol/csedu-nobackup/project/prooijendijk/huggingface/cache"
export TRITON_CACHE_DIR="vol/csedu-nobackup/project/prooijendijk/triton/cache"

# Navigate to the project directory
cd /vol/csedu-nobackup/project/prooijendijk

cd /vol/csedu-nobackup/project/prooijendijk/vader

# Commands to run your program go here, e.g.:
python evaluation_fine.py
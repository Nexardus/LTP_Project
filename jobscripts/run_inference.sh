#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --mem=32000

module purge
module load Python/3.11.3-GCCcore-12.3.0

#python3 -m venv $HOME/venvs/ltp
#source $HOME/venvs/ltp/bin/activate

export HF_HOME=/scratch/s3677923/hf_cache

python3 run_validation.py

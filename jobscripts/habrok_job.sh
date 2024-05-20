#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=ltp_job
#SBATCH --mem=3000
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --array=0-4

module purge

module load Python/3.9.6-GCCcore-11.2.0

# check if virtual environment exists and create one in home directory if not (can be adjusted ofc)
if [ ! -d ~/env ]; then
    python3 -m venv ~/env
    source ~/env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source ~/env/bin/activate
fi

python run_validation.py

deactivate
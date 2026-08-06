#!/bin/bash
#SBATCH -t 06:00:00                   # 6 hours walltime
#SBATCH -N 1                          # 1 node
#SBATCH -n 50                         # 20 CPU cores
#SBATCH --gres=gpu:1                  # 1 GPU
#SBATCH --partition=mit_preemptable   # partition
#SBATCH --mem=300G                    # adjust if needed
#SBATCH -o ./slurm_logs/inference-%j.out
#SBATCH -e ./slurm_logs/inference-%j.err
#SBATCH --mail-type=ALL

# Optional: dynamically set email
scontrol update job $SLURM_JOB_ID MailUser=$USER@mit.edu

# Go to the directory where the script lives (so relative imports & data paths work)
cd "$SLURM_SUBMIT_DIR"

echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "SLURM job ID: $SLURM_JOB_ID"

# (Optional) load your modules / conda env here
module load anaconda
source activate keras-pytorch-2.9

# Run your training script
python -u inference.py
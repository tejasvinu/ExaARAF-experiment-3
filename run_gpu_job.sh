#!/bin/bash
#SBATCH --job-name=cifar_train
#SBATCH --partition=gpu  # Or your desired partition
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32 # Adjust as needed
#SBATCH --mem=192131M        # Adjust as needed
#SBATCH --time=20:00:00   # Adjust as needed
#SBATCH --output=cifar_train_%j.out
#SBATCH --error=cifar_train_%j.err

# Load required modules (replace with your actual modules)
module load anaconda3/anaconda3
module load cuda/11.8

source /home/apps/anaconda3/etc/profile.d/conda.sh  # Use the correct path you found

conda activate pytorch-gpu

# Activate your Python environment if needed (e.g., conda)
# source activate your_env_name

# Run the Python script using torchrun for distributed training
# --nproc_per_node should match the number of GPUs requested by --gres
# Adjust arguments for cifar10_train.py as needed
torchrun --nproc_per_node=2 \
         --rdzv_id=$SLURM_JOB_ID \
         --rdzv_backend=c10d \
         --rdzv_endpoint=localhost:29500 \
         ./cifar10_train.py \
         --epochs 50 \
         --batch-size 128 \
         --output-dir ./cifar_output_distributed \
         --amp
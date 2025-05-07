#!/bin/bash

# --- Job Configuration (Adapt for your HPC Scheduler, e.g., Slurm) ---
# These SBATCH lines are for Slurm and can be removed or commented out for local execution.
# #SBATCH --job-name=cifar_train
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1 # Master process for torchrun
# #SBATCH --cpus-per-task=16   # Number of CPUs for data loaders (e.g., 4 per GPU if using 2 GPUs)
# #SBATCH --gres=gpu:2        # Request 2 GPUs. Adjust as needed (e.g., gpu:1 for single GPU)
# #SBATCH --mem=32G           # CPU memory
# #SBATCH --time=20:00:00     # Max 2 hours for CIFAR. Adjust as needed.
# #SBATCH --output=cifar_train_job_%j.out
# #SBATCH --error=cifar_train_job_%j.err
# #SBATCH --partition=gpu     # Or your specific GPU partition

# --- Environment Setup ---
echo "Setting up Python environment..."
echo "NOTE: Ensure you have activated your desired Python/Conda environment before running this script."
# # module load anaconda3/anaconda3 # Specific to HPC
# # module load cuda/11.8           # Specific to HPC

# # source /home/apps/anaconda3/etc/profile.d/conda.sh  # Adjust for your local Conda if needed
# # conda activate pytorch-gpu                          # Activate your local environment

echo "Python environment ready"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" # Check if this is set manually by the user

# --- Directory and Experiment Setup ---
OUTPUT_DIR="./cifar_output_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"

# --- nvidia-smi Logging ---
GPU_LOG_FILE="${OUTPUT_DIR}/gpu_stats.csv" # Changed OUTPUT_DIR_BASE to OUTPUT_DIR
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Starting nvidia-smi logging to ${GPU_LOG_FILE}"
    nvidia-smi --query-gpu=timestamp,name,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw --format=csv -l 5 -f "${GPU_LOG_FILE}" &
    NVIDIA_SMI_PID=$!
else
    # Attempt to detect GPUs if CUDA_VISIBLE_DEVICES is not set
    if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
        echo "CUDA_VISIBLE_DEVICES not set, but nvidia-smi found GPUs. Starting logging."
        echo "Consider setting CUDA_VISIBLE_DEVICES to control which GPUs are used."
        nvidia-smi --query-gpu=timestamp,name,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw --format=csv -l 5 -f "${GPU_LOG_FILE}" &
        NVIDIA_SMI_PID=$!
    else
        echo "No GPUs detected by CUDA_VISIBLE_DEVICES or nvidia-smi, skipping nvidia-smi logging."
        NVIDIA_SMI_PID=""
    fi
fi


# --- Training Command (Multi-GPU with torchrun if NUM_GPUS > 1) ---
# Determine number of GPUs available
# User can set CUDA_VISIBLE_DEVICES manually. If set, script respects it.
# Otherwise, it tries to count GPUs via nvidia-smi.
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
elif command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=0 # Default to 0 if no GPUs are assigned or detected
fi
echo "Number of GPUs to be used: $NUM_GPUS"


# Training Parameters
EPOCHS=50                 # CIFAR trains faster; 50 epochs can show good convergence
LEARNING_RATE=0.1        # Common starting LR for ResNet on CIFAR
BATCH_SIZE_PER_GPU=256   # CIFAR images are small; can often use larger batch sizes
                         # Adjust based on GPU VRAM and model size

# Determine available CPUs for workers
TOTAL_CPUS=$(nproc --all 2>/dev/null || echo 4) # Get total CPUs or default to 4
WORKERS_PER_GPU=2 # Or a sensible default
NUM_WORKERS=$(($TOTAL_CPUS / ($NUM_GPUS > 0 ? $NUM_GPUS : 1)))
NUM_WORKERS=$(($NUM_WORKERS > $WORKERS_PER_GPU ? $WORKERS_PER_GPU * ($NUM_GPUS > 0 ? $NUM_GPUS : 1) : $NUM_WORKERS)) # Cap workers per GPU
NUM_WORKERS=$(($NUM_WORKERS == 0 ? 1 : $NUM_WORKERS)) # Ensure at least 1 worker

echo "Starting PyTorch CIFAR10 training..."
echo "Epochs: ${EPOCHS}, Batch Size per GPU: ${BATCH_SIZE_PER_GPU}, Workers: ${NUM_WORKERS}"
echo "Output directory: ${OUTPUT_DIR}"

# Base Python command arguments
BASE_ARGS=" \
    cifar10_train.py \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE_PER_GPU} \
    --workers ${NUM_WORKERS} \
    --lr ${LEARNING_RATE} \
    --output-dir ${OUTPUT_DIR} \
    --amp" # Enable Automatic Mixed Precision
    # --resume ${OUTPUT_DIR}/checkpoint.pth # Uncomment to resume from a checkpoint

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Using ${NUM_GPUS} GPUs with torchrun."
    # For local execution, localhost is usually fine for rdzv_endpoint.
    RDZV_ENDPOINT="localhost:29500" # Simplified for local execution
    JOB_ID="localjob$$" # Simple job ID for local run

    PYTHON_COMMAND="torchrun --nproc_per_node=${NUM_GPUS} --nnodes=1 --rdzv_id=${JOB_ID} --rdzv_backend=c10d --rdzv_endpoint=${RDZV_ENDPOINT} ${BASE_ARGS}"
elif [ "$NUM_GPUS" -eq 1 ]; then
    echo "Using 1 GPU (direct script execution)."
    # Use 'python' from the activated environment
    PYTHON_COMMAND="python ${BASE_ARGS} --device cuda:0"
else # NUM_GPUS is 0
    echo "No GPUs specified or detected, running on CPU."
    # Use 'python' from the activated environment
    PYTHON_COMMAND="python ${BASE_ARGS} --device cpu"
fi


# Choose one profiling option if needed:
# Option 1: Run Python script directly
CMD_TO_RUN="${PYTHON_COMMAND}"

# Option 2: Run with Intel Vtune for HPC Performance Analysis
# VTUNE_RESULT_DIR="${OUTPUT_DIR}/vtune_hpc_cifar_resnet" # Corrected path
# mkdir -p "${VTUNE_RESULT_DIR}"
# CMD_TO_RUN="vtune -collect hpc-performance -result-dir \"${VTUNE_RESULT_DIR}\" -quiet -- ${PYTHON_COMMAND}"

echo "Executing: ${CMD_TO_RUN}"
eval "${CMD_TO_RUN}" # Use eval if your command has complex quoting or variables

TRAIN_EXIT_CODE=$?
echo "Training finished with exit code: ${TRAIN_EXIT_CODE}"

# --- Cleanup ---
if [ -n "$NVIDIA_SMI_PID" ]; then
    echo "Stopping nvidia-smi logging..."
    kill ${NVIDIA_SMI_PID}
    wait ${NVIDIA_SMI_PID} 2>/dev/null
fi

echo "Job finished. Output and logs are in ${OUTPUT_DIR}" # Changed OUTPUT_DIR_BASE to OUTPUT_DIR
# if [ -d "${VTUNE_RESULT_DIR}" ]; then
#     echo "Vtune results (if collected) are in ${VTUNE_RESULT_DIR}"
# fi

exit ${TRAIN_EXIT_CODE}
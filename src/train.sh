#!/bin/bash
# TODO under "construction"

# Inspired from https://github.com/Metacreation-Lab/MMM/blob/main/slurm/cedar/train_gpt2.sh
# which was in turn inspired from https://github.com/bigscience-workshop/bigscience/blob/7ccf7e42577fe71e88cf8bed3b9ca965c7afb8f7/train/tr11-176B-ml/tr11-176B-ml.slurm

# Set SLURM / hardware environment
#SBATCH --job-name=train-rwkv8192
#SBATCH --output=logs/train-rwkv8192.out
#SBATCH --error=logs/train-rwkv8192err.out
#SBATCH --account=def-pasquier
#SBATCH --mail-user=raa60@sfu.ca # Default mail
#SBATCH --nodes=1            # total nb of nodes
#SBATCH --ntasks-per-node=1  # nb of tasks per node
#SBATCH --gpus-per-node=v100l:4
#SBATCH --cpus-per-task=10   # nb of CPU cores per task
#SBATCH --mem=100G
#SBATCH --time=72:00:00

# Output GPUs and ram info
echo "START TIME: $(date)"
nvidia-smi
nvidia-smi topo -m
free -h

# Hardware vars
MASTER_HOSTNAME=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_IP=$(srun --nodes=1 --ntasks=1 -w "$MASTER_HOSTNAME" hostname --ip-address)
MASTER_PORT=9902
echo "Master hostname: $MASTER_HOSTNAME"
echo "Master addr: $MASTER_IP"
echo "Node list: $SLURM_JOB_NODELIST"

# Defining the right environment variables
# I have no idea whether these are still necessary, but I don't want to run the risk...
export PYTHONPATH=$SCRATCH/MMM
export HF_HOME=$SLURM_TMPDIR/.hf_cache
export HF_METRICS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
# The below variable is required to avoid a warning with the hf tokenizers lib and multiprocessing
# Weirdly, the tokenizer lib is used somewhere before that the dataloader create several workers,
# even when average_num_tokens_per_note is hardcoded in the Dataset class
# https://github.com/huggingface/transformers/issues/5486
# best explanation: https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996
export TOKENIZERS_PARALLELISM=0

echo "Starting venv"
# Load the python environment
module load python/3.11 scipy-stack/2025a gcc arrow/17.0.0 cudacore/.12.6.2 cudacompat/.12.6 
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip install --upgrade miditok

# Move hugging face dataset from scratch to local file system
# This is done on every nodes.
# Docs: https://docs.alliancecan.ca/wiki/Using_node-local_storage
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "mkdir $SLURM_TMPDIR/data && cp -r $SCRATCH/prefiltered $SLURM_TMPDIR/data/"

module list

echo "Path to CUDA core is $CUDA_CORE"

# Run the training
# Tensorboard can be access by running (with computenode replaced with the node hostname):
# ssh -N -f -L localhost:6006:computenode:6006 userid@cedar.computecanada.ca
tensorboard --logdir=outputs --host 0.0.0.0 --load_fast false & srun --jobid "$SLURM_JOBID" bash -c "python train.py"

echo "END TIME: $(date)"

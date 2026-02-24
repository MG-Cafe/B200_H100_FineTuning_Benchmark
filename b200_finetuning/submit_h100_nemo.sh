#!/bin/bash
#SBATCH --job-name=olmo3-nemo-h100
#SBATCH --partition=a3mega
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/logs/olmo3-nemo-h100-%j.out
#SBATCH --error=/home/%u/logs/olmo3-nemo-h100-%j.err

set -euo pipefail
WORK_DIR="/home/$(whoami)/olmo3-nemo"
CONTAINER="nvcr.io#nvidia/nemo-automodel:25.11.00"

HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
HEAD_ADDR=$(srun -N1 -n1 -w $HEAD_NODE hostname -I | awk '{print $1}')

# DCGM monitoring in background
srun --overlap -N $SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
  bash -c 'dcgmi dmon -e 155,150,203,204,1001,1002,1003,1004,1005 \
  -d 1000 > '${WORK_DIR}'/dcgm_$(hostname).csv 2>&1 &' &
sleep 2

srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
  --gpus-per-node=8 \
  --container-image=$CONTAINER \
  --container-writable \
  --container-mounts="${WORK_DIR}:${WORK_DIR}" \
  bash -c '
  set -e
  export HF_HOME='${WORK_DIR}'/.hf_cache
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export NCCL_DEBUG=WARN
  export NCCL_SOCKET_IFNAME=enp0s12
  export NCCL_IB_DISABLE=0
  export PYTHONPATH='${WORK_DIR}':$PYTHONPATH
  torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc-per-node=8 \
    --node-rank=$SLURM_PROCID \
    --master-addr='${HEAD_ADDR}' \
    --master-port=29500 \
    '${WORK_DIR}'/scripts/finetune.py \
    --config '${WORK_DIR}'/configs/olmo3_32b_h100.yaml
  '

echo 'TRAINING COMPLETE'

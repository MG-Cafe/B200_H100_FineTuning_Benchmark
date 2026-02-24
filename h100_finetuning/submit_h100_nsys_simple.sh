#!/bin/bash
#SBATCH --job-name=olmo3-nsys-h100
#SBATCH --partition=a3mega
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/logs/olmo3-nsys-h100-%j.out
#SBATCH --error=/home/%u/logs/olmo3-nsys-h100-%j.err

set -euo pipefail
WORK_DIR="/home/$(whoami)/olmo3-nemo"
CONTAINER="nvcr.io#nvidia/nemo-automodel:25.11.00"
NSYS_DIR="${}/nsys-h100"

HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
HEAD_ADDR=$(srun -N1 -n1 -w $HEAD_NODE hostname -I | awk '{print $1}')

srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
  --gpus-per-node=8 \
  --container-image=$CONTAINER \
  --container-writable \
  --container-mounts="${}:${}" \
  bash -c '
  set -e
  export HF_HOME='${}'/.hf_cache
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export NCCL_DEBUG=WARN
  export NCCL_SOCKET_IFNAME=enp0s12
  export NCCL_IB_DISABLE=0
  export PYTHONPATH='${}':$PYTHONPATH

  mkdir -p '${NSYS_DIR}'

  nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --gpu-metrics-frequency=10000 \
    --delay=30 \
    --duration=180 \
    --output='${NSYS_DIR}'/nsys_h100_rank${SLURM_PROCID}_%h \
    --force-overwrite=true \
    torchrun \
      --nnodes=$SLURM_JOB_NUM_NODES \
      --nproc-per-node=8 \
      --node-rank=$SLURM_PROCID \
      --master-addr='${}' \
      --master-port=29500 \
      '${}'/scripts/finetune.py \
      --config '${}'/configs/olmo3_32b_h100.yaml
  '

echo "NSYS PROFILING COMPLETE"

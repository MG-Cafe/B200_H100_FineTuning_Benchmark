#!/bin/bash
#SBATCH --job-name=olmo3-bench-b200
#SBATCH --partition=a4high
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/home/sa_101048180276343241939/logs/olmo3-bench-b200-%j.out
#SBATCH --error=/home/sa_101048180276343241939/logs/olmo3-bench-b200-%j.err

set -euo pipefail
WORK_DIR="/home/sa_101048180276343241939/olmo3-nemo"
BENCH_DIR="/home/sa_101048180276343241939/olmo3-nemo/benchmark-b200"
CONTAINER="nvcr.io#nvidia/nemo-automodel:25.11.00"

HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
HEAD_ADDR=$(srun -N1 -n1 -w $HEAD_NODE hostname -I | awk '{print $1}')

echo "=== B200 BENCHMARK START ==="
echo "HEAD_NODE: $HEAD_NODE"
echo "HEAD_ADDR: $HEAD_ADDR"
echo "NODES: $SLURM_JOB_NUM_NODES"
date -u
mkdir -p $BENCH_DIR

srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
  --gpus-per-node=8 \
  --container-image=$CONTAINER \
  --container-writable \
  --container-mounts="$WORK_DIR:$WORK_DIR" \
  bash -c '
  set -e
  export HF_HOME='$WORK_DIR'/.hf_cache
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export NCCL_DEBUG=WARN
  export NCCL_SOCKET_IFNAME=enp0s19
  export NCCL_IB_DISABLE=0
  export PYTHONPATH='$WORK_DIR':$PYTHONPATH

  mkdir -p '$BENCH_DIR'
  date -u > '$BENCH_DIR'/start_time.txt
  nvidia-smi dmon -s u -d 2 > '$BENCH_DIR'/gpu_util.csv 2>&1 &
  SMI_PID=$!
  nvidia-smi --query-gpu=name,memory.total --format=csv > '$BENCH_DIR'/gpu_info.csv 2>&1

  nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --delay=30 \
    --duration=300 \
    --output='$BENCH_DIR'/nsys_b200_rank$SLURM_PROCID \
    --force-overwrite=true \
    torchrun \
      --nnodes=$SLURM_JOB_NUM_NODES \
      --nproc-per-node=8 \
      --node-rank=$SLURM_PROCID \
      --master-addr='$HEAD_ADDR' \
      --master-port=29500 \
      '$WORK_DIR'/scripts/finetune.py \
      --config '$WORK_DIR'/configs/olmo3_32b_b200.yaml

  date -u > '$BENCH_DIR'/end_time.txt
  kill $SMI_PID 2>/dev/null || true
  '

echo "=== B200 BENCHMARK END ==="
date -u
echo "BENCHMARK COMPLETE"

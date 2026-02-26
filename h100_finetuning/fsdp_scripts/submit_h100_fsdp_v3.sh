#!/bin/bash
#SBATCH --job-name=olmo3-fsdp
#SBATCH --partition=a3mega
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/logs/olmo3-fsdp-%j.out
#SBATCH --error=/home/%u/logs/olmo3-fsdp-%j.err

set -euo pipefail

WORK_DIR="/home/$(whoami)/olmo3-nemo"
BENCH_DIR="${WORK_DIR}/benchmark-h100-fsdp"
SCRIPT_DIR="${WORK_DIR}/scripts"
DATA_DIR="${WORK_DIR}/data"
CONTAINER_IMAGE="${WORK_DIR}/nvidia+pytorch+24.04-py3.sqsh"
HF_CACHE="${WORK_DIR}/.hf_cache"
OUTPUT_DIR="${BENCH_DIR}/output"

# TCPXO env vars from host
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
NCCL_LIB_DIR="/var/lib/tcpxo/lib64" source /var/lib/tcpxo/lib64/nccl-env-profile.sh
export NCCL_DEBUG=WARN

HOST_VARS=$(sed 's/ \{1,\}/,/g' <<<"${!NCCL*}")

CONTAINER_MOUNTS="/var/tmp:/var/tmp"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${WORK_DIR}:${WORK_DIR}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS},/var/lib/tcpxo/lib64/"

HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
HEAD_ADDR=$(srun -N1 -n1 -w $HEAD_NODE hostname -I | awk '{print $1}')
MASTER_PORT=29500

echo "=== H100 FSDP + TCPXO (v3) ==="
echo "HEAD_NODE: $HEAD_NODE"
echo "Nodes: $SLURM_JOB_NUM_NODES"
date -u
mkdir -p $BENCH_DIR $OUTPUT_DIR ~/logs

srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
  --gpus-per-node=8 \
  --container-image=$CONTAINER_IMAGE \
  --container-writable \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-env="${HOST_VARS}" \
  bash -c '
  set -e

  # TCPXO: LD_PRELOAD overrides container NCCL
  export LD_PRELOAD=/var/lib/tcpxo/lib64/libnccl.so.2.28.7
  export LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:$LD_LIBRARY_PATH
  export HF_HOME='$HF_CACHE'
  export HUGGINGFACE_HUB_CACHE='$HF_CACHE'
  export TOKENIZERS_PARALLELISM=false
  export NCCL_DEBUG=WARN
  export TORCH_SHOW_CPP_STACKTRACES=1

  pip install --quiet "transformers>=4.57.0,<5.0.0" "sentencepiece" "datasets" 2>&1 | tail -5

  LOGDIR='$BENCH_DIR'/torchrun_logs_${SLURM_PROCID}_$(date +%s)
  mkdir -p $LOGDIR
  nvidia-smi dmon -s u -d 2 > '$BENCH_DIR'/gpu_util_$(hostname).csv 2>&1 &

  torchrun \
    --nnodes='$SLURM_JOB_NUM_NODES' \
    --nproc-per-node=8 \
    --node-rank=${SLURM_PROCID} \
    --master-addr='$HEAD_ADDR' \
    --master-port='$MASTER_PORT' \
    --redirects=3 --log-dir=$LOGDIR \
    '$SCRIPT_DIR'/train_fsdp_v3.py \
      --model_name allenai/Olmo-3-1125-32B \
      --data_path '$DATA_DIR'/train.jsonl \
      --output_dir '$OUTPUT_DIR' \
      --max_length 2048 --batch_size 1 \
      --gradient_accumulation_steps 4 \
      --num_epochs 1 --learning_rate 2e-5 \
      --warmup_steps 10 --log_every 1 --save_every 50
  '

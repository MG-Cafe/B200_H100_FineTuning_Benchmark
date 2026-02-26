#!/bin/bash
#SBATCH --job-name=olmo3-fsdp-nsys
#SBATCH --partition=a3mega
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/logs/olmo3-fsdp-nsys-%j.out
#SBATCH --error=/home/%u/logs/olmo3-fsdp-nsys-%j.err

set -euo pipefail

WORK_DIR="/home/$(whoami)/olmo3-nemo"
BENCH_DIR="${WORK_DIR}/benchmark-h100-fsdp-nsys"
SCRIPT_DIR="${WORK_DIR}/scripts"
DATA_DIR="${WORK_DIR}/data"
CONTAINER_IMAGE="${WORK_DIR}/nvidia+pytorch+24.04-py3.sqsh"
HF_CACHE="${WORK_DIR}/.hf_cache"
OUTPUT_DIR="${BENCH_DIR}/output"
NSYS_DIR="${BENCH_DIR}/nsys"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
NCCL_LIB_DIR="/var/lib/tcpxo/lib64" source /var/lib/tcpxo/lib64/nccl-env-profile.sh
export NCCL_DEBUG=WARN
HOST_VARS=$(sed 's/ \{1,\}/,/g' <<<"${!NCCL*}")

CONTAINER_MOUNTS="/var/tmp:/var/tmp,${WORK_DIR}:${WORK_DIR},/var/lib/tcpxo/lib64/"

HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
HEAD_ADDR=$(srun -N1 -n1 -w $HEAD_NODE hostname -I | awk '{print $1}')
MASTER_PORT=29500

echo "=== H100 FSDP + TCPXO + NSIGHT (v3) ==="
date -u
mkdir -p $BENCH_DIR $OUTPUT_DIR $NSYS_DIR ~/logs

srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
  --gpus-per-node=8 \
  --container-image=$CONTAINER_IMAGE \
  --container-writable \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-env="${HOST_VARS}" \
  bash -c '
  set -e

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

  # GPU monitoring
  nvidia-smi dmon -s u -d 2 > '$BENCH_DIR'/gpu_util_$(hostname).csv 2>&1 &
  SMI_PID=$!

  # DCGM monitoring (if available)
  if command -v dcgmi &> /dev/null; then
    dcgmi dmon -e 150,155,156,100,101,140,203,204,252 -d 1000 \
      > '$BENCH_DIR'/dcgm_$(hostname).csv 2>&1 &
    DCGM_PID=$!
  fi

  date -u > '$BENCH_DIR'/start_time.txt

  # Nsight: profile rank 0 only
  if [ "${SLURM_PROCID}" -eq 0 ]; then
    nsys profile \
      --trace=cuda,nvtx,osrt,cudnn,cublas \
      --gpu-metrics-device=all \
      --cuda-memory-usage=true \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop \
      -o '$NSYS_DIR'/h100_fsdp_tcpxo_$(hostname) \
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
  else
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
  fi

  TRAIN_EXIT=$?
  date -u > '$BENCH_DIR'/end_time.txt
  kill $SMI_PID 2>/dev/null || true
  kill $DCGM_PID 2>/dev/null || true
  exit $TRAIN_EXIT
  '

echo "=== BENCHMARK END ==="
date -u

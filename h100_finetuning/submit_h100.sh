#!/bin/bash
#SBATCH --job-name=olmo3-finetune-h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/logs/olmo3-h100-%j.out
#SBATCH --error=/home/%u/logs/olmo3-h100-%j.err

set -euo pipefail

WORK_DIR="/home/$(whoami)/olmo3-finetune"
DATA_DIR="${WORK_DIR}/simulated_training_data"
OUTPUT_DIR="${WORK_DIR}/output-h100-$(date +%Y%m%d-%H%M%S)"
SCRIPT_DIR="${WORK_DIR}/scripts"
HF_CACHE="${WORK_DIR}/.hf_cache"

mkdir -p "${OUTPUT_DIR}" "${HF_CACHE}" "/home/$(whoami)/logs"

CONTAINER_IMAGE="nvcr.io#nvidia/pytorch:25.02-py3"

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
HEAD_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
MASTER_PORT=29500

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $SLURM_JOB_NUM_NODES"
echo "GPUs/node:     8"
echo "Total GPUs:    $((SLURM_JOB_NUM_NODES * 8))"
echo "Head node:     $HEAD_NODE ($HEAD_ADDR)"
echo "Output dir:    $OUTPUT_DIR"
echo "Container:     $CONTAINER_IMAGE"
echo "Max Length:    2048"
echo "========================================"

srun --overlap -N $SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
  bash -c "
    dcgmi dmon -e 155,150,203,204,1001,1002,1003,1004,1005 \
      -d 1000 > ${OUTPUT_DIR}/dcgm_node_\$(hostname).csv 2>&1 &
    echo 'DCGM started on \$(hostname)'
  " &
DCGM_PID=$!
sleep 2

if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
  echo "ERROR: Training data not found at ${DATA_DIR}/train.jsonl"
  exit 1
else
  echo "Training data: $(wc -l < "${DATA_DIR}/train.jsonl") samples"
fi

echo "Launching fine-tuning..."
srun -N $SLURM_JOB_NUM_NODES \
  --ntasks-per-node=1 \
  --gpus-per-node=8 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-writable \
  --container-mounts="${WORK_DIR}:${WORK_DIR}" \
  bash -c "
    set -e
    pip install --quiet 'transformers>=4.57.0,<4.59.0' 'accelerate>=1.2.0,<1.6.0' 'datasets' 'sentencepiece' 2>/dev/null

    export HF_HOME=${HF_CACHE}
    export HUGGINGFACE_HUB_CACHE=${HF_CACHE}
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_DEBUG=WARN
    export NCCL_IB_DISABLE=0
    export NCCL_SOCKET_IFNAME=enp0s12
    export TORCH_SHOW_CPP_STACKTRACES=1

    echo \"===== ENV CHECK =====\"
    python -c \"import torch; print(f'PyTorch={torch.__version__}, CUDA={torch.version.cuda}, GPUs={torch.cuda.device_count()}')\"
    python -c \"import transformers; print(f'transformers={transformers.__version__}')\"
    python -c \"import accelerate; print(f'accelerate={accelerate.__version__}')\"
    echo \"====================\"

    accelerate launch \
      --config_file ${SCRIPT_DIR}/accelerate_fsdp_config.yaml \
      --num_machines ${SLURM_JOB_NUM_NODES} \
      --num_processes \$((SLURM_JOB_NUM_NODES * 8)) \
      --machine_rank \${SLURM_PROCID} \
      --main_process_ip ${HEAD_ADDR} \
      --main_process_port ${MASTER_PORT} \
      ${SCRIPT_DIR}/train.py \
        --model_name allenai/Olmo-3-1125-32B \
        --data_path ${DATA_DIR}/train.jsonl \
        --output_dir ${OUTPUT_DIR} \
        --gpu_type h100 \
        --max_length 2048 \
        --batch_size 1 \
        --gradient_accumulation_steps 4 \
        --num_epochs 1 \
        --learning_rate 2e-5 \
        --warmup_steps 10 \
        --log_every 1 \
        --save_every 50 \
        --profile_steps 5 \
        --profile_start_step 3
  "

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
  echo "ERROR: Training failed with exit code $TRAIN_EXIT"
  kill $DCGM_PID 2>/dev/null || true
  exit $TRAIN_EXIT
fi

echo "TRAINING COMPLETE — Starting Nsight profiling..."

NSYS_DIR="${OUTPUT_DIR}/nsight_traces_h100"
mkdir -p "${NSYS_DIR}"

srun -N $SLURM_JOB_NUM_NODES \
  --ntasks-per-node=1 \
  --gpus-per-node=8 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-writable \
  --container-mounts="${WORK_DIR}:${WORK_DIR}" \
  bash -c "
    set -e
    pip install --quiet 'transformers>=4.57.0,<4.59.0' 'accelerate>=1.2.0,<1.6.0' 'datasets' 'sentencepiece' 2>/dev/null

    export HF_HOME=${HF_CACHE}
    export HUGGINGFACE_HUB_CACHE=${HF_CACHE}
    export NCCL_DEBUG=WARN
    export NCCL_SOCKET_IFNAME=enp0s12
    export NCCL_IB_DISABLE=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TOKENIZERS_PARALLELISM=false

    nsys profile \
      --trace=cuda,nvtx,osrt,cudnn,cublas \
      --cuda-memory-usage=true \
      --sample=none \
      --delay=60 \
      --duration=300 \
      --kill=signal-handler \
      --force-overwrite=true \
      --output=${NSYS_DIR}/h100_rank\${SLURM_PROCID}_%h \
      accelerate launch \
        --config_file ${SCRIPT_DIR}/accelerate_fsdp_config.yaml \
        --num_machines ${SLURM_JOB_NUM_NODES} \
        --num_processes \$((SLURM_JOB_NUM_NODES * 8)) \
        --machine_rank \${SLURM_PROCID} \
        --main_process_ip ${HEAD_ADDR} \
        --main_process_port \$((MASTER_PORT + 1)) \
        ${SCRIPT_DIR}/train.py \
          --model_name allenai/Olmo-3-1125-32B \
          --data_path ${DATA_DIR}/train.jsonl \
          --output_dir ${OUTPUT_DIR}/nsight_run \
          --gpu_type h100 \
          --max_length 2048 \
          --batch_size 1 \
          --gradient_accumulation_steps 4 \
          --num_epochs 1 \
          --learning_rate 2e-5 \
          --warmup_steps 5 \
          --log_every 1 \
          --save_every 9999 \
          --profile_steps 0 \
          --max_steps 20
  "

echo "Nsight traces: ${NSYS_DIR}/"
ls -lh "${NSYS_DIR}/" 2>/dev/null || echo "(no traces)"
kill $DCGM_PID 2>/dev/null || true

srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
  bash -c "nvidia-smi --query-gpu=name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu \
    --format=csv > ${OUTPUT_DIR}/nvidia_smi_\$(hostname).csv"

echo "========================================"
echo "ALL DONE — Output: ${OUTPUT_DIR}"
echo "========================================"

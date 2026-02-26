# OLMo-3 32B Fine-Tuning Benchmark: H100 vs B200

> **Three-way GPU performance comparison** for fine-tuning [OLMo-3-1125-32B](https://huggingface.co/allenai/Olmo-3-1125-32B) (32.2B parameters) on Google Cloud Platform using Cluster Toolkit, Slurm, and distributed training across 16 GPUs.

## Headline Results

| Metric | H100 NeMo | H100 FSDP+TCPXO | B200 NeMo |
|---|---|---|---|
| **Training Time (31 steps)** | 6 min 31 sec | ~4 min 2 sec (projected) | **35 sec** |
| **Throughput (TPS)** | 2,522 | 15,572 | **28,887** |
| **Avg Step Time** | 13.0 sec | 7.7 sec | **1.2 sec** |
| **Peak GPU Memory** | 27.0 GiB | 24.9 GB | 27.0 GiB |
| **Framework** | NeMo AutoModel (FSDP2) | PyTorch FSDP (native) | NeMo AutoModel (FSDP2) |
| **Networking** | Standard | TCPXO (330 GB/s) | GPUDirect RDMA (RoCE) |

**B200 delivers 11.5× higher throughput than H100 NeMo baseline, and remains ~6.4× faster than the optimized H100+TCPXO run.**

---

## Table of Contents

1. [Prerequisites and Quotas](#1-prerequisites-and-quotas)
2. [Infrastructure Setup](#2-infrastructure-setup)
3. [Deploy GPU Clusters](#3-deploy-gpu-clusters)
4. [Post-Deployment Steps (Both Clusters)](#4-post-deployment-steps-both-clusters)
5. [Simulated Training Data Generation](#5-simulated-training-data-generation)
6. [Fine-Tuning with NeMo AutoModel (Recommended)](#6-fine-tuning-with-nemo-automodel-recommended)
7. [Fine-Tuning with Native PyTorch FSDP + TCPXO (H100)](#7-fine-tuning-with-native-pytorch-fsdp--tcpxo-h100)
8. [NCCL Communication Benchmarks](#8-nccl-communication-benchmarks)
9. [Nsight Systems Profiling and Analysis](#9-nsight-systems-profiling-and-analysis)
10. [Benchmark Results and Comparison](#10-benchmark-results-and-comparison)
11. [Switching to Reservations](#11-switching-to-reservations)
12. [Teardown](#12-teardown)
13. [Troubleshooting](#13-troubleshooting)
14. [Appendix: HuggingFace Accelerate FSDP (Reproducing OOM Failures)](#14-appendix-huggingface-accelerate-fsdp-reproducing-oom-failures)

---

## 1. Prerequisites and Quotas

### 1.1 GCP Project Requirements

- A GCP project with billing enabled.
- The `gcloud` CLI installed and configured locally or in Cloud Shell.

### 1.2 GPU Quotas

Request these quotas **before** deploying. Increases can take hours or days.

| Quota | Region | Needed For |
|---|---|---|
| NVIDIA H100 GPUs (16) | `us-central1` | H100 cluster (`a3-megagpu-8g`) |
| NVIDIA B200 GPUs (16) | `us-south1` | B200 cluster (`a4-highgpu-8g`) |
| `SsdStorageGibPerRegion` | `us-central1` | Filestore (BASIC_SSD fallback for H100) |
| `HighScaleSSDStorageGibPerRegion` | `us-south1` | Filestore for B200 (or use BASIC_SSD) |

### 1.3 Placeholder Conventions

Throughout this guide, replace placeholders with your own values:

| Placeholder | Description | Example |
|---|---|---|
| `<YOUR_PROJECT_ID>` | GCP project ID | `my-gpu-benchmark` |
| `<YOUR_NGC_API_KEY>` | NVIDIA NGC API key ([get one here](https://ngc.nvidia.com/setup/api-key)) | `bnZhcGkta2V5...` |
| `<YOUR_H100_TF_STATE_BUCKET>` | GCS bucket for H100 Terraform state | `olmo3-h100-tf-state` |
| `<YOUR_B200_TF_STATE_BUCKET>` | GCS bucket for B200 Terraform state | `olmo3-b200-tf-state` |
| `<YOUR_RESULTS_BUCKET>` | GCS bucket for benchmark results | `olmo3-benchmark-results` |
| `<YOUR_USER>` | Your Linux username on the cluster | `sa_12345` (check with `whoami`) |

---

## 2. Infrastructure Setup

### Step 1: Create Management Workstation (Cloud Shell)

Run all commands in this section from **GCP Cloud Shell**.

```bash
gcloud config set project <YOUR_PROJECT_ID>

# Enable required APIs
gcloud services enable \
  compute.googleapis.com file.googleapis.com storage.googleapis.com \
  serviceusage.googleapis.com cloudresourcemanager.googleapis.com \
  iam.googleapis.com servicenetworking.googleapis.com

# Create workstation VM
gcloud compute instances create cluster-mgmt \
  --project=<YOUR_PROJECT_ID> \
  --zone=us-central1-a \
  --machine-type=e2-standard-8 \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --scopes=cloud-platform \
  --metadata=enable-oslogin=TRUE

# SSH into it
gcloud compute ssh cluster-mgmt --zone=us-central1-a
```

### Step 2: Install Dependencies on Workstation

Run inside the `cluster-mgmt` VM:

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y git build-essential unzip wget curl jq

# Install Go
wget https://go.dev/dl/go1.23.6.linux-amd64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.23.6.linux-amd64.tar.gz
rm go1.23.6.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc && source ~/.bashrc

# Install Terraform and Packer
wget -O - https://apt.releases.hashicorp.com/gpg | \
  sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) \
  signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] \
  https://apt.releases.hashicorp.com $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt-get update && sudo apt-get install -y terraform packer

# Install Cluster Toolkit
cd ~ && git clone https://github.com/GoogleCloudPlatform/cluster-toolkit.git
cd cluster-toolkit && make
echo 'export PATH=$HOME/cluster-toolkit:$PATH' >> ~/.bashrc && source ~/.bashrc

# Verify (do NOT run gcloud auth login — Context-Aware Access blocks it on VMs)
gcloud config set project <YOUR_PROJECT_ID>
gcloud compute instances list --limit=1
```

### Step 3: IAM Permissions (Run from Cloud Shell)

> **Important:** Run this from **Cloud Shell**, not the workstation VM. The compute SA cannot modify its own IAM policies.

```bash
PROJECT_NUMBER=$(gcloud projects describe <YOUR_PROJECT_ID> \
  --format='value(projectNumber)')
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

for ROLE in \
  roles/compute.instanceAdmin.v1 roles/compute.securityAdmin \
  roles/iam.serviceAccountUser roles/monitoring.metricWriter \
  roles/logging.logWriter roles/storage.objectAdmin \
  roles/file.editor roles/servicenetworking.networksAdmin \
  roles/iap.tunnelResourceAccessor; do
  gcloud projects add-iam-policy-binding <YOUR_PROJECT_ID> \
    --member="serviceAccount:${SA}" --role="${ROLE}" \
    --condition=None --quiet
done
```

### Step 4: Create GCS Buckets for Terraform State

```bash
gcloud storage buckets create gs://<YOUR_H100_TF_STATE_BUCKET> \
  --project=<YOUR_PROJECT_ID> --location=us-central1 \
  --uniform-bucket-level-access

gcloud storage buckets create gs://<YOUR_B200_TF_STATE_BUCKET> \
  --project=<YOUR_PROJECT_ID> --location=us-south1 \
  --uniform-bucket-level-access
```

---

## 3. Deploy GPU Clusters

### Cluster 1: B200 (a4-highgpu-8g) — us-south1-b

Uses the official A4 High blueprint with first-class DWS Flex support. No patches needed.

#### Step 6A: Create B200 Deployment File

```bash
cd ~/cluster-toolkit
cat > b200-deployment.yaml << 'EOF'
---
terraform_backend_defaults:
  type: gcs
  configuration:
    bucket: <YOUR_B200_TF_STATE_BUCKET>
vars:
  deployment_name: olmo3-b200
  project_id: <YOUR_PROJECT_ID>
  region: us-south1
  zone: us-south1-b
  a4h_cluster_size: 2
  a4h_dws_flex_enabled: true
  a4h_enable_spot_vm: false
  a4h_reservation_name: ""
EOF
```

> **Remember** to replace `<YOUR_B200_TF_STATE_BUCKET>` and `<YOUR_PROJECT_ID>` with your actual values.

#### Step 7A: Deploy B200 Cluster

```bash
cd ~/cluster-toolkit
./gcluster deploy \
  -d b200-deployment.yaml \
  examples/machine-learning/a4-highgpu-8g/a4high-slurm-blueprint.yaml \
  --auto-approve
```

Total time: ~45–60 minutes (builds custom Slurm image ~35 min, then deploys cluster ~15 min).

#### Step 8A: Connect and Validate B200

Run from the workstation VM (`cluster-mgmt`):

```bash
# Find login node
gcloud compute instances list --project=<YOUR_PROJECT_ID> \
  --filter="name~olmo3.*b200.*login" --format="table(name,zone,status)"

# Create SSH firewall rule
NETWORK=$(gcloud compute instances describe olmo3b200-slurm-login-001 \
  --zone=us-south1-b --format='get(networkInterfaces[0].network)' \
  --project=<YOUR_PROJECT_ID> | awk -F/ '{print $NF}')

gcloud compute firewall-rules create allow-ssh-b200 \
  --network=$NETWORK --allow=tcp:22 \
  --source-ranges=0.0.0.0/0 --project=<YOUR_PROJECT_ID>

# SSH to login node
gcloud compute ssh olmo3b200-slurm-login-001 \
  --zone=us-south1-b --project=<YOUR_PROJECT_ID>
```

On the B200 login node:

```bash
# Verify GPUs (should show 8× B200)
srun -N 1 --gpus-per-node=8 --exclusive nvidia-smi
```

---

### Cluster 2: H100 (a3-megagpu-8g) — us-central1-a

Uses `a3-megagpu-8g` with TCPXO networking. Requires blueprint patches for DWS Flex compatibility and Filestore quota workaround.

> **Why a3-megagpu-8g instead of a3-highgpu-8g?** The `a3-highgpu-8g` blueprint requires private TCPX kernel credentials. `a3-megagpu-8g` uses TCPXO which needs no credentials. Same H100 80GB GPUs.

#### Step 6B: Patch the Blueprint

The blueprint needs two patches:
1. `enable_placement: false` — required for DWS Flex compatibility
2. `filestore_tier: BASIC_SSD` + `size_gb: 2560` — avoids quota exhaustion

```bash
cd ~/cluster-toolkit

# Copy the blueprint
cp examples/machine-learning/a3-megagpu-8g/a3mega-slurm-blueprint.yaml \
   a3mega-slurm-blueprint-patched.yaml

# Patch 1: Disable placement for DWS Flex
sed -i '/dws_flex:/i\      enable_placement: false' \
  a3mega-slurm-blueprint-patched.yaml

# Patch 2: Switch Filestore to BASIC_SSD
sed -i 's/filestore_tier: HIGH_SCALE_SSD/filestore_tier: BASIC_SSD/' \
  a3mega-slurm-blueprint-patched.yaml
sed -i 's/size_gb: 10240/size_gb: 2560/' \
  a3mega-slurm-blueprint-patched.yaml

# Verify patches
grep -B 1 -A 2 "enable_placement" a3mega-slurm-blueprint-patched.yaml
grep "filestore_tier\|size_gb" a3mega-slurm-blueprint-patched.yaml
```

#### Step 7B: Create H100 Deployment YAML

```bash
cd ~/cluster-toolkit
cat > h100-deployment.yaml << 'EOF'
---
terraform_backend_defaults:
  type: gcs
  configuration:
    bucket: <YOUR_H100_TF_STATE_BUCKET>
vars:
  deployment_name: olmo3-h100
  project_id: <YOUR_PROJECT_ID>
  region: us-central1
  zone: us-central1-a
  network_name_system: olmo3-h100-sys-net
  subnetwork_name_system: olmo3-h100-sys-subnet
  enable_ops_agent: true
  enable_nvidia_dcgm: true
  enable_nvidia_persistenced: true
  disk_size_gb: 200
  final_image_family: slurm-olmo3-h100
  slurm_cluster_name: olmo3h100
  a3mega_cluster_size: 2
  a3mega_reservation_name: ""
  a3mega_dws_flex_enabled: true
  a3mega_enable_spot_vm: false
EOF
```

> **Remember** to replace `<YOUR_H100_TF_STATE_BUCKET>` and `<YOUR_PROJECT_ID>`.

#### Step 8B: Deploy the H100 Cluster

```bash
cd ~/cluster-toolkit
./gcluster deploy \
  -d h100-deployment.yaml \
  a3mega-slurm-blueprint-patched.yaml \
  --auto-approve
```

Total time: ~45–60 minutes. If re-deploying (image already built), add: `--only primary,cluster --auto-approve -w`

#### Step 9B: Connect and Validate H100

Run from workstation VM (`cluster-mgmt`):

```bash
# Find login node
gcloud compute instances list --project=<YOUR_PROJECT_ID> \
  --filter="name~olmo3h100.*login" --format="table(name,zone,status)"

# Create SSH firewall rule
NETWORK=$(gcloud compute instances describe olmo3h100-login-001 \
  --zone=us-central1-a --format='get(networkInterfaces[0].network)' \
  --project=<YOUR_PROJECT_ID> | awk -F/ '{print $NF}')

gcloud compute firewall-rules create allow-ssh-h100 \
  --network=$NETWORK --allow=tcp:22 \
  --source-ranges=0.0.0.0/0 --project=<YOUR_PROJECT_ID>

# SSH to login node
gcloud compute ssh olmo3h100-login-001 \
  --zone=us-central1-a --project=<YOUR_PROJECT_ID>
```

On the H100 login node:

```bash
# Verify GPUs (should show 8× H100 80GB SXM)
srun -N 1 --gpus-per-node=8 --exclusive nvidia-smi
```

---

## 4. Post-Deployment Steps (Both Clusters)

SSH into **each** cluster's login node and run the following.

### Step 10: Set Up NGC Credentials

```bash
mkdir -p ~/.config/enroot/
cat > ~/.config/enroot/.credentials << 'CREDS'
machine nvcr.io login $oauthtoken password <YOUR_NGC_API_KEY>
machine authn.nvidia.com login $oauthtoken password <YOUR_NGC_API_KEY>
CREDS
chmod 600 ~/.config/enroot/.credentials
```

Get your key from: https://ngc.nvidia.com/setup/api-key

### Step 11: Verify OLMo-3 Readiness

```bash
srun -N 1 --gpus-per-node=8 --exclusive \
  --container-image=nvcr.io#nvidia/pytorch:24.12-py3 \
  --container-writable \
  bash -c '
    pip install "transformers<4.52" --quiet &&
    python -c "
from transformers import AutoTokenizer
print(\"OLMo-3 32B tokenizer loading...\")
tok = AutoTokenizer.from_pretrained(\"allenai/Olmo-3-1125-32B\")
print(f\"Vocab size: {tok.vocab_size}\")
print(\"SUCCESS: Ready for fine-tuning\")
"'
```

> Use `transformers<4.52` (not `>=4.57.0`) for compatibility with the PyTorch in the `24.12` container. For actual training, you will use different containers and versions as specified in later sections.

### Step 12: Verify DCGM (GPU Monitoring)

```bash
srun -N 1 --gpus-per-node=8 --exclusive dcgmi discovery -l
```

---

## 5. Simulated Training Data Generation

The benchmark uses a simulated multimodal malware analysis dataset that replicates the computational profile of a production workload (long-context hex-encoded binaries filling up to 65K tokens per sample).

### Clone the Repository

```bash
git clone https://github.com/MG-Cafe/B200_H100_FineTuning_Benchmark.git ~/B200_H100_FineTuning_Benchmark
```

### Generate the Dataset

```bash
mkdir -p ~/olmo3-nemo/data
cd ~/B200_H100_FineTuning_Benchmark

python h100_finetuning/fsdp_scripts/generate_training_data.py

mv simulated_training_data/train.jsonl ~/olmo3-nemo/data/train.jsonl
```

### Verify

```bash
wc -l ~/olmo3-nemo/data/train.jsonl
# Expected: 500

du -sh ~/olmo3-nemo/data/train.jsonl
# Expected: ~140 MB
```

### Dataset Properties

| Property | Value |
|---|---|
| Total Samples | 500 |
| Format | JSON Lines (.jsonl) |
| Label Distribution | ~50% malicious, ~50% benign |
| Binary Size Range | 64 KB – 512 KB per sample |
| Context Window Target | 65,000 tokens (OLMo-3 maximum) |
| Approx Input Tokens | 16,000 – 65,000 per sample |
| Approx Output Tokens | 50 – 150 per sample |

---

## 6. Fine-Tuning with NeMo AutoModel (Recommended)

This is the **recommended** approach. It uses NVIDIA's NeMo AutoModel container with FSDP2, which handles model wrapping and activation checkpointing correctly. This method is used for both the **H100 (without TCPXO)** and **B200** benchmark runs.

### 6.1 Critical Warnings

| ⚠️ Warning | Details |
|---|---|
| **CONTAINER** | Use `nemo-automodel:25.11.00`, **NOT** `nemo:25.11.01`. Completely different containers with incompatible APIs. |
| **NO PIP** | The container has everything pre-installed. Do **NOT** `pip install` anything. |
| **HEREDOC** | Do **NOT** write Python files using `cat << EOF`. Curly braces get eaten by bash. Use a Python writer script. |
| **MODEL DOWNLOAD** | Pre-download the model on a single GPU **BEFORE** multi-GPU training. 16 processes downloading simultaneously will fail. |
| **CORRUPT CHECKPOINTS** | If a job is killed mid-training, delete the checkpoint directory before resubmitting. |
| **DCGM OVERLAP** | Do **NOT** use `srun --overlap` for DCGM monitoring on B200. It sends SIGTERM to training processes. Use `nvidia-smi dmon` inside the container. |
| **NCCL_SOCKET_IFNAME** | B200 uses `enp0s19`, H100 uses `enp0s12`. Wrong NIC = NCCL timeout. |
| **LOSS** | Use `MaskedCrossEntropy`, **NOT** `FusedLinearCrossEntropy` (requires `output_hidden_states` which OLMo3 doesn't support). |
| **DATASET** | Return plain Python lists from the dataset adapter, **NOT** tensors. `default_collater` expects lists. |
| **GRADIENT CHECKPOINTING** | Set `activation_checkpointing: true` in the `distributed` section. A top-level `gradient_checkpointing` field is ignored. |

### 6.2 Initial Setup (Both Clusters)

SSH into the cluster login node and run:

```bash
# Create directory structure
mkdir -p ~/olmo3-nemo/{configs,scripts,data,output-h100,output-b200,benchmark-h100,benchmark-b200} ~/logs

# NGC credentials (if not already done in Step 10)
mkdir -p ~/.config/enroot
cat > ~/.config/enroot/.credentials << 'EOF'
machine nvcr.io login $oauthtoken password <YOUR_NGC_API_KEY>
machine authn.nvidia.com login $oauthtoken password <YOUR_NGC_API_KEY>
EOF
chmod 600 ~/.config/enroot/.credentials

# Copy training data
cp ~/B200_H100_FineTuning_Benchmark/simulated_training_data/train.jsonl ~/olmo3-nemo/data/train.jsonl
# Or if already generated:
# cp ~/olmo3-nemo/data/train.jsonl (already in place)

# Clone NeMo AutoModel repo for the entry point
cd ~/olmo3-nemo
git clone https://github.com/NVIDIA-NeMo/Automodel.git nemo-automodel-repo
cp nemo-automodel-repo/examples/llm_finetune/finetune.py scripts/finetune.py
touch ~/olmo3-nemo/scripts/__init__.py
```

### 6.3 Pre-Download Model (CRITICAL — Do This First)

All 16 processes trying to download simultaneously will fail with `OSError`. Run this once per cluster:

**On B200 cluster:**

```bash
srun -N1 -n1 --gpus=1 -p a4high --time=02:00:00 \
  --container-image="nvcr.io#nvidia/nemo-automodel:25.11.00" \
  --container-writable \
  --container-mounts="$HOME/olmo3-nemo:$HOME/olmo3-nemo" \
  bash -c '
  export HF_HOME='$HOME'/olmo3-nemo/.hf_cache
  python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print(\"Downloading tokenizer...\")
AutoTokenizer.from_pretrained(\"allenai/Olmo-3-1125-32B\")
print(\"Tokenizer done. Downloading model...\")
AutoModelForCausalLM.from_pretrained(\"allenai/Olmo-3-1125-32B\", torch_dtype=\"auto\")
print(\"DOWNLOAD COMPLETE\")
"
  '
```

**On H100 cluster:** Same command but change `-p a4high` to `-p a3mega`.

Verify download is complete:

```bash
ls ~/olmo3-nemo/.hf_cache/hub/models--allenai--Olmo-3-1125-32B/blobs/*.incomplete 2>/dev/null \
  && echo "STILL INCOMPLETE" || echo "ALL COMPLETE"

du -sh ~/olmo3-nemo/.hf_cache/hub/models--allenai--Olmo-3-1125-32B/
# Expected: ~61G and "ALL COMPLETE"
```

### 6.4 Deploy Configuration Files

#### Dataset Adapter (`scripts/malware_dataset.py`)

> **Do NOT write this using shell heredoc.** Use a Python writer or copy from the repo.

```bash
cp ~/B200_H100_FineTuning_Benchmark/b200_finetuning/scripts/malware_dataset.py \
   ~/olmo3-nemo/scripts/malware_dataset.py
```

If copying from the repo is not possible, create it with a Python writer:

```bash
cat > /tmp/write_dataset.py << 'PYEOF'
import os
home = os.environ["HOME"]

code = '''"""Custom dataset for malware analysis JSONL format."""
import json
from torch.utils.data import Dataset


class MalwareAnalysisDataset(Dataset):
    def __init__(self, path_or_dataset, tokenizer,
                 split="train", max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(path_or_dataset, "r") as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record = self.samples[idx]
        conversations = record["conversations"]
        system_msg = conversations[0]["content"]
        user_msg = conversations[1]["content"]
        assistant_msg = conversations[2]["content"]

        if "<binary>" in user_msg and "</binary>" in user_msg:
            start = user_msg.index("<binary>") + len("<binary>")
            end = user_msg.index("</binary>")
            binary_hex = user_msg[start:end]
            question = user_msg[end + len("</binary>"):]
            reserved_chars = (self.max_length - 200) * 2
            if len(binary_hex) > reserved_chars:
                binary_hex = binary_hex[:reserved_chars]
            user_msg = "<binary>" + binary_hex + "</binary>" + question

        full_text = (
            "<|system|>\\n" + system_msg + "\\n"
            + "<|user|>\\n" + user_msg + "\\n"
            + "<|assistant|>\\n" + assistant_msg + "<|endoftext|>"
        )

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = encoding["input_ids"]
        labels = list(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
'''

with open(os.path.join(home, "olmo3-nemo/scripts/malware_dataset.py"), "w") as f:
    f.write(code)
print("Dataset adapter written successfully")
PYEOF
python3 /tmp/write_dataset.py
```

#### Training Entry Point (`scripts/finetune.py`)

Already copied in step 6.2. Verify it exists:

```bash
head -5 ~/olmo3-nemo/scripts/finetune.py
# Should show: from __future__ import annotations
```

If missing, create it:

```bash
cat > ~/olmo3-nemo/scripts/finetune.py << 'EOF'
from __future__ import annotations

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


def main(default_config_path="examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag.yaml"):
    cfg = parse_args_and_load_config(default_config_path)
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
EOF
```

#### YAML Config — B200 (`configs/olmo3_32b_b200.yaml`)

Use a Python writer to generate the config with correct absolute paths:

```bash
cat > /tmp/write_b200_yaml.py << 'PYEOF'
import os
home = os.environ["HOME"]

yaml_content = """# OLMo-3 32B SFT on B200 (2 nodes x 8 GPUs = 16 GPUs)
# NeMo AutoModel - nemo-automodel:25.11.00 container

step_scheduler:
  global_batch_size: 16
  local_batch_size: 1
  ckpt_every_steps: 50
  val_every_steps: 9999
  num_epochs: 1

dist_env:
  backend: nccl
  timeout_minutes: 30

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 42
  ranked: true

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: allenai/Olmo-3-1125-32B

checkpoint:
  enabled: true
  checkpoint_dir: {home}/olmo3-nemo/output-b200/checkpoints
  model_save_format: safetensors
  save_consolidated: false

distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: null
  tp_size: 1
  cp_size: 1
  sequence_parallel: false
  activation_checkpointing: true

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

dataset:
  _target_: scripts.malware_dataset.MalwareAnalysisDataset
  path_or_dataset: {home}/olmo3-nemo/data/train.jsonl
  split: train
  max_length: 2048

packed_sequence:
  packed_sequence_size: 0

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  shuffle: true

optimizer:
  _target_: torch.optim.AdamW
  lr: 2.0e-5
  betas: [0.9, 0.95]
  weight_decay: 0.1
  eps: 1.0e-8
""".format(home=home)

with open(os.path.join(home, "olmo3-nemo/configs/olmo3_32b_b200.yaml"), "w") as f:
    f.write(yaml_content)
print("B200 YAML written successfully")
PYEOF
python3 /tmp/write_b200_yaml.py
```

#### YAML Config — H100 (`configs/olmo3_32b_h100.yaml`)

Identical to B200 except the checkpoint output path:

```bash
cat > /tmp/write_h100_yaml.py << 'PYEOF'
import os
home = os.environ["HOME"]

yaml_content = """# OLMo-3 32B SFT on H100 (2 nodes x 8 GPUs = 16 GPUs)
# NeMo AutoModel - nemo-automodel:25.11.00 container

step_scheduler:
  global_batch_size: 16
  local_batch_size: 1
  ckpt_every_steps: 50
  val_every_steps: 9999
  num_epochs: 1

dist_env:
  backend: nccl
  timeout_minutes: 30

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 42
  ranked: true

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: allenai/Olmo-3-1125-32B

checkpoint:
  enabled: true
  checkpoint_dir: {home}/olmo3-nemo/output-h100/checkpoints
  model_save_format: safetensors
  save_consolidated: false

distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: null
  tp_size: 1
  cp_size: 1
  sequence_parallel: false
  activation_checkpointing: true

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

dataset:
  _target_: scripts.malware_dataset.MalwareAnalysisDataset
  path_or_dataset: {home}/olmo3-nemo/data/train.jsonl
  split: train
  max_length: 2048

packed_sequence:
  packed_sequence_size: 0

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  shuffle: true

optimizer:
  _target_: torch.optim.AdamW
  lr: 2.0e-5
  betas: [0.9, 0.95]
  weight_decay: 0.1
  eps: 1.0e-8
""".format(home=home)

with open(os.path.join(home, "olmo3-nemo/configs/olmo3_32b_h100.yaml"), "w") as f:
    f.write(yaml_content)
print("H100 YAML written successfully")
PYEOF
python3 /tmp/write_h100_yaml.py
```

### 6.5 Deploy Benchmark Submit Scripts

#### B200 Benchmark Script (`submit_b200_benchmark.sh`)

```bash
cat > /tmp/write_b200_bench.py << 'PYEOF'
import os
home = os.environ["HOME"]

script = """#!/bin/bash
#SBATCH --job-name=olmo3-bench-b200
#SBATCH --partition=a4high
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output={home}/logs/olmo3-bench-b200-%j.out
#SBATCH --error={home}/logs/olmo3-bench-b200-%j.err

set -euo pipefail
WORK_DIR="{home}/olmo3-nemo"
BENCH_DIR="{home}/olmo3-nemo/benchmark-b200"
CONTAINER="nvcr.io#nvidia/nemo-automodel:25.11.00"

HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
HEAD_ADDR=$(srun -N1 -n1 -w $HEAD_NODE hostname -I | awk '{{print $1}}')

echo "=== B200 BENCHMARK START ==="
echo "HEAD_NODE: $HEAD_NODE"
echo "HEAD_ADDR: $HEAD_ADDR"
echo "NODES: $SLURM_JOB_NUM_NODES"
date -u
mkdir -p $BENCH_DIR

srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node=1 \\
  --gpus-per-node=8 \\
  --container-image=$CONTAINER \\
  --container-writable \\
  --container-mounts="$WORK_DIR:$WORK_DIR" \\
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

  nsys profile \\
    --trace=cuda,nvtx,osrt,cudnn,cublas \\
    --cuda-memory-usage=true \\
    --delay=30 \\
    --duration=300 \\
    --output='$BENCH_DIR'/nsys_b200_rank$SLURM_PROCID \\
    --force-overwrite=true \\
    torchrun \\
      --nnodes=$SLURM_JOB_NUM_NODES \\
      --nproc-per-node=8 \\
      --node-rank=$SLURM_PROCID \\
      --master-addr='$HEAD_ADDR' \\
      --master-port=29500 \\
      '$WORK_DIR'/scripts/finetune.py \\
      --config '$WORK_DIR'/configs/olmo3_32b_b200.yaml

  date -u > '$BENCH_DIR'/end_time.txt
  kill $SMI_PID 2>/dev/null || true
  '

echo "=== B200 BENCHMARK END ==="
date -u
echo "BENCHMARK COMPLETE"
""".format(home=home)

with open(os.path.join(home, "olmo3-nemo/submit_b200_benchmark.sh"), "w") as f:
    f.write(script)
os.chmod(os.path.join(home, "olmo3-nemo/submit_b200_benchmark.sh"), 0o755)
print("B200 benchmark script written successfully")
PYEOF
python3 /tmp/write_b200_bench.py
```

#### H100 NeMo Benchmark Script (`submit_h100_benchmark.sh`)

```bash
cat > /tmp/write_h100_bench.py << 'PYEOF'
import os
home = os.environ["HOME"]

script = """#!/bin/bash
#SBATCH --job-name=olmo3-bench-h100
#SBATCH --partition=a3mega
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output={home}/logs/olmo3-bench-h100-%j.out
#SBATCH --error={home}/logs/olmo3-bench-h100-%j.err

set -euo pipefail
WORK_DIR="{home}/olmo3-nemo"
BENCH_DIR="{home}/olmo3-nemo/benchmark-h100"
CONTAINER="nvcr.io#nvidia/nemo-automodel:25.11.00"

HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
HEAD_ADDR=$(srun -N1 -n1 -w $HEAD_NODE hostname -I | awk '{{print $1}}')

echo "=== H100 BENCHMARK START ==="
echo "HEAD_NODE: $HEAD_NODE"
echo "HEAD_ADDR: $HEAD_ADDR"
echo "NODES: $SLURM_JOB_NUM_NODES"
date -u
mkdir -p $BENCH_DIR

srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node=1 \\
  --gpus-per-node=8 \\
  --container-image=$CONTAINER \\
  --container-writable \\
  --container-mounts="$WORK_DIR:$WORK_DIR" \\
  bash -c '
  set -e
  export HF_HOME='$WORK_DIR'/.hf_cache
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export NCCL_DEBUG=WARN
  export NCCL_SOCKET_IFNAME=enp0s12
  export NCCL_IB_DISABLE=0
  export PYTHONPATH='$WORK_DIR':$PYTHONPATH

  mkdir -p '$BENCH_DIR'
  date -u > '$BENCH_DIR'/start_time.txt
  nvidia-smi dmon -s u -d 2 > '$BENCH_DIR'/gpu_util.csv 2>&1 &
  SMI_PID=$!
  nvidia-smi --query-gpu=name,memory.total --format=csv > '$BENCH_DIR'/gpu_info.csv 2>&1

  nsys profile \\
    --trace=cuda,nvtx,osrt,cudnn,cublas \\
    --cuda-memory-usage=true \\
    --delay=30 \\
    --duration=300 \\
    --output='$BENCH_DIR'/nsys_h100_rank$SLURM_PROCID \\
    --force-overwrite=true \\
    torchrun \\
      --nnodes=$SLURM_JOB_NUM_NODES \\
      --nproc-per-node=8 \\
      --node-rank=$SLURM_PROCID \\
      --master-addr='$HEAD_ADDR' \\
      --master-port=29500 \\
      '$WORK_DIR'/scripts/finetune.py \\
      --config '$WORK_DIR'/configs/olmo3_32b_h100.yaml

  date -u > '$BENCH_DIR'/end_time.txt
  kill $SMI_PID 2>/dev/null || true
  '

echo "=== H100 BENCHMARK END ==="
date -u
echo "BENCHMARK COMPLETE"
""".format(home=home)

with open(os.path.join(home, "olmo3-nemo/submit_h100_benchmark.sh"), "w") as f:
    f.write(script)
os.chmod(os.path.join(home, "olmo3-nemo/submit_h100_benchmark.sh"), 0o755)
print("H100 NeMo benchmark script written successfully")
PYEOF
python3 /tmp/write_h100_bench.py
```

### 6.6 Run the NeMo Benchmarks

#### Pre-Flight Checklist

```bash
# 1. Model cached?
ls ~/olmo3-nemo/.hf_cache/hub/models--allenai--Olmo-3-1125-32B/blobs/*.incomplete 2>/dev/null \
  && echo "INCOMPLETE" || echo "OK"

# 2. No corrupt checkpoints?
ls ~/olmo3-nemo/output-b200/checkpoints/ 2>/dev/null | grep -v "^$" \
  && echo "WARNING: checkpoints exist — clean them" || echo "OK: clean"

# 3. YAML paths correct?
grep -E "checkpoint_dir|path_or_dataset" ~/olmo3-nemo/configs/olmo3_32b_b200.yaml

# 4. Dataset adapter syntax OK?
python3 -c "import ast; ast.parse(open('$HOME/olmo3-nemo/scripts/malware_dataset.py').read()); print('OK')"
```

#### Clean Slate (Before Each Run)

```bash
# For B200:
rm -rf ~/olmo3-nemo/output-b200/checkpoints/
mkdir -p ~/olmo3-nemo/output-b200/checkpoints/
rm -rf ~/olmo3-nemo/benchmark-b200/
mkdir -p ~/olmo3-nemo/benchmark-b200/

# For H100:
rm -rf ~/olmo3-nemo/output-h100/checkpoints/
mkdir -p ~/olmo3-nemo/output-h100/checkpoints/
rm -rf ~/olmo3-nemo/benchmark-h100/
mkdir -p ~/olmo3-nemo/benchmark-h100/
```

#### Submit

```bash
# B200 cluster:
sbatch ~/olmo3-nemo/submit_b200_benchmark.sh

# H100 cluster:
sbatch ~/olmo3-nemo/submit_h100_benchmark.sh
```

#### Monitor

```bash
squeue --user=$(whoami)

# Watch error log (most important — all diagnostics go here):
tail -f $(ls -t ~/logs/olmo3-bench-*.err | head -1)

# Quick check for key events:
grep -iE 'loss|step|error|traceback|COMPLETE' \
  $(ls -t ~/logs/olmo3-bench-*.err | head -1) | tail -20
```

#### Verify Completion

```bash
ls -la ~/olmo3-nemo/output-b200/checkpoints/    # or output-h100
ls -la ~/olmo3-nemo/benchmark-b200/              # or benchmark-h100
```

#### Upload Results to GCS

```bash
gsutil -m cp -r ~/olmo3-nemo/benchmark-b200 gs://<YOUR_RESULTS_BUCKET>/b200/
gsutil -m cp -r ~/olmo3-nemo/benchmark-h100 gs://<YOUR_RESULTS_BUCKET>/h100/
```

---

## 7. Fine-Tuning with Native PyTorch FSDP + TCPXO (H100)

This section reproduces the **H100 FSDP+TCPXO** benchmark (Job 150 in the technical report). It uses native PyTorch FSDP — no NeMo, no Accelerate, no DeepSpeed — with TCPXO high-bandwidth networking enabled.

### 7.1 Why This Approach Exists

After discovering that NeMo AutoModel on the H100 cluster did **not** use TCPXO networking (due to container/NCCL incompatibilities), a native PyTorch FSDP approach was developed to leverage the TCPXO interconnect. This produced a **1.7× wall-time speedup** over the NeMo H100 run on the same hardware (7.7s/step vs 13.0s/step).

### 7.2 TCPXO Background

The H100 cluster (`a3-megagpu-8g`) has TCPXO high-bandwidth networking hardware installed at `/var/lib/tcpxo/lib64/`. The TCPXO directory ships **NCCL 2.28.7** alongside its FasTrak plugin (shim v7), but the container's PyTorch is built against a different NCCL version.

**The solution is `LD_PRELOAD`:** forcing the TCPXO NCCL to load first, overriding the container's version. This approach was validated at **330 GB/s** using NCCL all-reduce tests.

```
export LD_PRELOAD=/var/lib/tcpxo/lib64/libnccl.so.2.28.7
```

> **Do NOT use `LD_LIBRARY_PATH` alone** — it causes PyTorch to load the wrong NCCL version, crashing with ABI mismatches. **Do NOT symlink only the plugin files** — the plugin's internal API expects NCCL 2.28.7 function signatures.

### 7.3 Software Stack

| Component | Version |
|---|---|
| Container | `nvidia/pytorch:24.04-py3` |
| PyTorch | 2.3.0a0+6ddf5cf85e.nv24.04 |
| CUDA | 12.4 |
| NCCL (runtime, via LD_PRELOAD) | 2.28.7 |
| transformers | >=4.57.0,<5.0.0 (pip installed at runtime) |

> **⚠️ CRITICAL:** Do NOT install PyTorch via pip. The container's PyTorch is custom-built by NVIDIA. `transformers` must be pinned to `>=4.57.0,<5.0.0`. Version 5.x requires PyTorch ≥2.4 which is incompatible with this container.

### 7.4 Prerequisites

Ensure the following are in place on the **H100 cluster login node**:

```bash
# Create directories
mkdir -p ~/olmo3-nemo/{scripts,data,benchmark-h100-fsdp} ~/logs
mkdir -p ~/olmo3-nemo/benchmark-h100-fsdp/output

# Training data (same file used by NeMo runs)
ls ~/olmo3-nemo/data/train.jsonl  # must exist

# HuggingFace model cache (from the NeMo pre-download step)
ls ~/olmo3-nemo/.hf_cache/hub/models--allenai--Olmo-3-1125-32B/
```

### 7.5 Pre-Pull the Container Image

The FSDP script uses a `.sqsh` container image for faster startup. Pull it once:

```bash
srun -N1 -n1 -p a3mega --time=00:30:00 \
  --container-image="nvcr.io#nvidia/pytorch:24.04-py3" \
  --container-writable \
  bash -c 'echo "Container pulled successfully"'

# The enroot squashfs image will be cached. Find it:
ls /var/tmp/enroot-data/user-*/nvidia+pytorch+24.04-py3.sqsh 2>/dev/null \
  || echo "Check enroot cache location"

# Copy to home directory for the submit script
cp /var/tmp/enroot-data/user-*/nvidia+pytorch+24.04-py3.sqsh \
   ~/olmo3-nemo/nvidia+pytorch+24.04-py3.sqsh 2>/dev/null \
  || echo "NOTE: If sqsh not found, the submit script will use --container-image directly"
```

> If you cannot locate the `.sqsh` file, edit the submit script (Step 7.7) to use `--container-image="nvcr.io#nvidia/pytorch:24.04-py3"` instead of `--container-image=$CONTAINER_IMAGE`.

### 7.6 Training Script (`train_fsdp_v3.py`)

```bash
cat > ~/olmo3-nemo/scripts/train_fsdp_v3.py << 'TRAINEOF'
"""
OLMo-3 32B Fine-Tuning — Native PyTorch FSDP + TCPXO (v3)

No DeepSpeed. No Accelerate. No NeMo.
Just PyTorch FSDP + transformers for model loading.
"""

import argparse
import functools
import json
import logging
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def log_memory(tag, device=0):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        logger.info(f"[MEM {tag}] allocated={alloc:.1f}GB reserved={reserved:.1f}GB")


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_decoder_layer_class(model):
    """Dynamically find the transformer decoder layer class."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is not None and len(layers) > 0:
        cls = type(layers[0])
        if is_main():
            logger.info(f"Detected decoder layer class: {cls.__name__}")
        return cls
    raise RuntimeError("Could not detect decoder layer class.")


class MalwareAnalysisDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))
        if is_main():
            logger.info(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record = self.samples[idx]
        conversations = record["conversations"]
        system_msg = conversations[0]["content"]
        user_msg = conversations[1]["content"]
        assistant_msg = conversations[2]["content"]

        if "<binary>" in user_msg and "</binary>" in user_msg:
            start = user_msg.index("<binary>") + len("<binary>")
            end = user_msg.index("</binary>")
            binary_hex = user_msg[start:end]
            question = user_msg[end + len("</binary>"):]
            prompt_overhead = len(self.tokenizer.encode(
                f"System: {system_msg}\nUser: <binary></binary>{question}\nAssistant: {assistant_msg}",
                add_special_tokens=False
            ))
            hex_budget = max(200, self.max_length - prompt_overhead - 50)
            hex_tokens = self.tokenizer.encode(binary_hex, add_special_tokens=False)
            if len(hex_tokens) > hex_budget:
                truncated = self.tokenizer.decode(hex_tokens[:hex_budget])
                user_msg = f"<binary>{truncated}</binary>{question}"

        text = f"System: {system_msg}\nUser: {user_msg}\nAssistant: {assistant_msg}"
        encoded = self.tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="allenai/Olmo-3-1125-32B")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--cpu_offload", action="store_true")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        logger.info("=" * 60)
        logger.info("OLMo-3 32B Fine-Tuning — Native PyTorch FSDP v3")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Data: {args.data_path}")
        logger.info(f"World size: {world_size}")
        logger.info(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
        logger.info(f"GPUs per node: {torch.cuda.device_count()}")
        logger.info("=" * 60)

    log_memory("before-model-load", local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        use_cache=False, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if global_rank == 0:
        param_count = sum(p.numel() for p in model.parameters()) / 1e9
        logger.info(f"Model loaded: {param_count:.1f}B parameters")

    decoder_layer_cls = get_decoder_layer_class(model)
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={decoder_layer_cls},
    )

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16,
    )

    cpu_offload = CPUOffload(offload_params=True) if args.cpu_offload else None

    model = FSDP(
        model, auto_wrap_policy=auto_wrap_policy, mixed_precision=bf16_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD, cpu_offload=cpu_offload,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE, device_id=local_rank,
        limit_all_gathers=True, use_orig_params=True,
    )

    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda submodule: isinstance(submodule, decoder_layer_cls),
    )

    dataset = MalwareAnalysisDataset(args.data_path, tokenizer, args.max_length)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps

    model.train()
    global_step = 0
    total_tokens = 0
    step_times = []
    accumulated_loss = 0.0
    accumulated = 0

    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(dataloader):
            step_start = time.time()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            accumulated_loss += loss.item()
            accumulated += 1
            tokens_in_batch = attention_mask.sum().item()
            total_tokens += tokens_in_batch * world_size

            if accumulated >= args.gradient_accumulation_steps:
                model.clip_grad_norm_(1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                step_time = time.time() - step_start
                step_times.append(step_time * args.gradient_accumulation_steps)

                if global_rank == 0 and global_step % args.log_every == 0:
                    tps = (tokens_in_batch * world_size * args.gradient_accumulation_steps) / step_times[-1]
                    avg_tps = total_tokens / sum(step_times) if step_times else 0
                    alloc_gb = torch.cuda.memory_allocated(device) / 1e9
                    logger.info(
                        f"Step {global_step}/{total_steps} | Loss: {accumulated_loss:.4f} "
                        f"| TPS: {tps:.0f} (avg: {avg_tps:.0f}) "
                        f"| GPU mem: {alloc_gb:.1f}GB | Step time: {step_times[-1]:.2f}s"
                    )

                accumulated_loss = 0.0
                accumulated = 0
                dist.barrier()

    if global_rank == 0:
        total_time = sum(step_times)
        avg_tps = total_tokens / total_time if total_time > 0 else 0
        logger.info("TRAINING COMPLETE")
        logger.info(f"Average TPS: {avg_tps:.0f}")
        logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated(device) / 1e9:.1f}GB")
        metrics = {
            "total_steps": global_step, "total_tokens": total_tokens,
            "total_time_seconds": total_time, "average_tokens_per_second": avg_tps,
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9,
            "world_size": world_size, "model": args.model_name,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "framework": "PyTorch FSDP (native)",
        }
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
TRAINEOF
```

### 7.7 Submit Script — Basic Run (No Nsight)

```bash
cat > ~/olmo3-nemo/submit_h100_fsdp_v3.sh << 'SBATCHEOF'
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
SBATCHEOF

chmod +x ~/olmo3-nemo/submit_h100_fsdp_v3.sh
```

> **Note:** If you do not have the `.sqsh` file, change `CONTAINER_IMAGE` to use the registry path directly: `CONTAINER_IMAGE="nvcr.io#nvidia/pytorch:24.04-py3"`.

### 7.8 Submit Script — With Nsight Profiling + DCGM

```bash
cat > ~/olmo3-nemo/submit_h100_fsdp_v3_nsys.sh << 'SBATCHEOF'
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
SBATCHEOF

chmod +x ~/olmo3-nemo/submit_h100_fsdp_v3_nsys.sh
```

### 7.9 Run the FSDP+TCPXO Benchmark

```bash
# Basic run (metrics only)
sbatch ~/olmo3-nemo/submit_h100_fsdp_v3.sh

# Or full run with Nsight profiling
sbatch ~/olmo3-nemo/submit_h100_fsdp_v3_nsys.sh
```

#### Monitor

```bash
squeue -u $(whoami) --long
tail -f ~/logs/olmo3-fsdp-<JOB_ID>.out

# The real training output is in rank 0 stderr (torchrun redirects):
find ~/olmo3-nemo/benchmark-h100-fsdp/torchrun_logs_0_*/*/0/stderr.log \
  -exec tail -30 {} \;
```

#### What Success Looks Like

In rank 0 `stderr.log`:

```
Detected decoder layer class: Olmo3DecoderLayer
...
[MEM after-fsdp-wrap] allocated=4.0GB reserved=10.1GB
Activation checkpointing: ENABLED (non-reentrant)
Dataset: 500 samples
Step 1/8 | Loss: 5.0100 | TPS: 11897 | GPU mem: 16.6GB
Step 2/8 | Loss: 4.7737 | TPS: 16862 | GPU mem: 16.6GB
...
TRAINING COMPLETE
Average TPS: 15572
Peak GPU memory: 24.9GB
```

#### Collect Results

```bash
cat ~/olmo3-nemo/benchmark-h100-fsdp/output/metrics.json
ls -lh ~/olmo3-nemo/benchmark-h100-fsdp-nsys/nsys/*.nsys-rep  # if nsys run

# Upload to GCS
gsutil -m cp -r ~/olmo3-nemo/benchmark-h100-fsdp gs://<YOUR_RESULTS_BUCKET>/h100-fsdp-tcpxo/
gsutil -m cp -r ~/olmo3-nemo/benchmark-h100-fsdp-nsys gs://<YOUR_RESULTS_BUCKET>/h100-fsdp-tcpxo-nsys/ 2>/dev/null || true
```

---

## 8. NCCL Communication Benchmarks

Run these on **both** clusters to validate intra-node NVLink bandwidth and inter-node communication performance.

### 8.1 H100 Cluster

```bash
srun -N2 -n2 --gpus-per-node=8 -p a3mega --time=00:15:00 \
  --container-image="nvcr.io#nvidia/nemo-automodel:25.11.00" \
  --container-writable \
  --container-mounts="$HOME/olmo3-nemo:$HOME/olmo3-nemo" \
  bash -c '
  cd /tmp
  git clone https://github.com/NVIDIA/nccl-tests.git 2>/dev/null || true
  cd nccl-tests
  make MPI=0 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr 2>/dev/null

  unset NCCL_SOCKET_IFNAME
  export NCCL_IB_DISABLE=0
  export NCCL_NET_GDR_LEVEL=SYS
  export NCCL_P2P_LEVEL=NVL
  export NCCL_SHM_DISABLE=0
  export NCCL_BUFFSIZE=16777216
  export NCCL_NTHREADS=512
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export NCCL_DEBUG=WARN

  echo "=== GPU TOPOLOGY ==="
  nvidia-smi topo -m
  echo ""
  echo "=== NVLINK STATUS ==="
  nvidia-smi nvlink -s
  echo ""

  echo "=== ALL-REDUCE (large messages, 100 iters) ==="
  ./build/all_reduce_perf -b 256M -e 4G -f 2 -g 8 -n 100 -w 10

  echo ""
  echo "=== REDUCE-SCATTER (large messages, 100 iters) ==="
  ./build/reduce_scatter_perf -b 256M -e 4G -f 2 -g 8 -n 100 -w 10

  echo ""
  echo "=== ALL-REDUCE (full range for profile) ==="
  ./build/all_reduce_perf -b 8 -e 4G -f 2 -g 8 -n 50 -w 10
  '
```

### 8.2 B200 Cluster

Same commands but change `-p a3mega` to `-p a4high`:

```bash
srun -N2 -n2 --gpus-per-node=8 -p a4high --time=00:15:00 \
  --container-image="nvcr.io#nvidia/nemo-automodel:25.11.00" \
  --container-writable \
  --container-mounts="$HOME/olmo3-nemo:$HOME/olmo3-nemo" \
  bash -c '
  cd /tmp
  git clone https://github.com/NVIDIA/nccl-tests.git 2>/dev/null || true
  cd nccl-tests
  make MPI=0 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr 2>/dev/null

  unset NCCL_SOCKET_IFNAME
  export NCCL_IB_DISABLE=0
  export NCCL_NET_GDR_LEVEL=SYS
  export NCCL_P2P_LEVEL=NVL
  export NCCL_SHM_DISABLE=0
  export NCCL_BUFFSIZE=16777216
  export NCCL_NTHREADS=512
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export NCCL_DEBUG=WARN

  echo "=== GPU TOPOLOGY ==="
  nvidia-smi topo -m
  echo ""
  echo "=== NVLINK STATUS ==="
  nvidia-smi nvlink -s
  echo ""

  echo "=== ALL-REDUCE (large messages, 100 iters) ==="
  ./build/all_reduce_perf -b 256M -e 4G -f 2 -g 8 -n 100 -w 10

  echo ""
  echo "=== REDUCE-SCATTER (large messages, 100 iters) ==="
  ./build/reduce_scatter_perf -b 256M -e 4G -f 2 -g 8 -n 100 -w 10

  echo ""
  echo "=== ALL-REDUCE (full range for profile) ==="
  ./build/all_reduce_perf -b 8 -e 4G -f 2 -g 8 -n 50 -w 10
  '
```

### 8.3 Expected Results Summary

| Metric (at 4GB message size) | H100 (8×) | B200 (8×) | B200 Advantage |
|---|---|---|---|
| All-Reduce Bus BW | 477.3 GB/s | 833.8 GB/s | 74.7% higher |
| Reduce-Scatter Bus BW | 352.4 GB/s | 660.1 GB/s | 87.3% higher |

---

## 9. Nsight Systems Profiling and Analysis

The benchmark submit scripts (both NeMo and FSDP+TCPXO) capture Nsight Systems traces automatically. To visualize them, you need a graphical environment.

### 9.1 Set Up the Visualization Server

#### Step 1: Create SSH Tunnel (from local terminal or Cloud Shell)

```bash
gcloud compute ssh cluster-mgmt --zone=us-central1-a \
  --project=<YOUR_PROJECT_ID> -- -L 8080:localhost:8080
```

#### Step 2: Install Nsight Systems and Desktop Environment

On the workstation VM (`cluster-mgmt`):

```bash
sudo apt-get update
sudo apt-get install -y nsight-systems-2025.5.2 xfce4 xfce4-goodies tightvncserver novnc python3-websockify
```

#### Step 3: Initialize the VNC Server

```bash
vncserver
# Set a password when prompted. Choose 'n' for view-only.

vncserver -kill :1
echo "startxfce4 &" > ~/.vnc/xstartup
chmod +x ~/.vnc/xstartup
vncserver -geometry 1920x1080 :1
```

#### Step 4: Launch the Web Preview

```bash
/usr/share/novnc/utils/novnc_proxy --vnc localhost:5901 --listen 8080
```

In Cloud Shell, click **Web Preview** (top right) → **Preview on port 8080**. Select `vnc.html`, click **Connect**, and enter your VNC password.

#### Step 5: Open Nsight UI

Right-click the desktop → **Open Terminal Here** → type `nsys-ui`.

Inside Nsight UI, go to **File → Open** and navigate to your trace files. Download them from GCS first if needed:

```bash
# On the workstation VM
gsutil cp gs://<YOUR_RESULTS_BUCKET>/b200/benchmark/nsys_b200_rank0.nsys-rep ~/
gsutil cp gs://<YOUR_RESULTS_BUCKET>/h100/benchmark/nsys_h100_rank0.nsys-rep ~/
```

### 9.2 Key Rows to Analyze

| Row | What It Shows | What to Look For |
|---|---|---|
| **CUDA HW (Kernel)** | GPU hardware execution | Gaps = GPU idle, waiting for data |
| **pt_autograd / python3** | CPU threads driving training | Long blocks with empty GPU = CPU bottleneck |
| **NCCL** | Inter-GPU/node communication | Should overlap with or immediately follow math kernels |
| **pthread_cond_wait** | CPU waiting for GPU | Large tan boxes = high latency |
| **NCCL Progress** | RDMA background threads | Active during NCCL = RDMA/Zero-Copy working |
| **cuBLAS / CUDA API** | Matrix multiplication ops | Compare duration between H100 and B200 |

---

## 10. Benchmark Results and Comparison

### 10.1 Three-Way Summary

| Metric | H100 NeMo | H100 FSDP+TCPXO | B200 NeMo |
|---|---|---|---|
| Framework | NeMo AutoModel (FSDP2) | PyTorch FSDP (native) | NeMo AutoModel (FSDP2) |
| Avg Step Time | 13.0 sec | 7.7 sec | **1.2 sec** |
| TPS Total (mean) | 2,522 | 15,572 | **28,887** |
| Peak GPU Memory | 27.0 GiB | 24.9 GB | 27.0 GiB |
| Training Time (31 steps) | 6 min 31 sec | ~4 min 2 sec | **35 sec** |
| Networking | Standard | TCPXO (330 GB/s) | GPUDirect RDMA (RoCE) |

### 10.2 Speedup Matrix

| Comparison | Wall Time Speedup | Key Takeaway |
|---|---|---|
| B200 NeMo vs H100 NeMo | **11.2×** | Full generational gap |
| B200 NeMo vs H100 FSDP+TCPXO | **6.4×** | True hardware generation gap |
| H100 FSDP+TCPXO vs H100 NeMo | **1.7×** | Framework + TCPXO optimization gain |

### 10.3 Training Configuration (Identical Across All Runs)

| Parameter | Value |
|---|---|
| Model | allenai/Olmo-3-1125-32B (32.2B params) |
| Nodes × GPUs | 2 × 8 = 16 |
| Activation Checkpointing | ON |
| Precision | bf16 |
| global_batch_size | 16 |
| local_batch_size | 1 |
| max_length | 2048 |
| Optimizer | AdamW (lr=2e-5) |
| Training Samples | 500 |
| Total Steps | 31 |

> **TPS Note:** NeMo counts only label tokens (32,784/step). FSDP counts all non-padding tokens. **Wall time (step time) is the most reliable cross-framework comparison.**

### 10.4 B200 Optimization Opportunities

The apple-to-apple runs used identical settings. B200 has significant headroom:

| Optimization | Current | Possible on B200 | Cannot Do on H100 |
|---|---|---|---|
| Disable activation checkpointing | ON (27 GiB) | OFF (~65 GiB) | ❌ Exceeds 80 GiB |
| local_batch_size | 1 | 4–8 | ❌ OOM |
| max_length | 2048 | 4096–8192 | ❌ OOM |
| FP8 training | bf16 | FP8 (E4M3/E5M2) | ✅ Available |
| FP4 training | — | FP4 (Blackwell-exclusive) | ❌ Not supported |

---

## 11. Switching to Reservations

When dedicated reservations activate, update from DWS Flex:

### B200 Cluster

Edit `b200-deployment.yaml` to set:
- `a4h_dws_flex_enabled: false`
- `a4h_reservation_name: <YOUR_B200_RESERVATION_NAME>`

```bash
cd ~/cluster-toolkit
./gcluster deploy -d b200-deployment.yaml \
  examples/machine-learning/a4-highgpu-8g/a4high-slurm-blueprint.yaml \
  --auto-approve
```

### H100 Cluster

Edit `h100-deployment.yaml` to set:
- `a3mega_dws_flex_enabled: false`
- `a3mega_reservation_name: <YOUR_H100_RESERVATION_NAME>`

Edit `a3mega-slurm-blueprint-patched.yaml` to change `enable_placement: false` back to `true` (reservations support placement).

```bash
cd ~/cluster-toolkit
./gcluster deploy -d h100-deployment.yaml \
  a3mega-slurm-blueprint-patched.yaml \
  --only primary,cluster --auto-approve -w
```

---

## 12. Teardown

```bash
cd ~/cluster-toolkit

# H100 cluster
./gcluster destroy olmo3-h100 --auto-approve

# B200 cluster (disable Filestore deletion protection first)
gcloud filestore instances list --project=<YOUR_PROJECT_ID>
gcloud filestore instances update <FILESTORE_INSTANCE_NAME> \
  --no-deletion-protection --zone=us-south1-b --project=<YOUR_PROJECT_ID>
./gcluster destroy olmo3-b200 --auto-approve

# Clean up GCS buckets
gcloud storage buckets delete gs://<YOUR_H100_TF_STATE_BUCKET>
gcloud storage buckets delete gs://<YOUR_B200_TF_STATE_BUCKET>

# Delete workstation VM
gcloud compute instances delete cluster-mgmt --zone=us-central1-a --quiet
```

---

## 13. Troubleshooting

### Infrastructure Issues

| Issue | Fix |
|---|---|
| `dws_flex` field not recognized | Update Cluster Toolkit: `cd ~/cluster-toolkit && git pull && make` |
| `Cannot use DWS Flex with enable_placement` | Patch blueprint: `enable_placement: false` (Step 6B) |
| `a3-highgpu-8g` image build fails (TCPX) | Use `a3-megagpu-8g` instead (same H100 GPUs, uses TCPXO) |
| Filestore `HighScaleSSD` quota error | Switch to `BASIC_SSD` tier, `size_gb: 2560` |
| `servicenetworking` permission denied | Grant `roles/servicenetworking.networksAdmin` from Cloud Shell |
| SSH to login node times out | Create firewall rule: `tcp:22` from `0.0.0.0/0` |
| `gcloud auth login` blocked on VM | Use default compute SA (Context-Aware Access restriction) |
| IAM permission denied from VM | Run IAM commands from Cloud Shell, not the VM |
| DWS Flex nodes not provisioning | Check zone capacity: `gcloud compute machine-types list` |
| Slurm nodes in `down` state | `scontrol update nodename=<NODE> state=resume` |
| Deployment folder exists | Add `-w` flag, or `rm -rf olmo3-h100/cluster/` |
| Re-deploy wastes 35 min on image | Use `--only primary,cluster` to skip image build |

### NeMo Training Issues

| Issue | Fix |
|---|---|
| `OSError: does not appear to have model.safetensors` | Pre-download model on single GPU first (Section 6.3) |
| Model download timed out | Use `--time=02:00:00` for the download job |
| `FileNotFoundError: epoch_0_step_N/optim/.metadata` | Delete `output-*/checkpoints/` before resubmitting |
| DCGM kills training (SIGTERM) | Remove `srun --overlap` for DCGM; use `nvidia-smi dmon` inside container |
| NCCL timeout | Check `NCCL_SOCKET_IFNAME`: B200=`enp0s19`, H100=`enp0s12` |
| `FusedLinearCrossEntropy` error | Use `MaskedCrossEntropy` in YAML (OLMo3 incompatibility) |
| Dataset returns tensors error | Return plain Python lists from `__getitem__`, not tensors |

### FSDP+TCPXO Training Issues

| Issue | Fix |
|---|---|
| `KeyError: 'olmo3'` | Pin `transformers>=4.57.0,<5.0.0` |
| `PyTorch >= 2.4 is required` | `transformers` 5.x installed without cap; pin to `<5.0.0` |
| `Closing env plugin` (crash) | Use `LD_PRELOAD=/var/lib/tcpxo/lib64/libnccl.so.2.28.7` |
| `pip install read-only filesystem` | Add `--container-writable` to srun |
| OOM at 77GB (HF Accelerate) | Use native PyTorch FSDP with `transformer_auto_wrap_policy` |
| `Token indices > 65536` warning | Safe to ignore; model handles truncation |

---

## 14. Appendix: HuggingFace Accelerate FSDP (Reproducing OOM Failures)

> **This section is for reference only.** It reproduces the documented OOM failures caused by bugs in HuggingFace Accelerate 1.5.2. Use NeMo or native FSDP for actual training.

### Why Accelerate FSDP Fails for OLMo-3 32B

Three independent, silently-failing bugs in `accelerate` 1.5.2:

1. **`TRANSFORMER_BASED_WRAP` fails to resolve `Olmo3DecoderLayer`** — entire 32.2B model treated as a single FSDP unit, materializing 77GB during forward pass on 79.2GB GPUs.
2. **`fsdp_activation_checkpointing: true` is silently ignored** — no memory savings from checkpointing.
3. **`fsdp_offload_params: true` breaks during forward pass** — parameters move to CPU at init but are pulled back to GPU during forward.

### Reproduce the OOM

```bash
mkdir -p ~/olmo3-finetune/scripts
mkdir -p ~/olmo3-finetune/simulated_training_data
mkdir -p ~/logs

# Copy training data
cp ~/olmo3-nemo/data/train.jsonl ~/olmo3-finetune/simulated_training_data/train.jsonl
```

#### Deploy the Accelerate FSDP Config

```bash
cat > ~/olmo3-finetune/scripts/accelerate_fsdp_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: "no"
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_activation_checkpointing: true
  fsdp_auto_wrap_policy: SIZE_BASED_WRAP
  fsdp_min_num_params: 100000000
  fsdp_offload_params: true
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_forward_prefetch: true
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
  fsdp_cpu_ram_efficient_loading: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 2
num_processes: 16
rdzv_backend: c10d
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
```

#### Deploy Training and Submit Scripts

```bash
cp ~/B200_H100_FineTuning_Benchmark/h100_finetuning/fsdp_scripts/train.py \
   ~/olmo3-finetune/scripts/train.py
cp ~/B200_H100_FineTuning_Benchmark/h100_finetuning/submit_h100.sh \
   ~/olmo3-finetune/submit_h100.sh
chmod +x ~/olmo3-finetune/submit_h100.sh
```

#### Submit and Observe OOM

```bash
sbatch ~/olmo3-finetune/submit_h100.sh

# Watch for the characteristic 77GB memory jump:
tail -f $(ls -t ~/logs/olmo3-h100-*.err | head -1)

# Key diagnostic: look for the FSDP unit count
grep -E "FSDP units|MEM |OutOfMemory" ~/logs/olmo3-h100-*.err
```

Expected failure signature:
- `[MEM after-fsdp-wrap] allocated=12.xGB` → jumps to `77GB` during training
- `torch.OutOfMemoryError` during backward pass

---

## Project File Structure

```
~/olmo3-nemo/
├── configs/
│   ├── olmo3_32b_h100.yaml              # NeMo config for H100
│   └── olmo3_32b_b200.yaml              # NeMo config for B200
├── scripts/
│   ├── __init__.py
│   ├── finetune.py                       # NeMo training entry point
│   ├── malware_dataset.py                # Dataset adapter (NeMo)
│   └── train_fsdp_v3.py                  # Native FSDP training script (TCPXO)
├── data/
│   └── train.jsonl                       # 500 training samples (~140 MB)
├── output-h100/
│   └── checkpoints/                      # NeMo H100 checkpoints
├── output-b200/
│   └── checkpoints/                      # NeMo B200 checkpoints
├── benchmark-h100/                       # NeMo H100 artifacts (Nsight, GPU util)
├── benchmark-b200/                       # NeMo B200 artifacts
├── benchmark-h100-fsdp/                  # FSDP+TCPXO basic run artifacts
│   └── output/
│       └── metrics.json
├── benchmark-h100-fsdp-nsys/             # FSDP+TCPXO Nsight run artifacts
│   ├── nsys/
│   └── output/
├── submit_h100_benchmark.sh              # NeMo H100 submit script
├── submit_b200_benchmark.sh              # NeMo B200 submit script
├── submit_h100_fsdp_v3.sh                # FSDP+TCPXO basic submit script
├── submit_h100_fsdp_v3_nsys.sh           # FSDP+TCPXO Nsight submit script
├── nvidia+pytorch+24.04-py3.sqsh         # Cached container image (FSDP runs)
├── nemo-automodel-repo/                  # Cloned NeMo AutoModel repo
└── .hf_cache/                            # HuggingFace model cache (~61 GB)
```

---

## License

This repository is provided for benchmarking and educational purposes. The OLMo-3 model is licensed under its respective terms by Allen Institute for AI. NVIDIA software components are subject to NVIDIA's license agreements.

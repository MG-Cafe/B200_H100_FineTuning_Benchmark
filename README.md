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
| NVIDIA H100 GPUs (16) | `<H100_REGION>` | H100 cluster (`a3-megagpu-8g`) |
| NVIDIA B200 GPUs (16) | `<B200_REGION>` | B200 cluster (`a4-highgpu-8g`) |
| `SsdStorageGibPerRegion` | `<H100_REGION>` | Filestore (BASIC_SSD fallback for H100) |
| `HighScaleSSDStorageGibPerRegion` | `<B200_REGION>` | Filestore for B200 (or use BASIC_SSD) |

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
| `<WORKSTATION_VM_NAME>` | Name of your management workstation VM | `cluster-mgmt` |
| `<B200_LOGIN_NODE>` | B200 cluster Slurm login node name | `olmo3b200-slurm-login-001` |
| `<H100_LOGIN_NODE>` | H100 cluster Slurm login node name | `olmo3h100-login-001` |
| `<HOME_DIR>` | Your home directory on the cluster | `/home/sa_12345` (check with `echo $HOME`) |
| `<H100_REGION>` | GCP region for H100 cluster | `us-central1` |
| `<H100_ZONE>` | GCP zone for H100 cluster | `us-central1-a` |
| `<B200_REGION>` | GCP region for B200 cluster | `us-south1` |
| `<B200_ZONE>` | GCP zone for B200 cluster | `us-south1-b` |

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

# Create workstation VM (you can change the name)
gcloud compute instances create <WORKSTATION_VM_NAME> \
  --project=<YOUR_PROJECT_ID> \
  --zone=<H100_ZONE> \
  --machine-type=e2-standard-8 \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --scopes=cloud-platform \
  --metadata=enable-oslogin=TRUE

# SSH into it
gcloud compute ssh <WORKSTATION_VM_NAME> --zone=<H100_ZONE>
```

### Step 2: Install Dependencies on Workstation

Run inside the `<WORKSTATION_VM_NAME>` VM:

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
  --project=<YOUR_PROJECT_ID> --location=<H100_REGION> \
  --uniform-bucket-level-access

gcloud storage buckets create gs://<YOUR_B200_TF_STATE_BUCKET> \
  --project=<YOUR_PROJECT_ID> --location=<B200_REGION> \
  --uniform-bucket-level-access
```

---

## 3. Deploy GPU Clusters

### Cluster 1: B200 (a4-highgpu-8g) — <B200_ZONE>

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
  region: <B200_REGION>
  zone: <B200_ZONE>
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

Run from the workstation VM (`<WORKSTATION_VM_NAME>`):

```bash
# Find login node
gcloud compute instances list --project=<YOUR_PROJECT_ID> \
  --filter="name~olmo3.*b200.*login" --format="table(name,zone,status)"

# Create SSH firewall rule
NETWORK=$(gcloud compute instances describe <B200_LOGIN_NODE> \
  --zone=<B200_ZONE> --format='get(networkInterfaces[0].network)' \
  --project=<YOUR_PROJECT_ID> | awk -F/ '{print $NF}')

gcloud compute firewall-rules create allow-ssh-b200 \
  --network=$NETWORK --allow=tcp:22 \
  --source-ranges=0.0.0.0/0 --project=<YOUR_PROJECT_ID>

# SSH to login node
gcloud compute ssh <B200_LOGIN_NODE> \
  --zone=<B200_ZONE> --project=<YOUR_PROJECT_ID>
```

On the B200 login node:

```bash
# Verify GPUs (should show 8× B200)
srun -N 1 --gpus-per-node=8 --exclusive nvidia-smi
```

---

### Cluster 2: H100 (a3-megagpu-8g) — <H100_ZONE>

`a3-megagpu-8g` requires blueprint patches for DWS Flex compatibility and Filestore quota workaround.


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
  region: <H100_REGION>
  zone: <H100_ZONE>
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

Run from workstation VM (`<WORKSTATION_VM_NAME>`):

```bash
# Find login node
gcloud compute instances list --project=<YOUR_PROJECT_ID> \
  --filter="name~olmo3h100.*login" --format="table(name,zone,status)"

# Create SSH firewall rule
NETWORK=$(gcloud compute instances describe <H100_LOGIN_NODE> \
  --zone=<H100_ZONE> --format='get(networkInterfaces[0].network)' \
  --project=<YOUR_PROJECT_ID> | awk -F/ '{print $NF}')

gcloud compute firewall-rules create allow-ssh-h100 \
  --network=$NETWORK --allow=tcp:22 \
  --source-ranges=0.0.0.0/0 --project=<YOUR_PROJECT_ID>

# SSH to login node
gcloud compute ssh <H100_LOGIN_NODE> \
  --zone=<H100_ZONE> --project=<YOUR_PROJECT_ID>
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

> The source code is available at [`b200_finetuning/scripts/malware_dataset.py`](b200_finetuning/scripts/malware_dataset.py) in this repository.

#### Training Entry Point (`scripts/finetune.py`)

Already copied in step 6.2. Verify it exists:

```bash
head -5 ~/olmo3-nemo/scripts/finetune.py
# Should show: from __future__ import annotations
```

If missing, copy from this repository:

```bash
cp ~/B200_H100_FineTuning_Benchmark/b200_finetuning/scripts/finetune.py ~/olmo3-nemo/scripts/finetune.py
```

> The source code is available at [`b200_finetuning/scripts/finetune.py`](b200_finetuning/scripts/finetune.py) in this repository.

#### YAML Config — B200 (`configs/olmo3_32b_b200.yaml`)

Copy from this repository and replace `<HOME_DIR>` with your actual home directory:

```bash
cp ~/B200_H100_FineTuning_Benchmark/b200_finetuning/configs/olmo3_32b_b200.yaml \
   ~/olmo3-nemo/configs/olmo3_32b_b200.yaml

# Replace placeholder with your actual home directory
sed -i "s|<HOME_DIR>|$HOME|g" ~/olmo3-nemo/configs/olmo3_32b_b200.yaml
```

> The template is available at [`b200_finetuning/configs/olmo3_32b_b200.yaml`](b200_finetuning/configs/olmo3_32b_b200.yaml) in this repository.

#### YAML Config — H100 (`configs/olmo3_32b_h100.yaml`)

Identical to B200 except the checkpoint output path. Copy from this repository:

```bash
cp ~/B200_H100_FineTuning_Benchmark/b200_finetuning/configs/olmo3_32b_h100.yaml \
   ~/olmo3-nemo/configs/olmo3_32b_h100.yaml

# Replace placeholder with your actual home directory
sed -i "s|<HOME_DIR>|$HOME|g" ~/olmo3-nemo/configs/olmo3_32b_h100.yaml
```

> The template is available at [`b200_finetuning/configs/olmo3_32b_h100.yaml`](b200_finetuning/configs/olmo3_32b_h100.yaml) in this repository.

### 6.5 Deploy Benchmark Submit Scripts

#### B200 Benchmark Script (`submit_b200_benchmark.sh`)

Copy from this repository:

```bash
cp ~/B200_H100_FineTuning_Benchmark/b200_finetuning/submit_b200_benchmark.sh \
   ~/olmo3-nemo/submit_b200_benchmark.sh
chmod +x ~/olmo3-nemo/submit_b200_benchmark.sh
```

> The script is available at [`b200_finetuning/submit_b200_benchmark.sh`](b200_finetuning/submit_b200_benchmark.sh) in this repository.

#### H100 NeMo Benchmark Script (`submit_h100_benchmark.sh`)

Copy from this repository:

```bash
cp ~/B200_H100_FineTuning_Benchmark/h100_finetuning/submit_h100_benchmark.sh \
   ~/olmo3-nemo/submit_h100_benchmark.sh
chmod +x ~/olmo3-nemo/submit_h100_benchmark.sh
```

> The script is available at [`h100_finetuning/submit_h100_benchmark.sh`](h100_finetuning/submit_h100_benchmark.sh) in this repository.

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

Copy from this repository:

```bash
cp ~/B200_H100_FineTuning_Benchmark/h100_finetuning/fsdp_scripts/train_fsdp_v3.py \
   ~/olmo3-nemo/scripts/train_fsdp_v3.py
```

> The full training script is available at [`h100_finetuning/fsdp_scripts/train_fsdp_v3.py`](h100_finetuning/fsdp_scripts/train_fsdp_v3.py) in this repository. It implements native PyTorch FSDP with `transformer_auto_wrap_policy`, activation checkpointing, and `DistributedSampler`.

### 7.7 Submit Script — Basic Run (No Nsight)

Copy from this repository:

```bash
cp ~/B200_H100_FineTuning_Benchmark/h100_finetuning/fsdp_scripts/submit_h100_fsdp_v3.sh \
   ~/olmo3-nemo/submit_h100_fsdp_v3.sh
chmod +x ~/olmo3-nemo/submit_h100_fsdp_v3.sh
```

> The script is available at [`h100_finetuning/fsdp_scripts/submit_h100_fsdp_v3.sh`](h100_finetuning/fsdp_scripts/submit_h100_fsdp_v3.sh) in this repository.

> **Note:** If you do not have the `.sqsh` file, change `CONTAINER_IMAGE` to use the registry path directly: `CONTAINER_IMAGE="nvcr.io#nvidia/pytorch:24.04-py3"`.

### 7.8 Submit Script — With Nsight Profiling + DCGM

Copy from this repository:

```bash
cp ~/B200_H100_FineTuning_Benchmark/h100_finetuning/fsdp_scripts/submit_h100_fsdp_v3_nsys.sh \
   ~/olmo3-nemo/submit_h100_fsdp_v3_nsys.sh
chmod +x ~/olmo3-nemo/submit_h100_fsdp_v3_nsys.sh
```

> The script is available at [`h100_finetuning/fsdp_scripts/submit_h100_fsdp_v3_nsys.sh`](h100_finetuning/fsdp_scripts/submit_h100_fsdp_v3_nsys.sh) in this repository.

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
gcloud compute ssh <WORKSTATION_VM_NAME> --zone=<H100_ZONE> \
  --project=<YOUR_PROJECT_ID> -- -L 8080:localhost:8080
```

#### Step 2: Install Nsight Systems and Desktop Environment

On the workstation VM (`<WORKSTATION_VM_NAME>`):

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
  --no-deletion-protection --zone=<B200_ZONE> --project=<YOUR_PROJECT_ID>
./gcluster destroy olmo3-b200 --auto-approve

# Clean up GCS buckets
gcloud storage buckets delete gs://<YOUR_H100_TF_STATE_BUCKET>
gcloud storage buckets delete gs://<YOUR_B200_TF_STATE_BUCKET>

# Delete workstation VM
gcloud compute instances delete <WORKSTATION_VM_NAME> --zone=<H100_ZONE> --quiet
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
cp ~/B200_H100_FineTuning_Benchmark/h100_finetuning/fsdp_scripts/accelerate_fsdp_config.yaml \
   ~/olmo3-finetune/scripts/accelerate_fsdp_config.yaml
```

> The config is available at [`h100_finetuning/fsdp_scripts/accelerate_fsdp_config.yaml`](h100_finetuning/fsdp_scripts/accelerate_fsdp_config.yaml) in this repository.

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

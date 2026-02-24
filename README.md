# OLMo-3 32B Fine-Tuning Benchmark: H100 vs B200

**GCP Slurm Cluster Setup & Technical Deployment Guide**  
**Project:** gpu-launchpad-playground  

## 📌 Executive Summary

Two GPU clusters were deployed on Google Cloud Platform using **Cluster Toolkit** with Slurm for fine-tuning the **OLMo-3-1125-32B** (32B parameter) model. The primary goal is to benchmark **NVIDIA H100 vs B200 GPU** performance for distributed training.

Both clusters use DWS Flex Start (dynamic workload scheduling) until dedicated reservations activate.

| Cluster | GPU | Machine Type | Zone | Total GPUs | Networking |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **H100** | NVIDIA H100 80GB SXM | `a3-megagpu-8g` | `us-central1-a` | 16 (2x8) | GPUDirect-TCPXO |
| **B200** | NVIDIA B200 192GB | `a4-highgpu-8g` | `us-south1-b` | 16 (2x8) | GPUDirect-RDMA (gIB/RoCE) |

### Key Infrastructure Components
- **H100 Login Node:** `olmo3h100-login-001` (us-central1-a)
- **B200 Login Node:** `olmo3b200-slurm-login-001` (us-south1-b)
- **Workstation VM:** `cluster-mgmt` (e2-standard-8 in us-central1-a)
- **Container Runtime:** `enroot` + `pyxis` (NVIDIA NGC containers)
- **Shared Filesystem:** Filestore NFS mounted at `/home`

---

## 🚀 1. Infrastructure Setup (Full Reproduction Steps)

### Step 1: Create Management Workstation
Run the following from Cloud Shell to create the management workstation:

```bash
gcloud config set project gpu-launchpad-playground

# Enable required APIs
gcloud services enable \
  compute.googleapis.com file.googleapis.com storage.googleapis.com \
  serviceusage.googleapis.com cloudresourcemanager.googleapis.com \
  iam.googleapis.com servicenetworking.googleapis.com

# Create workstation VM
gcloud compute instances create cluster-mgmt \
  --project=gpu-launchpad-playground \
  --zone=us-central1-a \
  --machine-type=e2-standard-8 \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --scopes=cloud-platform \
  --metadata=enable-oslogin=TRUE

# SSH into the workstation VM
gcloud compute ssh cluster-mgmt --zone=us-central1-a
```

### Step 2: Install Dependencies on Workstation
From inside the workstation VM (`cluster-mgmt`):

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y git build-essential unzip wget curl jq

# Install Go
wget https://go.dev/dl/go1.23.6.linux-amd64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.23.6.linux-amd64.tar.gz
rm go1.23.6.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc && source ~/.bashrc

# Install Terraform & Packer
wget -O - https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt-get update && sudo apt-get install -y terraform packer

# Install Cluster Toolkit
cd ~ && git clone https://github.com/GoogleCloudPlatform/cluster-toolkit.git
cd cluster-toolkit && make
echo 'export PATH=$HOME/cluster-toolkit:$PATH' >> ~/.bashrc && source ~/.bashrc

# Set Project (DO NOT run `gcloud auth login` due to Context-Aware Access restrictions)
gcloud config set project gpu-launchpad-playground
```

### Step 3: Setup IAM Permissions
**⚠️ Run this from Cloud Shell, NOT the workstation VM:**
```bash
PROJECT_NUMBER=$(gcloud projects describe gpu-launchpad-playground --format='value(projectNumber)')
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

for ROLE in \
  roles/compute.instanceAdmin.v1 roles/compute.securityAdmin \
  roles/iam.serviceAccountUser roles/monitoring.metricWriter \
  roles/logging.logWriter roles/storage.objectAdmin \
  roles/file.editor roles/servicenetworking.networksAdmin \
  roles/iap.tunnelResourceAccessor; do
  gcloud projects add-iam-policy-binding gpu-launchpad-playground \
    --member="serviceAccount:${SA}" --role="${ROLE}" \
    --condition=None --quiet
done
```

### Step 4: Create Terraform State Buckets
```bash
gcloud storage buckets create gs://olmo3-h100-tf-state --location=us-central1 --uniform-bucket-level-access
gcloud storage buckets create gs://olmo3-b200-tf-state --location=us-south1 --uniform-bucket-level-access
```

---

## 🏗️ 2. Deploy GPU Clusters

Before deployment, ensure you have sufficient quota for `a3-megagpu-8g` (us-central1), `a4-highgpu-8g` (us-south1), and `HighScaleSSDStorageGibPerRegion`.

### A. Deploy B200 Cluster (us-south1-b)
The repo contains `infrastructure/b200-deployment.yaml`.
From the workstation VM:
```bash
cd ~/cluster-toolkit
cp ~/B200_H100_FineTuning_Benchmark/infrastructure/b200-deployment.yaml .

./gcluster deploy -d b200-deployment.yaml \
  examples/machine-learning/a4-highgpu-8g/a4high-slurm-blueprint.yaml \
  --auto-approve
```

### B. Deploy H100 Cluster (us-central1-a)
The H100 `a3-megagpu-8g` requires patched blueprints to disable placement groups for DWS Flex compatibility and manage Filestore tiers. We provide these patched files in `infrastructure/`.

```bash
cd ~/cluster-toolkit
cp ~/B200_H100_FineTuning_Benchmark/infrastructure/h100-deployment.yaml .
cp ~/B200_H100_FineTuning_Benchmark/infrastructure/a3mega-slurm-blueprint-patched.yaml .

./gcluster deploy -d h100-deployment.yaml \
  a3mega-slurm-blueprint-patched.yaml \
  --auto-approve
```

### C. Connect to Clusters & Authenticate NGC
```bash
# Example SSH to B200
gcloud compute ssh olmo3b200-slurm-login-001 --zone=us-south1-b

# Setup NGC Credentials (Run on each cluster login node)
mkdir -p ~/.config/enroot/
cat > ~/.config/enroot/.credentials << 'CREDS'
machine nvcr.io login $oauthtoken password YOUR_NGC_API_KEY
machine authn.nvidia.com login $oauthtoken password YOUR_NGC_API_KEY
CREDS
```

Verify GPU readiness:
```bash
srun -N 1 --gpus-per-node=8 --exclusive nvidia-smi
```

---

## 📊 3. Generating Simulated Training Data

We use a simulated **Multimodal Malware Analysis Dataset** to benchmark without exposing proprietary threat data. This dataset mimics production data perfectly, creating sequences up to the full **65,000 token context window** of OLMo-3 using hex-encoded binaries.

Run the generation script provided in the repository:
```bash
# Clone the repo on the cluster login node
git clone https://github.com/MG-Cafe/B200_H100_FineTuning_Benchmark.git ~/olmo3-finetune
cd ~/olmo3-finetune

# Generate 500 samples
python h100_finetuning/fsdp_scripts/generate_training_data.py
```
This produces `simulated_training_data/train.jsonl` (roughly 140MB), split equally between benign and malicious samples simulating deep PE binary analysis.

---

## 🚂 4. Fine-Tuning & Benchmarking Guide

Once data is generated, you can launch the distributed fine-tuning job across the 16 GPUs. FSDP configuration, training script, and submit scripts are pre-configured in the repository.

### Directory Structure & Required Files
```text
~/olmo3-finetune/
├── h100_finetuning/
│   ├── fsdp_scripts/
│   │   ├── train.py
│   │   └── accelerate_fsdp_config.yaml
│   └── submit_h100.sh
└── simulated_training_data/
    └── train.jsonl
```

### Step 1: Submit the Slurm Job
For H100 cluster:
```bash
cd ~/olmo3-finetune
sbatch h100_finetuning/submit_h100.sh
```

For B200 cluster:
```bash
cd ~/olmo3-finetune
sbatch b200_finetuning/submit_b200_benchmark.sh
```

### Step 2: Monitoring
```bash
squeue --user=$(whoami)
tail -f $(ls -t ~/logs/olmo3-h100-*.err | head -1) # FSDP logs & diagnostics
```

---

## ⚠️ 5. Critical Technical Learnings: OOM & FSDP Issues

During this benchmark deployment, we encountered significant technical challenges with **Accelerate 1.5.2's FSDP** implementation for OLMo-3. Over 9 jobs (Jobs 24–32), we observed extreme Out Of Memory (OOM) failures: allocating 77+ GB of 80 GB available memory during the backward pass regardless of sequence lengths (4096 vs 2048).

### The Root Cause: `TRANSFORMER_BASED_WRAP` Silent Failure
The accelerate FSDP configuration failed to resolve the class name `Olmo3DecoderLayer`. Because of this, **the entire 32.2B parameter model was treated as a single FSDP unit**.
During the forward pass, all 64GB of parameters were all-gathered onto each GPU simultaneously instead of one ~1GB layer at a time.

### Summary of Broken Features in Accelerate 1.5.2
| Feature | YAML Config | Actual Behavior in accelerate 1.5.2 |
| :--- | :--- | :--- |
| **Layer wrapping** | `TRANSFORMER_BASED_WRAP: Olmo3DecoderLayer` | **Silent failure:** entire model = 1 FSDP unit |
| **Activation checkpointing** | `fsdp_activation_checkpointing: true` | Completely ignored |
| **CPU offloading** | `fsdp_offload_params: true` | Works at init, **broken during forward pass** |
| **SIZE_BASED_WRAP** | `fsdp_min_num_params: 100000000` | **Works correctly** (196 units created) |

### Workarounds Implemented
1. Switched `fsdp_auto_wrap_policy` to `SIZE_BASED_WRAP` with `fsdp_min_num_params: 100000000` in `accelerate_fsdp_config.yaml`.
2. Verified per-layer wrapping by counting FSDP units in `train.py`:
   ```python
   from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
   fsdp_units = sum(1 for m in model.modules() if isinstance(m, FSDP))
   # 1-2 = BROKEN (will OOM)
   # 65+ = CORRECT (per-layer wrapping working)
   ```
3. Pinned to PyTorch 2.7+ (NGC `25.02-py3` container) and `transformers<4.52`. 
4. **Recommendation:** Switch to **DeepSpeed ZeRO-3** for working parameter partitioning and offloading if accelerate FSDP continues to struggle.

### Memory Logging 
We enforce memory logging at every stage inside `train.py` to trace explosions:
```
[MEM after-fsdp-wrap]    allocated=12.09GB   <- Shards only (good)
[MEM before-training]    allocated=12.09GB   <- No growth (good)
[During backward pass]   allocated=77.02GB   <- 65GB jump! (bad)
```

---

## 🧽 Teardown
To safely tear down the infrastructure:
```bash
cd ~/cluster-toolkit
./gcluster destroy olmo3-h100 --auto-approve

# Disable filestore deletion protection for B200 first
gcloud filestore instances update <INSTANCE_NAME> --no-deletion-protection --zone=us-south1-b
./gcluster destroy olmo3-b200 --auto-approve
```

## 📜 License
Please refer to the repository license for further usage conditions.
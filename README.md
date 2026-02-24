# OLMo-3 32B Fine-Tuning Benchmark: H100 vs B200

This repository contains the deployment and execution guidelines for fine-tuning the **OLMo-3-1125-32B** model on Google Cloud Platform, benchmarking NVIDIA H100 vs B200 GPU performance using Cluster Toolkit, Slurm, and PyTorch FSDP.

---

## 1. Prerequisites & Quotas

Before deploying, ensure you have the following in your GCP project (`<YOUR_PROJECT_ID>`):
- **NVIDIA H100 GPUs quota** in `us-central1` (for `a3-megagpu-8g` nodes, 16 GPUs total).
- **NVIDIA B200 GPUs quota** in `us-south1` (for `a4-highgpu-8g` nodes, 16 GPUs total).
- **HighScaleSSDStorageGibPerRegion** or **SsdStorageGibPerRegion** quota for Filestore.

---

## 2. Infrastructure Setup

### Step 1: Create Management Workstation (Cloud Shell)

```bash
gcloud config set project <YOUR_PROJECT_ID>

# Enable APIs
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

Run this inside the `cluster-mgmt` VM:

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y git build-essential unzip wget curl jq

# Install Go
wget https://go.dev/dl/go1.23.6.linux-amd64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.23.6.linux-amd64.tar.gz
rm go1.23.6.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc && source ~/.bashrc

# Install Terraform
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

# Auth: Do NOT run gcloud auth login (Context-Aware Access blocks it)
gcloud config set project <YOUR_PROJECT_ID>
gcloud compute instances list --limit=1  # verify SA works
```

### Step 3: IAM Permissions (Run from Cloud Shell)

*Run this from Cloud Shell (not the workstation VM). The compute SA cannot modify IAM policies.*

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

Run this from Cloud Shell:

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

**Step 6A: Create B200 Deployment File**
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

**Step 7A: Deploy B200 Cluster**
```bash
cd ~/cluster-toolkit
./gcluster deploy \
  -d b200-deployment.yaml \
  examples/machine-learning/a4-highgpu-8g/a4high-slurm-blueprint.yaml \
  --auto-approve
```

**Step 8A: Connect & Validate B200**
```bash
# Find login node and create firewall rule
NETWORK=$(gcloud compute instances describe olmo3b200-slurm-login-001 \
  --zone=us-south1-b --format='get(networkInterfaces[0].network)' \
  --project=<YOUR_PROJECT_ID> | awk -F/ '{print $NF}')

gcloud compute firewall-rules create allow-ssh-b200 \
  --network=$NETWORK --allow=tcp:22 \
  --source-ranges=0.0.0.0/0 --project=<YOUR_PROJECT_ID>

# SSH to login node
gcloud compute ssh olmo3b200-slurm-login-001 \
  --zone=us-south1-b --project=<YOUR_PROJECT_ID>

# Verify GPUs on the B200 login node:
srun -N 1 --gpus-per-node=8 --exclusive nvidia-smi
```

### Cluster 2: H100 (a3-megagpu-8g) — us-central1-a

**Step 6B: Patch the Blueprint**
The `a3-megagpu-8g` blueprint requires patches to work with DWS Flex and avoid Filestore quota limits.
```bash
cd ~/cluster-toolkit
cp examples/machine-learning/a3-megagpu-8g/a3mega-slurm-blueprint.yaml \
   a3mega-slurm-blueprint-patched.yaml

# Disable placement for DWS Flex
sed -i '/dws_flex:/i\      enable_placement: false' \
  a3mega-slurm-blueprint-patched.yaml

# Switch Filestore to BASIC_SSD
sed -i 's/filestore_tier: HIGH_SCALE_SSD/filestore_tier: BASIC_SSD/' \
  a3mega-slurm-blueprint-patched.yaml
sed -i 's/size_gb: 10240/size_gb: 2560/' \
  a3mega-slurm-blueprint-patched.yaml
```

**Step 7B: Create H100 Deployment YAML**
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

**Step 8B: Deploy the H100 Cluster**
```bash
cd ~/cluster-toolkit
./gcluster deploy \
  -d h100-deployment.yaml \
  a3mega-slurm-blueprint-patched.yaml \
  --auto-approve
```

**Step 9B: Connect & Validate H100**
```bash
NETWORK=$(gcloud compute instances describe olmo3h100-login-001 \
  --zone=us-central1-a --format='get(networkInterfaces[0].network)' \
  --project=<YOUR_PROJECT_ID> | awk -F/ '{print $NF}')

gcloud compute firewall-rules create allow-ssh-h100 \
  --network=$NETWORK --allow=tcp:22 \
  --source-ranges=0.0.0.0/0 --project=<YOUR_PROJECT_ID>

gcloud compute ssh olmo3h100-login-001 \
  --zone=us-central1-a --project=<YOUR_PROJECT_ID>

# Verify GPUs on the H100 login node:
srun -N 1 --gpus-per-node=8 --exclusive nvidia-smi
```

---

## 4. Post-Deployment Steps (Both Clusters)

**Step 10: Set Up NGC Credentials**
SSH into each cluster's login node and run:
```bash
mkdir -p ~/.config/enroot/
cat > ~/.config/enroot/.credentials << 'CREDS'
machine nvcr.io login $oauthtoken password <YOUR_NGC_API_KEY>
machine authn.nvidia.com login $oauthtoken password <YOUR_NGC_API_KEY>
CREDS
```

**Step 11: Verify OLMo-3 Readiness**
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

**Step 12: Verify DCGM (GPU Monitoring)**
```bash
srun -N 1 --gpus-per-node=8 --exclusive dcgmi discovery -l
```

---

## 5. Switching to Reservations

When dedicated reservations activate, update from DWS Flex:

**B200 Cluster:**
Edit `b200-deployment.yaml` to set `a4h_dws_flex_enabled: false` and `a4h_reservation_name: <YOUR_B200_RESERVATION_NAME>`. Then redeploy:
```bash
cd ~/cluster-toolkit
./gcluster deploy -d b200-deployment.yaml \
  examples/machine-learning/a4-highgpu-8g/a4high-slurm-blueprint.yaml \
  --auto-approve
```

**H100 Cluster:**
Edit `h100-deployment.yaml` to set `a3mega_dws_flex_enabled: false` and `a3mega_reservation_name: <YOUR_H100_RESERVATION_NAME>`. Edit `a3mega-slurm-blueprint-patched.yaml` to change `enable_placement: false` back to `true`.
```bash
cd ~/cluster-toolkit
./gcluster deploy -d h100-deployment.yaml \
  a3mega-slurm-blueprint-patched.yaml \
  --only primary,cluster --auto-approve -w
```

---

## 6. Simulated Training Data Generation

The benchmark requires long-context data to simulate multimodal malware analysis (up to 65K tokens per sample using hex-encoded binaries).

To generate the dataset, clone this repository on your cluster login node and run the generator:

```bash
git clone https://github.com/MG-Cafe/B200_H100_FineTuning_Benchmark.git ~/olmo3-finetune
cd ~/olmo3-finetune
python h100_finetuning/fsdp_scripts/generate_training_data.py
```
This script will produce `simulated_training_data/train.jsonl` containing 500 samples (roughly 140 MB).

---

## 7. Fine-Tuning Execution Guide

The following steps launch the distributed fine-tuning job across 16 GPUs using PyTorch FSDP.

**Step 1: SSH into the Slurm Login Node**
```bash
gcloud compute ssh olmo3h100-login-001 --zone=us-central1-a --project=<YOUR_PROJECT_ID>
```
Verify nodes are available:
```bash
sinfo
```

**Step 2: Create the Directory Structure**
```bash
mkdir -p ~/olmo3-finetune/scripts
mkdir -p ~/olmo3-finetune/simulated_training_data
mkdir -p ~/logs
```

**Step 3: Upload or Link Training Data**
Ensure your `train.jsonl` from the generation step is located at `~/olmo3-finetune/simulated_training_data/train.jsonl`.
```bash
wc -l ~/olmo3-finetune/simulated_training_data/train.jsonl
# Should output 500
```

**Step 4: Deploy the FSDP Config (YAML)**
Run this exact block to configure Accelerate:
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

**Step 5: Deploy the Training & Submit Scripts**
Copy the `train.py` and `submit_h100.sh` (or `submit_b200.sh`) scripts provided in this repository into your working directory:
```bash
cp ~/B200_H100_FineTuning_Benchmark/h100_finetuning/fsdp_scripts/train.py ~/olmo3-finetune/scripts/train.py
cp ~/B200_H100_FineTuning_Benchmark/h100_finetuning/submit_h100.sh ~/olmo3-finetune/submit_h100.sh
chmod +x ~/olmo3-finetune/submit_h100.sh
```

**Step 6: Verify All Files Are Correct**
```bash
# Check YAML has SIZE_BASED_WRAP
grep 'SIZE_BASED' ~/olmo3-finetune/scripts/accelerate_fsdp_config.yaml

# Check train.py has FSDP diagnostic
grep 'count_fsdp_units' ~/olmo3-finetune/scripts/train.py

# Check submit script has correct container
grep 'container-image' ~/olmo3-finetune/submit_h100.sh
```

**Step 7: Clean Up Previous Runs (If Any)**
```bash
scancel --user=$(whoami)
rm -rf ~/olmo3-finetune/output-h100-*/
rm -rf ~/olmo3-finetune/output-b200-*/
```

**Step 8: Submit the Job**
```bash
# For H100 cluster:
sbatch ~/olmo3-finetune/submit_h100.sh

# For B200 cluster (using the respective script):
# sbatch ~/olmo3-finetune/submit_b200.sh
```

**Step 9: Monitor the Job**
```bash
squeue --user=$(whoami)

# Watch the error log (diagnostics):
tail -f $(ls -t ~/logs/olmo3-h100-*.err | head -1)

# Watch the output log:
tail -f $(ls -t ~/logs/olmo3-h100-*.out | head -1)

# Search for key memory diagnostics in a completed job:
grep -E "OutOfMemory|MEM |FSDP units|Step |TRAINING COMPLETE" \
  ~/logs/olmo3-h100-*.err | head -30
```

**Step 10: Collect Results**
After training is complete, results are saved in the output directory.
```bash
ls ~/olmo3-finetune/output-h100-*/training_metrics_h100.csv
cat ~/olmo3-finetune/output-h100-*/training_summary_h100.json
ls ~/olmo3-finetune/output-h100-*/nsight_traces_h100/
ls ~/olmo3-finetune/output-*/dcgm_node_*.csv

# Upload to GCS
gsutil -m cp -r "$(ls -dt ~/olmo3-finetune/output-h100-*/ | head -1)" gs://<YOUR_RESULTS_BUCKET>/h100/
```

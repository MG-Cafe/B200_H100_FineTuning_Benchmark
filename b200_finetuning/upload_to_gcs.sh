#!/bin/bash
# ============================================================================
# OLMo-3 Benchmark Artifacts → GCS Bucket Upload
# ============================================================================
# Uploads all fine-tuning logs, benchmark artifacts, Nsight traces,
# configs, and the JSON benchmark report to a GCS bucket for:
#   1. B200 comparison benchmarking
#   2. Nsight Systems trace analysis on local machine
#   3. Long-term archival
#
# Usage:
#   chmod +x upload_to_gcs.sh
#   ./upload_to_gcs.sh [BUCKET_NAME] [JOB_ID]
#
# Example:
#   ./upload_to_gcs.sh gs://my-olmo3-benchmarks 67
# ============================================================================

set -euo pipefail

# ── Configuration ──
BUCKET="${1:-gs://olmo3-finetune-benchmarks}"
JOB_ID="${2:-}"
WORK_DIR="$HOME/olmo3-nemo"
LOG_DIR="$HOME/logs"
CLUSTER=$(hostname | grep -qi "b200" && echo "b200" || echo "h100")
TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
DEST="${BUCKET}/olmo3-32b/${CLUSTER}/${TIMESTAMP}"

echo "============================================"
echo "  OLMo-3 Benchmark → GCS Upload"
echo "============================================"
echo "  Bucket:    ${BUCKET}"
echo "  Cluster:   ${CLUSTER}"
echo "  Dest:      ${DEST}"
echo "  Work Dir:  ${WORK_DIR}"
echo "  Timestamp: ${TIMESTAMP}"
echo "============================================"

# ── Step 0: Verify gsutil access ──
echo ""
echo "[0/7] Verifying GCS access..."
if ! command -v gsutil &>/dev/null; then
    echo "ERROR: gsutil not found. Install Google Cloud SDK:"
    echo "  curl https://sdk.cloud.google.com | bash"
    exit 1
fi

# Create bucket if it doesn't exist (ignore error if it does)
gsutil ls "${BUCKET}" &>/dev/null || {
    echo "Creating bucket ${BUCKET}..."
    gsutil mb -l us-central1 "${BUCKET}" 2>/dev/null || true
}

# ── Step 1: Generate benchmark report JSON ──
echo ""
echo "[1/7] Generating benchmark report..."
REPORT_SCRIPT="${WORK_DIR}/benchmark_report.py"
if [ -f "${REPORT_SCRIPT}" ]; then
    if [ -n "${JOB_ID}" ]; then
        python3 "${REPORT_SCRIPT}" --job-id "${JOB_ID}" --cluster "${CLUSTER}" \
            --output "${WORK_DIR}/benchmark_report_${CLUSTER}.json" || true
    else
        python3 "${REPORT_SCRIPT}" --cluster "${CLUSTER}" \
            --output "${WORK_DIR}/benchmark_report_${CLUSTER}.json" || true
    fi
else
    echo "  SKIP: benchmark_report.py not found at ${REPORT_SCRIPT}"
fi

# ── Step 2: Upload training logs ──
echo ""
echo "[2/7] Uploading training logs..."
if [ -n "${JOB_ID}" ]; then
    # Upload specific job logs
    for f in ${LOG_DIR}/olmo3-*-${CLUSTER}-${JOB_ID}.{out,err}; do
        [ -f "$f" ] && gsutil cp "$f" "${DEST}/logs/" && echo "  ✓ $(basename $f)"
    done
else
    # Upload all recent olmo3 logs
    for f in $(ls -t ${LOG_DIR}/olmo3-*-${CLUSTER}-*.err 2>/dev/null | head -5); do
        gsutil cp "$f" "${DEST}/logs/" && echo "  ✓ $(basename $f)"
        # Also grab matching .out
        out_f="${f%.err}.out"
        [ -f "$out_f" ] && gsutil cp "$out_f" "${DEST}/logs/" && echo "  ✓ $(basename $out_f)"
    done
fi

# ── Step 3: Upload benchmark artifacts (Nsight traces, GPU util, timing) ──
echo ""
echo "[3/7] Uploading benchmark artifacts..."
BENCH_DIR="${WORK_DIR}/benchmark-${CLUSTER}"
if [ -d "${BENCH_DIR}" ]; then
    # Upload small files first
    for f in ${BENCH_DIR}/start_time.txt ${BENCH_DIR}/end_time.txt \
             ${BENCH_DIR}/gpu_info.csv ${BENCH_DIR}/gpu_util.csv; do
        [ -f "$f" ] && gsutil cp "$f" "${DEST}/benchmark/" && echo "  ✓ $(basename $f)"
    done

    # Upload Nsight traces (large files - use parallel composite upload)
    for f in ${BENCH_DIR}/nsys_*.nsys-rep; do
        if [ -f "$f" ]; then
            SIZE_GB=$(echo "scale=2; $(stat -c%s "$f") / 1073741824" | bc 2>/dev/null || echo "?")
            echo "  Uploading $(basename $f) (${SIZE_GB} GB)..."
            gsutil -o GSUtil:parallel_composite_upload_threshold=150M \
                cp "$f" "${DEST}/benchmark/" && echo "  ✓ $(basename $f)"
        fi
    done
else
    echo "  SKIP: No benchmark directory at ${BENCH_DIR}"
fi

# ── Step 4: Upload DCGM monitoring CSVs ──
echo ""
echo "[4/7] Uploading DCGM CSVs..."
DCGM_COUNT=0
for f in ${WORK_DIR}/dcgm_*.csv; do
    if [ -f "$f" ]; then
        gsutil cp "$f" "${DEST}/dcgm/" && echo "  ✓ $(basename $f)"
        DCGM_COUNT=$((DCGM_COUNT + 1))
    fi
done
[ ${DCGM_COUNT} -eq 0 ] && echo "  SKIP: No DCGM files found"

# ── Step 5: Upload config and scripts (for reproducibility) ──
echo ""
echo "[5/7] Uploading configs and scripts..."
for f in "${WORK_DIR}/configs/olmo3_32b_h100.yaml" \
         "${WORK_DIR}/scripts/malware_dataset.py" \
         "${WORK_DIR}/scripts/finetune.py" \
         "${WORK_DIR}/submit_h100_nemo.sh" \
         "${WORK_DIR}/submit_h100_benchmark.sh" \
         "${WORK_DIR}/benchmark_report.py"; do
    [ -f "$f" ] && gsutil cp "$f" "${DEST}/config/" && echo "  ✓ $(basename $f)"
done

# ── Step 6: Upload benchmark report JSON ──
echo ""
echo "[6/7] Uploading benchmark report JSON..."
REPORT_JSON="${WORK_DIR}/benchmark_report_${CLUSTER}.json"
if [ -f "${REPORT_JSON}" ]; then
    gsutil cp "${REPORT_JSON}" "${DEST}/" && echo "  ✓ $(basename ${REPORT_JSON})"
    # Also copy to a fixed "latest" location for easy comparison
    gsutil cp "${REPORT_JSON}" "${BUCKET}/olmo3-32b/${CLUSTER}/latest_report.json" \
        && echo "  ✓ → latest_report.json (overwritten)"
else
    echo "  SKIP: No report JSON found"
fi

# ── Step 7: Upload checkpoint metadata (not weights - too large) ──
echo ""
echo "[7/7] Uploading checkpoint metadata..."
CKPT_DIR="${WORK_DIR}/output-${CLUSTER}/checkpoints"
if [ -d "${CKPT_DIR}" ]; then
    [ -f "${CKPT_DIR}/training.jsonl" ] && \
        gsutil cp "${CKPT_DIR}/training.jsonl" "${DEST}/checkpoint/" \
        && echo "  ✓ training.jsonl"
    # List checkpoint contents for reference
    ls -la "${CKPT_DIR}/" > /tmp/checkpoint_listing.txt 2>&1
    gsutil cp /tmp/checkpoint_listing.txt "${DEST}/checkpoint/" \
        && echo "  ✓ checkpoint_listing.txt"
else
    echo "  SKIP: No checkpoint directory"
fi

# ── Summary ──
echo ""
echo "============================================"
echo "  UPLOAD COMPLETE"
echo "============================================"
echo "  Destination: ${DEST}"
echo ""
echo "  To list uploaded files:"
echo "    gsutil ls -lh ${DEST}/**"
echo ""
echo "  To download Nsight traces for local analysis:"
echo "    gsutil cp ${DEST}/benchmark/nsys_*.nsys-rep ./"
echo "    nsys-ui nsys_h100_rank0.nsys-rep"
echo ""
echo "  To compare H100 vs B200 reports:"
echo "    gsutil cp ${BUCKET}/olmo3-32b/h100/latest_report.json h100_report.json"
echo "    gsutil cp ${BUCKET}/olmo3-32b/b200/latest_report.json b200_report.json"
echo "    python3 -c \""
echo "      import json"
echo "      h = json.load(open('h100_report.json'))"
echo "      b = json.load(open('b200_report.json'))"
echo "      hm = h['aggregate_metrics']"
echo "      bm = b['aggregate_metrics']"
echo "      print(f'Memory:  H100={hm[\\\"mem_steady_gib\\\"]} vs B200={bm[\\\"mem_steady_gib\\\"]}')"
echo "      print(f'TPS:     H100={hm[\\\"tps_total_mean\\\"]} vs B200={bm[\\\"tps_total_mean\\\"]}')"
echo "      print(f'TPS/GPU: H100={hm[\\\"tps_per_gpu_mean\\\"]} vs B200={bm[\\\"tps_per_gpu_mean\\\"]}')"
echo "    \""
echo "============================================"

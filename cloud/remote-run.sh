#!/usr/bin/env bash
# Runs on the pod, inside the study image. Expects /work/code (study code) and /work/data (inputs).
# Usage: remote-run.sh IMAGE_REF PLATFORM_ID CODE_SHA256 DEVICE TASKS [TIME_LIMIT_S]
set -euo pipefail
image=$1 platform_id=$2 code_sha=$3 device=$4 tasks=$5 time_limit=${6:-}
mkdir -p /work/out /work/hf

{
  echo "uname=$(uname -a)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"
  nproc
  df -h /work | tail -1
  free -g | sed -n 2p
} > /work/out/host.txt 2>&1 || true

limit_args=()
[[ -n $time_limit ]] && limit_args=(--time-limit-s "$time_limit")
cd /work
PYTHONPATH=/work/code/src PLATFORM_ID="$platform_id" IMAGE_REF="$image" CODE_SHA256="$code_sha" \
  python -m evo2size.run --device "$device" --tasks "$tasks" "${limit_args[@]}" 2>&1 | tee /work/out/workloads.log

#!/usr/bin/env bash
# Run one tier's tasks on a Runpod Secure Cloud pod, copy results back, and delete the pod.
# The pod runs the study image itself, so there is no Docker step. Uses Runpod REST API v2.
#
# Required environment:
#   PLATFORM_ID      tier id, e.g. T2
#   RUNPOD_API_KEY   API key (never commit it)
#   IMAGE_REF, BUNDLE, OUT, CODE_SHA256, TASKS, DEVICE
#   and either GPU_TYPE_IDS (JSON array, tried in order) or CPU_FLAVOR and VCPUS
# Optional: MAX_HOURS (default 6), DISK_GB (default 200), TIME_LIMIT_S
set -euo pipefail
source "$(dirname "$0")/../common.sh"

MAX_HOURS=${MAX_HOURS:-6}
DISK_GB=${DISK_GB:-200}
NAME="evo2size-$(lower "$PLATFORM_ID")-$(date -u +%Y%m%d%H%M%S)"
API=https://api.runpod.io/v2
mkdir -p "$OUT"
KEY="$OUT/.ssh-key"

# Call the API; print the body, and fail with the API's own error message on a non-2xx status.
rp() {
  local body status
  body=$(curl -sS -w '\n%{http_code}' -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" "$@")
  status=${body##*$'\n'}; body=${body%$'\n'*}
  if [[ $status != 2* ]]; then log "Runpod API $status: $body"; return 1; fi
  printf '%s' "$body"
}

POD_ID=""
WATCHDOG=""
cleanup() {
  set +e
  [[ -n $WATCHDOG ]] && kill "$WATCHDOG" 2>/dev/null
  if [[ -n $POD_ID ]]; then
    log "Deleting pod $POD_ID"
    rp -X DELETE "$API/pods/$POD_ID" >/dev/null
    mark terminated
  fi
  rm -f "$KEY" "$KEY.pub"
}
trap cleanup EXIT

# Body for one pod request. $1 is a jq expression for the compute part: a gpu or a cpu object.
pod_body() {
  jq -n --arg name "$NAME" --arg image "$IMAGE_REF" --arg key "$(cat "$KEY.pub")" \
    --argjson disk "$DISK_GB" --arg max "$((MAX_HOURS * 3600))" "{
      name: \$name, image: \$image, cloud: \"SECURE\", disk: \$disk, ports: [\"22/tcp\"],
      env: {PUBLIC_KEY: \$key, MAX_SECONDS: \$max},
      cmd: [\"/opt/evo2ops/pod-start.sh\"]} + $1"
}

new_ssh_key
mark launch_requested
if [[ -n ${CPU_FLAVOR:-} ]]; then
  BODY=$(pod_body "{cpu: {id: \"$CPU_FLAVOR\", vcpuCount: ${VCPUS:?VCPUS is required}}}")
  if POD=$(rp -X POST "$API/pods" -d "$BODY"); then POD_ID=$(echo "$POD" | jq -r .id); fi
  compute="$CPU_FLAVOR x$VCPUS"
else
  for gpu in $(echo "${GPU_TYPE_IDS:?GPU_TYPE_IDS or CPU_FLAVOR is required}" | jq -r '.[] | @base64'); do
    gpu=$(echo "$gpu" | base64 --decode)
    BODY=$(pod_body "{gpu: {id: \"$gpu\", count: 1}}")
    if POD=$(rp -X POST "$API/pods" -d "$BODY"); then
      POD_ID=$(echo "$POD" | jq -r .id)
      compute=$gpu
      break
    fi
  done
fi
[[ -n $POD_ID ]] || { log "No pod could be created"; exit 3; }
mark pod_created
log "Pod $POD_ID created ($compute); waiting for the image pull and SSH port"

# Local watchdog: delete the pod after MAX_HOURS even if this script hangs. The image's own
# self-delete backstop calls Runpod's retired v1 endpoint and cannot be relied on.
( sleep $((MAX_HOURS * 3600)); log "Watchdog: MAX_HOURS reached"; rp -X DELETE "$API/pods/$POD_ID" >/dev/null ) &
WATCHDOG=$!

for _ in $(seq 1 120); do
  POD=$(rp "$API/pods/$POD_ID" || true)
  SSH_HOST=$(echo "$POD" | jq -r '.ssh.direct.host // empty' 2>/dev/null)
  SSH_PORT=$(echo "$POD" | jq -r '.ssh.direct.port // empty' 2>/dev/null)
  [[ -n $SSH_HOST && -n $SSH_PORT ]] && break
  sleep 15
done
[[ -n ${SSH_HOST:-} ]] || { log "Pod never exposed SSH"; exit 3; }
mark running
echo "$POD" | jq --argjson disk "$DISK_GB" '{provider: "runpod", instance_type: (.gpu.id // .cpu.id),
  gpu: .gpu, cpu: .cpu, region: .dataCenterId, instance_id: .id, image: .image, purchase_option: "on-demand",
  cost_per_hr: .cost, disk_gb: $disk, host_cuda: .cudaVersion}' > "$OUT/instance.json"
SSH_USER=root
log "Pod at $SSH_HOST:$SSH_PORT, $(jq -r .cost_per_hr "$OUT/instance.json") USD/h"

wait_for_ssh
run_bundle

# Shared launcher steps, sourced by the provider launch scripts. Works with macOS bash 3.2.
# The caller sets KEY, SSH_USER, SSH_HOST and SSH_PORT once the machine is up.

log() { printf '%s [%s] %s\n' "$(date -u +%H:%M:%S)" "${PLATFORM_ID:-?}" "$*" >&2; }

# Record a named UTC timestamp in the run's events file.
mark() { printf '{"event": "%s", "utc": "%s"}\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/events.jsonl"; }

lower() { echo "$1" | tr '[:upper:]' '[:lower:]'; }

new_ssh_key() { ssh-keygen -q -t ed25519 -N "" -C evo2size -f "$KEY"; }

ssh_opts() {
  echo -i "$KEY" -p "${SSH_PORT:-22}" -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile="$OUT/.known_hosts" -o ConnectTimeout=10 -o ServerAliveInterval=30
}
ssh_cmd() { ssh $(ssh_opts) "$SSH_USER@$SSH_HOST" "$@"; }
# copy_to LOCAL... REMOTE_DIR
copy_to() {
  local dest=${!#}
  scp -q -r $(ssh_opts | sed 's/-p /-P /') "${@:1:$#-1}" "$SSH_USER@$SSH_HOST:$dest"
}
# copy_from REMOTE LOCAL_DIR
copy_from() { scp -q -r $(ssh_opts | sed 's/-p /-P /') "$SSH_USER@$SSH_HOST:$1" "$2"; }

wait_for_ssh() {
  for _ in $(seq 1 60); do
    if ssh_cmd true 2>/dev/null; then mark ssh_ready; return 0; fi
    sleep 10
  done
  log "SSH did not become available"; return 1
}

# Copy the bundle (code.tar.gz, remote-run.sh, data/) to /work, run it, copy /work/out back.
run_bundle() {
  local status=0
  ssh_cmd "sudo mkdir -p /work 2>/dev/null || mkdir -p /work; sudo chown \$(id -u):\$(id -g) /work 2>/dev/null || true"
  copy_to "$BUNDLE/code.tar.gz" "$BUNDLE/remote-run.sh" "$BUNDLE/data" /work/
  ssh_cmd "mkdir -p /work/code && tar -xzf /work/code.tar.gz -C /work/code"
  mark workload_start
  ssh_cmd "bash /work/remote-run.sh $IMAGE_REF $PLATFORM_ID $CODE_SHA256 $DEVICE $TASKS ${TIME_LIMIT_S:-}" || status=$?
  mark workload_end
  copy_from "/work/out/." "$OUT/" || log "Copying outputs back failed"
  return $status
}

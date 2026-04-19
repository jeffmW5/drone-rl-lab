#!/bin/bash
# =============================================================================
# Drone RL Lab — RunPod Pod Manager
# =============================================================================
# Starts the pod, waits for SSH, copies deploy key, runs the full pipeline,
# then exits (pod auto-stops via its own safety timer).
#
# Usage:
#   bash scripts/manage_pod.sh              # check inbox, start pod, run pipeline
#   bash scripts/manage_pod.sh --status     # just print pod status
#   bash scripts/manage_pod.sh --stop       # force stop the pod
#   bash scripts/manage_pod.sh --dry-run    # show inbox queue, don't start pod
#
# Required env vars (set in ~/.bashrc on VM — never commit these):
#   RUNPOD_API_KEY   — your RunPod API key
#   RUNPOD_POD_ID    — your pod ID (e.g. "abc1234def5")
#
# Optional env vars:
#   RUNPOD_GPU_COUNT — GPUs to resume with (default: 1)
#   RUNPOD_CLOUD_TYPE — pod cloud type (default: SECURE)
#   DRONE_RL_DEPLOY_KEY — SSH private key to inject into/run against pods
#   DRONE_RL_RUNPOD_POD_ID_FILE — local file used to persist the current pod ID
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GPU_COUNT="${RUNPOD_GPU_COUNT:-1}"
RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-SECURE}"
POLL_INTERVAL=10   # seconds between status checks
MAX_WAIT=300       # max seconds to wait for pod to start (5 min)
SSH_TIMEOUT=120    # max seconds to wait for SSH to be ready

# GPU types to try when creating a new pod (order = preference)
GPU_TYPES=(
    "NVIDIA GeForce RTX 3090"
)
PYTORCH_IMAGE="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
DEFAULT_POD_ID_FILE="${HOME}/.config/drone-rl-lab/runpod_pod_id"
POD_ID_FILE="${DRONE_RL_RUNPOD_POD_ID_FILE:-$DEFAULT_POD_ID_FILE}"

# Prefer a normal home-directory key. Fall back to older shared-folder paths only
# if they are explicitly readable in this VM session.
DEPLOY_KEY="${DRONE_RL_DEPLOY_KEY:-}"
if [ -z "$DEPLOY_KEY" ]; then
    for candidate in \
        "${HOME}/.ssh/id_ed25519_runpod" \
        "${HOME}/.ssh/id_ed25519" \
        "/media/id_ed25519_runpod"
    do
        if [ -r "$candidate" ]; then
            DEPLOY_KEY="$candidate"
            break
        fi
    done
fi

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log()  { echo -e "${GREEN}[pod]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[err]${NC} $*" >&2; }

# ── Validate env vars ─────────────────────────────────────────────────────────
check_env() {
    if [ -z "${RUNPOD_API_KEY:-}" ]; then
        err "RUNPOD_API_KEY not set. Add to ~/.bashrc:"
        err "  export RUNPOD_API_KEY=\"your_key_here\""
        exit 1
    fi
    if [ -z "$DEPLOY_KEY" ]; then
        err "No readable SSH key found for RunPod access."
        err "Set DRONE_RL_DEPLOY_KEY, or place a readable key at:"
        err "  ${HOME}/.ssh/id_ed25519_runpod"
        err "  ${HOME}/.ssh/id_ed25519"
        exit 1
    fi
    # Load persisted pod ID from file if env var not set
    if [ -z "${RUNPOD_POD_ID:-}" ] && [ -f "$POD_ID_FILE" ]; then
        RUNPOD_POD_ID=$(cat "$POD_ID_FILE")
        warn "Loaded pod ID from $POD_ID_FILE: $RUNPOD_POD_ID"
    fi
    if [ -z "${RUNPOD_POD_ID:-}" ]; then
        err "RUNPOD_POD_ID not set. Add to ~/.bashrc:"
        err "  export RUNPOD_POD_ID=\"your_pod_id_here\""
        exit 1
    fi
}

# ── RunPod API helpers ────────────────────────────────────────────────────────
runpod_query() {
    local query="$1"
    curl -s \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
        -d "{\"query\": \"$query\"}" \
        "https://api.runpod.io/graphql"
}

pod_exists() {
    local pod_id="$1"
    local status_json
    status_json=$(runpod_query "{ pod(input: {podId: \\\"${pod_id}\\\"}) { id } }")
    echo "$status_json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
sys.exit(0 if ((d.get('data') or {}).get('pod') or {}).get('id') else 1)
" 2>/dev/null
}

get_pod_status() {
    runpod_query "{ pod(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) { id desiredStatus runtime { uptimeInSeconds ports { ip privatePort publicPort type } } } }"
}

save_pod_id() {
    local pod_id="$1"
    mkdir -p "$(dirname "$POD_ID_FILE")"
    echo "$pod_id" > "$POD_ID_FILE"
    warn "New pod ID saved to $POD_ID_FILE"
    warn "Update ~/.bashrc when convenient: export RUNPOD_POD_ID=\"$pod_id\""
}

create_new_pod() {
    local gpu_type="$1"
    log "Trying to create new pod with GPU: $gpu_type"
    local public_key
    public_key=$(ssh-keygen -y -f "$DEPLOY_KEY" 2>/dev/null | tr -d '\n')
    if [ -z "$public_key" ]; then
        err "Could not derive public key from $DEPLOY_KEY"
        return 1
    fi
    local result
    result=$(runpod_query "mutation { podFindAndDeployOnDemand(input: { \
name: \\\"drone-rl-lab\\\", \
imageName: \\\"${PYTORCH_IMAGE}\\\", \
gpuTypeId: \\\"${gpu_type}\\\", \
cloudType: ${RUNPOD_CLOUD_TYPE}, \
gpuCount: 1, \
containerDiskInGb: 50, \
ports: \\\"22/tcp\\\", \
startSsh: true, \
env: [{key: \\\"PUBLIC_KEY\\\", value: \\\"${public_key}\\\"}] \
}) { id } }")
    echo "$result" | python3 -c "
import sys, json
d = json.load(sys.stdin)
pod = (d.get('data') or {}).get('podFindAndDeployOnDemand') or {}
pid = pod.get('id', '')
if pid:
    print(pid)
" 2>/dev/null
}

start_pod() {
    if ! pod_exists "$RUNPOD_POD_ID"; then
        warn "Configured pod $RUNPOD_POD_ID does not exist anymore — creating a fresh RTX 3090 pod..."
        local new_id=""
        for gpu_type in "${GPU_TYPES[@]}"; do
            new_id=$(create_new_pod "$gpu_type")
            if [ -n "$new_id" ]; then
                log "New pod created: $new_id (GPU: $gpu_type)"
                RUNPOD_POD_ID="$new_id"
                save_pod_id "$new_id"
                return 0
            fi
            warn "  $gpu_type — unavailable"
        done
        err "RTX 3090 pod unavailable."
        exit 1
    fi

    log "Sending podResume request for $RUNPOD_POD_ID..."
    local result
    result=$(runpod_query "mutation { podResume(input: {podId: \\\"${RUNPOD_POD_ID}\\\", gpuCount: ${GPU_COUNT}}) { id desiredStatus } }")

    # Detect GPU availability failure and fall back to creating a new pod
    if echo "$result" | python3 -c "
import sys, json
d = json.load(sys.stdin)
errors = d.get('errors') or []
msg = ' '.join(e.get('message','') for e in errors).lower()
sys.exit(0 if ('not enough free gpu' in msg or 'no available' in msg or 'unavailable' in msg) else 1)
" 2>/dev/null; then
        warn "GPU unavailable for existing pod — trying to create a new one..."
        local new_id=""
        for gpu_type in "${GPU_TYPES[@]}"; do
            new_id=$(create_new_pod "$gpu_type")
            if [ -n "$new_id" ]; then
                log "New pod created: $new_id (GPU: $gpu_type)"
                RUNPOD_POD_ID="$new_id"
                save_pod_id "$new_id"
                return 0
            fi
            warn "  $gpu_type — unavailable"
        done
        err "All GPU types exhausted. No pod available."
        err "Try again later or check RunPod availability."
        exit 1
    fi
}

stop_pod() {
    log "Sending podStop request..."
    runpod_query "mutation { podStop(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) { id desiredStatus } }"
}

# ── Parse SSH endpoint from pod status JSON ───────────────────────────────────
get_ssh_info() {
    local status_json="$1"
    # Extract TCP port for port 22
    local public_port
    public_port=$(echo "$status_json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
pod = d.get('data', {}).get('pod', {})
runtime = pod.get('runtime') or {}
ports = runtime.get('ports') or []
for p in ports:
    if p.get('privatePort') == 22 and p.get('type') == 'tcp':
        print(p['publicPort'])
        break
" 2>/dev/null)

    local host
    host=$(echo "$status_json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
pod = d.get('data', {}).get('pod', {})
runtime = pod.get('runtime') or {}
ports = runtime.get('ports') or []
for p in ports:
    if p.get('privatePort') == 22 and p.get('type') == 'tcp':
        print(p['ip'])
        break
" 2>/dev/null)

    if [ -n "$public_port" ] && [ -n "$host" ]; then
        echo "${host}:${public_port}"
    fi
}

get_desired_status() {
    local status_json="$1"
    echo "$status_json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d.get('data', {}).get('pod', {}).get('desiredStatus', 'UNKNOWN'))
" 2>/dev/null
}

# ── Wait for pod to be running and SSH-ready ──────────────────────────────────
wait_for_pod() {
    log "Waiting for pod to start (max ${MAX_WAIT}s)..."
    local elapsed=0
    local ssh_info=""

    while [ $elapsed -lt $MAX_WAIT ]; do
        local status_json
        status_json=$(get_pod_status)
        local desired_status
        desired_status=$(get_desired_status "$status_json")

        if [ "$desired_status" = "RUNNING" ]; then
            ssh_info=$(get_ssh_info "$status_json")
            if [ -n "$ssh_info" ]; then
                log "Pod running. SSH endpoint: ${ssh_info}"
                echo "$ssh_info"
                return 0
            fi
        fi

        echo -n "  [${elapsed}s] Status: ${desired_status}... "
        sleep $POLL_INTERVAL
        elapsed=$((elapsed + POLL_INTERVAL))
        echo ""
    done

    err "Pod did not start within ${MAX_WAIT}s"
    return 1
}

wait_for_ssh() {
    local host="$1"
    local port="$2"
    log "Waiting for SSH on ${host}:${port} (max ${SSH_TIMEOUT}s)..."
    local elapsed=0

    while [ $elapsed -lt $SSH_TIMEOUT ]; do
        if ssh -o StrictHostKeyChecking=no \
               -o ConnectTimeout=5 \
               -o BatchMode=yes \
               -i "$DEPLOY_KEY" \
               -p "$port" "root@${host}" \
               "echo ok" &>/dev/null; then
            log "SSH is ready!"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done

    err "SSH not ready within ${SSH_TIMEOUT}s"
    return 1
}

# ── Copy deploy key to pod ────────────────────────────────────────────────────
copy_deploy_key() {
    local host="$1"
    local port="$2"
    log "Copying deploy key to pod..."

    if [ ! -f "$DEPLOY_KEY" ]; then
        err "Deploy key not found at $DEPLOY_KEY"
        err "Set DRONE_RL_DEPLOY_KEY to override the default key search path."
        exit 1
    fi

    ssh -o StrictHostKeyChecking=no \
        -i "$DEPLOY_KEY" \
        -p "$port" "root@${host}" \
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh" 2>/dev/null

    scp -o StrictHostKeyChecking=no \
        -i "$DEPLOY_KEY" \
        -P "$port" \
        "$DEPLOY_KEY" "root@${host}:~/.ssh/id_ed25519"

    ssh -o StrictHostKeyChecking=no \
        -i "$DEPLOY_KEY" \
        -p "$port" "root@${host}" \
        "chmod 600 ~/.ssh/id_ed25519 && ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null"

    log "Deploy key installed."
}

# ── Run pipeline on pod ───────────────────────────────────────────────────────
run_pipeline_on_pod() {
    local host="$1"
    local port="$2"

    log "Running setup + pipeline on pod..."
    log "This may take a while. Pod will auto-stop when done."
    echo ""

    ssh -o StrictHostKeyChecking=no \
        -i "$DEPLOY_KEY" \
        -p "$port" "root@${host}" \
        -t "bash -lc '
            set -e
            cd /root
            # Clone or pull
            if [ -d drone-rl-lab ]; then
                cd drone-rl-lab && git pull && cd /root
            else
                git clone git@github.com:jeffmW5/drone-rl-lab.git
            fi

            # Setup if not already done
            if [ ! -f /root/.drone_rl_runpod_setup_complete ]; then
                bash /root/drone-rl-lab/scripts/setup_runpod.sh
            else
                echo \"[skip] Setup already done.\"
            fi

            # Run the experiment pipeline
            cd /root/drone-rl-lab
            bash scripts/run_experiment.sh
        '"
}

# ── Check inbox has work ──────────────────────────────────────────────────────
check_inbox() {
    cd "$REPO_DIR"
    git pull 2>/dev/null || true

    local next
    next=$(python3 scripts/parse_queue.py --next 2>/dev/null) || true

    if [ -z "$next" ]; then
        warn "INBOX is empty — no tasks to run."
        warn "Add tasks to inbox/INBOX.md first."
        return 1
    fi

    log "Found task: $next"
    return 0
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    check_env

    case "${1:-}" in
        --status)
            log "Checking pod status..."
            status_json=$(get_pod_status)
            desired=$(get_desired_status "$status_json")
            ssh_info=$(get_ssh_info "$status_json")
            echo "  Pod ID:     $RUNPOD_POD_ID"
            echo "  Status:     $desired"
            echo "  SSH:        ${ssh_info:-not available}"
            exit 0
            ;;
        --stop)
            log "Stopping pod..."
            stop_pod
            log "Stop request sent."
            exit 0
            ;;
        --dry-run)
            log "Dry run — showing inbox queue:"
            cd "$REPO_DIR"
            git pull 2>/dev/null || true
            python3 scripts/parse_queue.py || true
            exit 0
            ;;
    esac

    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║  Drone RL Lab — Pod Manager              ║"
    echo "╚══════════════════════════════════════════╝"
    echo ""

    # Check inbox first — don't start pod if nothing to do
    if ! check_inbox; then
        exit 0
    fi

    # Start pod
    log "Starting pod ${RUNPOD_POD_ID}..."
    start_pod > /dev/null

    # Wait for running + SSH endpoint
    ssh_info=$(wait_for_pod)
    SSH_HOST="${ssh_info%%:*}"
    SSH_PORT="${ssh_info##*:}"

    # Wait for SSH to accept connections
    wait_for_ssh "$SSH_HOST" "$SSH_PORT"

    # Install deploy key on pod
    copy_deploy_key "$SSH_HOST" "$SSH_PORT"

    # Run the pipeline (pod auto-stops when done via setup_runpod.sh timer)
    run_pipeline_on_pod "$SSH_HOST" "$SSH_PORT"

    echo ""
    log "Pipeline complete. Pod will auto-stop (4hr safety timer running on pod)."
    log "Check outbox/STATUS.md for results."
}

main "$@"

#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="/root/drone-rl-lab"

CONFIG=""
AUTO_STOP=1
PULL_RESULTS=1

usage() {
    cat <<'EOF'
Usage: bash scripts/runpod_ai_gp_train.sh --config configs/ai_gp_NNN.yaml [options]

Runs an AI-GP training config on RunPod from Linux, pulls results/<experiment>,
and stops the pod by default.

Options:
  --config PATH       Config to upload and train.
  --keep-running     Do not stop the pod after training.
  --no-pull          Do not pull results after training exits.
  -h, --help         Show this help.

Required environment:
  RUNPOD_API_KEY, plus RUNPOD_POD_ID or ~/.config/drone-rl-lab/runpod_pod_id.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="${2:?--config requires a path}"
            shift 2
            ;;
        --keep-running)
            AUTO_STOP=0
            shift
            ;;
        --no-pull)
            PULL_RESULTS=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "--config is required." >&2
    usage >&2
    exit 1
fi
if [[ "$CONFIG" = /* ]]; then
    echo "Use a repo-relative config path." >&2
    exit 1
fi

cd "$REPO_DIR"
if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

experiment="$(
    python3 - "$CONFIG" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r"(?m)^\s*name:\s*([A-Za-z0-9_.-]+)\s*$", text)
if not match:
    raise SystemExit("could not parse experiment name")
print(match.group(1))
PY
)"

extra_paths=()
initial_checkpoint="$(
    python3 - "$CONFIG" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r"(?m)^\s*initial_actor_checkpoint:\s*['\"]?([^'\"\n]+?)['\"]?\s*$", text)
print(match.group(1).strip() if match else "")
PY
)"
if [[ -n "$initial_checkpoint" && "$initial_checkpoint" != /* ]]; then
    if [[ -f "$initial_checkpoint" ]]; then
        extra_paths+=("$initial_checkpoint")
    else
        echo "Initial actor checkpoint not found: $initial_checkpoint" >&2
        exit 1
    fi
fi

for path in "$CONFIG" scripts/manage_pod.sh scripts/smoke_ai_gp.py ai_gp_rl \
    train.py train_ai_gp.py train_ai_gp_swift_bc.py "${extra_paths[@]}"
do
    if [[ ! -e "$path" ]]; then
        echo "Required path not found: $path" >&2
        exit 1
    fi
done

# Reuse the existing RunPod API, SSH, and fallback-pod helpers.
# shellcheck source=scripts/manage_pod.sh
source "$REPO_DIR/scripts/manage_pod.sh"

POD_STARTED=0
archive=""
cleanup() {
    if [[ -n "$archive" ]]; then
        rm -f "$archive"
    fi
    if [[ "$AUTO_STOP" == "1" && "$POD_STARTED" == "1" ]]; then
        stop_pod >/dev/null || true
        log "Stop request sent."
    fi
}
trap cleanup EXIT

check_env
start_pod >/dev/null
POD_STARTED=1
ssh_info=$(wait_for_pod | tail -n 1)
SSH_HOST="${ssh_info%%:*}"
SSH_PORT="${ssh_info##*:}"
wait_for_ssh "$SSH_HOST" "$SSH_PORT"

archive="$(mktemp --suffix=.tar.gz)"
tar -czf "$archive" \
    --exclude=__pycache__ \
    --exclude=*.pyc \
    ai_gp_rl \
    train.py \
    train_ai_gp.py \
    train_ai_gp_swift_bc.py \
    scripts/smoke_ai_gp.py \
    "$CONFIG" \
    "${extra_paths[@]}"

log "Uploading AI-GP training bundle for $experiment..."
scp -o StrictHostKeyChecking=no \
    -i "$DEPLOY_KEY" \
    -P "$SSH_PORT" \
    "$archive" "root@${SSH_HOST}:/root/drone-rl-ai-gp-train.tar.gz"

ssh -o StrictHostKeyChecking=no \
    -i "$DEPLOY_KEY" \
    -p "$SSH_PORT" "root@${SSH_HOST}" \
    "mkdir -p '$REMOTE_ROOT' && tar -xzf /root/drone-rl-ai-gp-train.tar.gz -C '$REMOTE_ROOT' && rm -f /root/drone-rl-ai-gp-train.tar.gz"

log "Running CUDA smoke and training $experiment..."
ssh -o StrictHostKeyChecking=no \
    -i "$DEPLOY_KEY" \
    -p "$SSH_PORT" "root@${SSH_HOST}" \
    "cd '$REMOTE_ROOT' && \
     python3 scripts/smoke_ai_gp.py '$CONFIG' && \
     python3 -u train.py '$CONFIG'"

if [[ "$PULL_RESULTS" == "1" ]]; then
    mkdir -p results
    log "Pulling results/$experiment..."
    rm -rf "results/$experiment"
    scp -r -o StrictHostKeyChecking=no \
        -i "$DEPLOY_KEY" \
        -P "$SSH_PORT" \
        "root@${SSH_HOST}:${REMOTE_ROOT}/results/${experiment}" \
        results/
fi

log "Training complete: results/$experiment"

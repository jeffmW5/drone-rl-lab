#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="/root/drone-rl-lab"

CONFIG="configs/ai_gp_039_all_gate_soft_floor_ppo_30m.yaml"
CHECKPOINT="results/ai_gp_039_all_gate_soft_floor_ppo_30m/best_policy.pt"
EPISODES=512
TRAJECTORIES=32
SEED=1001
RANDOMIZATION=1
AUTO_STOP=1
MODE="near_gate_teacher"
TAKEOVER_DISTANCE_M=10.0
LATERAL_THRESHOLD_M=0.40
VERTICAL_THRESHOLD_M=0.40
GATE_INDICES="all"
OUTPUT=""

usage() {
    cat <<'EOF'
Usage: bash scripts/runpod_ai_gp_hybrid_eval.sh [options]

Runs a policy/geometric-teacher hybrid evaluation on RunPod, pulls the telemetry
JSON, and stops the pod by default.

Options:
  --config PATH              Config to upload and evaluate.
  --checkpoint PATH          Checkpoint to upload and evaluate.
  --output PATH              Local/remote report path under the repo.
  --episodes N              Episode count, default 512.
  --trajectories N           Tracked trajectory count, default 32.
  --seed N                   Evaluation seed, default 1001.
  --mode MODE                policy, teacher, near_gate_teacher, or
                             off_center_near_gate_teacher.
  --takeover-distance-m N    Near-gate teacher takeover distance, default 10.
  --lateral-threshold-m N    Off-center lateral threshold, default 0.40.
  --vertical-threshold-m N   Off-center vertical threshold, default 0.40.
  --gate-indices LIST        Comma-separated gate indices or "all", default all.
  --no-randomization         Disable environment randomization.
  --keep-running             Do not stop the pod after the eval.
  -h, --help                 Show this help.

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
        --checkpoint)
            CHECKPOINT="${2:?--checkpoint requires a path}"
            shift 2
            ;;
        --output)
            OUTPUT="${2:?--output requires a path}"
            shift 2
            ;;
        --episodes)
            EPISODES="${2:?--episodes requires a value}"
            shift 2
            ;;
        --trajectories)
            TRAJECTORIES="${2:?--trajectories requires a value}"
            shift 2
            ;;
        --seed)
            SEED="${2:?--seed requires a value}"
            shift 2
            ;;
        --mode)
            MODE="${2:?--mode requires a value}"
            shift 2
            ;;
        --takeover-distance-m)
            TAKEOVER_DISTANCE_M="${2:?--takeover-distance-m requires a value}"
            shift 2
            ;;
        --lateral-threshold-m)
            LATERAL_THRESHOLD_M="${2:?--lateral-threshold-m requires a value}"
            shift 2
            ;;
        --vertical-threshold-m)
            VERTICAL_THRESHOLD_M="${2:?--vertical-threshold-m requires a value}"
            shift 2
            ;;
        --gate-indices)
            GATE_INDICES="${2:?--gate-indices requires a value}"
            shift 2
            ;;
        --no-randomization)
            RANDOMIZATION=0
            shift
            ;;
        --keep-running)
            AUTO_STOP=0
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

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="results/ai_gp_039_all_gate_soft_floor_ppo_30m/evals/hybrid_${MODE}_${SEED}.json"
fi

if [[ "$CONFIG" = /* || "$CHECKPOINT" = /* || "$OUTPUT" = /* ]]; then
    echo "Use repo-relative paths for --config, --checkpoint, and --output." >&2
    exit 1
fi

cd "$REPO_DIR"
for path in "$CONFIG" "$CHECKPOINT" scripts/manage_pod.sh scripts/smoke_ai_gp.py \
    scripts/evaluate_ai_gp_hybrid_teacher.py scripts/evaluate_ai_gp_checkpoint.py \
    ai_gp_rl train_ai_gp.py train.py
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
cleanup() {
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
trap 'rm -f "$archive"; cleanup' EXIT
tar -czf "$archive" \
    --exclude=__pycache__ \
    --exclude=*.pyc \
    ai_gp_rl \
    train.py \
    train_ai_gp.py \
    scripts/smoke_ai_gp.py \
    scripts/evaluate_ai_gp_checkpoint.py \
    scripts/evaluate_ai_gp_hybrid_teacher.py \
    "$CONFIG" \
    "$CHECKPOINT"

log "Uploading AI-GP hybrid eval bundle..."
scp -o StrictHostKeyChecking=no \
    -i "$DEPLOY_KEY" \
    -P "$SSH_PORT" \
    "$archive" "root@${SSH_HOST}:/root/drone-rl-ai-gp-hybrid-eval.tar.gz"

ssh -o StrictHostKeyChecking=no \
    -i "$DEPLOY_KEY" \
    -p "$SSH_PORT" "root@${SSH_HOST}" \
    "mkdir -p '$REMOTE_ROOT' && tar -xzf /root/drone-rl-ai-gp-hybrid-eval.tar.gz -C '$REMOTE_ROOT' && rm -f /root/drone-rl-ai-gp-hybrid-eval.tar.gz"

randomization_arg=()
if [[ "$RANDOMIZATION" == "1" ]]; then
    randomization_arg=(--randomization)
fi

log "Running CUDA smoke and hybrid eval mode=${MODE} seed=${SEED} episodes=${EPISODES}..."
ssh -o StrictHostKeyChecking=no \
    -i "$DEPLOY_KEY" \
    -p "$SSH_PORT" "root@${SSH_HOST}" \
    "cd '$REMOTE_ROOT' && \
     python3 scripts/smoke_ai_gp.py '$CONFIG' && \
     python3 scripts/evaluate_ai_gp_hybrid_teacher.py \
       '$CONFIG' \
       '$CHECKPOINT' \
       '$OUTPUT' \
       --mode '$MODE' \
       --episodes '$EPISODES' \
       --trajectories '$TRAJECTORIES' \
       --seed '$SEED' \
       --takeover-distance-m '$TAKEOVER_DISTANCE_M' \
       --lateral-threshold-m '$LATERAL_THRESHOLD_M' \
       --vertical-threshold-m '$VERTICAL_THRESHOLD_M' \
       --gate-indices '$GATE_INDICES' \
       ${randomization_arg[*]}"

mkdir -p "$(dirname "$OUTPUT")"
log "Pulling hybrid telemetry report to $OUTPUT..."
scp -o StrictHostKeyChecking=no \
    -i "$DEPLOY_KEY" \
    -P "$SSH_PORT" \
    "root@${SSH_HOST}:${REMOTE_ROOT}/${OUTPUT}" \
    "$OUTPUT"

log "Hybrid eval complete: $OUTPUT"

#!/bin/bash
set -euo pipefail

ROOT="/root/drone-rl-lab"
LOG_DIR="${ROOT}/logs"

usage() {
    echo "Usage: $0 bootstrap|smoke|start|status|logs|stop [arguments]"
}

bootstrap() {
    command -v python3 >/dev/null 2>&1 || {
        echo "python3 is required"
        exit 1
    }
    if ! python3 -c "import yaml" >/dev/null 2>&1; then
        python3 -m pip install --quiet --disable-pip-version-check pyyaml
    fi
    python3 - <<'PY'
import torch
import yaml

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
print(f"torch={torch.__version__}")
print(f"cuda={torch.cuda.get_device_name(0)}")
print(f"pyyaml={yaml.__version__}")
PY
}

validate_name() {
    [[ "$1" =~ ^[A-Za-z0-9_.-]+$ ]] || {
        echo "Invalid experiment name: $1"
        exit 1
    }
}

start_job() {
    local config_path="${1:?config path required}"
    local experiment="${2:?experiment name required}"
    local auto_stop="${3:-false}"
    validate_name "$experiment"

    local full_config="${ROOT}/${config_path}"
    local log_path="${LOG_DIR}/${experiment}.log"
    local pid_path="${LOG_DIR}/${experiment}.pid"
    local status_path="${LOG_DIR}/${experiment}.exit"
    local job_path="${LOG_DIR}/${experiment}.job.sh"

    [[ -f "$full_config" ]] || {
        echo "Config not found: $full_config"
        exit 1
    }
    mkdir -p "$LOG_DIR"
    if [[ -f "$pid_path" ]] && kill -0 "$(cat "$pid_path")" 2>/dev/null; then
        echo "Training already running: pid=$(cat "$pid_path")"
        exit 1
    fi

    cat > "$job_path" <<EOF
#!/bin/bash
set -o pipefail
cd "$ROOT"
python3 -u train.py "$config_path" > "$log_path" 2>&1
status=\$?
echo "\$status" > "$status_path"
if [[ "$auto_stop" == "true" ]] && command -v runpodctl >/dev/null 2>&1 && [[ -n "\${RUNPOD_POD_ID:-}" ]]; then
    runpodctl stop pod "\$RUNPOD_POD_ID" || true
fi
exit "\$status"
EOF
    chmod +x "$job_path"
    rm -f "$status_path"
    nohup setsid "$job_path" >/dev/null 2>&1 < /dev/null &
    echo "$!" > "$pid_path"
    echo "started experiment=$experiment pid=$(cat "$pid_path") log=$log_path"
}

status_job() {
    local experiment="${1:?experiment name required}"
    validate_name "$experiment"
    local pid_path="${LOG_DIR}/${experiment}.pid"
    local status_path="${LOG_DIR}/${experiment}.exit"
    local log_path="${LOG_DIR}/${experiment}.log"

    if [[ -f "$pid_path" ]] && kill -0 "$(cat "$pid_path")" 2>/dev/null; then
        echo "state=running pid=$(cat "$pid_path")"
    elif [[ -f "$status_path" ]]; then
        echo "state=finished exit_code=$(cat "$status_path")"
    else
        echo "state=not_started"
    fi
    [[ -f "$log_path" ]] && tail -n 25 "$log_path"
}

logs_job() {
    local experiment="${1:?experiment name required}"
    validate_name "$experiment"
    touch "${LOG_DIR}/${experiment}.log"
    tail -n 100 -f "${LOG_DIR}/${experiment}.log"
}

stop_job() {
    local experiment="${1:?experiment name required}"
    validate_name "$experiment"
    local pid_path="${LOG_DIR}/${experiment}.pid"
    if [[ ! -f "$pid_path" ]]; then
        echo "No pid file for $experiment"
        return 0
    fi
    local pid
    pid="$(cat "$pid_path")"
    if kill -0 "$pid" 2>/dev/null; then
        kill -- "-$pid"
        echo "stop requested for process_group=$pid"
    else
        echo "process is not running"
    fi
}

case "${1:-}" in
    bootstrap) bootstrap ;;
    smoke) shift; cd "$ROOT"; python3 scripts/smoke_ai_gp.py "$@" ;;
    start) shift; start_job "$@" ;;
    status) shift; status_job "$@" ;;
    logs) shift; logs_job "$@" ;;
    stop) shift; stop_job "$@" ;;
    *) usage; exit 1 ;;
esac

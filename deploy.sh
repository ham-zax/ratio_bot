#!/usr/bin/env bash
# Deployment helper for crypto_ratio_alerter

set -euo pipefail

SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}")
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

APP_FILE="crypto_ratio_alerter.py"
SCREEN_SESSION="ratio_bot"
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-pip}"

RESTART=false
SKIP_INSTALL=false
REFRESH_VENV=false

usage() {
    cat <<EOF
Usage: ${SCRIPT_NAME} [options]

Options:
  --restart        Stop the running screen session (if any) and relaunch the bot.
  --skip-install   Skip dependency installation (useful when requirements have not changed).
  --refresh-venv   Rebuild the virtual environment from scratch before installation.
  -h, --help       Show this help message.

Environment overrides:
  PYTHON_BIN=<path>   Override python interpreter (default: python3)
  PIP_BIN=<path>      Override pip executable (default: pip)
EOF
}

log() {
    local level="$1"; shift
    printf "%s [%s] %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "${level}" "$*"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --restart)
                RESTART=true
                shift
                ;;
            --skip-install)
                SKIP_INSTALL=true
                shift
                ;;
            --refresh-venv)
                REFRESH_VENV=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log "ERR" "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

ensure_prerequisites() {
    if [[ ! -f "${PROJECT_ROOT}/${APP_FILE}" ]]; then
        log "ERR" "Cannot find ${APP_FILE} in project root (${PROJECT_ROOT})."
        exit 1
    fi

    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        log "ERR" "Python interpreter '${PYTHON_BIN}' not found."
        exit 1
    fi
}

prepare_virtualenv() {
    if [[ "${REFRESH_VENV}" == true && -d "${PROJECT_ROOT}/${VENV_DIR}" ]]; then
        log "INF" "Removing existing virtual environment (${VENV_DIR})."
        rm -rf "${PROJECT_ROOT:?}/${VENV_DIR}"
    fi

    if [[ ! -d "${PROJECT_ROOT}/${VENV_DIR}" ]]; then
        log "INF" "Creating virtual environment (${VENV_DIR})."
        "${PYTHON_BIN}" -m venv "${PROJECT_ROOT}/${VENV_DIR}"
    fi

    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/${VENV_DIR}/bin/activate"
}

install_dependencies() {
    if [[ "${SKIP_INSTALL}" == true ]]; then
        log "INF" "Skipping dependency installation (--skip-install)."
        return
    fi

    if [[ ! -f "${PROJECT_ROOT}/${REQUIREMENTS_FILE}" ]]; then
        log "INF" "No ${REQUIREMENTS_FILE} found; skipping dependency installation."
        return
    fi

    log "INF" "Upgrading pip."
    pip install --upgrade pip

    log "INF" "Installing dependencies from ${REQUIREMENTS_FILE}."
    pip install -r "${PROJECT_ROOT}/${REQUIREMENTS_FILE}"
}

screen_session_exists() {
    screen -list | grep -q "[.]${SCREEN_SESSION}[[:space:]]"
}

stop_screen_session() {
    if screen_session_exists; then
        log "INF" "Stopping screen session '${SCREEN_SESSION}'."
        screen -S "${SCREEN_SESSION}" -X quit || true
        sleep 1
    else
        log "INF" "No existing screen session '${SCREEN_SESSION}' found."
    fi
}

stop_running_process() {
    if pgrep -f "${APP_FILE}" >/dev/null 2>&1; then
        log "INF" "Terminating running process ${APP_FILE}."
        pkill -f "${APP_FILE}" || true
        sleep 1
    else
        log "INF" "No running process ${APP_FILE} detected."
    fi
}

start_bot() {
    if [[ ! -d "${PROJECT_ROOT}/${VENV_DIR}" ]]; then
        log "ERR" "Virtual environment missing; cannot start bot."
        exit 1
    fi

    local start_cmd="source ${PROJECT_ROOT}/${VENV_DIR}/bin/activate && python ${PROJECT_ROOT}/${APP_FILE}"

    log "INF" "Launching bot in detached screen session '${SCREEN_SESSION}'."
    screen -dmS "${SCREEN_SESSION}" bash -lc "${start_cmd}"
    log "INF" "Bot started; attach with 'screen -r ${SCREEN_SESSION}'."
}

main() {
    parse_args "$@"
    ensure_prerequisites

    log "INF" "Starting deployment (restart=${RESTART}, skip_install=${SKIP_INSTALL}, refresh_venv=${REFRESH_VENV})."

    prepare_virtualenv
    install_dependencies

    if [[ "${RESTART}" == true ]]; then
        stop_screen_session
        stop_running_process
        start_bot
    else
        log "INF" "Deployment finished without restart. Use '--restart' to relaunch via screen."
    fi

    log "INF" "Deployment complete."
}

main "$@"

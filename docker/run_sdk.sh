#!/usr/bin/env bash
# Run RBY1 Isaac Sim GUI container (rby1-sdk UDP integration mode)
#
# Behavior:
#   1. Start the Isaac Sim container in the background (with UDP bridge)
#       · 5005/udp : Isaac Sim → rby1-sdk (sends RobotState)
#       · 5006/udp : rby1-sdk → Isaac Sim (receives RobotCommand)
#   2. Once the Isaac Sim UDP bridge (port 5006) is open, automatically run app_main isaac
#   3. When either side exits, both are cleaned up
#
# Since --network=host is used, the host and container share the same network namespace
# → the SDK can connect to the container via 127.0.0.1.
#
# app_main is bundled inside the image (/opt/rby1-sim-isaac/sdk/app/app_main),
# so no separate build on the host is required. Only the rby1-sim-isaac image is needed.
#
# Usage:
#   ./docker/run_sdk.sh
#   ./docker/run_sdk.sh --image 0.10.7-a_v1.2
#   ./docker/run_sdk.sh --image rainbowroboticsofficial/rby1-sim-isaac:0.10.7-a_v1.2
#   IMAGE_NAME=0.10.7-a_v1.2 ./docker/run_sdk.sh
#   ./docker/run_sdk.sh --gripper --gripper-name rb_gripper
#   GRIPPER_NAME=rb_gripper ./docker/run_sdk.sh
#
# All arguments except --image are forwarded to simulation.py through docker/run.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_IMAGE_NAME_ENV="${IMAGE_NAME:-}"
_IMAGE_EXPLICIT=0
SIM_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            [[ $# -ge 2 ]] || { echo "[run_sdk] ERROR: a tag must be specified after --image." >&2; exit 1; }
            IMAGE_NAME="$2"
            _IMAGE_EXPLICIT=1
            export IMAGE_NAME
            shift 2
            ;;
        *)
            SIM_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${SIM_ARGS[@]+"${SIM_ARGS[@]}"}"  # replace $@ with the remaining args after parsing

IMAGE_NAME="${IMAGE_NAME:-0.10.7-a_v1.2-isaac6.1.0-r1}"
export IMAGE_NAME
# Expand a bare version tag (e.g. "0.10.7-a_v1.2") to the full registry image name.
if [[ "${IMAGE_NAME}" != */* && "${IMAGE_NAME}" != *:* ]]; then
    IMAGE_NAME="rainbowroboticsofficial/rby1-sim-isaac:${IMAGE_NAME}"
fi
if [[ $_IMAGE_EXPLICIT -eq 0 && -z "${_IMAGE_NAME_ENV}" ]]; then
    echo "[run_sdk] WARNING: No image tag specified. Using default (${IMAGE_NAME})." >&2
    echo "[run_sdk]          To specify: ./docker/run_sdk.sh --image <TAG>" >&2
    echo "[run_sdk]                     IMAGE_NAME=<TAG> ./docker/run_sdk.sh" >&2
fi

# Path to app_main inside the container (installed to /opt/rby1-sim-isaac/sdk/app/ in the Dockerfile)
APP_MAIN_IN_CONTAINER="${APP_MAIN_IN_CONTAINER:-/opt/rby1-sim-isaac/sdk/app/app_main}"
CONTAINER_NAME="${CONTAINER_NAME:-rby1-sim-isaac}"

APP_MAIN_PID=""

cleanup() {
    echo ""
    echo "[run_sdk] Shutting down..."
    # Kill the docker exec wrapper process (app_main inside the container is cleaned up by docker stop)
    [[ -n "${APP_MAIN_PID}" ]] && kill "${APP_MAIN_PID}" 2>/dev/null || true
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Start the Isaac Sim container in the background (non-interactive)
echo "[run_sdk] Starting Isaac Sim container..."

DOCKER_TTY=0 "${SCRIPT_DIR}/run.sh" \
    --udp \
    --state-ip 127.0.0.1 \
    --state-port 5005 \
    --cmd-port 5006 \
    "$@" &
DOCKER_PID=$!

# Wait until the Isaac Sim UDP bridge (port 5006) is ready
# Since --network=host is used, the container socket is visible in the host namespace
echo "[run_sdk] Waiting for Isaac Sim to initialize (port 5006)..."
TIMEOUT=300
ELAPSED=0
while ! ss -uln 2>/dev/null | grep -q ':5006 '; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [[ $ELAPSED -ge $TIMEOUT ]]; then
        echo "[run_sdk] ERROR: Isaac Sim did not start within ${TIMEOUT}s." >&2
        exit 1
    fi
    # Early exit if docker exited unexpectedly
    if ! kill -0 "${DOCKER_PID}" 2>/dev/null; then
        echo "[run_sdk] ERROR: Isaac Sim container exited unexpectedly." >&2
        exit 1
    fi
done
echo "[run_sdk] Isaac Sim is ready (${ELAPSED}s)"

# Run app_main isaac inside the container (docker exec)
# - LD_LIBRARY_PATH, RBY1_SKIP_* are already set via ENV in the Dockerfile
# - docker exec inherits the container's environment variables
echo "[run_sdk] Starting app_main isaac (inside container)..."
docker exec "${CONTAINER_NAME}" \
    "${APP_MAIN_IN_CONTAINER}" isaac &
APP_MAIN_PID=$!
echo "[run_sdk] app_main docker-exec PID=${APP_MAIN_PID}"

# Wait until either side exits first → EXIT trap cleans up the other
wait -n "${DOCKER_PID}" "${APP_MAIN_PID}"

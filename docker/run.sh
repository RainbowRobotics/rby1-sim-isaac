#!/usr/bin/env bash
# Run RBY1 Isaac Sim GUI container (Standalone mode)
# - Isaac Sim GUI is displayed on the host display via X11
# - RBY1 robot is automatically spawned and operated via internal PD control
#
# Usage:
#   ./docker/run.sh                       # standalone, no gripper (default)
#   ./docker/run.sh --image 0.10.7-a_v1.2
#   ./docker/run.sh --image rainbowroboticsofficial/rby1-sim-isaac:0.10.7-a_v1.2
#   ./docker/run.sh --gripper --gripper-name rb_gripper
#   ./docker/run.sh --gripper --no-sim-gripper
#   GRIPPER_NAME=rb_gripper ./docker/run.sh
#
# Optional environment variables:
#   IMAGE_NAME       docker image (default: rainbowroboticsofficial/rby1-sim-isaac:0.10.7-a_v1.2)
#   ISAAC_HOST_DIR  persistent cache/config directory (default: ${HOME}/docker/isaac-sim)
#   CONTAINER_NAME   container name (default: rby1-sim-isaac)
#   GRIPPER_NAME     default gripper folder under assets/gripper; overridden by --gripper-name
#   RBY1_ISAAC_DIR   path to rby1-sim-isaac repo (containing src/, assets/).
#                    If unset, the repo root (../) relative to this script is used.
#
# Prerequisites:
#   - NVIDIA driver + nvidia-container-toolkit installed
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

_IMAGE_NAME_ENV="${IMAGE_NAME:-}"
CONTAINER_NAME="${CONTAINER_NAME:-rby1-sim-isaac}"

# Parse --image argument (remaining args are forwarded to simulation.py)
_IMAGE_EXPLICIT=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { echo "[run] ERROR: a tag must be specified after --image." >&2; exit 1; }
            IMAGE_NAME="$2"
            _IMAGE_EXPLICIT=1
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

IMAGE_NAME="${IMAGE_NAME:-rainbowroboticsofficial/rby1-sim-isaac:0.10.7-a_v1.2}"
# Expand a bare version tag (e.g. "0.10.7-a_v1.2") to the full registry image name.
if [[ "${IMAGE_NAME}" != */* && "${IMAGE_NAME}" != *:* ]]; then
    IMAGE_NAME="rainbowroboticsofficial/rby1-sim-isaac:${IMAGE_NAME}"
fi
if [[ $_IMAGE_EXPLICIT -eq 0 && -z "${_IMAGE_NAME_ENV}" ]]; then
    echo "[run] WARNING: No image tag specified. Using default (${IMAGE_NAME})." >&2
    echo "[run]          To specify: ./docker/run.sh --image <TAG>" >&2
    echo "[run]                     IMAGE_NAME=<TAG> ./docker/run.sh" >&2
fi

# Path to rby1-sim-isaac repo (containing src/, assets/).
# If unset, automatically uses the repo root relative to this script.
RBY1_ISAAC_DIR="${RBY1_ISAAC_DIR:-${REPO_ROOT}}"
RBY1_ISAAC_DIR="$(cd "${RBY1_ISAAC_DIR}" && pwd)"
if [[ ! -d "${RBY1_ISAAC_DIR}/src" ]]; then
    echo "[run] ERROR: ${RBY1_ISAAC_DIR}/src directory not found." >&2
    exit 1
fi
if [[ ! -d "${RBY1_ISAAC_DIR}/assets" ]]; then
    echo "[run] ERROR: ${RBY1_ISAAC_DIR}/assets directory not found." >&2
    exit 1
fi

# Isaac Sim cache/config directory (persists shader cache etc. on the host)
ISAAC_HOST_DIR="${ISAAC_HOST_DIR:-${HOME}/docker/isaac-sim}"

# Resolve relative paths before passing them to Docker bind mounts.
ISAAC_HOST_DIR="$(realpath -m -- "${ISAAC_HOST_DIR}")"

CACHE_DIRS=(
    "${ISAAC_HOST_DIR}"
    "${ISAAC_HOST_DIR}/cache"
    "${ISAAC_HOST_DIR}/cache/main"
    "${ISAAC_HOST_DIR}/cache/main/ov"
    "${ISAAC_HOST_DIR}/cache/main/warp"
    "${ISAAC_HOST_DIR}/cache/computecache"
    "${ISAAC_HOST_DIR}/cache/kit"
    "${ISAAC_HOST_DIR}/config"
    "${ISAAC_HOST_DIR}/data"
    "${ISAAC_HOST_DIR}/data/documents"
    "${ISAAC_HOST_DIR}/data/Kit"
    "${ISAAC_HOST_DIR}/logs"
    "${ISAAC_HOST_DIR}/pkg"
)
for cache_dir in "${CACHE_DIRS[@]}"; do
    [[ -d "${cache_dir}" ]] && continue
    # Find the nearest existing parent, including for a new cache root.
    cache_parent="$(dirname -- "${cache_dir}")"
    while [[ ! -e "${cache_parent}" ]]; do
        cache_parent="$(dirname -- "${cache_parent}")"
    done
    if [[ -w "${cache_parent}" && -x "${cache_parent}" ]]; then
        mkdir -p -- "${cache_dir}"
    else
        sudo mkdir -p -- "${cache_dir}"
    fi
done

# Check only managed directories; use sudo if a private parent blocks access.
for cache_dir in "${CACHE_DIRS[@]}"; do
    cache_stat="$(stat -L -c '%u:%g:%a' -- "${cache_dir}" 2>/dev/null)" ||
        cache_stat="$(sudo stat -L -c '%u:%g:%a' -- "${cache_dir}")"
    IFS=: read -r cache_uid cache_gid cache_mode <<< "${cache_stat}"
    if [[ "${cache_uid}:${cache_gid}" != "1234:1234" ]]; then
        echo "[run] Changing ownership of ${cache_dir} to 1234:1234 (requires sudo)"
        sudo chown 1234:1234 -- "${cache_dir}"
    fi
    if (( (8#${cache_mode} & 0700) != 0700 )); then
        sudo chmod u+rwx -- "${cache_dir}"
    fi
done

# Allow X11 local access (grant uid=1234 for container user + local socket)
xhost +SI:localuser:\#1234 +local: >/dev/null 2>&1 || true

# Mount src/ and assets/ from the rby1-sim-isaac repo into the container
SRC_MOUNT=(-v "${RBY1_ISAAC_DIR}/src:/opt/rby1-sim-isaac/app:ro")
ASSETS_MOUNT=(-v "${RBY1_ISAAC_DIR}/assets:/opt/rby1-sim-isaac/assets:ro")
echo "[run] src    mount: ${RBY1_ISAAC_DIR}/src"
echo "[run] assets mount: ${RBY1_ISAAC_DIR}/assets"

# TTY option: default is -it (interactive). Set DOCKER_TTY=0 for non-interactive (background) use
TTY_FLAGS=(-it)
if [[ "${DOCKER_TTY:-1}" == "0" ]]; then
    TTY_FLAGS=()
fi

_has_arg() { local n="$1"; shift; for a in "$@"; do [[ "$a" == "$n" ]] && return 0; done; return 1; }
_has_arg_with_value() {
    local n="$1"
    shift
    for a in "$@"; do
        [[ "$a" == "$n" || "$a" == "$n="* ]] && return 0
    done
    return 1
}

if [[ -n "${GRIPPER_NAME:-}" ]] \
   && ! _has_arg_with_value "--gripper-name" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
   && ! _has_arg "--no-gripper" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; then
    EXTRA_ARGS+=("--gripper-name" "${GRIPPER_NAME}")
    echo "[run] gripper asset selected from GRIPPER_NAME=${GRIPPER_NAME}"
fi

# Passing a gripper name through this runner is treated as a request to load a gripper.
if _has_arg_with_value "--gripper-name" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
   && ! _has_arg "--gripper" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
   && ! _has_arg "--no-gripper" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; then
    EXTRA_ARGS=("--gripper" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")
    echo "[run] gripper enabled because --gripper-name was provided"
fi

# Enable sim-gripper bridge by default only when a gripper is loaded.
# To disable, set SIM_GRIPPER=0 environment variable.
# If the user explicitly passed --sim-gripper / --no-sim-gripper, respect that.
if [[ "${SIM_GRIPPER:-1}" != "0" ]] \
   && _has_arg "--gripper" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"\
   && ! _has_arg "--sim-gripper" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"\
   && ! _has_arg "--no-sim-gripper" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"\
   && ! _has_arg "--no-gripper" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; then
    EXTRA_ARGS=("--sim-gripper" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}") 
    echo "[run] sim-gripper bridge enabled by default (disable with: SIM_GRIPPER=0)"
fi

echo "[run] Starting Isaac Sim GUI container..."
echo "      Image: ${IMAGE_NAME}"
echo ""

docker run \
    --name "${CONTAINER_NAME}" \
    --rm "${TTY_FLAGS[@]}" \
    --gpus all \
    --network=host \
    --ipc=host \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -e "DISPLAY=${DISPLAY}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${HOME}/.Xauthority:/isaac-sim/.Xauthority:rw" \
    "${SRC_MOUNT[@]}" \
    "${ASSETS_MOUNT[@]}" \
    -v "${ISAAC_HOST_DIR}/cache/main:/isaac-sim/.cache:rw" \
    -v "${ISAAC_HOST_DIR}/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
    -v "${ISAAC_HOST_DIR}/cache/kit:/isaac-sim/kit/cache:rw" \
    -v "${ISAAC_HOST_DIR}/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
    -v "${ISAAC_HOST_DIR}/config:/isaac-sim/.nvidia-omniverse/config:rw" \
    -v "${ISAAC_HOST_DIR}/data:/isaac-sim/.local/share/ov/data:rw" \
    -v "${ISAAC_HOST_DIR}/pkg:/isaac-sim/.local/share/ov/pkg:rw" \
    -u 1234:1234 \
    "${IMAGE_NAME}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

# rby1-sim-isaac

Rainbow Robotics **RBY1** robot simulator powered by **NVIDIA Isaac Sim 5.1.0**.

<img width="2267" height="1275" alt="RBY1 robot in Isaac Sim" src="docs/Isaac_sim_scene.png?v=3" />

## Requirements

* NVIDIA GPU + driver
* [Docker](https://docs.docker.com/engine/install/)
* [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Minimal install check

```bash
nvidia-smi
docker --version
dpkg -l | grep -E 'nvidia-container-toolkit|nvidia-container-runtime'
```

If `nvidia-smi` prints the GPU table and the last command lists an NVIDIA
container package, the NVIDIA driver, Docker, and NVIDIA Container Toolkit are
installed.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/RainbowRobotics/rby1-sim-isaac.git
cd rby1-sim-isaac
```

### 2. Run

| Script | Robot control | Robot UDP bridge | `app_main isaac` |
|--------|---------------|------------------|------------------|
| `docker/run.sh` | Built-in standalone PD trajectory | Disabled by default | Not started |
| `docker/run_sdk.sh` | rby1-sdk / rby1 core commands | Enabled via `--udp` | Started automatically |

#### rby1-sdk UDP integration mode

Connects rby1-sdk with the Isaac Sim container to enable robot control via rby1 core.

```bash
./docker/run_sdk.sh --image 0.10.7-a_v1.2-isaac6.1.0-r1   # Model A v1.2
./docker/run_sdk.sh --image 0.10.7-m_v1.2-isaac6.1.0-r1   # Model M v1.2
./docker/run_sdk.sh --image 0.10.7-a_v1.2-isaac6.1.0-r1 --gripper --gripper-name rb_gripper
HEADLESS=1 ./docker/run_sdk.sh --image 0.10.7-a_v1.2-isaac6.1.0-r1   # no GUI window (faster lockstep)
```

#### Standalone mode

Runs Isaac Sim only. The robot is controlled by the built-in PD trajectory in `RBY1Task`; no `app_main` process is started and the robot command/state UDP bridge is disabled.

```bash
./docker/run.sh --image 0.10.7-a_v1.2-isaac6.1.0-r1   # Model A v1.2
./docker/run.sh --image 0.10.7-m_v1.2-isaac6.1.0-r1   # Model M v1.2
GRIPPER_NAME=rb_gripper ./docker/run.sh --image 0.10.7-m_v1.2-isaac6.1.0-r1
```

The default launch uses the no-gripper base USD. Additional arguments after
`--image` are forwarded to `simulation.py`, including `--gripper`,
`--gripper-name <name>`, and `--no-sim-gripper`. A gripper can also be selected
and enabled with the `GRIPPER_NAME` environment variable.

### Run options

| Environment / argument | Default | Meaning |
|---|---|---|
| `HEADLESS=1` (`--headless`) | off | No GUI windows (Isaac Sim with `--headless`, rby1 core with `RBY1_HEADLESS=1`) and no rendering. Recommended for SDK/lockstep runs: drawing the window can slow the lockstep loop far below real time (seen at 1/40 on a VM X server). `--render-hz` renders anyway. |
| `RBY1_SERVO_PROFILE` | `delayed` | Joint servo model. `delayed`: per-group equivalent command delay (0-16 ms) matched to the robot's step response. `no_delay`: same gains without delay (arm steps ~10 ms faster than the robot). |
| `RBY1_FF_DELAY` | `fitted` | Delay of torque commands (feedforward torque, feedback gain): `fitted` per group from joint impedance steps, `equal` = the target delay, `measured` = the drivers' 14-17 ms (over-delays). |
| `RBY1_UDP_INLINE_RECV` | `1` | Read UDP commands in the simulation thread (no receive-thread handoff per 2 ms tick). `0` = background thread. |
| `RBY1_PHYSICS_DEVICE` (`--physics-device`) | `gpu` | PhysX pipeline, `gpu` or `cpu` (no measured speed difference for this single articulation). |
| `RBY1_FRICTION_MODEL` | `physx` | `physx`: PhysX joint-axis friction. `torque`: the same Coulomb/viscous values as an explicit torque in the PD loop. |
| `RBY1_TIMING_LOG=1` | off | Print the time per lockstep step by part (physics, UDP wait, state, effort, render) every 500 steps. |
| `RBY1_REALTIME` | on with `--udp`, off standalone | Pace the physics steps to wall-clock time (one step per 2 ms); a late step does not wait, so the loop catches up. `0` runs as fast as possible, `1` paces standalone runs too. |
| `CPUSET` | - | Pin the container to CPU cores (`--cpuset-cpus`); give it 16+ cores, Kit runs a task thread per core. |
| `CACHE_CHOWN=1` | off | `run.sh` grants the container user (uid 1234) access to cache folders owned by the invoking user with an ACL (`setfacl`, no sudo); set this to chown them to 1234 with sudo instead. |

The Isaac Sim 5.1 images (`0.10.7-*_v1.2`) crash in the RTX renderer (`librtx.scenedb`) at startup on an RTX 5090
with driver 615.71, also headless; use the Isaac Sim 6.1 images there.

### Joint servo model

The simulated joint servo (`src/rby1_task.py`, `src/rby1_controller.py`, `src/config.py`) reproduces the drivers
identified on an RB-Y1 Model A v1.2: stiffness `kp` as measured (about 2x the motor-datasheet values used before),
a per-group delay of the command (target, feedback gain and feedforward torque) and a D term on the undelayed joint
velocity. The previous D term used a 10-sample moving-average velocity (~9 ms lag), which excited the elbow and
wrist at the identified gains. Motor profiles (`src/motor_profiles.py`) and drive effort limits in the USD
(shoulder 136.5, elbow 78, wrist roll 44 Nm = measured driver saturation) come from the same measurements.

Checked through the rby1 core in SDK mode (`-isaac6.1.0-r1` image, rb_gripper mounted) against the same experiments on
the robot:

| Experiment | Robot | Simulation |
|---|---|---|
| Position step 0.5/1 deg, overshoot / 90 % rise: shoulder | 11.5 % / 31 ms | 11.2 % / 34 ms |
| elbow | 12.3 % / 29 ms | 12.9 % / 31 ms |
| wrist | 24.2 % / 28 ms | 26.7 % / 31 ms |
| wrist roll | 55.2 % / 26 ms | 57.6 % / 30 ms |
| Position step 0.2 deg, lower / upper torso overshoot | 4.6 % / 4.0 % | 3.8 % / 5.2 % |
| Chirp 0.3 deg 0.2-15 Hz, amplitude ratio 1-10 Hz (shoulder, elbow) | 0.99-1.04 | 0.97-1.02 |
| Joint impedance step, overshoot / 50 % delay: shoulder K 900 | 7.8 % / 27.7 ms | 7.7 % / 23 ms |
| elbow K 500 | 14.5 % / 31.7 ms | 12 % / 20 ms |
| wrist K 500, right_arm_4 / right_arm_5 | 30 % / 36 %, 21.5 ms | 29 % / 25 %, 18-20 ms |
| wrist roll K 1000 | 25 % / 58 ms | 25 % / 41 ms |

The robot's structural resonances (torso ~2.5 Hz, arm ~10-12 Hz) are not modeled (rigid links). In joint impedance
mode the robot follows the reference later than the simulation (elbow 31.7 vs 20 ms to 50 %, wrist roll 58 vs 41 ms)
with similar overshoot; a longer torque delay does not close that gap (it only raises the overshoot), so it is
left open.

### Custom gripper assets

The simulator loads gripper assets by folder name, so a different hand can be
attached without modifying the base RBY1 USD. Place the hand USD files and mount
configuration under `assets/gripper/<gripper-name>/`:

<img width="1280" height="900" alt="RBY1 custom gripper attachment structure" src="docs/gripper_attachment_overview.svg?v=2" />

```text
assets/gripper/<gripper-name>/
├── <gripper-name>_left.usd
├── <gripper-name>_right.usd
└── gripper.json
```

`gripper.json` defines which prim in each hand is fixed to the robot wrist:

```json
{
  "left": {
    "body": "link_mount_left",
    "mount_pos": [0.0, 0.0, -0.1261],
    "mount_rot": [0.0, 0.70710678, -0.70710678, 0.0]
  },
  "right": {
    "body": "link_mount_right",
    "mount_pos": [0.0, 0.0, -0.1261],
    "mount_rot": [0.0, 0.70710678, 0.70710678, 0.0]
  }
}
```

For example, a user-provided hand can be tested by placing USD files under
`assets/gripper/custom_gripper/` and launching:

```bash
./docker/run.sh --image 0.10.7-a_v1.2 --gripper --gripper-name custom_gripper
```

User-provided USD files are not included in this repository; only the attachment
pattern and `gripper_servers/custom_gripper.py` adapter skeleton are provided.

## Supported images

The following images are currently supported. Each image automatically loads the
USD asset matching its model (via the `ROBOT_MODEL_NAME` env var), so no extra
flag is required.

| Tag | Model | Isaac Sim | rby1 core (`feature/isaac-sim`) |
|-----|-------|-----------|------------|
| `0.10.7-a_v1.2-isaac6.1.0-r1` | Model A v1.2 | 6.1.0 | df9d04d: hardware velocity estimation, `RBY1_HEADLESS`, lockstep without the core's period wait |
| `0.10.7-m_v1.2-isaac6.1.0-r1` | Model M v1.2 | 6.1.0 | df9d04d |
| `0.10.7-a_v1.2-isaac6.1.0` | Model A v1.2 | 6.1.0 | ad75916 |
| `0.10.7-m_v1.2-isaac6.1.0` | Model M v1.2 | 6.1.0 | ad75916 |
| `0.10.7-a_v1.2` | Model A v1.2 | 5.1.0 | ad75916 |
| `0.10.7-m_v1.2` | Model M v1.2 | 5.1.0 | ad75916 |

The joint servo model of `src/` (torque command delays, real-time pacing) is fitted with the `-r1` images. With the
older images the rby1 core phase-locks the lockstep loop to its own 2 ms clock, SDK clients see stretched state
intervals under load, and joint impedance steps of the wrist and wrist roll overshoot more than on the robot.

## Docker image tag convention

```
rainbowroboticsofficial/rby1-sim-isaac:<version>-<model>_v<model_version>[-isaac<isaac_version>][-r<rebuild>]

e.g.) rainbowroboticsofficial/rby1-sim-isaac:0.10.7-a_v1.2-isaac6.1.0-r1   # Model A v1.2, Isaac Sim 6.1.0, rebuild 1
      rainbowroboticsofficial/rby1-sim-isaac:0.10.7-a_v1.2-isaac6.1.0      # Model A v1.2, Isaac Sim 6.1.0
      rainbowroboticsofficial/rby1-sim-isaac:0.10.7-a_v1.2                 # Model A v1.2, Isaac Sim 5.1.0

-r<rebuild>: same release and Isaac Sim version, rebuilt with a newer rby1 core of the feature/isaac-sim branch.
```

## UDP ports

Since `--network=host` is used, the host and container share the same network namespace.
Ports `5005/5006` are used by rby1-sdk integration mode; ports `5007/5008` are used by the sim gripper bridge.

| Port | Direction | Description |
|------|-----------|-------------|
| `5005/udp` | Isaac Sim → rby1-sdk | RobotState (`ControlState`) |
| `5006/udp` | rby1-sdk → Isaac Sim | RobotCommand (`ControlInput`) |
| `5007/udp` | User code → Isaac Sim | Gripper command |
| `5008/udp` | Isaac Sim → User code | Gripper state |

## Gripper example

When Isaac Sim is running (with `--sim-gripper` enabled by default), you can directly control the gripper from the host.

```bash
cd examples
pip install numpy          # dependency

# In IsaacSim
python3 gripper_example.py --sim

# Real robot
python3 gripper_example.py
```

The `SimDynamixelBus` class in `sim_gripper_bridge.py` provides the same interface as `rby1_sdk.DynamixelBus`. By changing just one line (`--sim`), you can control both the real robot and simulation with the same code.

## License

Source files containing the NVIDIA SPDX header are licensed under Apache-2.0.

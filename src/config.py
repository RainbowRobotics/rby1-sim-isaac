# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Central configuration constants for rby1-sim-isaac.

Common simulation constants (physics rates, UDP ports, PD gains) live here so
they can be tuned in one place. Gripper-specific calibration belongs to each
gripper adapter.
"""
from __future__ import annotations

import os
from typing import Final

import numpy as np


# ============================================================
# Simulation configuration
# ============================================================

PHYSICS_HZ: Final[float] = 500.0
RENDER_HZ: Final[float] = 30.0

PHYSICS_DT: Final[float] = 1.0 / PHYSICS_HZ
RENDER_DT: Final[float] = 1.0 / RENDER_HZ
STAGE_UNITS_IN_METERS: Final[float] = 1.0

# Note: the RBY1 USD asset is selected automatically per robot model.
# See ``RBY1_MODEL_CONFIGS`` and ``_resolve_usd_path`` in ``rby1_task.py``.


# ============================================================
# UDP bridge defaults (rby1-sdk integration)
# ============================================================

# Robot state/command bridge (matches rby1-sdk real_time_control_protocol.h)
ROBOT_STATE_HOST: Final[str] = "127.0.0.1"   # Isaac Sim → C++ rby1-sdk
ROBOT_STATE_PORT: Final[int] = 5005
ROBOT_CMD_PORT: Final[int] = 5006            # C++ rby1-sdk → Isaac Sim

# Gripper bridge defaults (SimDynamixelBus ↔ gripper-specific server)
GRIPPER_HOST: Final[str] = os.environ.get("RBY1_SIM_GRIPPER_HOST", "127.0.0.1")
GRIPPER_CMD_PORT: Final[int] = int(os.environ.get("RBY1_SIM_GRIPPER_CMD_PORT", "5007"))
GRIPPER_STATE_PORT: Final[int] = int(os.environ.get("RBY1_SIM_GRIPPER_STATE_PORT", "5008"))


# ============================================================
# Control configuration (PD controller)
# ============================================================

# C++ kRobotJointNames order (rby1-sdk/include/rby1-sdk/model.h, class A).
# Isaac Sim controls these 24 DOF; gripper joints are handled separately.
CPP_JOINT_NAMES: Final[list[str]] = [
    "right_wheel", "left_wheel",
    "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3",
    "right_arm_4", "right_arm_5", "right_arm_6",
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3",
    "left_arm_4", "left_arm_5", "left_arm_6",
    "head_0", "head_1",
]
N_CTRL: Final[int] = len(CPP_JOINT_NAMES)  # 24

# Per-joint PD gains in CPP_JOINT_NAMES order. Multiplied by 10 when applied
# (kept here as base values so they read identically to the original source).
JOINT_KP_BASE: Final[np.ndarray] = np.array([
    100.0, 100.0,
    391.1, 391.1, 391.1, 57.36, 13.03, 57.36,
    20.86, 20.86, 20.86, 9.127, 3.911, 3.911, 3.911,
    20.86, 20.86, 20.86, 9.127, 3.911, 3.911, 3.911,
    50.0, 50.0,
], dtype=np.float32)

JOINT_KD_BASE: Final[np.ndarray] = np.array([
    1.0, 1.0,
    352.0, 352.0, 352.0, 104.3, 104.3, 104.3,
    52.15, 52.15, 52.15, 20.86, 9.126, 9.126, 9.126,
    52.15, 52.15, 52.15, 20.86, 9.126, 9.126, 9.126,
    2.0, 2.0,
], dtype=np.float32)

# Fixed dt used by the PD torque controller. Kept as a control-domain alias of PHYSICS_DT.
PD_CONTROL_DT: Final[float] = PHYSICS_DT


# ============================================================
# Joint servo model (identified on an RB-Y1 Model A v1.2)
# ============================================================
#
# The real joint drivers act on the current measured position and velocity and on a position target that
# reached them some ticks ago (servo fit: tau = kp * (target[k - d] - q[k]) - kd * qd[k], r^2 0.99 for this
# structure). The simulated servo reproduces that: the target is delayed per motor group, the D term uses the
# undelayed finite-difference velocity. SERVO_PROFILE selects the command delays:
#   "delayed"  - equivalent command delay per group (default; matches the robot's rise time and overshoot)
#   "no_delay" - no command delay (arm steps ~10-13 ms faster than the robot; same overshoot)
SERVO_PROFILE: Final[str] = os.environ.get("RBY1_SERVO_PROFILE", "delayed").strip().lower()

# Equivalent command delay per motor group [PD ticks of PD_CONTROL_DT], fitted together with kd so that the rigid
# model reproduces the robot's step rise time and overshoot (rb_gripper mounted). The drivers' measured
# target-to-torque delay is 14-17 ms, but the encoders are motor-side: the rotor moves before the harmonic drive and
# link do, so the rigid model needs less delay on the large joints (none on the lower torso, whose steps are also
# shaped by a 0.05 rad/s command ramp) and about the full delay on the wrists.
COMMAND_DELAY_TICKS: Final[dict[str, int]] = {
    "lower_torso": 0,
    "upper_torso": 2,
    "arm_shoulder": 2,
    "arm_elbow": 5,
    "arm_wrist": 7,
    "arm_wrist2": 8,
    "head": 7,
    "wheel": 0,
}

# Delay of the feedback gain and feedforward torque per motor group [ticks] (torque commands: joint impedance and
# other feedback-gain-0 modes). Fitted through the rby1 core (lockstep without the core's period wait, real-time
# paced) against the robot's joint impedance steps, overshoot sim vs robot: shoulder 1 (K 900: 7.7 vs 7.8 %; 0 ticks
# 6.7 %), elbow 5 (K 500: 12 vs 14.5 %; 4 and 6 ticks 8 and 18 %), wrist 5 (K 500: 29 vs 30 %; 4 ticks 19 %; the
# other wrist joint 25 vs 36 %), wrist roll 3 (K 1000: 25 vs 25 %; 2 ticks 20 %). The robot reaches 50 % of an
# impedance step later than the simulation (elbow 32 vs 20 ms, wrist roll 58 vs 41 ms); a longer torque delay only
# raises the overshoot. The drivers' measured delay (7-8 ticks for every group, RBY1_FF_DELAY=measured) over-delays.
# RBY1_FF_DELAY=equal uses COMMAND_DELAY_TICKS for all groups.
FEEDFORWARD_DELAY_TICKS: Final[dict[str, int]] = {
    "lower_torso": 0,
    "upper_torso": 2,
    "arm_shoulder": 1,
    "arm_elbow": 5,
    "arm_wrist": 5,
    "arm_wrist2": 3,
    "head": 7,
    "wheel": 0,
}
MEASURED_DRIVER_DELAY_TICKS: Final[dict[str, int]] = {
    "lower_torso": 7,
    "upper_torso": 8,
    "arm_shoulder": 8,
    "arm_elbow": 8,
    "arm_wrist": 8,
    "arm_wrist2": 8,
    "head": 8,
    "wheel": 0,
}
FEEDFORWARD_DELAY_MODE: Final[str] = os.environ.get("RBY1_FF_DELAY", "fitted").strip().lower()

# Samples of the moving average applied to the finite-difference joint velocity that is reported as robot
# state. The PD D term does not use this average (a 10-sample average lags ~9 ms, which turns the D term into
# an excitation at the identified gains); it uses the raw difference (PD_VELOCITY_AVG_WINDOW samples).
STATE_VELOCITY_AVG_WINDOW: Final[int] = 10
PD_VELOCITY_AVG_WINDOW: Final[int] = int(os.environ.get("RBY1_PD_VELOCITY_WINDOW", "1"))

# Joint friction model.
#   "physx"  - PhysX joint-axis friction (staticFrictionEffort / dynamicFrictionEffort / viscous), solved in PGS.
#   "torque" - the same Coulomb + viscous values applied as an explicit torque in the PD loop,
#              tau_f = -(tau_c * tanh(v / FRICTION_VELOCITY_EPS) + b * v); PhysX joint friction is set to 0.
FRICTION_MODEL: Final[str] = os.environ.get("RBY1_FRICTION_MODEL", "physx").strip().lower()
FRICTION_VELOCITY_EPS: Final[float] = float(os.environ.get("RBY1_FRICTION_VELOCITY_EPS", "0.03"))  # [rad/s]


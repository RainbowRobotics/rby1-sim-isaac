# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Joint motor model profiles for the RBY1 robot.

Each profile describes the friction/armature parameters of a joint actuator,
expressed at the joint (output shaft): ``armature`` is the reflected rotor and
gear inertia, ``static_effort``/``dynamic_effort`` the Coulomb friction torque
and ``viscous_nm_s_per_rad`` the viscous friction. Profiles are looked up by
joint base name (stripped of ``left_``/``right_`` prefix).

Values come from joint identification on an RB-Y1 Model A v1.2 (ramp and
coast-down experiments with the rb_gripper mounted; link inertia of the
rby1-sdk URDF removed). Coulomb friction is the mean of that identification and
the per-motor-type identification; the torso armature is the per-motor-type
value. ``arm_wrist2`` (wrist roll) has its own values instead of the wrist's.
"""
from __future__ import annotations

from typing import Final


# Output-shaft inertia and friction parameters.
JOINT_MOTOR_MODEL_PROFILES: Final[dict[str, dict[str, float]]] = {
    "arm_shoulder": {
        "joint_friction": 0.0,
        "static_effort": 3.33369,
        "dynamic_effort": 3.33369,
        "viscous_nm_s_per_rad": 6.92762,
        "armature": 1.00397,
    },
    "arm_elbow": {
        "joint_friction": 0.0,
        "static_effort": 2.1804,
        "dynamic_effort": 2.1804,
        "viscous_nm_s_per_rad": 4.65157,
        "armature": 0.242567,
    },
    "arm_wrist": {
        "joint_friction": 0.0,
        "static_effort": 1.63119,
        "dynamic_effort": 1.63119,
        "viscous_nm_s_per_rad": 3.46067,
        "armature": 0.118914,
    },
    "arm_wrist2": {
        "joint_friction": 0.0,
        "static_effort": 4.03049,
        "dynamic_effort": 4.03049,
        "viscous_nm_s_per_rad": 3.13564,
        "armature": 0.303734,
    },
    "lower_torso": {
        "joint_friction": 0.0,
        "static_effort": 9.11269,
        "dynamic_effort": 9.11269,
        "viscous_nm_s_per_rad": 89.1002,
        "armature": 8.8327,
    },
    "upper_torso": {
        "joint_friction": 0.0,
        "static_effort": 7.97434,
        "dynamic_effort": 7.97434,
        "viscous_nm_s_per_rad": 34.6797,
        "armature": 2.12243,
    },
    "default": {
        "joint_friction": 0.0,
        "static_effort": 2.0,
        "dynamic_effort": 2.0,
        "viscous_nm_s_per_rad": 3.0,
        "armature": 0.5,
    },
}


# Map joint base name → motor profile key. Names not present here use "default".
JOINT_MODEL_TO_MOTOR_MODEL_PROFILE: Final[dict[str, str]] = {
    "arm_0": "arm_shoulder",
    "arm_1": "arm_shoulder",
    "arm_2": "arm_shoulder",
    "arm_3": "arm_elbow",
    "arm_4": "arm_wrist",
    "arm_5": "arm_wrist",
    "arm_6": "arm_wrist2",
    "torso_0": "lower_torso",
    "torso_1": "lower_torso",
    "torso_2": "lower_torso",
    "torso_3": "upper_torso",
    "torso_4": "upper_torso",
    "torso_5": "upper_torso",
    "head_0": "arm_wrist",
    "head_1": "arm_wrist",
}


def strip_side_prefix(joint_name: str) -> str:
    """Remove ``left_``/``right_`` prefix so symmetric joints share a profile."""
    for prefix in ("left_", "right_"):
        if joint_name.startswith(prefix):
            return joint_name.removeprefix(prefix)
    return joint_name


def get_profile_for_joint(joint_name: str) -> dict[str, float]:
    """Look up the motor profile for a fully qualified joint name."""
    base = strip_side_prefix(joint_name)
    profile_key = JOINT_MODEL_TO_MOTOR_MODEL_PROFILE.get(base, "default")
    return JOINT_MOTOR_MODEL_PROFILES[profile_key]

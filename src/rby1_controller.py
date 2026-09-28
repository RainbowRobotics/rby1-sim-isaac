# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PD torque controller used by ``RBY1Task``."""
from __future__ import annotations

import numpy as np

from config import PD_CONTROL_DT


class PDController:
    """Per-joint PD controller with integrated velocity targets for wheel joints.

    For each joint the computed torque is::

        torque = feedback_gain * (kp * pos_err + kd * vel_err)
                 + feedforward_term

    Gains use kp [N*m/rad] and kd [N*m*s/rad]; the D term has no dt factor.

    Normal joints receive absolute position targets. Wheel joints receive
    velocity targets, which are integrated into ``target_pos``.
    """

    def __init__(self, kp: np.ndarray, kd: np.ndarray, num_joints: int):
        self.kp = kp
        self.kd = kd
        self.num_joints = num_joints
        self.integrated_velocity_mode = np.zeros(num_joints, dtype=bool)
        self.target_pos       = np.zeros(num_joints, dtype=float)
        self.target_vel       = np.zeros(num_joints, dtype=float)
        self.feedback_gain    = np.ones(num_joints, dtype=float)
        self.feedforward_term = np.zeros(num_joints, dtype=float)
        self.target_torque    = np.zeros(num_joints, dtype=float)

    def compute_torque(self, curr_pos: np.ndarray, curr_vel: np.ndarray) -> np.ndarray:
        """Compute the per-joint torque from the current joint state arrays."""
        pos_error = self.target_pos - curr_pos
        vel_error = self.target_vel - curr_vel

        self.target_torque = (
            self.feedback_gain
            * (self.kp * pos_error + self.kd * vel_error)
            + self.feedforward_term
        )
        return self.target_torque

    def update_target(self, target: np.ndarray, integrated_velocity_mode: np.ndarray) -> None:
        """Update absolute targets, integrating wheel velocity commands."""
        absolute_position_mode = ~integrated_velocity_mode

        self.target_pos[absolute_position_mode] = target[absolute_position_mode]
        self.target_pos[integrated_velocity_mode] += (target[integrated_velocity_mode] * PD_CONTROL_DT)

        self.target_vel[absolute_position_mode] = 0.0
        self.target_vel[integrated_velocity_mode] = target[integrated_velocity_mode]

    def update_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self.kp = kp
        self.kd = kd

    def update_feedforward_term(self, feedforward_term: np.ndarray) -> None:
        self.feedforward_term = feedforward_term

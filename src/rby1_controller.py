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

    The command is delayed per joint like the robot's joint drivers: they act on
    the current measured position and velocity and on a command that reached them
    some ticks ago. ``delay_ticks`` delays the position target, ``ff_delay_ticks``
    the feedback gain and feedforward torque (defaults to ``delay_ticks``).
    ``0`` disables a delay.
    """

    def __init__(
        self,
        kp: np.ndarray,
        kd: np.ndarray,
        num_joints: int,
        delay_ticks: np.ndarray | None = None,
        ff_delay_ticks: np.ndarray | None = None,
    ):
        self.kp = kp
        self.kd = kd
        self.num_joints = num_joints
        self.set_delay_ticks(np.zeros(num_joints, dtype=int) if delay_ticks is None else delay_ticks, ff_delay_ticks)
        self.integrated_velocity_mode = np.zeros(num_joints, dtype=bool)
        self.target_pos       = np.zeros(num_joints, dtype=float)
        self.target_vel       = np.zeros(num_joints, dtype=float)
        self.feedback_gain    = np.ones(num_joints, dtype=float)
        self.feedforward_term = np.zeros(num_joints, dtype=float)
        self.target_torque    = np.zeros(num_joints, dtype=float)

    def set_delay_ticks(self, delay_ticks: np.ndarray, ff_delay_ticks: np.ndarray | None = None) -> None:
        """Set the per-joint target / feedforward delays [ticks] and clear the delay lines."""
        self.delay_ticks = np.asarray(delay_ticks, dtype=int).copy()
        self.ff_delay_ticks = (
            self.delay_ticks.copy() if ff_delay_ticks is None else np.asarray(ff_delay_ticks, dtype=int).copy()
        )
        depth = int(max(self.delay_ticks.max(initial=0), self.ff_delay_ticks.max(initial=0))) + 1
        self._history = np.zeros((depth, self.num_joints), dtype=float)
        self._fbg_history = np.ones((depth, self.num_joints), dtype=float)
        self._ff_history = np.zeros((depth, self.num_joints), dtype=float)
        self._history_idx = 0
        self._history_filled = 0
        self._joint_range = np.arange(self.num_joints)

    def reset_delay(self, target: np.ndarray | None = None) -> None:
        """Fill the delay line with ``target`` (default: the current target)."""
        self._history[:] = self.target_pos if target is None else target
        self._fbg_history[:] = self.feedback_gain
        self._ff_history[:] = self.feedforward_term
        self._history_filled = len(self._history)

    def _delayed(self, history: np.ndarray, ticks: np.ndarray | None = None) -> np.ndarray:
        rows = (self._history_idx - (self.delay_ticks if ticks is None else ticks)) % len(history)
        return history[rows, self._joint_range]

    def delayed_target(self) -> np.ndarray:
        """The position target each joint's P term sees this tick (``target_pos`` ``delay_ticks`` ago)."""
        if len(self._history) == 1:
            return self.target_pos
        return self._delayed(self._history)

    def _push_target(self) -> None:
        if len(self._history) == 1:
            return
        self._history_idx = (self._history_idx + 1) % len(self._history)
        self._history[self._history_idx] = self.target_pos
        self._fbg_history[self._history_idx] = self.feedback_gain
        self._ff_history[self._history_idx] = self.feedforward_term
        if self._history_filled < len(self._history):
            # until the line has been filled once, older slots repeat the first command
            self._history_filled += 1
            if self._history_filled == 1:
                self._history[:] = self.target_pos
                self._fbg_history[:] = self.feedback_gain
                self._ff_history[:] = self.feedforward_term

    def compute_torque(self, curr_pos: np.ndarray, curr_vel: np.ndarray) -> np.ndarray:
        """Compute the per-joint torque from the current joint state arrays.

        ``curr_vel`` should be an undelayed velocity estimate; the target is delayed per joint. Called once per
        physics tick, so the delay counts ticks even when no new command arrived.
        """
        self._push_target()
        if len(self._history) == 1:
            feedback_gain, feedforward = self.feedback_gain, self.feedforward_term
        else:
            feedback_gain = self._delayed(self._fbg_history, self.ff_delay_ticks)
            feedforward = self._delayed(self._ff_history, self.ff_delay_ticks)
        pos_error = self.delayed_target() - curr_pos
        vel_error = self.target_vel - curr_vel

        self.target_torque = (
            feedback_gain
            * (self.kp * pos_error + self.kd * vel_error)
            + feedforward
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

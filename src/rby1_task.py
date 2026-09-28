# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Sim ``BaseTask`` implementing the RBY1 simulation loop (model M/A)."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, Sdf, UsdPhysics

from config import PD_CONTROL_DT
from gripper_servers import BaseGripperServer
from rby1_controller import PDController
from rby1_robot import RBY1Robot
from rby1_udp_bridge import RBY1UdpBridge


# ============================================================
# Per-model configuration
# ============================================================

@dataclass(frozen=True)
class RBY1ModelConfig:
    model: str
    base_usd_file_name: str
    modular_gripper_supported: bool
    cpp_joint_names: tuple[str, ...]
    joint_kp: tuple[float, ...]
    joint_kd: tuple[float, ...]
    mobility_dof: int
    reference_wheel_target: tuple[float, ...]


_BODY_JOINT_NAMES = (
    "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6",
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6",
    "head_0", "head_1",
)

# Simulation PD gains: kp [N*m/rad], kd [N*m*s/rad].
_BODY_JOINT_KP = (
    199500.1, 99750.05, 99750.05, 29260.1, 29260.1, 29260.1,
    10640.0, 12662.4963, 10640.0, 4655.0, 2625.996, 1982.147, 4365.12,
    10640.0, 10640.0, 10640.0, 4655.0, 2393.5159, 2054.4908, 4365.12,
    2151.3, 2151.3,
)

_BODY_JOINT_KD = (
    359.1, 374.7954, 398.2526, 106.4, 106.4, 106.4,
    53.2, 56.5315, 53.2, 21.28, 6.024, 6.024, 5.9024,
    53.2, 53.2, 53.2, 21.28, 6.024, 6.024, 4.9104,
    10.04, 10.04,
)

RBY1_MODEL_CONFIGS = {
    "a": RBY1ModelConfig(
        model="a",
        base_usd_file_name="model_v_1_2_a.usd",
        modular_gripper_supported=True,
        cpp_joint_names=("right_wheel", "left_wheel", *_BODY_JOINT_NAMES),
        joint_kp=(262.8, 262.8, *_BODY_JOINT_KP),
        joint_kd=(7.510, 7.510, *_BODY_JOINT_KD),
        mobility_dof=2,
        reference_wheel_target=(-1.0, -0.5),
    ),
    "m": RBY1ModelConfig(
        model="m",
        base_usd_file_name="model_v_1_2_m.usd",
        modular_gripper_supported=True,
        cpp_joint_names=("wheel_fr", "wheel_fl", "wheel_rr", "wheel_rl", *_BODY_JOINT_NAMES),
        joint_kp=(262.8, 262.8, 262.8, 262.8, *_BODY_JOINT_KP),
        joint_kd=(7.510, 7.510, 7.510, 7.510, *_BODY_JOINT_KD),
        mobility_dof=4,
        reference_wheel_target=(0.5, -0.5, -0.5, 0.5),
    ),
}

JOINT_VELOCITY_AVG_WINDOW = 10
DEFAULT_GRIPPER_NAME = "rb_gripper"


@dataclass(frozen=True)
class GripperSideConfig:
    body: str
    mount_pos: tuple[float, float, float]
    mount_rot: tuple[float, float, float, float]


@dataclass(frozen=True)
class GripperAssetConfig:
    name: str
    left_usd_path: str
    right_usd_path: str
    sides: dict[str, GripperSideConfig]


GRIPPER_SIDES = ("left", "right")
GRIPPER_MOUNT_BODIES = {
    "left": "link_left_arm_6",
    "right": "link_right_arm_6",
}
GRIPPER_TOOL_JOINTS = {
    "left": "tool_left",
    "right": "tool_right",
}
GRIPPER_ROOT_NAMES = {
    "left": "left_gripper",
    "right": "right_gripper",
}


def normalize_robot_model(robot_model: str) -> str:
    model = str(robot_model).lower()
    if model not in RBY1_MODEL_CONFIGS:
        valid_models = ", ".join(sorted(RBY1_MODEL_CONFIGS))
        raise ValueError(f"Unsupported RBY1 model '{robot_model}'. Valid models: {valid_models}")
    return model


def _resolve_usd_path(usd_file_name: str) -> str:
    """Locate the RBY1 USD asset for the selected model.

    Resolution order:
      1. ``RBY1_USD_PATH`` env var pointing to a directory -> ``<dir>/<model usd>``
         (keeps per-model auto-selection when an external asset dir is supplied)
      2. ``RBY1_USD_PATH`` env var pointing to a file -> used as-is
         (single-asset override; bypasses per-model selection)
      3. ``<repo>/assets/<model usd>``
      4. ``<src/>/<model usd>`` (legacy fallback)
    """
    env_path = os.environ.get("RBY1_USD_PATH")
    if env_path:
        if os.path.isdir(env_path):
            return os.path.join(env_path, usd_file_name)
        return env_path
    src_dir = os.path.dirname(os.path.abspath(__file__))
    repo_assets = os.path.join(os.path.dirname(src_dir), "assets", usd_file_name)
    if os.path.isfile(repo_assets):
        return repo_assets
    return os.path.join(src_dir, usd_file_name)


def _repo_assets_dir() -> str:
    src_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(src_dir), "assets")


def _validate_gripper_name(gripper_name: str) -> str:
    name = str(gripper_name).strip()
    if not name or name in {".", ".."} or os.path.basename(name) != name:
        raise ValueError(f"Invalid gripper name '{gripper_name}'. Use a folder name under assets/gripper/.")
    return name


def _read_float_tuple(raw, length: int, field_name: str, config_path: str) -> tuple[float, ...]:
    if not isinstance(raw, list) or len(raw) != length:
        raise ValueError(f"{config_path}: '{field_name}' must be a list of {length} numbers.")
    try:
        return tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{config_path}: '{field_name}' must contain only numbers.") from exc


def _load_gripper_config(config_path: str) -> dict[str, GripperSideConfig]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    if not isinstance(raw_config, dict):
        raise ValueError(f"{config_path}: gripper config must be a JSON object.")

    sides: dict[str, GripperSideConfig] = {}
    for side in GRIPPER_SIDES:
        raw_side = raw_config.get(side)
        if not isinstance(raw_side, dict):
            raise ValueError(f"{config_path}: missing object field '{side}'.")

        body = raw_side.get("body")
        if not isinstance(body, str) or not body.strip():
            raise ValueError(f"{config_path}: '{side}.body' must be a non-empty string.")

        sides[side] = GripperSideConfig(
            body=body.strip(),
            mount_pos=_read_float_tuple(raw_side.get("mount_pos"), 3, f"{side}.mount_pos", config_path),
            mount_rot=_read_float_tuple(raw_side.get("mount_rot"), 4, f"{side}.mount_rot", config_path),
        )

    return sides


def _resolve_gripper_asset(gripper_name: str) -> GripperAssetConfig:
    name = _validate_gripper_name(gripper_name)
    gripper_dir = os.path.join(_repo_assets_dir(), "gripper", name)
    config_path = os.path.join(gripper_dir, "gripper.json")
    left_usd_path = os.path.join(gripper_dir, f"{name}_left.usd")
    right_usd_path = os.path.join(gripper_dir, f"{name}_right.usd")

    missing_paths = [
        path for path in (config_path, left_usd_path, right_usd_path)
        if not os.path.isfile(path)
    ]
    if missing_paths:
        missing = ", ".join(missing_paths)
        raise FileNotFoundError(f"Missing gripper asset file(s) for '{name}': {missing}")

    return GripperAssetConfig(
        name=name,
        left_usd_path=left_usd_path,
        right_usd_path=right_usd_path,
        sides=_load_gripper_config(config_path),
    )


# ============================================================
# Task definition
# ============================================================

class RBY1Task(BaseTask):
    """Drives the RBY1 articulation from external UDP commands (or a built-in test trajectory)."""

    def __init__(
        self,
        udp_bridge: Optional[RBY1UdpBridge] = None,
        robot_model: str = "m",
        gripper_enabled: bool = False,
        gripper_name: str = DEFAULT_GRIPPER_NAME,
        gripper_server: Optional[BaseGripperServer] = None,
    ):
        super().__init__(name="rby1_task", offset=None)
        self.robot_model = normalize_robot_model(robot_model)
        self.model_config = RBY1_MODEL_CONFIGS[self.robot_model]
        self.robot = None
        self.robot_prim_path = "/World/RBY1"
        self.gripper_enabled = gripper_enabled
        self.gripper_name = _validate_gripper_name(gripper_name) if gripper_enabled else DEFAULT_GRIPPER_NAME
        self.gripper_asset_config: Optional[GripperAssetConfig] = None

        if self.gripper_enabled:
            if not self.model_config.modular_gripper_supported:
                raise RuntimeError(
                    f"[RBY1Task] model '{self.robot_model}' does not support modular gripper attachment."
                )
            self.gripper_asset_config = _resolve_gripper_asset(self.gripper_name)

        self.usd_path = _resolve_usd_path(self.model_config.base_usd_file_name)
        self.step_counter = 0
        self.udp_bridge = udp_bridge          # None -> standalone mode (no UDP)
        self.gripper_server = gripper_server if gripper_enabled else None
        self.force_ui = None                  # optional ExternalForceUI

        self._last_command_seq = 0
        self._new_command_count_since_log = 0
        self._command_wait_timeout: Optional[float] = None
        self._command_wait_miss_count_since_log = 0
        self._state_sent_once = False
        self._command_stream_started = False
        self._last_pre_step_sim_time: Optional[float] = None
        self._prev_diff_position: Optional[np.ndarray] = None
        self._prev_diff_time: Optional[float] = None
        self._joint_velocity_samples: Optional[np.ndarray] = None
        self._joint_velocity_sample_sum: Optional[np.ndarray] = None
        self._joint_velocity_sample_idx = 0
        self._wheel_raw_position_prev: Optional[np.ndarray] = None
        self._wheel_multiturn_position: Optional[np.ndarray] = None
        self._joint_indices_np: Optional[np.ndarray] = None
        self._ready_flags: Optional[np.ndarray] = None
        self._zero_ctrl_state: Optional[np.ndarray] = None
        self._state_q: Optional[np.ndarray] = None
        self._state_dq: Optional[np.ndarray] = None
        self._state_tau: Optional[np.ndarray] = None
        self._efforts_full: Optional[np.ndarray] = None
        self._external_force_error_logged = False
        self._last_log_wall_time = time.monotonic()
        self._last_log_sim_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def set_up_scene(self, scene: Scene) -> None:
        """Spawn the ground plane and the RBY1 robot in the world."""
        super().set_up_scene(scene)

        ground_plane = scene.add_default_ground_plane(
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.0,
        )
        ground_plane._collision_prim.set_contact_offset(0.002)
        ground_plane._collision_prim.set_rest_offset(0.0)

        if self.gripper_asset_config is None:
            gripper_label = "off" if not self.gripper_enabled else "base-usd"
        else:
            gripper_label = self.gripper_asset_config.name
        print(
            f"[RBY1Task] model={self.robot_model}, gripper={gripper_label}, "
            f"usd_path={self.usd_path}"
        )
        add_reference_to_stage(usd_path=self.usd_path, prim_path=self.robot_prim_path)
        if self.gripper_asset_config is not None:
            self._add_gripper_to_robot(scene.stage)

        self.rby_robot = RBY1Robot(
            prim_path=self.robot_prim_path,
            name="RBY1",
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            robot_model=self.robot_model,
        )
        self.robot = scene.add(self.rby_robot)
        self.rby_robot.set_joint_properties(scene.stage)
        self.rby_robot.set_properties(scene.stage)
        # Self-collision left disabled — enabling it with the current USD collision
        # geometry causes joint instability. A USD collision fix is planned.
        self.robot.set_enabled_self_collisions(False)
        self.robot.set_solver_position_iteration_count(4)
        self.robot.set_solver_velocity_iteration_count(1)

    def _add_gripper_to_robot(self, stage) -> None:
        """Compose the selected gripper into the robot articulation and mount it."""
        if self.gripper_asset_config is None:
            return

        print(
            f"[RBY1Task] loading gripper '{self.gripper_asset_config.name}': "
            f"{self.gripper_asset_config.left_usd_path}, "
            f"{self.gripper_asset_config.right_usd_path}"
        )
        side_usd_paths = {
            "left": self.gripper_asset_config.left_usd_path,
            "right": self.gripper_asset_config.right_usd_path,
        }
        for side in GRIPPER_SIDES:
            gripper_root_path = self._gripper_side_root_path(side)
            add_reference_to_stage(
                usd_path=side_usd_paths[side],
                prim_path=gripper_root_path,
            )
            print(f"[RBY1Task] gripper {side} root: {gripper_root_path}")

        for side in GRIPPER_SIDES:
            self._create_gripper_mount_joint(
                stage=stage,
                side=side,
                side_config=self.gripper_asset_config.sides[side],
                gripper_root_path=self._gripper_side_root_path(side),
            )

    def _create_gripper_mount_joint(
        self,
        stage,
        side: str,
        side_config: GripperSideConfig,
        gripper_root_path: str,
    ) -> UsdPhysics.FixedJoint:
        """Create the fixed tool joint between a wrist link and gripper base."""
        joint_path = self._child_prim_path(self.robot_prim_path, f"joints/{GRIPPER_TOOL_JOINTS[side]}")
        body0_path = self._child_prim_path(self.robot_prim_path, GRIPPER_MOUNT_BODIES[side])
        body1_path = self._child_prim_path(gripper_root_path, side_config.body)

        fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        fixed_joint.GetBody0Rel().SetTargets([Sdf.Path(body0_path)])
        fixed_joint.GetBody1Rel().SetTargets([Sdf.Path(body1_path)])

        fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*side_config.mount_pos))
        fixed_joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(side_config.mount_rot[0], Gf.Vec3f(*side_config.mount_rot[1:]))
        )
        fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        fixed_joint.CreateBreakForceAttr().Set(float("inf"))
        fixed_joint.CreateBreakTorqueAttr().Set(float("inf"))

        return fixed_joint

    def _gripper_side_root_path(self, side: str) -> str:
        return self._child_prim_path(self.robot_prim_path, GRIPPER_ROOT_NAMES[side])

    @staticmethod
    def _child_prim_path(root_path: str, child_path: str) -> str:
        if child_path.startswith("/"):
            return child_path
        return f"{root_path.rstrip('/')}/{child_path.lstrip('/')}"

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """Callback invoked before each PhysX step."""
        self.step_counter += 1
        self._last_pre_step_sim_time = simulation_time
        if self._last_log_sim_time is None:
            self._last_log_sim_time = simulation_time

        if self.udp_bridge is None:
            # Standalone mode: drive a built-in test trajectory (C++ DOF order).
            self._update_full_state(simulation_time, update_velocity=True)
            wheel_velocity_mode, target = self._build_reference_target(simulation_time)
            self.pd_controller.update_target(target, wheel_velocity_mode)
        else:
            # UDP mode: consume the latest command (waiting for a fresh one once the stream started).
            if self._state_sent_once and self._command_stream_started:
                cmd, new_command_received, command_seq = self.udp_bridge.wait_for_command_after(
                    self._last_command_seq,
                    timeout=self._command_wait_timeout,
                )
            else:
                cmd, new_command_received, command_seq = self.udp_bridge.get_latest_command_with_status(
                    self._last_command_seq
                )
            if new_command_received and cmd is not None:
                self._command_stream_started = True
                self._new_command_count_since_log += command_seq - self._last_command_seq
                self._last_command_seq = command_seq
                self._apply_udp_command(cmd)
            elif self._state_sent_once and self._command_stream_started:
                self._command_wait_miss_count_since_log += 1

        # External PD torque → max_efforts clipping → expand to the full Isaac Sim DOF buffer.
        self.robot.set_joint_efforts(self._compute_reordered_efforts())

        # Optional sim gripper bridge.
        if self.gripper_enabled and self.gripper_server is not None:
            self._apply_gripper_commands(simulation_time)

        self._apply_external_force_ui()

        if self.step_counter % 500 == 0:
            now_wall = time.monotonic()
            wall_dt = now_wall - self._last_log_wall_time
            sim_dt = simulation_time - self._last_log_sim_time
            realtime_factor = sim_dt / wall_dt if wall_dt > 0.0 else 0.0
            print("wall_dt={:.3f}s, sim_dt={:.3f}s, rtf={:.3f}".format(wall_dt, sim_dt, realtime_factor))
            self._last_log_wall_time = now_wall
            self._last_log_sim_time = simulation_time

    def _apply_external_force_ui(self) -> None:
        """Apply force/torque from the ExternalForceUI to the selected body only."""
        force_ui = getattr(self, "force_ui", None)
        if force_ui is None:
            return

        if hasattr(force_ui, "consume_apply_once"):
            apply_once = force_ui.consume_apply_once()
        else:
            apply_once = bool(getattr(force_ui, "apply_trigger", False))
            force_ui.apply_trigger = False

        continuous_enabled = bool(getattr(force_ui, "continuous_enabled", False))
        if not apply_once and not continuous_enabled:
            return

        body_name = getattr(force_ui, "selected_body", "link_torso_5")
        body_prims = {
            "link_torso_5": self.robot.torso_5,
            "link_left_arm_6": self.robot.link_left_arm_6,
            "link_right_arm_6": self.robot.link_right_arm_6,
        }
        rigid_prim = body_prims.get(body_name)
        if rigid_prim is None:
            self._disable_external_force_ui(force_ui, f"selected body '{body_name}' is not initialized")
            return

        try:
            force = np.asarray(force_ui.forces[body_name], dtype=np.float32).reshape(1, 3)
            torque = np.asarray(force_ui.torques[body_name], dtype=np.float32).reshape(1, 3)
            if not np.any(force) and not np.any(torque):
                return

            rigid_prim_view = rigid_prim._rigid_prim_view
            if hasattr(rigid_prim_view, "is_physics_handle_valid") and not rigid_prim_view.is_physics_handle_valid():
                self._disable_external_force_ui(force_ui, f"selected body '{body_name}' physics handle is not valid")
                return

            rigid_prim_view.apply_forces_and_torques_at_pos(
                forces=force,
                torques=torque,
                is_global=True,
            )
        except Exception as exc:
            self._disable_external_force_ui(force_ui, f"failed for {body_name}: {exc}")

    def _disable_external_force_ui(self, force_ui, reason: str) -> None:
        """Disable continuous force application and avoid repeating the same error."""
        force_ui.continuous_enabled = False
        force_ui.apply_trigger = False
        if hasattr(force_ui, "apply_once_requested"):
            force_ui.apply_once_requested = False
        if hasattr(force_ui, "_update_continuous_status"):
            force_ui._update_continuous_status()
        if not self._external_force_error_logged:
            print(f"[RBY1Task] External force disabled: {reason}")
            self._external_force_error_logged = True

    def post_physics_step(self, physics_dt: float) -> None:
        """After a physics step, send the state to C++ and prepare the next command."""
        if self.udp_bridge is None or self.robot is None or not hasattr(self, "joint_indices"):
            return

        state_time = 0.0
        if self._last_pre_step_sim_time is not None:
            state_time = self._last_pre_step_sim_time + physics_dt

        self._update_udp_state(state_time, update_velocity=True)

        self.udp_bridge.send_state(
            t=state_time,
            is_ready=self._ready_flags,
            position=self._state_q,
            velocity=self._state_dq,
            current=self._zero_ctrl_state,  # Isaac Sim does not expose motor current
            torque=self._state_tau,
        )
        self._state_sent_once = True

    def _apply_udp_command(self, cmd) -> None:
        """Store a received RobotCommand as the PD reference used by the next pre_step."""
        _mode, cmd_target, feedback_gain, feedforward_torque, _finished, _kp, _kd = cmd
        target = np.asarray(cmd_target, dtype=float)
        integrated_velocity_mode = np.asarray(_mode, dtype=bool)
        self.pd_controller.update_target(target, integrated_velocity_mode)
        self.pd_controller.feedback_gain = np.clip(feedback_gain.astype(float) / 10.0, 0.0, 1.0)
        self.pd_controller.feedforward_term = feedforward_torque

    def _apply_gripper_commands(self, simulation_time: float) -> None:
        """Apply gripper-server joint targets and publish current joint state."""
        if not self.gripper_enabled or self.gripper_server is None:
            return
        if not self.gripper_command_indices:
            return

        current_positions = None
        try:
            joints_state = self.robot.get_joints_state()
            positions = np.asarray(joints_state.positions, dtype=np.float64)
            current_positions = positions[self.gripper_command_indices]
        except Exception:
            pass

        targets = self.gripper_server.get_target_positions(current_positions, simulation_time)
        if targets is not None:
            self.robot._articulation_view.set_joint_position_targets(
                np.asarray(targets, dtype=np.float64),
                joint_indices=self.gripper_command_indices,
            )

        if current_positions is not None:
            self.gripper_server.publish_state(current_positions, simulation_time)

    def _build_reference_target(self, simulation_time: float) -> tuple[np.ndarray, np.ndarray]:
        """Generate a built-in test trajectory (C++ model DOF order) for standalone mode."""
        wheel_velocity_mode = np.zeros(len(self.joint_indices), dtype=bool)
        target = np.zeros(len(self.joint_indices))

        wheel_velocity_mode[: self.model_config.mobility_dof] = True
        target[: self.model_config.mobility_dof] = self.model_config.reference_wheel_target
        target[: self.model_config.mobility_dof] *= np.clip(simulation_time * 0.05, a_min=0.0, a_max=1.0)

        if self.gripper_enabled and self.gripper_command_indices:
            if simulation_time < 4.0:
                gripper_target = np.array([[0.0, -0.0]], dtype=np.float32)
                self.robot._articulation_view.set_joint_position_targets(
                    positions=gripper_target,
                    joint_indices=self.gripper_command_indices,
                )
            elif simulation_time < 8.0:
                gripper_target = np.array([[-0.05, -0.05]], dtype=np.float32)
                self.robot._articulation_view.set_joint_position_targets(
                    positions=gripper_target,
                    joint_indices=self.gripper_command_indices,
                )

        return wheel_velocity_mode, target

    def _compute_reordered_efforts(self) -> np.ndarray:
        """Expand the C++ model-DOF torques into the full Isaac Sim DOF buffer."""
        if self._state_q is None or self._state_dq is None or self._efforts_full is None:
            raise RuntimeError("[RBY1Task] joint state buffer not initialised.")

        efforts = self.pd_controller.compute_torque(self._state_q, self._state_dq)
        efforts = np.clip(efforts, -self.max_efforts, self.max_efforts)
        self._efforts_full.fill(0.0)
        self._efforts_full[self.joint_indices] = efforts
        return self._efforts_full

    def coswave(self, t):
        return 0.5 * (1 - np.cos(2 * np.pi / 4.0 * t))

    def sinwave(self, t):
        return np.sin(2 * np.pi / 4.0 * t)

    # ------------------------------------------------------------------
    # State collection
    # ------------------------------------------------------------------

    def _update_full_state(self, sample_time: Optional[float] = None, update_velocity: bool = True) -> None:
        """Update the member state buffers needed by the standalone path."""
        joints_state = self.robot.get_joints_state()
        # joints_tau = self.robot.get_measured_joint_efforts()
        joints_tau = self.robot.get_applied_joint_efforts()
        pos_IB, quat_IB = self.robot.get_world_pose()
        base_lin_vel_b, base_ang_vel_b = self._get_base_velocities_in_body_frame(quat_IB)
        raw_q = np.asarray(joints_state.positions, dtype=float)[self.joint_indices]
        q = self._update_wheel_multiturn_position(raw_q)
        np.copyto(self._state_q, q)
        self._update_joint_velocity(sample_time, update_velocity)

        raw_tau = joints_tau
        if raw_tau is None or np.ndim(raw_tau) == 0:
            self._state_tau.fill(0.0)
        else:
            tau = np.array(raw_tau)[self.joint_indices]
            np.copyto(self._state_tau, tau)

        _ = pos_IB, base_lin_vel_b, base_ang_vel_b

    def _update_udp_state(self, sample_time: Optional[float] = None, update_velocity: bool = True) -> None:
        """Update only the minimal joint state needed by the UDP lockstep path."""
        raw_q = np.asarray(self.robot.get_joint_positions(joint_indices=self._joint_indices_np), dtype=float)
        q = self._update_wheel_multiturn_position(raw_q)
        np.copyto(self._state_q, q)
        self._update_joint_velocity(sample_time, update_velocity)
        # joints_tau = self.robot.get_measured_joint_efforts()
        joints_tau = self.robot.get_applied_joint_efforts()
        raw_tau = joints_tau
        if raw_tau is None or np.ndim(raw_tau) == 0:
            self._state_tau.fill(0.0)
        else:
            tau = np.array(raw_tau)[self.joint_indices]
            np.copyto(self._state_tau, tau)

    def _update_wheel_multiturn_position(self, raw_q: np.ndarray) -> np.ndarray:
        """Return joint positions with wheel joints unwrapped into a multi-turn frame."""
        q = np.asarray(raw_q, dtype=float).copy()
        mobility_dof = self.model_config.mobility_dof
        if mobility_dof <= 0:
            return q

        raw_wheel = q[:mobility_dof]
        if (
            self._wheel_raw_position_prev is None
            or self._wheel_multiturn_position is None
            or self._wheel_raw_position_prev.shape != raw_wheel.shape
            or self._wheel_multiturn_position.shape != raw_wheel.shape
        ):
            self._wheel_raw_position_prev = raw_wheel.copy()
            self._wheel_multiturn_position = raw_wheel.copy()
            q[:mobility_dof] = self._wheel_multiturn_position
            self._sync_wheel_target_to_multiturn_position()
            return q

        raw_delta = raw_wheel - self._wheel_raw_position_prev
        wheel_delta = (raw_delta + np.pi) % (2.0 * np.pi) - np.pi
        self._wheel_multiturn_position += wheel_delta
        np.copyto(self._wheel_raw_position_prev, raw_wheel)
        q[:mobility_dof] = self._wheel_multiturn_position
        return q

    def _sync_wheel_target_to_multiturn_position(self) -> None:
        """Align wheel position references to the first observed multi-turn wheel state."""
        if self._wheel_multiturn_position is None:
            return
        if not hasattr(self, "pd_controller"):
            return
        if self._command_stream_started:
            return

        mobility_dof = self.model_config.mobility_dof
        self.pd_controller.target_pos[:mobility_dof] = self._wheel_multiturn_position

    def _update_joint_velocity(self, sample_time: Optional[float], update_velocity: bool) -> None:
        """Derive fixed-step joint velocity and smooth it with a moving average."""
        if self._state_q is None or self._state_dq is None:
            raise RuntimeError("[RBY1Task] joint state buffer not initialised.")

        if not update_velocity:
            return

        if self._prev_diff_position is None:
            self._prev_diff_position = self._state_q.copy()
            self._prev_diff_time = sample_time
            self._state_dq.fill(0.0)
            if self._joint_velocity_samples is not None:
                self._joint_velocity_samples.fill(0.0)
            if self._joint_velocity_sample_sum is not None:
                self._joint_velocity_sample_sum.fill(0.0)
            self._joint_velocity_sample_idx = 0
            return

        np.subtract(self._state_q, self._prev_diff_position, out=self._state_dq)
        self._state_dq /= PD_CONTROL_DT

        expected_sample_shape = (JOINT_VELOCITY_AVG_WINDOW, self._state_dq.shape[0])
        if (
            self._joint_velocity_samples is None
            or self._joint_velocity_samples.shape != expected_sample_shape
        ):
            self._joint_velocity_samples = np.zeros(expected_sample_shape, dtype=float)
            self._joint_velocity_sample_sum = np.zeros_like(self._state_dq)
            self._joint_velocity_sample_idx = 0
        elif (
            self._joint_velocity_sample_sum is None
            or self._joint_velocity_sample_sum.shape != self._state_dq.shape
        ):
            self._joint_velocity_sample_sum = np.zeros_like(self._state_dq)
            self._joint_velocity_samples.fill(0.0)
            self._joint_velocity_sample_idx = 0

        old_sample = self._joint_velocity_samples[self._joint_velocity_sample_idx]
        self._joint_velocity_sample_sum += self._state_dq - old_sample
        np.copyto(old_sample, self._state_dq)
        self._joint_velocity_sample_idx = (
            self._joint_velocity_sample_idx + 1
        ) % JOINT_VELOCITY_AVG_WINDOW
        np.copyto(self._state_dq, self._joint_velocity_sample_sum / JOINT_VELOCITY_AVG_WINDOW)

        np.copyto(self._prev_diff_position, self._state_q)
        self._prev_diff_time = sample_time

    def _get_base_velocities_in_body_frame(self, quat_IB: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lin_I = self.robot.get_linear_velocity()
        ang_I = self.robot.get_angular_velocity()
        R_BI = quat_to_rot_matrix(quat_IB).transpose()
        return np.matmul(R_BI, lin_I), np.matmul(R_BI, ang_I)

    # ------------------------------------------------------------------
    # BaseTask lifecycle
    # ------------------------------------------------------------------

    def calculate_metrics(self) -> dict:
        raise NotImplementedError

    def is_done(self) -> bool:
        return False

    def cleanup(self) -> None:
        if self._state_q is not None:
            self._state_q.fill(0.0)
        if self._state_dq is not None:
            self._state_dq.fill(0.0)
        if self._joint_velocity_samples is not None:
            self._joint_velocity_samples.fill(0.0)
        if self._joint_velocity_sample_sum is not None:
            self._joint_velocity_sample_sum.fill(0.0)
        self._joint_velocity_sample_idx = 0
        self._wheel_raw_position_prev = None
        self._wheel_multiturn_position = None
        if self._state_tau is not None:
            self._state_tau.fill(0.0)

    def post_reset(self) -> None:
        """Initialise joint index mapping and PD controller after a reset."""
        self.step_counter = 0
        self._last_pre_step_sim_time = None
        self._prev_diff_position = None
        self._prev_diff_time = None
        self._joint_velocity_samples = None
        self._joint_velocity_sample_sum = None
        self._joint_velocity_sample_idx = 0
        self._wheel_raw_position_prev = None
        self._wheel_multiturn_position = None
        self._new_command_count_since_log = 0
        self._command_wait_miss_count_since_log = 0
        self._external_force_error_logged = False
        self._state_sent_once = False
        self._command_stream_started = False
        self._last_log_wall_time = time.monotonic()
        self._last_log_sim_time = None
        if self.udp_bridge is not None:
            _, _, self._last_command_seq = self.udp_bridge.get_latest_command_with_status(self._last_command_seq)

        self.num_joints = self.robot.num_dof
        self.dof_names = self.robot.dof_names
        print(f"[RBY1Task] num_joints={self.num_joints} dof_names={list(self.dof_names)}")

        # Map C++ joint names → Isaac Sim DOF indices.
        self.joint_indices = []
        for name in self.model_config.cpp_joint_names:
            idx = self.robot._articulation_view.get_dof_index(name)
            if idx < 0:
                raise RuntimeError(
                    f"[RBY1Task] C++ joint '{name}' not found in Isaac Sim DOF. "
                    f"dof_names: {list(self.dof_names)}"
                )
            self.joint_indices.append(idx)
        self.isaac_to_cpp_idx = self.joint_indices  # alias kept for UDP-bridge compatibility
        print(f"[RBY1Task] joint_indices: {self.joint_indices}")
        self._joint_indices_np = np.asarray(self.joint_indices, dtype=np.int64)

        n_ctrl = len(self.joint_indices)
        self._ready_flags = np.ones(n_ctrl, dtype=bool)
        self._zero_ctrl_state = np.zeros(n_ctrl, dtype=float)
        self._state_q = np.zeros(n_ctrl, dtype=float)
        self._state_dq = np.zeros(n_ctrl, dtype=float)
        self._joint_velocity_samples = np.zeros((JOINT_VELOCITY_AVG_WINDOW, n_ctrl), dtype=float)
        self._joint_velocity_sample_sum = np.zeros(n_ctrl, dtype=float)
        self._joint_velocity_sample_idx = 0
        self._state_tau = np.zeros(n_ctrl, dtype=float)
        self._efforts_full = np.zeros(self.num_joints, dtype=float)

        self.gripper_command_indices = []
        if self.gripper_enabled:
            if self.gripper_server is None:
                raise RuntimeError("[RBY1Task] gripper is enabled, but no gripper server was configured.")
            self.gripper_command_indices = self.gripper_server.resolve_joint_indices(
                self.robot._articulation_view
            )
            print(f"[RBY1Task] gripper_command_indices: {self.gripper_command_indices}")
        else:
            print("[RBY1Task] gripper disabled; skipping gripper joint lookup.")

        # Effort mode for the controlled joints; position mode for the gripper.
        self.robot._articulation_view.switch_control_mode("effort", joint_indices=self.joint_indices)
        if self.gripper_enabled and self.gripper_command_indices:
            self.robot._articulation_view.switch_control_mode("position", joint_indices=self.gripper_command_indices)

        self.max_efforts = self.robot._articulation_view.get_max_efforts()[0, self.joint_indices]

        self._initialize_pd_controller()
        self._set_default_robot_state()

    def _initialize_pd_controller(self) -> None:
        """Create the PD controller using per-joint gains (C++ model DOF order)."""
        joint_kp = np.asarray(self.model_config.joint_kp, dtype=np.float32)
        joint_kd = np.asarray(self.model_config.joint_kd, dtype=np.float32)
        if len(joint_kp) != len(self.joint_indices) or len(joint_kd) != len(self.joint_indices):
            raise RuntimeError("[RBY1Task] PD gain array length does not match the C++ model joint count.")

        self.pd_controller = PDController(kp=joint_kp, kd=joint_kd, num_joints=len(self.joint_indices))

        # Initialise the controller to a neutral state to avoid a sudden torque step on reset.
        self.pd_controller.integrated_velocity_mode.fill(False)
        self.pd_controller.target_pos.fill(0.0)
        self.pd_controller.target_vel.fill(0.0)
        self.pd_controller.feedback_gain.fill(1.0)
        self.pd_controller.feedforward_term.fill(0.0)
        self.pd_controller.target_torque.fill(0.0)

    def _set_default_robot_state(self) -> None:
        """Reset the robot pose/joint state explicitly right after a reset."""
        self.robot.set_world_pose(position=np.array([0.0, 0.0, 1.31]))
        self.robot.set_joint_positions(np.zeros(self.num_joints, dtype=float))
        self.robot.set_joint_velocities(np.zeros(self.num_joints, dtype=float))
        self.robot.set_joint_efforts(np.zeros(self.num_joints, dtype=float))

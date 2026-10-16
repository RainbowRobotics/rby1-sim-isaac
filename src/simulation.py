# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Top-level Isaac Sim entry point for the RBY1 simulator."""
from __future__ import annotations

import os
import time

import numpy as np

from isaacsim import SimulationApp

from config import (
    GRIPPER_CMD_PORT,
    GRIPPER_HOST,
    GRIPPER_STATE_PORT,
    PHYSICS_DT,
    RENDER_DT,
    ROBOT_CMD_PORT,
    ROBOT_STATE_HOST,
    ROBOT_STATE_PORT,
    STAGE_UNITS_IN_METERS,
)


def _read_nonnegative_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return max(0.0, float(value))
    except ValueError:
        print(f"[Simulation] Invalid {name}={value!r}; using {default}.")
        return default


class Simulation:
    """Owns the Isaac Sim ``World`` and the optional UDP bridges."""

    def __init__(
        self,
        sim_app,
        use_udp: bool = False,
        udp_state_send_ip: str = ROBOT_STATE_HOST,
        udp_state_send_port: int = ROBOT_STATE_PORT,
        udp_cmd_recv_port: int = ROBOT_CMD_PORT,
        use_sim_gripper: bool = False,
        sim_gripper_cmd_port: int = GRIPPER_CMD_PORT,
        sim_gripper_state_host: str = GRIPPER_HOST,
        sim_gripper_state_port: int = GRIPPER_STATE_PORT,
        robot_model: str = "m",
        gripper_enabled: bool = False,
        gripper_name: str = "rb_gripper",
        headless: bool = False,
        physics_device: str = "gpu",
        render_hz: float | None = None,
    ):
        # Imports kept lazy: many of these depend on SimulationApp being initialised.
        from isaacsim.core.api import World

        from rby1_task import RBY1Task, normalize_robot_model
        from rby1_udp_bridge import RBY1UdpBridge

        self.simulator = sim_app
        self.robot_model = normalize_robot_model(robot_model)
        self.gripper_name = gripper_name
        self._physics_dt = PHYSICS_DT
        # Headless runs do not render at all unless render_hz asks for it (rendering costs lockstep time).
        self._headless = headless
        if render_hz is None:
            render_hz = 0.0 if headless else 1.0 / RENDER_DT
        self._rendering_dt = (1.0 / render_hz) if render_hz > 0.0 else None
        # RBY1_TIMING_LOG=1: print where each lockstep step spends its time, every 500 steps.
        self._timing_log = os.environ.get("RBY1_TIMING_LOG", "0") not in ("", "0")
        # Real-time pacing: one physics step per PHYSICS_DT of wall-clock time. A step that ends early waits for its
        # slot; a late step does not wait, so the loop catches up instead of losing time. Default on in rby1-sdk mode
        # (SDK clients run in wall-clock time), off standalone. RBY1_REALTIME=1/0 forces it.
        realtime_env = os.environ.get("RBY1_REALTIME", "").strip()
        self._realtime = (realtime_env not in ("0", "false")) if realtime_env else use_udp
        self._next_step_wall_time: float | None = None
        self._timing = {"step": 0.0, "post": 0.0, "render": 0.0, "steps": 0}
        self._last_render_wall_time = 0.0
        self._grpc_reset_settle_seconds = _read_nonnegative_float_env(
            "RBY1_GRPC_RESET_SETTLE_SECONDS",
            0.5,
        )

        # Robot UDP bridge is enabled only in rby1-sdk integration mode (--udp).
        # In standalone mode this stays None, so RBY1Task uses its built-in trajectory.
        self.udp_bridge = None
        if use_udp:
            self.udp_bridge = RBY1UdpBridge(
                state_send_ip=udp_state_send_ip,
                state_send_port=udp_state_send_port,
                cmd_recv_port=udp_cmd_recv_port,
            )

        # Optional gRPC client to the C++ app_main (used to reset the C++ ControlManager).
        self.grpc_robot = None
        if use_udp:
            try:
                from rby1_sdk import create_robot
                robot = create_robot("127.0.0.1:50051", self.robot_model)
                if robot.connect():
                    self.grpc_robot = robot
                    print("[Simulation] Connected to C++ app_main gRPC server at 127.0.0.1:50051")
                else:
                    print("[Simulation] gRPC client created, but failed to connect to 127.0.0.1:50051 "
                          "(server may not be running)")
            except Exception as exc:
                print(f"[Simulation] gRPC client unavailable: {exc}")

        self.gripper_enabled = gripper_enabled

        # Gripper command/state server (optional).
        self.gripper_server = None
        if self.gripper_enabled:
            from gripper_servers import create_gripper_server
            self.gripper_server = create_gripper_server(
                self.gripper_name,
                cmd_bind_port=sim_gripper_cmd_port,
                state_send_host=sim_gripper_state_host,
                state_send_port=sim_gripper_state_port,
            )
            if use_sim_gripper:
                self.gripper_server.start()

        self.world = World(
            physics_dt=PHYSICS_DT,
            rendering_dt=RENDER_DT,
            stage_units_in_meters=STAGE_UNITS_IN_METERS,
            sim_params={
                "solver_type": 0,
                # One articulation: the CPU pipeline avoids a GPU->CPU sync on every state read in the 2 ms loop.
                "use_gpu_pipeline": physics_device == "gpu",
                "enable_scene_query_support": False,
                "use_fabric": True,
            },
            **({"device": "cpu"} if physics_device == "cpu" else {}),
        )
        self.task = RBY1Task(
            udp_bridge=self.udp_bridge,
            robot_model=self.robot_model,
            gripper_enabled=self.gripper_enabled,
            gripper_name=self.gripper_name,
            gripper_server=self.gripper_server,
        )
        self.world.add_task(self.task)
        self.robot = None
        self.reset_needed = False

        self._carb_input = None
        self._input = None
        self._keyboard = None
        self._keyboard_sub = None
        try:
            self._register_keyboard_callbacks()
        except Exception as exc:  # headless runs may have no app window/keyboard
            print(f"[Simulation] keyboard callbacks unavailable: {exc}")

        # GUI external-force controller (best-effort — skipped if the UI is unavailable).
        self.force_ui = None
        try:
            self.force_ui = ExternalForceUI()
            self.task.force_ui = self.force_ui
        except Exception as exc:
            print(f"[Simulation] ExternalForceUI unavailable: {exc}")

    # ------------------------------------------------------------------
    # Keyboard handling (Backspace = request reset)
    # ------------------------------------------------------------------

    def _register_keyboard_callbacks(self) -> None:
        import carb.input as carb_input
        import omni.appwindow

        self._carb_input = carb_input
        self._input = carb_input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

    def _cleanup_keyboard_callbacks(self) -> None:
        if self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None
        self._keyboard = None
        self._input = None

    def _on_keyboard_event(self, event) -> bool:
        if event.type != self._carb_input.KeyboardEventType.KEY_PRESS:
            return False
        if event.input == self._carb_input.KeyboardInput.BACKSPACE:
            self.reset_needed = True
            return True
        return False

    # ------------------------------------------------------------------
    # World lifecycle
    # ------------------------------------------------------------------

    def initial_reset(self) -> None:
        self._next_step_wall_time = None
        self.world.reset()
        self.world.initialize_physics()
        self.robot = self.world.scene.get_object("RBY1")
        self._last_render_wall_time = 0.0
        self.world.pause()

    def reset_simulation(self) -> None:
        if self.force_ui is not None:
            self.force_ui._reset_all()

        # Disable the C++ ControlManager and power off via gRPC (MuJoCo-style reset).
        if self.grpc_robot is not None:
            try:
                if not self.grpc_robot.is_connected():
                    self.grpc_robot.connect()
                if self.grpc_robot.is_connected():
                    self.grpc_robot.disable_control_manager()
                    self.grpc_robot.power_off(".*")
                    print("[Simulation] gRPC: Control Manager disabled and powered off.")
                    if self._grpc_reset_settle_seconds > 0.0:
                        print(
                            "[Simulation] Waiting "
                            f"{self._grpc_reset_settle_seconds:.3f}s for gRPC reset to settle."
                        )
                        time.sleep(self._grpc_reset_settle_seconds)
                else:
                    print("[Simulation] gRPC robot not connected. Skipping C++ reset commands.")
            except Exception as exc:
                print(f"[Simulation] gRPC request failed during reset: {exc}")

        self.world.reset()
        self.world.initialize_physics()
        self.robot = self.world.scene.get_object("RBY1")
        self._last_render_wall_time = 0.0
        self.reset_needed = False
        self._next_step_wall_time = None
        self.world.pause()
        print("[Simulation] Reset")

    def simulation_loop(self) -> None:
        self.world.play()
        try:
            while self.simulator.is_running():
                if self.reset_needed:
                    self.reset_simulation()
                self.simulation_step()
        finally:
            self._cleanup_keyboard_callbacks()
            if self.udp_bridge is not None:
                self.udp_bridge.close()
            if self.gripper_server is not None:
                self.gripper_server.close()
            self.simulator.close()

    def simulation_step(self) -> None:
        """Run one physics step and decouple rendering from the physics rate."""
        if self.world.is_playing():
            # Isaac lockstep: World.step(render=False) drives one physics step and the task pre_step.
            if not self._timing_log:
                self.world.step(render=False)
                self.task.post_physics_step(self._physics_dt)
                self._render_if_due()
                self._pace_realtime()
                return
            t0 = time.perf_counter()
            self.world.step(render=False)
            t1 = time.perf_counter()
            self.task.post_physics_step(self._physics_dt)
            t2 = time.perf_counter()
            self._render_if_due()
            t3 = time.perf_counter()
            timing = self._timing
            timing["step"] += t1 - t0
            timing["post"] += t2 - t1
            timing["render"] += t3 - t2
            timing["steps"] += 1
            if timing["steps"] == 500:
                per = {k: 1e3 * v / 500 for k, v in timing.items() if k != "steps"}
                task_timing = self.task.consume_timing() if hasattr(self.task, "consume_timing") else {}
                print(
                    "[timing] ms/step: world.step {step:.3f} (of which ".format(**per)
                    + ", ".join(f"{k} {v:.3f}" for k, v in task_timing.items())
                    + "), post {post:.3f}, render {render:.3f}".format(**per)
                )
                for k in timing:
                    timing[k] = 0.0 if k != "steps" else 0
            self._pace_realtime()
        else:
            self._next_step_wall_time = None
            if self._rendering_dt is None:
                time.sleep(0.01)  # paused and headless: nothing to draw
            self._render_if_due()

    # Wall-clock lag after which the pacing schedule restarts instead of catching up (e.g. after a stall).
    _MAX_PACING_LAG = 0.05

    def _pace_realtime(self) -> None:
        """Hold the step rate at real time with catch-up (see RBY1_REALTIME)."""
        if not self._realtime:
            return
        now = time.perf_counter()
        if self._next_step_wall_time is None or now - self._next_step_wall_time > self._MAX_PACING_LAG:
            self._next_step_wall_time = now
        self._next_step_wall_time += self._physics_dt
        remaining = self._next_step_wall_time - time.perf_counter()
        if remaining > 0.0003:
            time.sleep(remaining - 0.0002)
        while time.perf_counter() < self._next_step_wall_time:
            pass

    def _render_if_due(self) -> None:
        """Render at most once per ``rendering_dt`` (wall-clock based); never when headless without render_hz."""
        if self._rendering_dt is None:
            return
        now = time.monotonic()
        if now - self._last_render_wall_time >= self._rendering_dt:
            self.world.render()
            self._last_render_wall_time = now


class ExternalForceUI:
    """omni.ui panel to apply external force/torque to selected rigid bodies."""

    def __init__(self):
        import omni.ui as ui

        self.ui = ui
        self.window = ui.Window("External Force/Torque Controller", width=420, height=470)

        self.body_names = ["link_torso_5", "link_left_arm_6", "link_right_arm_6"]
        self.selected_body = "link_torso_5"

        self.forces = {
            "link_torso_5": np.zeros(3, dtype=float),
            "link_left_arm_6": np.zeros(3, dtype=float),
            "link_right_arm_6": np.zeros(3, dtype=float),
        }
        self.torques = {
            "link_torso_5": np.zeros(3, dtype=float),
            "link_left_arm_6": np.zeros(3, dtype=float),
            "link_right_arm_6": np.zeros(3, dtype=float),
        }

        self.apply_once_requested = False
        self.apply_trigger = False
        self.continuous_enabled = False

        self.force_fields = []
        self.torque_fields = []
        self.continuous_status_label = None
        self._build_ui()

    def _build_ui(self):
        ui = self.ui
        with self.window.frame:
            with ui.VStack(spacing=8):
                ui.Spacer(height=5)
                with ui.HStack(height=0):
                    ui.Label("Target Body:", width=100, style={"font_weight": "bold"})
                    combo = ui.ComboBox(0, *self.body_names)
                    combo.model.get_item_value_model().add_value_changed_fn(self._on_body_index_changed)

                ui.Spacer(height=5)
                ui.Label("=== Apply Forces (Newton) ===",
                         style={"color": 0xFF8080FF, "font_weight": "bold"})

                for i, axis in enumerate(["X", "Y", "Z"]):
                    with ui.HStack(height=0):
                        ui.Label(f"Force {axis}:", width=80)
                        field = ui.FloatField()
                        field.model.add_value_changed_fn(
                            lambda m, idx=i: self._on_force_changed(idx, m.as_float)
                        )
                        self.force_fields.append(field)

                ui.Spacer(height=5)
                ui.Label("=== Apply Torques (N-m) ===",
                         style={"color": 0xFF80FF80, "font_weight": "bold"})

                for i, axis in enumerate(["X", "Y", "Z"]):
                    with ui.HStack(height=0):
                        ui.Label(f"Torque {axis}:", width=80)
                        field = ui.FloatField()
                        field.model.add_value_changed_fn(
                            lambda m, idx=i: self._on_torque_changed(idx, m.as_float)
                        )
                        self.torque_fields.append(field)

                ui.Spacer(height=10)

                with ui.VStack(spacing=6):
                    ui.Button("APPLY ONCE", clicked_fn=self._trigger_apply_once,
                              style={"background_color": 0xFF4444FF, "color": 0xFFFFFFFF,
                                     "font_weight": "bold"})
                    with ui.HStack(height=0, spacing=10):
                        ui.Button("Start Continuous", clicked_fn=self._start_continuous)
                        ui.Button("Stop Continuous", clicked_fn=self._stop_continuous)
                    self.continuous_status_label = ui.Label("Continuous: OFF",
                                                            style={"font_weight": "bold"})
                    with ui.HStack(height=0, spacing=10):
                        ui.Button("Zero Selected Body", clicked_fn=self._reset_selected)
                        ui.Button("Zero All Bodies", clicked_fn=self._reset_all)

    def _trigger_apply_once(self):
        self.apply_once_requested = True
        self.apply_trigger = True

    def consume_apply_once(self):
        should_apply = self.apply_once_requested or self.apply_trigger
        self.apply_once_requested = False
        self.apply_trigger = False
        return should_apply

    def _start_continuous(self):
        self.continuous_enabled = True
        self._update_continuous_status()

    def _stop_continuous(self):
        self.continuous_enabled = False
        self._update_continuous_status()

    def _update_continuous_status(self):
        if self.continuous_status_label is not None:
            state = "ON" if self.continuous_enabled else "OFF"
            try:
                self.continuous_status_label.text = f"Continuous: {state}"
            except Exception:
                pass

    def _on_body_index_changed(self, model):
        idx = model.as_int
        self.selected_body = self.body_names[idx]
        self._sync_selected_fields()

    def _sync_selected_fields(self):
        for i in range(3):
            self.force_fields[i].model.set_value(float(self.forces[self.selected_body][i]))
            self.torque_fields[i].model.set_value(float(self.torques[self.selected_body][i]))

    def _on_force_changed(self, axis_idx, value):
        self.forces[self.selected_body][axis_idx] = value

    def _on_torque_changed(self, axis_idx, value):
        self.torques[self.selected_body][axis_idx] = value

    def _reset_selected(self):
        self.forces[self.selected_body].fill(0.0)
        self.torques[self.selected_body].fill(0.0)
        self.apply_once_requested = False
        self.apply_trigger = False
        self._sync_selected_fields()

    def _reset_all(self):
        for body in self.forces:
            self.forces[body].fill(0.0)
            self.torques[body].fill(0.0)
        self._sync_selected_fields()
        self.apply_once_requested = False
        self.apply_trigger = False
        self.continuous_enabled = False
        self._update_continuous_status()


def _build_argparser():
    import argparse

    p = argparse.ArgumentParser(description="RBY1 Isaac Sim Simulation")
    # Default model comes from the ROBOT_MODEL_NAME env var that the per-model
    # image sets (A/M -> a/m), so the a/m images each load their own USD asset
    # without requiring an explicit --model flag. Can still be overridden.
    default_model = os.environ.get("ROBOT_MODEL_NAME", "m").strip().lower() or "m"
    p.add_argument("--model", type=str.lower, choices=("m", "a"), default=default_model,
                   help="RBY1 model selection: m or a (default from ROBOT_MODEL_NAME env)")
    p.add_argument("--udp", action="store_true",
               help="Enable rby1-sdk UDP bridge")
    p.add_argument("--state-ip", default=ROBOT_STATE_HOST,
                   help="Target IP for RobotState transmission")
    p.add_argument("--state-port", type=int, default=ROBOT_STATE_PORT,
                   help="Target port for RobotState transmission")
    p.add_argument("--cmd-port", type=int, default=ROBOT_CMD_PORT,
                   help="Port for receiving RobotCommand")
    p.add_argument("--gripper", dest="gripper_enabled", action="store_true", default=False,
                   help="Load the selected gripper asset and allow gripper control")
    p.add_argument("--no-gripper", dest="gripper_enabled", action="store_false",
                   help="Load only the no-gripper robot USD and disable all gripper control (default)")
    p.add_argument("--gripper-name", default="rb_gripper",
                   help="Gripper folder name under assets/gripper (default: rb_gripper)")
    p.add_argument("--sim-gripper", action="store_true",
                   help="Enable receiving gripper commands from SimDynamixelBus")
    p.add_argument("--no-sim-gripper", action="store_true",
                   help="Ignore --sim-gripper (used to override the docker default)")
    p.add_argument("--gripper-cmd-port", type=int, default=GRIPPER_CMD_PORT,
                   help="Port for receiving gripper commands")
    p.add_argument("--gripper-state-ip", default=GRIPPER_HOST,
                   help="IP for sending gripper state")
    p.add_argument("--headless", action="store_true",
                   help="Run without the GUI window (no rendering unless --render-hz is given)")
    p.add_argument("--physics-device", choices=("gpu", "cpu"), default=os.environ.get("RBY1_PHYSICS_DEVICE", "gpu"),
                   help="PhysX pipeline. 'cpu' is faster for this single articulation in lockstep mode")
    p.add_argument("--render-hz", type=float, default=None,
                   help="Render rate [Hz] (default 30 with the GUI, 0 = never when headless)")
    p.add_argument("--gripper-state-port", type=int, default=GRIPPER_STATE_PORT,
                   help="Port for sending gripper state")
    return p


def main() -> None:
    args, _unknown = _build_argparser().parse_known_args()

    # --no-sim-gripper overrides --sim-gripper (used by docker default args).
    gripper_enabled = args.gripper_enabled
    sim_gripper_enabled = gripper_enabled and args.sim_gripper and not args.no_sim_gripper
    if args.sim_gripper and not gripper_enabled:
        print("[Simulation] --sim-gripper ignored because --no-gripper is set.")

    simulation_app = SimulationApp({"headless": args.headless})

    simulation = Simulation(
        simulation_app,
        use_udp=args.udp,
        udp_state_send_ip=args.state_ip,
        udp_state_send_port=args.state_port,
        udp_cmd_recv_port=args.cmd_port,
        gripper_enabled=gripper_enabled,
        use_sim_gripper=sim_gripper_enabled,
        sim_gripper_cmd_port=args.gripper_cmd_port,
        sim_gripper_state_host=args.gripper_state_ip,
        sim_gripper_state_port=args.gripper_state_port,
        robot_model=args.model,
        gripper_name=args.gripper_name,
        headless=args.headless,
        physics_device=args.physics_device,
        render_hz=args.render_hz,
    )
    simulation.initial_reset()

    if args.udp:
        mode = f"model={args.model}, UDP(-> {args.state_ip}:{args.state_port}, <- :{args.cmd_port})"
    else:
        mode = f"model={args.model}, standalone"
    mode += f", Gripper({args.gripper_name if gripper_enabled else 'off'})"
    if sim_gripper_enabled:
        mode += (f" + SimGripper(cmd←:{args.gripper_cmd_port}, "
                 f"state→{args.gripper_state_ip}:{args.gripper_state_port})")
    elif not gripper_enabled:
        mode += " + SimGripper(off: no gripper)"
    elif args.no_sim_gripper:
        mode += " + SimGripper(off)"
    print(f"[Simulation] Starting... [{mode}]")

    simulation.simulation_loop()


if __name__ == "__main__":
    main()

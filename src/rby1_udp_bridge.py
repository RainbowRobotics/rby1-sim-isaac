# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UDP transport for the rby1-sdk real-time control protocol.

Wraps :mod:`udp_protocol` with socket I/O and a background receive thread.

The wire-format codec lives in the :mod:`udp_protocol` module, which is shipped
as a pre-compiled extension (``udp_protocol.*.so``) and made importable via
``PYTHONPATH`` at runtime. Its plaintext Python source is intentionally not part
of this repository, so the codec is imported here rather than inlined.
"""
from __future__ import annotations

import os
import socket
import select
import threading
import time
from typing import Optional, Tuple

import numpy as np

from udp_protocol import (
    build_robot_state_packet,
    parse_robot_command_packet,
)


# ==========================================================
# UDP bridge class
# ==========================================================

class RBY1UdpBridge:
    """
    UDP bridge between Isaac Sim and C++ rby1 Core.

    Python (Isaac Sim) side:
      - sends RobotState packets to ``state_send_addr``
      - receives RobotCommand packets on ``cmd_recv_port`` (background thread)
    """

    def __init__(
        self,
        state_send_ip:   str = "127.0.0.1",  # IP the C++ IsaacSimulation listens on
        state_send_port: int = 5005,          # port the C++ IsaacSimulation listens on
        cmd_recv_port:   int = 5006,          # port Python listens on (C++ sends commands here)
        inline_recv: Optional[bool] = None,
    ):
        self.state_send_addr = (state_send_ip, state_send_port)
        self.cmd_recv_port   = cmd_recv_port

        # send socket
        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # recv socket (non-blocking)
        self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_sock.bind(("0.0.0.0", cmd_recv_port))
        self._recv_sock.setblocking(False)

        # latest received command (thread-safe)
        self._lock    = threading.Lock()
        self._command_cv = threading.Condition(self._lock)
        self._command: Optional[Tuple] = None  # (mode, target, fbg, fft, finished)
        self._command_seq = 0
        self._command_recv_time = 0.0
        self._command_ready_time = 0.0
        self._timing_wait_count = 0
        self._timing_immediate_count = 0
        self._timing_wait_total_sum = 0.0
        self._timing_wait_total_max = 0.0
        self._timing_recv_to_ready_sum = 0.0
        self._timing_recv_to_ready_max = 0.0
        self._timing_ready_to_return_sum = 0.0
        self._timing_ready_to_return_max = 0.0
        self._timing_recv_to_return_sum = 0.0
        self._timing_recv_to_return_max = 0.0

        # Inline receive (default): the simulation thread reads the socket itself while it waits for the next
        # command, which removes the receive-thread -> condition-variable handoff from every lockstep tick.
        # RBY1_UDP_INLINE_RECV=0 restores the background receive thread.
        if inline_recv is None:
            inline_recv = os.environ.get("RBY1_UDP_INLINE_RECV", "1") != "0"
        self._inline_recv = inline_recv
        self._recv_buf = bytearray(4096)
        self._running = True
        self._recv_thread = None
        if not self._inline_recv:
            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._recv_thread.start()

        print(f"[RBY1UdpBridge] State send -> {state_send_ip}:{state_send_port}")
        print(f"[RBY1UdpBridge] Command recv <- 0.0.0.0:{cmd_recv_port}")

    def send_state(
        self,
        t:        float,
        is_ready: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        current:  np.ndarray,
        torque:   np.ndarray,
    ) -> None:
        """Send a RobotState packet to C++."""
        pkt = build_robot_state_packet(t, is_ready, position, velocity, current, torque)
        try:
            self._send_sock.sendto(pkt, self.state_send_addr)
        except OSError:
            pass

    def get_latest_command(self) -> Optional[Tuple]:
        """Return the most recently received RobotCommand, or None."""
        with self._lock:
            return self._command

    def _drain_socket(self, timeout: Optional[float]) -> bool:
        """Inline mode: read every queued command packet, waiting up to ``timeout`` for the first one.

        Returns True if at least one command was parsed. ``timeout=None`` blocks.
        """
        got = False
        wait = timeout
        while True:
            readable, _, _ = select.select([self._recv_sock], [], [], wait)
            if not readable:
                return got
            try:
                n, _ = self._recv_sock.recvfrom_into(self._recv_buf)
            except BlockingIOError:
                return got
            except OSError:
                self._running = False
                return got
            recv_time = time.perf_counter()
            result = parse_robot_command_packet(bytes(self._recv_buf[:n]))
            if result is not None:
                with self._lock:
                    self._command = result
                    self._command_seq += 1
                    self._command_recv_time = recv_time
                    self._command_ready_time = recv_time
                got = True
            wait = 0.0  # keep draining whatever is already queued, then return

    def get_latest_command_with_status(self, last_seq: int = 0) -> Tuple[Optional[Tuple], bool, int]:
        """Return the latest command plus whether a new one arrived after ``last_seq``."""
        if self._inline_recv and self._running:
            self._drain_socket(0.0)
        with self._lock:
            return self._command, self._command is not None and self._command_seq != last_seq, self._command_seq

    def wait_for_command_after(self, last_seq: int, timeout: Optional[float] = None) -> Tuple[Optional[Tuple], bool, int]:
        """Block until a command newer than ``last_seq`` arrives. ``timeout=None`` waits indefinitely."""
        deadline = None if timeout is None else time.monotonic() + timeout
        wait_start = time.perf_counter()
        if self._inline_recv:
            self._drain_socket(0.0)
            while self._running and self._command_seq == last_seq:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    break
                # wake up at least every 100 ms so close() is noticed
                self._drain_socket(0.1 if remaining is None else min(remaining, 0.1))
        with self._command_cv:
            waited = self._command_seq == last_seq
            while self._running and self._command_seq == last_seq:
                if deadline is None:
                    self._command_cv.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    self._command_cv.wait(timeout=remaining)
            wait_done = time.perf_counter()
            new_command = self._command is not None and self._command_seq != last_seq
            if new_command:
                if waited:
                    wait_total = wait_done - wait_start
                    recv_to_ready = self._command_ready_time - self._command_recv_time
                    ready_to_return = wait_done - self._command_ready_time
                    recv_to_return = wait_done - self._command_recv_time
                    self._timing_wait_count += 1
                    self._timing_wait_total_sum += wait_total
                    self._timing_wait_total_max = max(self._timing_wait_total_max, wait_total)
                    self._timing_recv_to_ready_sum += recv_to_ready
                    self._timing_recv_to_ready_max = max(self._timing_recv_to_ready_max, recv_to_ready)
                    self._timing_ready_to_return_sum += ready_to_return
                    self._timing_ready_to_return_max = max(self._timing_ready_to_return_max, ready_to_return)
                    self._timing_recv_to_return_sum += recv_to_return
                    self._timing_recv_to_return_max = max(self._timing_recv_to_return_max, recv_to_return)
                else:
                    self._timing_immediate_count += 1
            return self._command, new_command, self._command_seq

    def consume_timing_stats(self) -> dict:
        """Return accumulated wait_for_command_after timing stats (ms) and reset them."""
        with self._lock:
            count = self._timing_wait_count
            immediate = self._timing_immediate_count

            def avg_ms(value: float) -> float:
                return value / count * 1000.0 if count > 0 else 0.0

            stats = {
                "wait_count": count,
                "immediate_count": immediate,
                "wait_total_avg_ms": avg_ms(self._timing_wait_total_sum),
                "wait_total_max_ms": self._timing_wait_total_max * 1000.0,
                "recv_to_ready_avg_ms": avg_ms(self._timing_recv_to_ready_sum),
                "recv_to_ready_max_ms": self._timing_recv_to_ready_max * 1000.0,
                "ready_to_return_avg_ms": avg_ms(self._timing_ready_to_return_sum),
                "ready_to_return_max_ms": self._timing_ready_to_return_max * 1000.0,
                "recv_to_return_avg_ms": avg_ms(self._timing_recv_to_return_sum),
                "recv_to_return_max_ms": self._timing_recv_to_return_max * 1000.0,
            }

            self._timing_wait_count = 0
            self._timing_immediate_count = 0
            self._timing_wait_total_sum = 0.0
            self._timing_wait_total_max = 0.0
            self._timing_recv_to_ready_sum = 0.0
            self._timing_recv_to_ready_max = 0.0
            self._timing_ready_to_return_sum = 0.0
            self._timing_ready_to_return_max = 0.0
            self._timing_recv_to_return_sum = 0.0
            self._timing_recv_to_return_max = 0.0
            return stats

    def _recv_loop(self) -> None:
        """Background thread: receive command packets from C++ and update ``_command``."""
        buf = bytearray(4096)
        while self._running:
            try:
                readable, _, _ = select.select([self._recv_sock], [], [], 0.1)
                if not readable:
                    continue
                n, _ = self._recv_sock.recvfrom_into(buf)
                recv_time = time.perf_counter()
                result = parse_robot_command_packet(bytes(buf[:n]))
                if result is not None:
                    with self._command_cv:
                        self._command = result
                        self._command_seq += 1
                        self._command_recv_time = recv_time
                        self._command_ready_time = time.perf_counter()
                        self._command_cv.notify_all()
            except BlockingIOError:
                continue
            except (OSError, ValueError):
                break

    def close(self) -> None:
        """Tear down sockets and the receive thread."""
        with self._command_cv:
            self._running = False
            self._command_cv.notify_all()
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=0.5)
        self._send_sock.close()
        self._recv_sock.close()

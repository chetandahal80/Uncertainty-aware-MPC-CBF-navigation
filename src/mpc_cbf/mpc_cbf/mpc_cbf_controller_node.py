"""Robust real-time MPC-DCBF controller for timestamped human tracks."""

import json
import math
import time
from typing import Dict, List, Optional

import cvxpy as cp
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from . import geometry_utils as geo
from .mpc_controller.kalman_tracker import make_d_safe_array
from .mpc_controller.mpc_core import (
    NU,
    rollout_reference,
    initial_reference_guess,
    shift_trajectory,
    wrap_to_pi,
)
from .mpc_controller.mpc_core_dpp import DPPMPCSolver
from .mpc_controller.route_reference import RouteReference, guide_controls
from .mpc_controller.safety_monitor import assess_solution_safety
from .mpc_controller.world_track_manager import WorldTrackManager
from .tracking_io import decode_world_tracks
from .visualization_state import controller_snapshot


def default_mpc_weights():
    return (
        np.diag([50.0, 50.0, 5.0]),
        np.diag([1.0, 0.5]),
        np.diag([200.0, 200.0, 20.0]),
    )


class MPCCBFControllerNode(Node):
    """Track humans in the world frame and command a differential-drive robot."""

    def __init__(self) -> None:
        super().__init__("mpc_cbf_controller_node")

        # Goal and model.
        self.declare_parameter("goal_x", 5.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_theta_deg", 0.0)
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("robot_v_min", -0.5)
        self.declare_parameter("robot_v_max", 1.0)
        self.declare_parameter("robot_omega_max", 2.0)
        self.declare_parameter("acceleration_limits_enabled", True)
        self.declare_parameter("robot_linear_acceleration_max", 1.0)
        self.declare_parameter("robot_angular_acceleration_max", 3.0)
        self.declare_parameter("N", 20)
        self.declare_parameter("Ts", 0.1)
        self.declare_parameter("gamma", 0.3)
        self.declare_parameter("route_reference_enabled", False)
        self.declare_parameter("route_padding", 0.2)
        self.declare_parameter("route_memory_sec", 10.0)
        self.declare_parameter("cruise_speed", 0.6)
        self.declare_parameter("route_lookahead", 0.7)

        # Human prediction and safety radius.
        self.declare_parameter("d_safe_base", 0.5)
        self.declare_parameter("kappa", 1.0)
        self.declare_parameter("slack_penalty", 1.0e6)
        self.declare_parameter("emergency_stop_distance", 0.65)
        self.declare_parameter("max_allowed_safety_slack", 0.05)
        self.declare_parameter("safety_guard_horizon_sec", 0.5)
        self.declare_parameter("predicted_clearance_tolerance", 0.03)
        self.declare_parameter("max_humans", 8)

        # World-frame tracking and identifier recovery.
        self.declare_parameter("kf_process_noise_std", 0.25298221281347033)
        self.declare_parameter("kf_measurement_noise_std", 0.15)
        self.declare_parameter("kf_initial_vel_std", 1.0)
        self.declare_parameter("track_stale_timeout_sec", 3.0)
        self.declare_parameter("association_mahalanobis_gate", 9.21)
        self.declare_parameter("association_max_distance", 1.0)
        self.declare_parameter("assumed_human_speed_max", 2.0)
        self.declare_parameter("duplicate_measurement_distance", 0.20)
        self.declare_parameter("source_id_bonus", 2.0)

        # Runtime checks and solver behavior.
        self.declare_parameter("perception_timeout_sec", 2.0)
        self.declare_parameter("odom_timeout_sec", 0.5)
        self.declare_parameter("max_input_age_sec", 1.5)
        self.declare_parameter("max_future_stamp_sec", 0.25)
        self.declare_parameter("linearization_iterations", 1)
        self.declare_parameter("reference_reset_distance", 0.75)
        self.declare_parameter("reference_reset_angle", 0.8)
        self.declare_parameter("pos_tol", 0.15)
        self.declare_parameter("theta_tol", 0.2)

        # Topics.
        self.declare_parameter("tracked_humans_topic", "/tracked_humans_world")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("goal_topic", "/goal_pose")

        gp = self.get_parameter
        self.goal = np.array(
            [
                float(gp("goal_x").value),
                float(gp("goal_y").value),
                math.radians(float(gp("goal_theta_deg").value)),
            ]
        )
        self.world_frame = str(gp("world_frame").value)
        self.v_min = float(gp("robot_v_min").value)
        self.v_max = float(gp("robot_v_max").value)
        self.omega_max = abs(float(gp("robot_omega_max").value))
        self.acceleration_limits_enabled = bool(gp("acceleration_limits_enabled").value)
        self.acceleration_limits = np.array(
            [
                float(gp("robot_linear_acceleration_max").value),
                float(gp("robot_angular_acceleration_max").value),
            ]
        )
        if not np.all(np.isfinite(self.acceleration_limits)) or np.any(
            self.acceleration_limits <= 0
        ):
            raise ValueError("Robot acceleration limits must be finite and positive")
        self.N = int(gp("N").value)
        self.Ts = float(gp("Ts").value)
        self.gamma = float(gp("gamma").value)
        self.route_reference_enabled = bool(gp("route_reference_enabled").value)
        self.route_padding = float(gp("route_padding").value)
        self.route_memory_sec = float(gp("route_memory_sec").value)
        self.cruise_speed = float(gp("cruise_speed").value)
        self.route_lookahead = float(gp("route_lookahead").value)
        if (
            not math.isfinite(self.route_padding)
            or self.route_padding < 0
            or not math.isfinite(self.cruise_speed)
            or self.cruise_speed <= 0
            or not math.isfinite(self.route_lookahead)
            or self.route_lookahead <= 0
        ):
            raise ValueError("Invalid route-reference parameters")
        self.route_reference = RouteReference(self.route_padding, self.route_memory_sec)
        self.d_safe_base = float(gp("d_safe_base").value)
        self.kappa = float(gp("kappa").value)
        self.slack_penalty = float(gp("slack_penalty").value)
        self.emergency_stop_distance = float(gp("emergency_stop_distance").value)
        self.max_allowed_safety_slack = float(gp("max_allowed_safety_slack").value)
        self.safety_guard_horizon_sec = float(gp("safety_guard_horizon_sec").value)
        self.predicted_clearance_tolerance = float(
            gp("predicted_clearance_tolerance").value
        )
        self.max_humans = int(gp("max_humans").value)
        self.perception_timeout_sec = float(gp("perception_timeout_sec").value)
        self.odom_timeout_sec = float(gp("odom_timeout_sec").value)
        self.max_input_age_sec = float(gp("max_input_age_sec").value)
        self.max_future_stamp_sec = float(gp("max_future_stamp_sec").value)
        self.linearization_iterations = max(
            1, int(gp("linearization_iterations").value)
        )
        self.reference_reset_distance = float(gp("reference_reset_distance").value)
        self.reference_reset_angle = float(gp("reference_reset_angle").value)
        self.pos_tol = float(gp("pos_tol").value)
        self.theta_tol = float(gp("theta_tol").value)

        self._validate_parameters()

        self.track_manager = WorldTrackManager(
            process_noise_std=float(gp("kf_process_noise_std").value),
            measurement_noise_std=float(gp("kf_measurement_noise_std").value),
            initial_vel_std=float(gp("kf_initial_vel_std").value),
            stale_timeout_sec=float(gp("track_stale_timeout_sec").value),
            mahalanobis_gate=float(gp("association_mahalanobis_gate").value),
            association_max_distance=float(gp("association_max_distance").value),
            assumed_human_speed_max=float(gp("assumed_human_speed_max").value),
            duplicate_measurement_distance=float(
                gp("duplicate_measurement_distance").value
            ),
            source_id_bonus=float(gp("source_id_bonus").value),
        )

        self._create_solver(self.max_humans)
        self._precompile_solver()

        self.robot_pose: Optional[np.ndarray] = None
        self.last_odom_receive_sec: Optional[float] = None
        self.last_perception_receive_sec: Optional[float] = None
        self.last_track_stamp_sec: Optional[float] = None
        self.last_control_pose: Optional[np.ndarray] = None
        self.x_ref = None
        self.u_ref = None
        self.u_prev = np.zeros(NU)
        self.last_command = np.zeros(NU)
        self.last_command_time = None
        self.goal_reached = False
        self.goal_alignment_active = False
        self.warn_times: Dict[str, float] = {}
        self.last_stop_reason = None

        tracked_topic = str(gp("tracked_humans_topic").value)
        odom_topic = str(gp("odom_topic").value)
        cmd_topic = str(gp("cmd_vel_topic").value)
        goal_topic = str(gp("goal_topic").value)
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.on_odom, qos_profile_sensor_data
        )
        self.sub_tracks = self.create_subscription(
            String, tracked_topic, self.on_tracked_frame, 10
        )
        self.sub_goal = self.create_subscription(
            PoseStamped, goal_topic, self.on_goal_pose, 10
        )
        self.pub_cmd = self.create_publisher(Twist, cmd_topic, 10)
        self.declare_parameter("visualization_topic", "/mpc_cbf/visualization")
        self.pub_visualization = self.create_publisher(
            String, str(self.get_parameter("visualization_topic").value), 1
        )
        self.visualization_timer = self.create_timer(0.2, self.publish_visualization)
        watchdog_period = max(0.05, min(self.Ts, 0.2))
        self.watchdog = self.create_timer(watchdog_period, self.on_watchdog)
        self.control_timer = self.create_timer(self.Ts, self.on_control_timer)

        self.get_logger().info(
            f"Controller ready: goal={self.goal.tolist()}, horizon={self.N}x{self.Ts:.3f}s, "
            f"max_humans={self.max_humans}, velocity=[{self.v_min}, {self.v_max}], "
            f"solver={self.solver.solver}"
        )

    def publish_visualization(self):
        if self.pub_visualization.get_subscription_count() == 0:
            return
        try:
            message = String()
            message.data = json.dumps(controller_snapshot(self), allow_nan=False)
            self.pub_visualization.publish(message)
        except (ValueError, TypeError, ArithmeticError) as exc:
            self._warn_throttled(
                "visualization", f"Cannot publish plot snapshot: {exc}"
            )

    def _validate_parameters(self) -> None:
        if self.N < 2 or self.Ts <= 0.0:
            raise ValueError("N must be at least 2 and Ts must be positive")
        if self.v_min >= self.v_max or self.omega_max <= 0.0:
            raise ValueError("Robot velocity bounds are invalid")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if self.max_humans < 1:
            raise ValueError("max_humans must be positive")
        if self.d_safe_base <= 0.0 or self.emergency_stop_distance <= 0.0:
            raise ValueError("Safety distances must be positive")
        if not math.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError("kappa must be finite and nonnegative [1/m]")
        if not math.isfinite(self.slack_penalty) or self.slack_penalty <= 0:
            raise ValueError("slack_penalty must be finite and positive")
        if self.max_allowed_safety_slack < 0.0:
            raise ValueError("max_allowed_safety_slack cannot be negative")
        if self.safety_guard_horizon_sec <= 0.0:
            raise ValueError("safety_guard_horizon_sec must be positive")
        if self.safety_guard_horizon_sec > self.N * self.Ts:
            raise ValueError("safety_guard_horizon_sec exceeds the MPC horizon")
        if self.perception_timeout_sec > self.N * self.Ts:
            raise ValueError("perception_timeout_sec exceeds the MPC horizon")
        if self.max_input_age_sec > self.N * self.Ts:
            raise ValueError("max_input_age_sec exceeds the MPC horizon")
        if self.predicted_clearance_tolerance < 0.0:
            raise ValueError("predicted_clearance_tolerance cannot be negative")
        positive_times = (
            self.perception_timeout_sec,
            self.odom_timeout_sec,
            self.max_input_age_sec,
            self.max_future_stamp_sec,
            self.reference_reset_distance,
            self.reference_reset_angle,
            self.pos_tol,
            self.theta_tol,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_times):
            raise ValueError("Controller timing and tolerance values must be positive")

    def _create_solver(self, capacity):
        Q, R, Qf = default_mpc_weights()
        self.solver = DPPMPCSolver(
            self.N,
            self.Ts,
            Q,
            R,
            Qf,
            (self.v_min, self.v_max),
            (-self.omega_max, self.omega_max),
            max_humans=capacity,
            slack_penalty=self.slack_penalty,
            acceleration_limits=(
                self.acceleration_limits if self.acceleration_limits_enabled else None
            ),
        )

    def _command_interval(self):
        if self.last_command_time is None:
            return self.Ts
        return float(np.clip(self._now_sec() - self.last_command_time, 0.0, self.Ts))

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1.0e9

    def _precompile_solver(self) -> None:
        """Compile every fixed-capacity constraint before perception starts."""
        state = np.zeros(3)
        times = np.arange(self.N + 1, dtype=float) * self.Ts
        x_ref = np.column_stack((0.1 * times, 0.03 * times, 0.02 * times))
        u_ref = np.tile(np.array([0.1, 0.02]), (self.N, 1))
        # Populate all human slots with far-away synthetic trajectories. Using
        # nonzero DCBF coefficients here establishes the complete sparse
        # structure; compiling only the no-human case would defer part of the
        # numerical setup until the first live detection frame.
        dummy_humans = [
            {
                "x": np.full(self.N + 1, 20.0 + index),
                "y": np.full(self.N + 1, 5.0 + 0.5 * index),
                "d_safe": np.full(self.N + 1, self.d_safe_base),
            }
            for index in range(self.max_humans)
        ]
        started = time.perf_counter()
        u_opt, x_opt, status, _ = self.solver.solve(
            state,
            state,
            x_ref,
            u_ref,
            self.N,
            self.Ts,
            humans=dummy_humans,
            gamma=self.gamma,
            u_prev=np.zeros(NU),
            warm_start=False,
        )
        elapsed = time.perf_counter() - started
        if (
            status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            or u_opt is None
            or x_opt is None
        ):
            raise RuntimeError(f"MPC precompilation failed with status={status}")
        self.get_logger().info(
            f"MPC problem compiled with {self.solver.last_solver_used} "
            f"in {elapsed:.3f} s"
        )

    def _warn_throttled(self, key: str, message: str, seconds: float = 5.0) -> None:
        now = self._now_sec()
        if now - self.warn_times.get(key, -1.0e30) >= seconds:
            self.warn_times[key] = now
            self.get_logger().warn(message)

    def on_odom(self, msg: Odometry) -> None:
        frame_id = str(msg.header.frame_id)
        if frame_id and frame_id != self.world_frame:
            self._warn_throttled(
                "odom_frame",
                f"Ignoring odometry in frame '{frame_id}'; expected '{self.world_frame}'",
            )
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        try:
            values = np.array(
                [p.x, p.y, geo.quaternion_to_yaw(q.x, q.y, q.z, q.w)],
                dtype=float,
            )
        except (TypeError, ValueError, OverflowError):
            self._warn_throttled("odom_invalid", "Ignoring invalid odometry pose")
            return
        if not np.all(np.isfinite(values)):
            self._warn_throttled("odom_nan", "Ignoring non-finite odometry")
            return
        self.robot_pose = values
        self.last_odom_receive_sec = self._now_sec()

    def on_goal_pose(self, msg: PoseStamped) -> None:
        frame_id = str(msg.header.frame_id)
        if frame_id and frame_id != self.world_frame:
            self._warn_throttled(
                "goal_frame",
                f"Rejected goal in frame '{frame_id}'; expected '{self.world_frame}'",
            )
            return
        q = msg.pose.orientation
        try:
            goal = np.array(
                [
                    msg.pose.position.x,
                    msg.pose.position.y,
                    geo.quaternion_to_yaw(q.x, q.y, q.z, q.w),
                ],
                dtype=float,
            )
        except (TypeError, ValueError, OverflowError):
            self._warn_throttled("goal_invalid", "Rejected invalid goal pose")
            return
        if not np.all(np.isfinite(goal)):
            self._warn_throttled("goal_nan", "Rejected non-finite goal")
            return
        self.goal = goal
        self.route_reference.path = None
        self.goal_reached = False
        self.goal_alignment_active = False
        self.x_ref = None
        self.u_ref = None
        self.get_logger().info(f"New goal: {self.goal.tolist()} in {self.world_frame}")

    def _measurement_dicts(self, rows) -> List[dict]:
        measurements = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                source_id = int(row.get("id", row.get("source_id")))
                x = float(row["x"])
                y = float(row["y"])
                std = float(
                    row.get(
                        "measurement_std",
                        self.track_manager.default_measurement_noise_std,
                    )
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if all(math.isfinite(value) for value in (x, y, std)) and std > 0.0:
                measurement = {"id": source_id, "x": x, "y": y, "measurement_std": std}
                if "measurement_covariance" in row:
                    measurement["measurement_covariance"] = row[
                        "measurement_covariance"
                    ]
                measurements.append(measurement)
        return measurements

    def on_tracked_frame(self, msg: String) -> None:
        receive_ns = self.get_clock().now().nanoseconds
        receive_sec = receive_ns / 1.0e9
        try:
            stamp_ns, frame_id, rows, legacy = decode_world_tracks(msg.data, receive_ns)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self._warn_throttled(
                "track_payload", f"Rejected world-track payload: {exc}"
            )
            self._publish_stop("invalid perception payload")
            return
        if frame_id and frame_id != self.world_frame:
            self._warn_throttled(
                "track_frame",
                f"Rejected tracks in frame '{frame_id}'; expected '{self.world_frame}'",
            )
            self._publish_stop("perception frame mismatch")
            return
        stamp_sec = stamp_ns / 1.0e9
        age_sec = receive_sec - stamp_sec
        if not legacy and age_sec < -self.max_future_stamp_sec:
            self._warn_throttled(
                "track_future",
                f"Rejected world tracks timestamped {-age_sec:.3f} s in the future",
            )
            self._publish_stop("future perception timestamp")
            return
        if not legacy and age_sec > self.max_input_age_sec:
            self._warn_throttled(
                "track_age", f"Rejected stale world tracks ({age_sec:.3f} s old)"
            )
            self._publish_stop("stale perception input")
            return
        if (
            self.last_track_stamp_sec is not None
            and stamp_sec < self.last_track_stamp_sec - 1.0
        ):
            self.get_logger().warn(
                "Detected simulation-clock reset; clearing world tracks and controller reference"
            )
            self.track_manager.clear()
            self.route_reference = RouteReference(
                self.route_padding, self.route_memory_sec
            )
            self.x_ref = None
            self.u_ref = None
            self.u_prev[:] = 0.0
            self.last_track_stamp_sec = None
        measurements = self._measurement_dicts(rows)
        try:
            tracks = self.track_manager.update(measurements, stamp_sec)
        except ValueError as exc:
            self._warn_throttled("track_order", f"Rejected track frame: {exc}")
            return

        self.last_perception_receive_sec = receive_sec
        self.last_track_stamp_sec = stamp_sec
        # Perception updates the native-time filter; MPC has its own Ts timer.

    def on_control_timer(self):
        now = self._now_sec()
        if (
            self.last_perception_receive_sec is None
            or self.last_track_stamp_sec is None
        ):
            self._publish_stop("waiting for perception")
            return
        age = now - self.last_track_stamp_sec
        if age < -self.max_future_stamp_sec:
            self._publish_stop("simulation clock reset")
            return
        if (
            now - self.last_perception_receive_sec > self.perception_timeout_sec
            or age > self.max_input_age_sec
        ):
            self._publish_stop("perception timeout")
            return
        tracks = [
            t
            for t in self.track_manager.tracks
            if now - t.last_seen_time <= self.track_manager.stale_timeout_sec
        ]
        self._run_control(tracks, measurement_age_sec=max(age, 0.0))

    def _build_humans(self, tracks, measurement_age_sec: float) -> List[dict]:
        humans = []
        robot_xy = self.robot_pose[:2]
        for track in tracks:
            px, py, trace_pos = track.kf.predict_horizon(
                self.N,
                self.Ts,
                start_time_offset=measurement_age_sec,
            )
            d_safe = make_d_safe_array(trace_pos, self.d_safe_base, self.kappa)
            distances = np.hypot(px - robot_xy[0], py - robot_xy[1])
            risk = float(np.min(distances - d_safe))
            humans.append(
                {
                    "x": px,
                    "y": py,
                    "d_safe": d_safe,
                    "_observed_position": track.last_observed_position.tolist(),
                    "_last_seen_time": track.last_seen_time,
                    "_risk": risk,
                    "_radius_now": float(d_safe[0]),
                    "_radius_end": float(d_safe[-1]),
                    "_trace_now": float(trace_pos[0]),
                    "_trace_end": float(trace_pos[-1]),
                    "_age": float(measurement_age_sec),
                    "_speed": float(np.linalg.norm(track.kf.velocity)),
                    "_current_distance": float(distances[0]),
                    "_internal_id": track.internal_id,
                    "_source_id": track.source_id,
                }
            )
        humans.sort(key=lambda item: item["_risk"])
        if len(humans) > self.solver.max_humans:
            # Capacity is an allocation detail, never a reason to omit a person.
            self._create_solver(len(humans))
        return humans

    def _reference_is_stale(self, current: np.ndarray) -> bool:
        if self.x_ref is None or self.u_ref is None:
            return True
        if self.x_ref.shape != (self.N + 1, 3) or self.u_ref.shape != (self.N, NU):
            return True
        distance = float(np.linalg.norm(current[:2] - self.x_ref[0, :2]))
        angle = abs(wrap_to_pi(current[2] - self.x_ref[0, 2]))
        return (
            distance > self.reference_reset_distance
            or angle > self.reference_reset_angle
        )

    def _run_control(self, tracks, measurement_age_sec: float = 0.0) -> None:
        if self.robot_pose is None or self.last_odom_receive_sec is None:
            self._publish_stop("waiting for odometry")
            return
        if self._now_sec() - self.last_odom_receive_sec > self.odom_timeout_sec:
            self._publish_stop("odometry timeout")
            return
        if self.goal_reached:
            self._publish_stop("goal reached", log=False)
            return

        current = self.robot_pose.copy()
        position_error = float(np.linalg.norm(current[:2] - self.goal[:2]))
        heading_error = abs(wrap_to_pi(current[2] - self.goal[2]))
        if position_error >= self.pos_tol:
            self.goal_alignment_active = False
        # Finish a normal goal only after the bounded QP has slowed down.
        ready_to_stop = not self.acceleration_limits_enabled or np.all(
            np.abs(self.last_command)
            <= self.acceleration_limits * self._command_interval() + 1e-6
        )
        if (
            position_error < self.pos_tol
            and heading_error < self.theta_tol
            and ready_to_stop
        ):
            self.goal_reached = True
            self._publish_stop("goal reached")
            return

        humans_with_meta = self._build_humans(tracks, measurement_age_sec)
        if humans_with_meta:
            closest = min(item["_current_distance"] for item in humans_with_meta)
            if closest <= self.emergency_stop_distance:
                self._publish_stop(f"human inside emergency distance ({closest:.3f} m)")
                return

        humans_qp = [
            {
                "x": item["x"],
                "y": item["y"],
                "d_safe": item["d_safe"],
                "id": item["_internal_id"],
                "observed_position": item["_observed_position"],
                "last_seen_time": item["_last_seen_time"],
            }
            for item in humans_with_meta
        ]

        if self._reference_is_stale(current):
            self.x_ref = initial_reference_guess(current, self.N, humans=humans_qp)
            self.u_ref = np.zeros((self.N, NU))

        current[2] = self.x_ref[0, 2] + wrap_to_pi(current[2] - self.x_ref[0, 2])
        reference_x = rollout_reference(current, self.u_ref, self.Ts)
        reference_u = self.u_ref.copy()
        task_reference = self.goal
        docking_velocity = None
        if self.route_reference_enabled:
            path, guide = self.route_reference.build(
                current,
                self.goal,
                humans_qp,
                self.N,
                self.Ts,
                min(self.cruise_speed, self.v_max),
                self._now_sec(),
            )
            if path is not None:
                task_reference = guide
                reference_u = guide_controls(
                    current,
                    path,
                    self.N,
                    self.Ts,
                    min(self.cruise_speed, self.v_max),
                    self.omega_max,
                    self.route_lookahead,
                    acceleration_limits=(
                        self.acceleration_limits
                        if self.acceleration_limits_enabled
                        else None
                    ),
                    u_prev=self.u_prev,
                    first_step_dt=self._command_interval(),
                )
                reference_x = rollout_reference(current, reference_u, self.Ts)
                if (
                    self.acceleration_limits_enabled
                    and position_error < 0.5
                    and np.linalg.norm(path[-1, :2] - self.goal[:2]) < 1e-6
                    and np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))
                    < 0.75
                ):
                    # First capture position, then align heading. Simultaneously
                    # trading lateral error against final heading can stall a
                    # nonholonomic robot with bounded input changes.
                    stopping_distance = self.u_prev[0] ** 2 / (
                        2 * self.acceleration_limits[0]
                    )
                    if position_error + stopping_distance < self.pos_tol:
                        self.goal_alignment_active = True
                    if self.goal_alignment_active:
                        task_reference = self.goal
                        times = self._command_interval() + np.arange(self.N) * self.Ts
                        docking_velocity = np.sign(self.u_prev[0]) * np.maximum(
                            abs(self.u_prev[0]) - self.acceleration_limits[0] * times,
                            0.0,
                        )
                        reference_u = np.column_stack(
                            [docking_velocity, np.zeros(self.N)]
                        )
                        reference_x = rollout_reference(current, reference_u, self.Ts)
                    else:
                        bearing = math.atan2(
                            self.goal[1] - current[1], self.goal[0] - current[0]
                        )
                        task_reference = np.array([self.goal[0], self.goal[1], bearing])
        solution = None

        for _ in range(self.linearization_iterations):
            try:
                result = self.solver.solve(
                    current,
                    task_reference,
                    reference_x,
                    reference_u,
                    self.N,
                    self.Ts,
                    humans=humans_qp,
                    gamma=self.gamma,
                    u_prev=self.u_prev,
                    first_step_dt=(
                        self._command_interval()
                        if self.acceleration_limits_enabled
                        else None
                    ),
                    linear_velocity_profile=docking_velocity,
                )
            except Exception as exc:  # solver backends expose several exception types
                self.get_logger().error(f"MPC solver exception: {exc}")
                self._publish_stop("optimization exception")
                return
            u_opt, x_opt, status, zeta_opt = result
            acceptable = status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            if (
                not acceptable
                or u_opt is None
                or x_opt is None
                or not np.all(np.isfinite(u_opt))
                or not np.all(np.isfinite(x_opt))
            ):
                self.get_logger().error(
                    f"MPC solve failed or invalid (status={status})"
                )
                self._publish_stop("optimization failure")
                return
            solution = (u_opt, x_opt, status, zeta_opt)
            reference_x = rollout_reference(current, u_opt, self.Ts)
            reference_u = u_opt

        u_opt, x_opt, status, zeta_opt = solution
        human_count = len(humans_qp)
        guard_steps = min(
            self.N,
            max(1, int(math.ceil(self.safety_guard_horizon_sec / self.Ts))),
        )
        try:
            safety = assess_solution_safety(
                rollout_reference(current, u_opt, self.Ts),
                zeta_opt,
                humans_qp,
                guard_steps,
            )
        except (KeyError, TypeError, ValueError) as exc:
            self.get_logger().error(f"Invalid optimized safety data: {exc}")
            self._publish_stop("invalid optimized safety data")
            return

        if safety.near_slack > self.max_allowed_safety_slack:
            worst_human = int(
                np.argmax(np.max(zeta_opt[:guard_steps, :human_count], axis=0))
            )
            diagnostic = humans_with_meta[worst_human]
            self._warn_throttled(
                "safety_slack",
                "Near-term safety constraint required excessive relaxation "
                f"({safety.near_slack:.4f} over {guard_steps * self.Ts:.2f} s; "
                f"horizon maximum={safety.horizon_slack:.4f} at "
                f"{(safety.worst_slack_step + 1) * self.Ts:.2f} s); "
                f"track={diagnostic['_internal_id']}, "
                f"distance={diagnostic['_current_distance']:.2f}m, "
                f"radius(now/end)={diagnostic['_radius_now']:.2f}/{diagnostic['_radius_end']:.2f}m, "
                f"trace(now/end)={diagnostic['_trace_now']:.3f}/{diagnostic['_trace_end']:.3f}m^2, "
                f"image_age={diagnostic['_age']:.2f}s, "
                f"human_speed={diagnostic['_speed']:.2f}m/s",
                seconds=1.0,
            )
            self.x_ref, self.u_ref = shift_trajectory(x_opt, u_opt)
            self._publish_stop(
                "excessive near-term safety slack", reset_reference=False
            )
            return
        if safety.minimum_near_clearance < -self.predicted_clearance_tolerance:
            self._warn_throttled(
                "predicted_clearance",
                "Near-term optimized trajectory violates the safety radius by "
                f"{-safety.minimum_near_clearance:.3f} m",
                seconds=1.0,
            )
            self.x_ref, self.u_ref = shift_trajectory(x_opt, u_opt)
            self._publish_stop(
                "predicted near-term safety violation", reset_reference=False
            )
            return
        if safety.horizon_slack > self.max_allowed_safety_slack:
            self._warn_throttled(
                "future_safety_slack",
                "Future-horizon safety relaxation will be replanned: "
                f"maximum={safety.horizon_slack:.4f} at "
                f"{(safety.worst_slack_step + 1) * self.Ts:.2f} s; "
                f"near-term={safety.near_slack:.4f}",
                seconds=2.0,
            )

        command = np.asarray(u_opt[0], dtype=float)
        command[0] = np.clip(command[0], self.v_min, self.v_max)
        command[1] = np.clip(command[1], -self.omega_max, self.omega_max)
        self.u_prev = command.copy()
        self._publish_cmd(command[0], command[1])
        self.last_command = command.copy()
        self.last_stop_reason = None
        self.last_control_pose = current.copy()
        self.x_ref, self.u_ref = shift_trajectory(x_opt, u_opt)

        if humans_with_meta:
            closest_item = min(
                humans_with_meta, key=lambda item: item["_current_distance"]
            )
            detail = (
                f"closest={closest_item['_current_distance']:.2f}m "
                f"internal_id={closest_item['_internal_id']} "
                f"source_id={closest_item['_source_id']}"
            )
        else:
            detail = "no active human tracks"
        self.get_logger().debug(
            f"status={status}, humans={human_count}, {detail}, "
            f"near_slack={safety.near_slack:.5f}, "
            f"horizon_slack={safety.horizon_slack:.5f}, "
            f"clearance={safety.minimum_near_clearance:.3f}, "
            f"cmd=({command[0]:.3f},{command[1]:.3f})"
        )

    def on_watchdog(self) -> None:
        now = self._now_sec()
        if self.last_odom_receive_sec is None:
            self._publish_stop("waiting for odometry")
            return
        if now - self.last_odom_receive_sec > self.odom_timeout_sec:
            self._publish_stop("odometry watchdog timeout")
            return
        if self.last_perception_receive_sec is None:
            self._publish_stop("waiting for perception")
            return
        if now - self.last_perception_receive_sec > self.perception_timeout_sec:
            self._publish_stop("perception watchdog timeout")

    def _publish_cmd(self, velocity: float, yaw_rate: float) -> None:
        msg = Twist()
        msg.linear.x = float(velocity)
        msg.angular.z = float(yaw_rate)
        self.pub_cmd.publish(msg)
        self.last_command_time = self._now_sec()

    def _publish_stop(
        self,
        reason: str,
        log: bool = True,
        *,
        reset_reference: bool = True,
    ) -> None:
        self._publish_cmd(0.0, 0.0)
        self.last_command[:] = 0.0
        self.u_prev[:] = 0.0
        if reset_reference:
            self.x_ref = None
            self.u_ref = None
        if log and reason != self.last_stop_reason:
            self.get_logger().warn(f"Commanded stop: {reason}")
        self.last_stop_reason = reason


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MPCCBFControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node._publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

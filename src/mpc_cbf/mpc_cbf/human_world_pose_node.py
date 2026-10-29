"""Convert timestamped RGB-D tracks into world-frame human measurements."""

from collections import deque
import json
import math
import numpy as np
from typing import Dict, Optional, Tuple

import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from . import geometry_utils as geo
from .tracking_io import (
    decode_tracked_boxes,
    encode_world_tracks,
    stamp_to_nanoseconds,
)


class HumanWorldPoseNode(Node):
    """Project image-space tracks into a configurable fixed world frame."""

    def __init__(self) -> None:
        super().__init__("human_world_pose_node")

        self.declare_parameter("tracked_boxes_topic", "/tracked_boxes")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("output_topic", "/tracked_humans_world")
        self.declare_parameter("output_frame", "odom")
        self.declare_parameter("camera_optical_frame", "")
        self.declare_parameter("use_tf", True)
        self.declare_parameter("tf_timeout_sec", 0.05)
        self.declare_parameter("allow_odom_fallback", True)
        self.declare_parameter("max_odom_time_error_sec", 0.15)
        self.declare_parameter("max_tf_time_error_sec", 0.05)
        self.declare_parameter("max_detection_age_sec", 1.5)
        self.declare_parameter("max_future_stamp_sec", 0.25)
        self.declare_parameter("accept_legacy_payloads", False)
        self.declare_parameter("min_valid_depth", 0.15)
        self.declare_parameter("max_valid_depth", 9.9)
        self.declare_parameter("camera_offset_forward", 0.205)
        self.declare_parameter("camera_offset_left", 0.0)
        self.declare_parameter("camera_yaw", 0.0)
        # Raw sensor fallback only when a publisher supplies no tracker covariance.
        self.declare_parameter("pixel_measurement_std", 1.0)
        self.declare_parameter("depth_measurement_std", 0.05)
        self.declare_parameter("noise_reference_width", 0)
        self.declare_parameter("depth_bias_correction", 0.0)
        self.declare_parameter("measurement_covariance_scale", 1.0)
        self.declare_parameter("world_position_noise_floor", 0.0)

        gp = self.get_parameter
        self.output_frame = str(gp("output_frame").value)
        self.camera_optical_frame = str(gp("camera_optical_frame").value)
        self.use_tf = bool(gp("use_tf").value)
        self.tf_timeout_sec = float(gp("tf_timeout_sec").value)
        self.allow_odom_fallback = bool(gp("allow_odom_fallback").value)
        self.max_odom_time_error_sec = float(gp("max_odom_time_error_sec").value)
        self.max_tf_time_error_sec = float(gp("max_tf_time_error_sec").value)
        self.max_detection_age_sec = float(gp("max_detection_age_sec").value)
        self.max_future_stamp_sec = float(gp("max_future_stamp_sec").value)
        self.accept_legacy_payloads = bool(gp("accept_legacy_payloads").value)
        self.min_valid_depth = float(gp("min_valid_depth").value)
        self.max_valid_depth = float(gp("max_valid_depth").value)
        self.camera_offset_forward = float(gp("camera_offset_forward").value)
        self.camera_offset_left = float(gp("camera_offset_left").value)
        self.camera_yaw = float(gp("camera_yaw").value)
        self.pixel_measurement_std = float(gp("pixel_measurement_std").value)
        self.depth_measurement_std = float(gp("depth_measurement_std").value)
        self.noise_reference_width = int(gp("noise_reference_width").value)
        if self.noise_reference_width < 0:
            raise ValueError("noise_reference_width cannot be negative")
        self.depth_bias_correction = float(gp("depth_bias_correction").value)
        self.measurement_covariance_scale = float(
            gp("measurement_covariance_scale").value
        )
        self.world_position_noise_floor = float(gp("world_position_noise_floor").value)
        if (
            not math.isfinite(self.world_position_noise_floor)
            or self.world_position_noise_floor < 0
        ):
            raise ValueError(
                "world_position_noise_floor must be finite and nonnegative"
            )
        if not math.isfinite(self.depth_bias_correction):
            raise ValueError("depth_bias_correction must be finite")
        if (
            not math.isfinite(self.measurement_covariance_scale)
            or self.measurement_covariance_scale <= 0
        ):
            raise ValueError("measurement_covariance_scale must be positive")

        if not self.output_frame:
            raise ValueError("output_frame cannot be empty")
        positive_values = (
            self.tf_timeout_sec,
            self.max_odom_time_error_sec,
            self.max_tf_time_error_sec,
            self.max_detection_age_sec,
            self.max_future_stamp_sec,
            self.min_valid_depth,
            self.max_valid_depth,
            self.pixel_measurement_std,
            self.depth_measurement_std,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_values):
            raise ValueError(
                "World-pose timing, depth, and noise values must be positive"
            )
        if self.min_valid_depth >= self.max_valid_depth:
            raise ValueError("min_valid_depth must be smaller than max_valid_depth")
        self.fx, self.fy = geo.DEFAULT_FX, geo.DEFAULT_FY
        self.cx, self.cy = geo.DEFAULT_CX, geo.DEFAULT_CY
        self.image_width = geo.DEFAULT_WIDTH
        self.image_height = geo.DEFAULT_HEIGHT
        self.camera_info_frame = ""
        self.have_live_intrinsics = False
        self.odom_history = deque(maxlen=300)
        self.warn_times: Dict[str, int] = {}

        self.tf_buffer: Optional[Buffer] = None
        self.tf_listener: Optional[TransformListener] = None
        if self.use_tf:
            self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
            self.tf_listener = TransformListener(self.tf_buffer, self)

        tracked_boxes_topic = str(gp("tracked_boxes_topic").value)
        camera_info_topic = str(gp("camera_info_topic").value)
        odom_topic = str(gp("odom_topic").value)
        output_topic = str(gp("output_topic").value)

        self.sub_boxes = self.create_subscription(
            String, tracked_boxes_topic, self.on_tracked_boxes, 1
        )
        self.sub_camera_info = self.create_subscription(
            CameraInfo, camera_info_topic, self.on_camera_info, qos_profile_sensor_data
        )
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self.on_odom, qos_profile_sensor_data
        )
        self.pub_world = self.create_publisher(String, output_topic, 10)

        self.get_logger().info(
            f"World-pose integration ready: {tracked_boxes_topic} -> {output_topic}; "
            f"target frame={self.output_frame}, tf={self.use_tf}"
        )

    def _warn_throttled(self, key: str, message: str, seconds: float = 5.0) -> None:
        now_ns = self.get_clock().now().nanoseconds
        last_ns = self.warn_times.get(key, -(10**30))
        if now_ns - last_ns >= int(seconds * 1.0e9):
            self.warn_times[key] = now_ns
            self.get_logger().warn(message)

    def on_camera_info(self, msg: CameraInfo) -> None:
        values = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
        if not all(math.isfinite(float(value)) for value in values):
            self._warn_throttled("intrinsics_nan", "Ignoring non-finite CameraInfo")
            return
        if msg.k[0] <= 0.0 or msg.k[4] <= 0.0:
            self._warn_throttled("intrinsics_zero", "Ignoring invalid CameraInfo")
            return
        self.fx, self.fy, self.cx, self.cy = (float(value) for value in values)
        self.image_width = int(msg.width) if msg.width else self.image_width
        self.image_height = int(msg.height) if msg.height else self.image_height
        self.camera_info_frame = str(msg.header.frame_id)
        if not self.have_live_intrinsics:
            self.get_logger().info(
                f"Using live camera intrinsics fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}"
            )
        self.have_live_intrinsics = True

    def on_odom(self, msg: Odometry) -> None:
        frame_id = str(msg.header.frame_id)
        if frame_id and frame_id != self.output_frame:
            self._warn_throttled(
                "odom_frame",
                f"Ignoring fallback odometry in frame '{frame_id}'; expected '{self.output_frame}'",
            )
            return
        try:
            stamp_ns = stamp_to_nanoseconds(msg.header.stamp)
        except (TypeError, ValueError):
            stamp_ns = self.get_clock().now().nanoseconds
        if stamp_ns == 0:
            stamp_ns = self.get_clock().now().nanoseconds
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        try:
            sample = (
                stamp_ns,
                float(p.x),
                float(p.y),
                geo.quaternion_to_yaw(q.x, q.y, q.z, q.w),
            )
        except (TypeError, ValueError, OverflowError):
            self._warn_throttled("odom_invalid", "Ignoring invalid odometry pose")
            return
        if not all(math.isfinite(value) for value in sample[1:]):
            self._warn_throttled("odom_invalid", "Ignoring non-finite odometry pose")
            return
        if self.odom_history and stamp_ns < self.odom_history[-1][0]:
            if self.odom_history[-1][0] - stamp_ns > 1_000_000_000:
                self.get_logger().warn(
                    "Detected simulation-clock reset; clearing odometry history"
                )
                self.odom_history.clear()
            else:
                self._warn_throttled("odom_order", "Ignoring out-of-order odometry")
                return
        self.odom_history.append(sample)

    def _odom_at(self, stamp_ns: int) -> Optional[Tuple[float, float, float]]:
        if not self.odom_history:
            return None
        samples = self.odom_history
        if stamp_ns <= samples[0][0]:
            chosen = samples[0]
            error_ns = samples[0][0] - stamp_ns
            return (
                chosen[1:] if error_ns <= self.max_odom_time_error_sec * 1e9 else None
            )
        if stamp_ns >= samples[-1][0]:
            chosen = samples[-1]
            error_ns = stamp_ns - samples[-1][0]
            return (
                chosen[1:] if error_ns <= self.max_odom_time_error_sec * 1e9 else None
            )
        for newer_index in range(1, len(samples)):
            newer = samples[newer_index]
            if newer[0] < stamp_ns:
                continue
            older = samples[newer_index - 1]
            span = newer[0] - older[0]
            alpha = 0.0 if span <= 0 else (stamp_ns - older[0]) / span
            return (
                older[1] + alpha * (newer[1] - older[1]),
                older[2] + alpha * (newer[2] - older[2]),
                geo.interpolate_yaw(older[3], newer[3], alpha),
            )
        return None

    def _lookup_transform(self, source_frame: str, stamp_ns: int):
        if not self.use_tf or self.tf_buffer is None or not source_frame:
            return None
        try:
            return self.tf_buffer.lookup_transform(
                self.output_frame,
                source_frame,
                Time(nanoseconds=stamp_ns),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except TransformException as exact_error:
            # A camera stamp can lead the newest dynamic transform by a few
            # milliseconds. Bounded latest-transform fallback avoids dropping
            # that frame while still rejecting materially mistimed geometry.
            try:
                latest = self.tf_buffer.lookup_transform(
                    self.output_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.0),
                )
                latest_ns = stamp_to_nanoseconds(latest.header.stamp)
                error_sec = abs(latest_ns - stamp_ns) / 1.0e9
                if latest_ns != 0 and error_sec <= self.max_tf_time_error_sec:
                    self._warn_throttled(
                        "tf_near_time",
                        "Using nearest available transform "
                        f"({error_sec:.3f} s from image timestamp)",
                    )
                    return latest
            except (AttributeError, TransformException, TypeError, ValueError):
                pass
            self._warn_throttled(
                "tf_lookup",
                f"Transform {source_frame} -> {self.output_frame} unavailable: "
                f"{exact_error}",
            )
            return None

    @staticmethod
    def _parse_row(row):
        if isinstance(row, dict):
            if "bbox" in row:
                bbox = list(row["bbox"])
                if len(bbox) != 4:
                    raise ValueError("bbox must contain four coordinates")
                return (*bbox, row["depth"], row.get("id", row.get("source_id")))
            return (
                row["x1"],
                row["y1"],
                row["x2"],
                row["y2"],
                row["depth"],
                row.get("id", row.get("source_id")),
            )
        values = list(row)
        if len(values) < 6:
            raise ValueError("track row requires six values")
        return tuple(values[:6])

    def on_tracked_boxes(self, msg: String) -> None:
        receive_ns = self.get_clock().now().nanoseconds
        try:
            stamp_ns, payload_frame, rows, legacy = decode_tracked_boxes(
                msg.data, receive_ns
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self._warn_throttled("payload", f"Rejected tracking payload: {exc}")
            return

        if legacy and not self.accept_legacy_payloads:
            self._warn_throttled(
                "legacy_payload",
                "Ignoring legacy /tracked_boxes payload without an image timestamp; "
                "stop the duplicate legacy tracker and run only rgbd_tracking_adapter",
            )
            return
        if legacy:
            self._warn_throttled(
                "legacy_payload_allowed",
                "Legacy /tracked_boxes payload has no image timestamp; "
                "using receipt time",
            )

        age_sec = (receive_ns - stamp_ns) / 1.0e9
        if age_sec < -self.max_future_stamp_sec:
            self._warn_throttled(
                "future_detection",
                f"Dropping detection frame timestamped {-age_sec:.3f} s in the future",
            )
            return
        if age_sec > self.max_detection_age_sec:
            self._warn_throttled(
                "old_detection",
                f"Dropping stale detection frame ({age_sec:.3f} s old)",
            )
            return

        # An empty frame is still valuable: it advances existing world tracks
        # through a temporary detector miss. No coordinate transform is needed
        # when there are no points to transform.
        if not rows:
            output = String()
            output.data = encode_world_tracks([], stamp_ns, self.output_frame)
            self.pub_world.publish(output)
            return

        source_frame = (
            self.camera_optical_frame or payload_frame or self.camera_info_frame
        )
        transform = self._lookup_transform(source_frame, stamp_ns)
        fallback_pose = None
        if transform is None and self.allow_odom_fallback:
            fallback_pose = self._odom_at(stamp_ns)
        if transform is None and fallback_pose is None:
            self._warn_throttled(
                "no_pose",
                "Cannot transform detections: neither a timed transform nor matching odometry is available",
            )
            return

        parsed_by_id = {}
        rejected = 0
        for row in rows:
            try:
                x1, y1, x2, y2, depth, source_id = self._parse_row(row)
                values = [float(x1), float(y1), float(x2), float(y2), float(depth)]
                if source_id is None or not all(
                    math.isfinite(value) for value in values
                ):
                    raise ValueError("non-finite or missing value")
                source_id_float = float(source_id)
                source_id = int(source_id_float)
                if not math.isfinite(source_id_float) or source_id_float != source_id:
                    raise ValueError("track identifier must be an integer")
                x1, y1, x2, y2, depth = values
                x1 = min(max(x1, 0.0), float(self.image_width - 1))
                x2 = min(max(x2, 0.0), float(self.image_width - 1))
                y1 = min(max(y1, 0.0), float(self.image_height - 1))
                y2 = min(max(y2, 0.0), float(self.image_height - 1))
                if x2 <= x1 or y2 <= y1:
                    raise ValueError("empty bounding box")
                if not self.min_valid_depth <= depth <= self.max_valid_depth:
                    raise ValueError("depth outside configured range")
                covariance_uvd = (
                    row.get("covariance_uvd") if isinstance(row, dict) else None
                )
                if covariance_uvd is None:
                    pixel_std = self.pixel_measurement_std
                    if self.noise_reference_width:
                        pixel_std *= self.image_width / self.noise_reference_width
                    covariance_uvd = np.diag(
                        [pixel_std**2, pixel_std**2, self.depth_measurement_std**2]
                    )
                candidate = (x1, y1, x2, y2, depth, source_id, covariance_uvd)
                previous = parsed_by_id.get(source_id)
                if previous is None or depth < previous[4]:
                    parsed_by_id[source_id] = candidate
            except (KeyError, TypeError, ValueError, OverflowError):
                rejected += 1

        world_tracks = []
        for x1, y1, x2, y2, depth, source_id, covariance_uvd in parsed_by_id.values():
            u = 0.5 * (x1 + x2)
            v = 0.5 * (y1 + y2)
            depth += self.depth_bias_correction
            try:
                optical_point = geo.project_pixel_to_optical(
                    u, v, depth, self.fx, self.fy, self.cx, self.cy
                )
                if transform is not None:
                    t = transform.transform.translation
                    q = transform.transform.rotation
                    rotation = np.column_stack(
                        [
                            geo.rotate_vector_by_quaternion(axis, (q.x, q.y, q.z, q.w))
                            for axis in np.eye(3)
                        ]
                    )
                    x_world, y_world, _ = geo.transform_point(
                        optical_point,
                        (t.x, t.y, t.z),
                        (q.x, q.y, q.z, q.w),
                    )
                else:
                    robot_x, robot_y, robot_yaw = fallback_pose
                    rotation = geo.optical_to_world_planar_rotation(
                        robot_yaw, self.camera_yaw
                    )
                    x_world, y_world = geo.pixel_depth_to_world_planar(
                        u,
                        v,
                        depth,
                        robot_x,
                        robot_y,
                        robot_yaw,
                        fx=self.fx,
                        fy=self.fy,
                        cx=self.cx,
                        cy=self.cy,
                        camera_offset_forward=self.camera_offset_forward,
                        camera_offset_left=self.camera_offset_left,
                        camera_yaw=self.camera_yaw,
                    )
                measurement_covariance = geo.covariance_uvd_to_world(
                    u,
                    v,
                    depth,
                    self.fx,
                    self.fy,
                    self.cx,
                    self.cy,
                    covariance_uvd,
                    rotation,
                )
                measurement_covariance *= self.measurement_covariance_scale
                # Bounding-box/body-surface geometry has world-frame error
                # beyond pinhole sensor noise; include it in measurement covariance.
                measurement_covariance += np.eye(2) * self.world_position_noise_floor**2
                measurement_std = math.sqrt(
                    max(float(np.trace(measurement_covariance)) / 2, 1e-12)
                )
                world_tracks.append(
                    {
                        "id": source_id,
                        "x": float(x_world),
                        "y": float(y_world),
                        "depth": float(depth),
                        "measurement_std": float(measurement_std),
                        "measurement_covariance": measurement_covariance.tolist(),
                    }
                )
            except (TypeError, ValueError, OverflowError):
                rejected += 1

        if rejected:
            self._warn_throttled(
                "rejected_rows", f"Rejected {rejected} invalid track row(s)"
            )
        output = String()
        output.data = encode_world_tracks(world_tracks, stamp_ns, self.output_frame)
        self.pub_world.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HumanWorldPoseNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

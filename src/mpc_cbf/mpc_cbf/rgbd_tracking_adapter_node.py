"""Timestamped RGB-D observations with image-frame identity association.

The image Kalman filter supports association. World projection receives current
matched pixel/depth measurements and sensor covariance; world-coordinate
filtering is performed downstream.
"""

import math
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from typing import Any

import rclpy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer

from human_tracking.rgbd_subscription import RGBDSubscriber

from .image_tracker_noise import configure_image_filter, scale_pixel_noise
from .detector_runtime import RuntimeDetector
from .tracker_lifecycle import configure_tracker_lifecycle
from .perception_measurements import depth_mode, current_frame_rows

from .tracking_io import (
    encode_tracked_boxes,
    stamp_to_nanoseconds,
    timestamp_age_seconds,
)


class RGBDTrackingAdapterNode(RGBDSubscriber):
    """Publish timestamped observations from the RGB-D detection pipeline."""

    def __init__(self) -> None:
        self._captured_rows: Any = []
        self._last_processed_stamp_ns = None
        self._frame_dt = 1.0
        super().__init__()
        self.declare_parameter("rgbd_sync_slop_sec", 0.03)
        slop = float(self.get_parameter("rgbd_sync_slop_sec").value)
        if not math.isfinite(slop) or not 0 <= slop <= 0.2:
            raise ValueError("RGB-D synchronization slop must lie in [0,.2]")
        rgb_topic = self.subscription_rgb.getTopic()
        depth_topic = self.subscription_depth.getTopic()
        self.destroy_subscription(self.subscription_rgb.sub)
        self.destroy_subscription(self.subscription_depth.sub)
        latest_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.subscription_rgb = Subscriber(
            self, Image, rgb_topic, qos_profile=latest_qos
        )
        self.subscription_depth = Subscriber(
            self, Image, depth_topic, qos_profile=latest_qos
        )
        self.ts = ApproximateTimeSynchronizer(
            [self.subscription_rgb, self.subscription_depth], queue_size=4, slop=slop
        )
        self.ts.registerCallback(self.sync_callback)
        self.declare_parameter("image_tracker_min_streak", 0)
        self.declare_parameter("image_tracker_max_age", 5)
        configure_tracker_lifecycle(
            self.tracker,
            self.get_parameter("image_tracker_min_streak").value,
            self.get_parameter("image_tracker_max_age").value,
        )
        self.declare_parameter("inference_device", "auto")
        self.declare_parameter("inference_threads", 2)
        self.declare_parameter("detector_weights", "")
        self.declare_parameter("inference_image_size", 0)
        threads = int(self.get_parameter("inference_threads").value)
        if threads < 1:
            raise ValueError("inference_threads must be positive")
        torch.set_num_threads(threads)
        device = str(self.get_parameter("inference_device").value)
        if device == "auto":
            device = 0 if torch.cuda.is_available() else "cpu"
        weights = str(self.get_parameter("detector_weights").value)
        image_size = int(self.get_parameter("inference_image_size").value)
        if image_size < 0 or image_size % 32:
            raise ValueError("inference_image_size must be zero or a multiple of 32")
        if weights:
            if not Path(weights).is_file():
                raise ValueError(f"Detector weights not found: {weights}")
            self.model = YOLO(weights)
            self.classNames = list(self.model.names.values())
        self.model = RuntimeDetector(self.model, device, image_size)
        self.get_logger().info(f"Detector device={device}, CPU threads={threads}")
        self.declare_parameter("tracked_boxes_topic", "/tracked_boxes")
        self.declare_parameter("max_frame_start_age_sec", 0.25)
        output_topic = self.get_parameter("tracked_boxes_topic").value
        self.max_frame_start_age_sec = float(
            self.get_parameter("max_frame_start_age_sec").value
        )
        if (
            not math.isfinite(self.max_frame_start_age_sec)
            or self.max_frame_start_age_sec <= 0.0
        ):
            raise ValueError("max_frame_start_age_sec must be positive")
        noise_defaults = {
            "pixel_measurement_std": 1.0,
            "depth_measurement_std": 0.05,
            "pixel_process_std": 50.0,
            "depth_process_std": 0.8,
            "initial_pixel_velocity_std": 50.0,
            "initial_depth_velocity_std": 1.0,
        }
        self.image_noise = {}
        for name, default in noise_defaults.items():
            self.declare_parameter(name, default)
            value = float(self.get_parameter(name).value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            self.image_noise[name] = value
        self.image_noise["pixel_std"] = self.image_noise.pop("pixel_measurement_std")
        self.image_noise["depth_std"] = self.image_noise.pop("depth_measurement_std")
        self.declare_parameter("noise_reference_width", 0)
        self.noise_reference_width = int(
            self.get_parameter("noise_reference_width").value
        )
        if self.noise_reference_width < 0:
            raise ValueError("noise_reference_width cannot be negative")
        self._frame_width = self.noise_reference_width
        self._stale_frames_skipped = 0
        self.publisher_tracks = self.create_publisher(String, output_topic, 1)

        original_tracking = self.tracker.tracking

        def capture_tracking_result(*args, **kwargs):
            """Run image tracking and retain current measurements with assigned IDs."""
            self._tracking_called = True
            # Use elapsed frame time in the image constant-rate model.
            for track in self.tracker.trackers:
                self._configure_track_noise(track)
            detections = np.asarray(
                args[0] if args else kwargs.get("detections", []), dtype=float
            )
            detections = (
                detections.reshape(-1, 6) if detections.size else np.empty((0, 6))
            )
            detections = detections[np.all(np.isfinite(detections), axis=1)]
            rows = original_tracking(detections)
            # New filters are created inside tracking(); initialize them in
            # sensor units before their covariance is published for the first time.
            for track in self.tracker.trackers:
                if not getattr(track, "_metric_noise_configured", False):
                    self._configure_track_noise(track)
            noise = scale_pixel_noise(
                self.image_noise, self._frame_width, self.noise_reference_width
            )
            self._captured_rows = current_frame_rows(
                detections, rows, noise["pixel_std"], noise["depth_std"]
            )
            for measurement in self._captured_rows:
                spread = getattr(self, "_depth_spreads", {}).get(
                    tuple(measurement["bbox"]), noise["depth_std"]
                )
                measurement["covariance_uvd"][2][2] = spread**2
            return rows

        # Wrap tracking to retain current measurements alongside assigned IDs.
        self.tracker.tracking = capture_tracking_result
        self.get_logger().info(
            f"Publishing current-frame RGB-D observations on {output_topic}"
        )

    def get_depth(self, depth_image, bbox_arngd):
        # Return NaN on unsupported regions; the projector must not mistake
        # no-return pixels, floor or a far-plane sentinel for a person.
        """Estimate supported depths and retain per-box depth spread."""
        result = {}
        self._depth_spreads = {}
        for key, box in bbox_arngd.items():
            measurement = depth_mode(
                depth_image, box, noise_floor=self.image_noise["depth_std"]
            )
            result[key] = float("nan") if measurement is None else measurement[0]
            if measurement is not None:
                self._depth_spreads[tuple(float(v) for v in box[:4])] = measurement[1]
        return result

    def _configure_track_noise(self, track):
        """Apply frame timing and noise units to an image Kalman filter."""
        initialize = not getattr(track, "_metric_noise_configured", False)
        configure_image_filter(
            track.kf,
            self._frame_dt,
            initialize=initialize,
            **scale_pixel_noise(
                self.image_noise, self._frame_width, self.noise_reference_width
            ),
        )
        track._metric_noise_configured = True

    def sync_callback(self, rgb_msg, depth_msg) -> None:
        """Process a synchronized frame and publish one observation message."""
        # Reject stale queued frames before running detector inference.
        try:
            stamp_ns = stamp_to_nanoseconds(rgb_msg.header.stamp)
            depth_stamp_ns = stamp_to_nanoseconds(depth_msg.header.stamp)
        except (AttributeError, TypeError, ValueError) as exc:
            self.get_logger().warn(f"Skipping frame with invalid timestamp: {exc}")
            return
        if abs(depth_stamp_ns - stamp_ns) > int(
            self.get_parameter("rgbd_sync_slop_sec").value * 1e9
        ):
            self.get_logger().warn(
                "Rejecting RGB-D frame pair with excessive timestamp skew"
            )
            return
        if rgb_msg.width != depth_msg.width or rgb_msg.height != depth_msg.height:
            self.get_logger().warn(
                "Rejecting RGB-D frame pair with different image dimensions"
            )
            return
        if self._last_processed_stamp_ns is not None:
            if stamp_ns == self._last_processed_stamp_ns:
                return
            if stamp_ns < self._last_processed_stamp_ns:
                self.tracker.trackers.clear()
                self._last_processed_stamp_ns = None
        now_ns = self.get_clock().now().nanoseconds
        if stamp_ns != 0:
            start_age_sec = timestamp_age_seconds(stamp_ns, now_ns)
            if start_age_sec > self.max_frame_start_age_sec:
                self._stale_frames_skipped += 1
                if (
                    self._stale_frames_skipped == 1
                    or self._stale_frames_skipped % 30 == 0
                ):
                    self.get_logger().warn(
                        "Draining stale RGB-D backlog: skipped "
                        f"{self._stale_frames_skipped} frame(s); newest skipped frame "
                        f"was {start_age_sec:.3f} s old"
                    )
                return

        if (
            self._last_processed_stamp_ns is not None
            and stamp_ns > self._last_processed_stamp_ns
        ):
            self._frame_dt = (stamp_ns - self._last_processed_stamp_ns) / 1e9
        else:
            self._frame_dt = 1.0
        self._last_processed_stamp_ns = stamp_ns
        self._frame_width = int(rgb_msg.width)
        self._captured_rows = []
        self._tracking_called = False
        try:
            super().sync_callback(rgb_msg, depth_msg)
        finally:
            # Advance track prediction and retirement on empty detection frames.
            if not self._tracking_called:
                self.tracker.tracking(np.empty((0, 6)))
            # Publish frame metadata even when no usable measurement is available.
            # An empty result may also follow a processing error in the callback.
            if stamp_ns == 0:
                stamp_ns = self.get_clock().now().nanoseconds
            frame_id = depth_msg.header.frame_id or rgb_msg.header.frame_id
            output = String()
            output.data = encode_tracked_boxes(
                self._captured_rows,
                stamp_nanoseconds=stamp_ns,
                frame_id=frame_id,
            )
            self.publisher_tracks.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RGBDTrackingAdapterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

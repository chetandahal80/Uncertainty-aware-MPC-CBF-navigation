"""Geometry helpers for the RGB-D-to-world integration.

The preferred path uses the live camera calibration and the ROS transform tree.
The planar odometry-based conversion remains available only as a configurable
fallback for simulations that do not publish the camera transform.
"""

import math
from typing import Iterable, Tuple

DEFAULT_HFOV = 1.089
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FX = DEFAULT_WIDTH / (2.0 * math.tan(DEFAULT_HFOV / 2.0))
DEFAULT_FY = DEFAULT_FX
DEFAULT_CX = DEFAULT_WIDTH / 2.0
DEFAULT_CY = DEFAULT_HEIGHT / 2.0


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Return planar yaw from a finite, normalized quaternion."""
    qx, qy, qz, qw = (float(value) for value in (qx, qy, qz, qw))
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("Quaternion is invalid")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def interpolate_yaw(yaw0: float, yaw1: float, alpha: float) -> float:
    """Interpolate yaw along the shortest angular arc."""
    return wrap_to_pi(yaw0 + float(alpha) * wrap_to_pi(yaw1 - yaw0))


def project_pixel_to_optical(
    u: float,
    v: float,
    depth: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[float, float, float]:
    """Back-project a pixel using optical-axis depth.

    The returned point follows the standard optical-frame convention:
    ``x`` right, ``y`` down, and ``z`` forward.
    """
    values = (u, v, depth, fx, fy, cx, cy)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Pixel, depth, and camera intrinsics must be finite")
    if depth <= 0.0:
        raise ValueError("Depth must be positive")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("Focal lengths must be positive")
    return (
        (float(u) - float(cx)) * float(depth) / float(fx),
        (float(v) - float(cy)) * float(depth) / float(fy),
        float(depth),
    )


def rotate_vector_by_quaternion(
    vector: Iterable[float],
    quaternion: Iterable[float],
) -> Tuple[float, float, float]:
    """Rotate a three-dimensional vector by quaternion ``(x, y, z, w)``."""
    vx, vy, vz = (float(value) for value in vector)
    qx, qy, qz, qw = (float(value) for value in quaternion)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("Transform quaternion is invalid")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def transform_point(
    point: Iterable[float],
    translation: Iterable[float],
    quaternion: Iterable[float],
) -> Tuple[float, float, float]:
    """Apply a rigid transform to a point."""
    rx, ry, rz = rotate_vector_by_quaternion(point, quaternion)
    tx, ty, tz = (float(value) for value in translation)
    return rx + tx, ry + ty, rz + tz


def pixel_depth_to_world_planar(
    u: float,
    v: float,
    depth: float,
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    *,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
    camera_offset_forward: float = 0.205,
    camera_offset_left: float = 0.0,
    camera_yaw: float = 0.0,
) -> Tuple[float, float]:
    """Fallback planar transform when a ROS transform is unavailable.

    Camera height is intentionally absent because the controller consumes only
    ground-plane ``x`` and ``y``. Camera forward, lateral, and yaw offsets must
    match the robot description whenever this fallback is enabled.
    """
    x_opt, _y_opt, z_opt = project_pixel_to_optical(u, v, depth, fx, fy, cx, cy)
    camera_forward = z_opt
    camera_left = -x_opt

    cos_mount = math.cos(camera_yaw)
    sin_mount = math.sin(camera_yaw)
    forward = (
        camera_offset_forward + cos_mount * camera_forward - sin_mount * camera_left
    )
    left = camera_offset_left + sin_mount * camera_forward + cos_mount * camera_left

    cos_robot = math.cos(robot_theta)
    sin_robot = math.sin(robot_theta)
    return (
        robot_x + forward * cos_robot - left * sin_robot,
        robot_y + forward * sin_robot + left * cos_robot,
    )


# Backward-compatible name used by older callers.
pixel_depth_to_world = pixel_depth_to_world_planar


def covariance_uvd_to_world(
    u, v, depth, fx, fy, cx, cy, covariance_uvd, rotation_optical_to_world
):
    """Propagate pixel/depth covariance: R = J_g Sigma_uvd J_g.T.

    rotation_optical_to_world is the complete camera-optical to world rotation;
    translation has zero measurement Jacobian. Keep the horizontal two rows.
    """
    import numpy as np

    project_pixel_to_optical(u, v, depth, fx, fy, cx, cy)
    covariance = np.asarray(covariance_uvd, dtype=float)
    rotation = np.asarray(rotation_optical_to_world, dtype=float)
    if (
        covariance.shape != (3, 3)
        or not np.all(np.isfinite(covariance))
        or not np.allclose(covariance, covariance.T)
        or np.linalg.eigvalsh(covariance)[0] < -1e-10
    ):
        raise ValueError("Invalid u,v,depth covariance")
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("Invalid optical-to-world rotation")
    jacobian_optical = np.array(
        [[depth / fx, 0, (u - cx) / fx], [0, depth / fy, (v - cy) / fy], [0, 0, 1]]
    )
    jacobian_world = (rotation @ jacobian_optical)[:2]
    result = jacobian_world @ covariance @ jacobian_world.T
    return (result + result.T) / 2


def optical_to_world_planar_rotation(robot_yaw, camera_yaw=0.0):
    """Optical (right,down,forward) to world, for a level planar camera."""
    import numpy as np

    theta = robot_yaw + camera_yaw
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[s, 0, c], [-c, 0, s], [0, -1, 0]], dtype=float)

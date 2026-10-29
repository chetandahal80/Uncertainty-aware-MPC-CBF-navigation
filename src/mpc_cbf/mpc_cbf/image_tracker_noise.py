"""Time-dependent noise configuration for the OpenCV image Kalman filter."""

import math
import numpy as np


def configure_image_filter(
    kf,
    dt,
    *,
    pixel_std=1.0,
    depth_std=0.05,
    pixel_process_std=50.0,
    depth_process_std=0.8,
    initial_pixel_velocity_std=50.0,
    initial_depth_velocity_std=1.0,
    initialize=False
):
    """Configure u/v in pixels, depth in metres, and rate states per second.

    Retain area/aspect-ratio dynamics. For u/v/depth, use the
    continuous white-acceleration model with q=process_std**2, matching
    the constant-rate state model. Q is recomputed for actual dt.
    """
    values = (
        dt,
        pixel_std,
        depth_std,
        pixel_process_std,
        depth_process_std,
        initial_pixel_velocity_std,
        initial_depth_velocity_std,
    )
    if any(not math.isfinite(float(v)) or v <= 0 for v in values):
        raise ValueError("Image tracker times and noise values must be positive")
    transition = np.asarray(kf.transitionMatrix).copy()
    for position, rate in ((0, 5), (1, 6), (2, 7), (4, 8)):
        transition[position, rate] = dt
    kf.transitionMatrix = transition
    R = np.asarray(kf.measurementNoiseCov).copy()
    for index, std in ((0, pixel_std), (1, pixel_std), (4, depth_std)):
        R[index, index] = std**2
    kf.measurementNoiseCov = R
    Q = np.asarray(kf.processNoiseCov).copy()
    covariance = np.asarray(kf.errorCovPost).copy()
    for position, rate, std, process_std, velocity_std in (
        (0, 5, pixel_std, pixel_process_std, initial_pixel_velocity_std),
        (1, 6, pixel_std, pixel_process_std, initial_pixel_velocity_std),
        (4, 8, depth_std, depth_process_std, initial_depth_velocity_std),
    ):
        Q[position, :] = Q[:, position] = 0
        Q[rate, :] = Q[:, rate] = 0
        block = process_std**2 * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        Q[np.ix_([position, rate], [position, rate])] = block
        if initialize:
            covariance[position, :] = covariance[:, position] = 0
            covariance[rate, :] = covariance[:, rate] = 0
            covariance[position, position] = std**2
            covariance[rate, rate] = velocity_std**2
    kf.processNoiseCov = Q
    if initialize:
        kf.errorCovPost = covariance


def scale_pixel_noise(noise, image_width, reference_width=0):
    """Keep calibrated angular uncertainty consistent if image width changes.

    The calibration assumes the same camera FOV and detector inference size.
    Metric-depth parameters are never scaled by image resolution.
    """
    result = dict(noise)
    if reference_width == 0:
        return result
    if reference_width < 0 or image_width <= 0:
        raise ValueError("Image and noise-reference widths must be positive")
    factor = float(image_width) / float(reference_width)
    for key in ("pixel_std", "pixel_process_std", "initial_pixel_velocity_std"):
        result[key] *= factor
    return result

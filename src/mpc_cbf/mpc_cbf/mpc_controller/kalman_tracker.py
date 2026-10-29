"""Numerically robust constant-velocity tracking and horizon prediction."""

import math
from typing import Optional, Tuple

import numpy as np


class ConstantVelocityKF:
    """Two-dimensional constant-velocity Kalman filter.

    State order is ``[px, py, vx, vy]``. Prediction accepts the measured time
    interval rather than assuming that camera frames arrive at a perfect rate.
    """

    def __init__(
        self,
        px0: float,
        py0: float,
        vx0: float = 0.0,
        vy0: float = 0.0,
        process_noise_std: float = 0.25298221281347033,
        measurement_noise_std: float = 0.15,
        initial_pos_std: float = 0.25,
        initial_vel_std: float = 1.0,
        initial_measurement_covariance=None,
    ) -> None:
        initial = np.asarray([px0, py0, vx0, vy0], dtype=float)
        if not np.all(np.isfinite(initial)):
            raise ValueError("Initial Kalman state must be finite")
        positive = (
            process_noise_std,
            measurement_noise_std,
            initial_pos_std,
            initial_vel_std,
        )
        if any(not math.isfinite(float(value)) or value <= 0.0 for value in positive):
            raise ValueError("Kalman noise values must be finite and positive")
        self.s = initial
        self.P = np.diag(
            [
                initial_pos_std**2,
                initial_pos_std**2,
                initial_vel_std**2,
                initial_vel_std**2,
            ]
        )
        self.q_std = float(process_noise_std)
        self.default_measurement_std = float(measurement_noise_std)
        # Initialize position covariance from the first measurement covariance.
        self.P[:2, :2] = self.measurement_covariance(initial_measurement_covariance)
        self.H = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=float,
        )

    def _F_Q(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        dt = float(dt)
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("Prediction interval must be finite and nonnegative")
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # Continuous white-acceleration spectral density q [m^2/s^3].
        q = self.q_std**2
        Q = q * np.array(
            [
                [dt**3 / 3.0, 0.0, dt**2 / 2.0, 0.0],
                [0.0, dt**3 / 3.0, 0.0, dt**2 / 2.0],
                [dt**2 / 2.0, 0.0, dt, 0.0],
                [0.0, dt**2 / 2.0, 0.0, dt],
            ]
        )
        return F, Q

    def predict(self, dt: float) -> None:
        """Advance the filter by ``dt`` seconds."""
        if dt <= 0.0:
            return
        F, Q = self._F_Q(dt)
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def measurement_covariance(self, measurement_std=None):
        """Accept a world-frame measurement covariance or scalar standard deviation."""
        if measurement_std is None:
            return np.eye(2) * self.default_measurement_std**2
        value = np.asarray(measurement_std, dtype=float)
        if value.ndim == 0:
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Measurement standard deviation must be positive")
            return np.eye(2) * float(value) ** 2
        if (
            value.shape != (2, 2)
            or not np.all(np.isfinite(value))
            or not np.allclose(value, value.T, atol=1e-10)
            or np.linalg.eigvalsh(value)[0] < -1e-10
        ):
            raise ValueError("Measurement covariance must be finite, symmetric PSD")
        return (value + value.T) / 2

    def innovation_statistics(
        self,
        measurement,
        measurement_std: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        z = np.asarray(measurement, dtype=float).reshape(2)
        if not np.all(np.isfinite(z)):
            raise ValueError("Measurement must be finite")
        innovation = z - self.H @ self.s
        covariance = self.H @ self.P @ self.H.T + self.measurement_covariance(
            measurement_std
        )
        distance_squared = float(innovation.T @ np.linalg.solve(covariance, innovation))
        return innovation, covariance, distance_squared

    def update(self, measurement, measurement_std: Optional[float] = None) -> None:
        """Correct the filter using a world-frame position measurement."""
        innovation, innovation_cov, _ = self.innovation_statistics(
            measurement, measurement_std
        )
        # Solve instead of explicitly inverting the innovation covariance.
        K = np.linalg.solve(innovation_cov, self.H @ self.P).T
        self.s = self.s + K @ innovation

        # Joseph form preserves symmetry and positive semidefiniteness better
        # than the abbreviated (I-KH)P update.
        identity = np.eye(4)
        residual = identity - K @ self.H
        R = self.measurement_covariance(measurement_std)
        self.P = residual @ self.P @ residual.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def step(
        self,
        measurement,
        dt: float,
        measurement_std: Optional[float] = None,
    ) -> None:
        self.predict(dt)
        self.update(measurement, measurement_std)

    @property
    def position(self) -> np.ndarray:
        return self.s[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.s[2:].copy()

    @property
    def position_covariance(self) -> np.ndarray:
        return self.P[:2, :2].copy()

    def predict_horizon(
        self,
        steps: int,
        sample_time: float,
        start_time_offset: float = 0.0,
    ):
        """Predict position and uncertainty from a delayed measurement time.

        ``start_time_offset`` advances a copy of the filter to the current
        control time before horizon sample zero. The stored filter state is not
        mutated, so a later camera measurement can still be processed at its
        true acquisition timestamp.
        """
        steps = int(steps)
        sample_time = float(sample_time)
        start_time_offset = float(start_time_offset)
        if (
            steps < 0
            or sample_time <= 0.0
            or not math.isfinite(start_time_offset)
            or start_time_offset < 0.0
        ):
            raise ValueError("Horizon values and start offset are invalid")
        state = self.s.copy()
        covariance = self.P.copy()
        if start_time_offset > 0.0:
            F_offset, Q_offset = self._F_Q(start_time_offset)
            state = F_offset @ state
            covariance = F_offset @ covariance @ F_offset.T + Q_offset
            covariance = 0.5 * (covariance + covariance.T)
        px = np.zeros(steps + 1)
        py = np.zeros(steps + 1)
        trace_pos = np.zeros(steps + 1)
        F, Q = self._F_Q(sample_time)
        for index in range(steps + 1):
            px[index], py[index] = state[0], state[1]
            trace_pos[index] = max(float(np.trace(covariance[:2, :2])), 0.0)
            if index < steps:
                state = F @ state
                covariance = F @ covariance @ F.T + Q
                covariance = 0.5 * (covariance + covariance.T)
        # Exact covariance recursion retains Qpv, including its cross terms.
        # No monotonic clipping: valid cross-covariance can reduce the trace.
        return px, py, trace_pos


def make_d_safe_array(trace_pos, d_safe_base: float, kappa: float = 1.0):
    """Compute d_safe = d_min + kappa * trace(P_pos).

    trace(P_pos) has units m^2, so kappa has units 1/m. No sigma,
    acceleration add-on, cap, or monotonic clipping changes this formula.
    """
    trace = np.asarray(trace_pos, dtype=float)
    if trace.ndim != 1 or not np.all(np.isfinite(trace)) or np.any(trace < 0):
        raise ValueError("Position covariance traces must be finite and nonnegative")
    if (
        not np.isfinite(d_safe_base)
        or d_safe_base <= 0
        or not np.isfinite(kappa)
        or kappa < 0
    ):
        raise ValueError("Invalid d_min or kappa")
    return float(d_safe_base) + float(kappa) * trace

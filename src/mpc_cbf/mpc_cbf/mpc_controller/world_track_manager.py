"""World-frame track continuity layer for safety-critical control.

The image tracker identifiers are treated as useful hints, not unquestionable
identity. Position/covariance gating can reconnect a new source identifier to
an existing physical track after an upstream identifier change.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List, Set, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman_tracker import ConstantVelocityKF


@dataclass
class WorldTrack:
    internal_id: int
    source_id: int
    kf: ConstantVelocityKF
    last_seen_time: float
    state_time: float
    hits: int = 1
    misses: int = 0
    source_id_history: Set[int] = field(default_factory=set)
    last_observed_position: Optional[np.ndarray] = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        self.source_id_history.add(int(self.source_id))
        if self.last_observed_position is None:
            self.last_observed_position = self.kf.position


class WorldTrackManager:
    """Associate timestamped world measurements and maintain stable tracks."""

    INVALID_COST = 1.0e9

    def __init__(
        self,
        *,
        process_noise_std: float = 0.25298221281347033,
        measurement_noise_std: float = 0.15,
        initial_pos_std: float = 0.25,
        initial_vel_std: float = 1.0,
        stale_timeout_sec: float = 1.0,
        mahalanobis_gate: float = 9.21,
        association_max_distance: float = 1.0,
        assumed_human_speed_max: float = 2.0,
        source_id_bonus: float = 2.0,
        max_filter_speed: float = 0.0,
        duplicate_measurement_distance: float = 0.20,
        max_tracks: int = 50,
    ) -> None:
        self.kf_kwargs = {
            "process_noise_std": process_noise_std,
            "measurement_noise_std": measurement_noise_std,
            "initial_pos_std": initial_pos_std,
            "initial_vel_std": initial_vel_std,
        }
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.kf_kwargs.values()
        ):
            raise ValueError("Kalman noise values must be finite and positive")
        self.default_measurement_noise_std = float(measurement_noise_std)
        self.stale_timeout_sec = float(stale_timeout_sec)
        self.mahalanobis_gate = float(mahalanobis_gate)
        self.association_max_distance = float(association_max_distance)
        self.assumed_human_speed_max = float(assumed_human_speed_max)
        self.source_id_bonus = float(source_id_bonus)
        self.max_filter_speed = float(max_filter_speed)
        self.duplicate_measurement_distance = float(duplicate_measurement_distance)
        self.max_tracks = int(max_tracks)
        positive_values = (
            self.stale_timeout_sec,
            self.mahalanobis_gate,
            self.association_max_distance,
            self.assumed_human_speed_max,
            self.duplicate_measurement_distance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_values):
            raise ValueError(
                "World-track manager thresholds must be finite and positive"
            )
        if self.max_tracks < 1:
            raise ValueError("max_tracks must be positive")
        nonnegative_values = (
            self.source_id_bonus,
            self.max_filter_speed,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in nonnegative_values):
            raise ValueError(
                "World-track manager bonuses and limits must be finite and nonnegative"
            )
        self.tracks: List[WorldTrack] = []
        self.next_internal_id = 1
        self.last_update_time = None

    def _validated_measurements(self, measurements: Iterable[Dict]) -> List[Dict]:
        validated = []
        for measurement in measurements:
            try:
                source_id = int(measurement.get("id", measurement.get("source_id")))
                x = float(measurement["x"])
                y = float(measurement["y"])
                std = float(
                    measurement.get(
                        "measurement_std", self.default_measurement_noise_std
                    )
                )
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(value) for value in (x, y, std)) or std <= 0.0:
                continue
            covariance = measurement.get("measurement_covariance", np.eye(2) * std**2)
            covariance = np.asarray(covariance, dtype=float)
            if (
                covariance.shape != (2, 2)
                or not np.all(np.isfinite(covariance))
                or not np.allclose(covariance, covariance.T)
                or np.linalg.eigvalsh(covariance)[0] < -1e-10
            ):
                continue
            validated.append(
                {
                    "source_id": source_id,
                    "x": x,
                    "y": y,
                    "measurement_std": std,
                    "measurement_covariance": covariance,
                }
            )
        return validated

    def _deduplicate_measurements(self, measurements: List[Dict]) -> List[Dict]:
        """Suppress physically coincident rows before association.

        Different upstream identifiers can briefly describe the same person.
        Keeping the lower-noise row prevents duplicate tracks, including on
        the first frame when no existing track is available for comparison.
        """
        deduplicated: List[Dict] = []
        for measurement in sorted(
            measurements, key=lambda item: item["measurement_std"]
        ):
            position = np.array([measurement["x"], measurement["y"]])
            if any(
                np.linalg.norm(position - np.array([existing["x"], existing["y"]]))
                < self.duplicate_measurement_distance
                for existing in deduplicated
            ):
                continue
            deduplicated.append(measurement)
        return deduplicated

    def _new_track(self, measurement: Dict, timestamp: float) -> WorldTrack:
        track = WorldTrack(
            internal_id=self.next_internal_id,
            source_id=measurement["source_id"],
            kf=ConstantVelocityKF(
                measurement["x"],
                measurement["y"],
                initial_measurement_covariance=measurement["measurement_covariance"],
                **self.kf_kwargs,
            ),
            last_seen_time=timestamp,
            state_time=timestamp,
        )
        self.next_internal_id += 1
        return track

    def _predict_to(self, track: WorldTrack, timestamp: float) -> None:
        dt = timestamp - track.state_time
        if dt < -1.0e-6:
            raise ValueError("Out-of-order world-track timestamp")
        if dt > 0.0:
            track.kf.predict(dt)
            track.state_time = timestamp

    def update(
        self, measurements: Iterable[Dict], timestamp: float
    ) -> List[WorldTrack]:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("Track timestamp must be finite and nonnegative")
        if (
            self.last_update_time is not None
            and timestamp < self.last_update_time - 1.0e-6
        ):
            raise ValueError("Track frames must be processed in timestamp order")
        self.last_update_time = timestamp

        parsed = self._deduplicate_measurements(
            self._validated_measurements(measurements)
        )
        self.tracks = [
            track
            for track in self.tracks
            if timestamp - track.last_seen_time <= self.stale_timeout_sec
        ]
        for track in self.tracks:
            self._predict_to(track, timestamp)

        if not self.tracks:
            self.tracks = [self._new_track(item, timestamp) for item in parsed]
            return list(self.tracks)

        if not parsed:
            for track in self.tracks:
                track.misses += 1
            return list(self.tracks)

        cost = np.full((len(self.tracks), len(parsed)), self.INVALID_COST)
        for track_index, track in enumerate(self.tracks):
            time_missing = max(timestamp - track.last_seen_time, 0.0)
            distance_gate = (
                self.association_max_distance
                + self.assumed_human_speed_max * time_missing
            )
            for measurement_index, measurement in enumerate(parsed):
                delta = (
                    np.array([measurement["x"], measurement["y"]]) - track.kf.position
                )
                euclidean = float(np.linalg.norm(delta))
                if euclidean > distance_gate:
                    continue
                try:
                    _, _, distance_squared = track.kf.innovation_statistics(
                        (measurement["x"], measurement["y"]),
                        measurement["measurement_covariance"],
                    )
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if distance_squared > self.mahalanobis_gate:
                    continue
                value = distance_squared
                if measurement["source_id"] == track.source_id:
                    value = max(0.0, value - self.source_id_bonus)
                cost[track_index, measurement_index] = value

        row_indices, column_indices = linear_sum_assignment(cost)
        matched_tracks = set()
        matched_measurements = set()
        for track_index, measurement_index in zip(row_indices, column_indices):
            if cost[track_index, measurement_index] >= self.INVALID_COST:
                continue
            track = self.tracks[track_index]
            measurement = parsed[measurement_index]
            track.kf.update(
                (measurement["x"], measurement["y"]),
                measurement["measurement_covariance"],
            )
            track.source_id = measurement["source_id"]
            track.source_id_history.add(measurement["source_id"])
            track.last_observed_position = np.array(
                [measurement["x"], measurement["y"]]
            )
            track.last_seen_time = timestamp
            track.hits += 1
            track.misses = 0
            matched_tracks.add(track_index)
            matched_measurements.add(measurement_index)

        for index, track in enumerate(self.tracks):
            if index not in matched_tracks:
                track.misses += 1

        for index, measurement in enumerate(parsed):
            if index in matched_measurements:
                continue
            measurement_position = np.array([measurement["x"], measurement["y"]])
            if any(
                np.linalg.norm(measurement_position - track.kf.position)
                < self.duplicate_measurement_distance
                for track in self.tracks
            ):
                continue
            self.tracks.append(self._new_track(measurement, timestamp))

        self.tracks.sort(key=lambda item: item.internal_id)
        return list(self.tracks)

    def clear(self, *, reset_ids: bool = True) -> None:
        self.tracks.clear()
        self.last_update_time = None
        if reset_ids:
            self.next_internal_id = 1

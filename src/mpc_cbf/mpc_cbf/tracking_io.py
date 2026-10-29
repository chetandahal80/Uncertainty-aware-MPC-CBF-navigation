"""Versioned JSON payload helpers shared by the integration nodes.

The decoder intentionally accepts the original list-only payload so existing
publishers keep working. New publishers should use the versioned envelope so
the image timestamp and camera frame are preserved.
"""

import json
import math
from typing import Any, Dict, Iterable, List, Tuple

SCHEMA_VERSION = 1


def timestamp_age_seconds(stamp_nanoseconds: int, now_nanoseconds: int) -> float:
    """Return signed message age using integer nanosecond timestamps."""
    return (int(now_nanoseconds) - int(stamp_nanoseconds)) / 1.0e9


def stamp_to_nanoseconds(stamp: Any) -> int:
    """Convert a ROS-like stamp or stamp dictionary to integer nanoseconds."""
    if isinstance(stamp, dict):
        sec = int(stamp.get("sec", 0))
        nanosec = int(stamp.get("nanosec", 0))
    else:
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError("Invalid timestamp")
    return sec * 1_000_000_000 + nanosec


def nanoseconds_to_stamp_dict(nanoseconds: int) -> Dict[str, int]:
    nanoseconds = max(int(nanoseconds), 0)
    sec, nanosec = divmod(nanoseconds, 1_000_000_000)
    return {"sec": sec, "nanosec": nanosec}


def encode_tracked_boxes(
    rows: Iterable[Iterable[float]],
    stamp_nanoseconds: int,
    frame_id: str,
) -> str:
    """Encode tracker rows as a timestamped, versioned JSON envelope."""
    tracks = []
    for row in () if rows is None else rows:
        try:
            values = (
                list(row["bbox"]) + [row["depth"], row["id"]]
                if isinstance(row, dict)
                else list(row)
            )
            if len(values) < 6:
                continue
            numeric = [float(value) for value in values[:5]]
            source_id_float = float(values[5])
            source_id = int(source_id_float)
        except (TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(value) for value in numeric + [source_id_float]):
            continue
        if source_id_float != source_id:
            continue
        if isinstance(row, dict) and "covariance_uvd" in row:
            # Validate dimensions/PSD in the world projector, preserve here.
            tracks.append(
                {
                    "bbox": numeric[:4],
                    "depth": numeric[4],
                    "id": source_id,
                    "covariance_uvd": row["covariance_uvd"],
                }
            )
        else:
            tracks.append(numeric + [source_id])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stamp": nanoseconds_to_stamp_dict(stamp_nanoseconds),
        "frame_id": str(frame_id or ""),
        "tracks": tracks,
    }
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def decode_tracked_boxes(
    payload: str,
    fallback_stamp_nanoseconds: int,
) -> Tuple[int, str, List[Any], bool]:
    """Decode either the new envelope or the legacy list-only payload.

    Returns ``(stamp_nanoseconds, frame_id, tracks, used_legacy_format)``.
    Row-level validation is deliberately left to the consuming node so it can
    report precise rejection counts.
    """
    decoded = json.loads(payload)
    if isinstance(decoded, list):
        return int(fallback_stamp_nanoseconds), "", decoded, True
    if not isinstance(decoded, dict):
        raise ValueError("Tracking payload must be a list or object")
    version = int(decoded.get("schema_version", SCHEMA_VERSION))
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported tracking schema version: {version}")
    tracks = decoded.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("Tracking payload field 'tracks' must be a list")
    stamp = decoded.get("stamp")
    stamp_ns = (
        stamp_to_nanoseconds(stamp)
        if stamp is not None
        else int(fallback_stamp_nanoseconds)
    )
    return stamp_ns, str(decoded.get("frame_id", "")), tracks, False


def encode_world_tracks(
    tracks: Iterable[Dict[str, Any]],
    stamp_nanoseconds: int,
    frame_id: str,
) -> str:
    """Encode validated world-frame measurements."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stamp": nanoseconds_to_stamp_dict(stamp_nanoseconds),
        "frame_id": str(frame_id),
        "tracks": list(tracks),
    }
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def decode_world_tracks(
    payload: str,
    fallback_stamp_nanoseconds: int,
) -> Tuple[int, str, List[Any], bool]:
    """Decode timestamped world tracks, with legacy-list compatibility."""
    decoded = json.loads(payload)
    if isinstance(decoded, list):
        # Legacy messages never carried a frame. An empty frame lets the
        # consumer apply its configured world frame instead of incorrectly
        # assuming that every deployment uses ``odom``.
        return int(fallback_stamp_nanoseconds), "", decoded, True
    if not isinstance(decoded, dict):
        raise ValueError("World-track payload must be a list or object")
    version = int(decoded.get("schema_version", SCHEMA_VERSION))
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported world-track schema version: {version}")
    tracks = decoded.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("World-track payload field 'tracks' must be a list")
    stamp = decoded.get("stamp")
    stamp_ns = (
        stamp_to_nanoseconds(stamp)
        if stamp is not None
        else int(fallback_stamp_nanoseconds)
    )
    return stamp_ns, str(decoded.get("frame_id", "")), tracks, False

"""Obstacle-aware route references and dynamically consistent control seeds.

The geometric route guides the MPC objective. Time-indexed collision constraints
remain part of the optimization and are checked separately after solving.
"""

import heapq
import math
import numpy as np
from .mpc_core import unicycle_step, wrap_to_pi


def segment_clear(a, b, centers, radii, tolerance=1e-07):
    if len(centers) == 0:
        return True
    delta = np.asarray(b) - a
    t = np.clip((centers - a) @ delta / max(float(delta @ delta), 1e-16), 0.0, 1.0)
    return bool(
        np.all(
            np.linalg.norm(centers - (a + t[:, None] * delta), axis=1)
            >= radii - tolerance
        )
    )


def visibility_route(
    start,
    goal,
    centers,
    radii,
    padding=0.12,
    vertices=12,
    allow_goal_proxy=False,
    allow_start_recovery=False,
):
    """Visibility guide, optionally with a separating start-recovery edge."""
    start = np.asarray(start, dtype=float).reshape(2)
    goal = np.asarray(goal, dtype=float).reshape(2)
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if (
        len(centers) != len(radii)
        or np.any(radii <= 0)
        or (not all((np.all(np.isfinite(a)) for a in (start, goal, centers, radii))))
    ):
        raise ValueError("Invalid guide geometry")
    if segment_clear(start, goal, centers, radii):
        return np.array([start, goal])
    # An initial state inside a planning circle requires an outward recovery edge.
    if not allow_start_recovery and np.any(
        np.linalg.norm(centers - start, axis=1) < radii - 1e-06
    ):
        return None
    angles = np.arange(vertices) * 2 * np.pi / vertices
    offsets = np.column_stack([np.cos(angles), np.sin(angles)])
    ring = (
        centers[:, None, :]
        + ((radii + padding) / np.cos(np.pi / vertices))[:, None, None] * offsets
    )
    nodes = np.vstack([start, goal, ring.reshape(-1, 2)])
    # Discard vertices inside any other circle.
    valid = np.ones(len(nodes), dtype=bool)
    for c, r in zip(centers, radii):
        valid[2:] &= np.linalg.norm(nodes[2:] - c, axis=1) >= r - 1e-07
    nodes = np.vstack([nodes[:2], nodes[2:][valid[2:]]])
    ii, jj = np.triu_indices(len(nodes), 1)
    a, b = (nodes[ii], nodes[jj])
    v = b - a
    length2 = np.sum(v * v, axis=1)
    visible = length2 > 1e-14
    for c, r in zip(centers, radii):
        t = np.clip(np.sum((c - a) * v, axis=1) / np.maximum(length2, 1e-16), 0.0, 1.0)
        clear = np.linalg.norm(c - (a + t[:, None] * v), axis=1) >= r - 1e-07
        if allow_start_recovery and np.linalg.norm(start - c) < r:
            # Only start edges may leave an already-entered planning margin.
            # The squared distance derivative is nonnegative at the start,
            # so separation increases monotonically until the edge exits.
            clear |= (
                (ii == 0)
                & (np.linalg.norm(b - c, axis=1) >= r - 1e-7)
                & (np.sum((start - c) * v, axis=1) >= -1e-9)
            )
        visible &= clear
    graph = [[] for _ in nodes]
    for i, j, d in zip(ii[visible], jj[visible], np.sqrt(length2[visible])):
        graph[i].append((j, d))
        graph[j].append((i, d))
    dist = np.full(len(nodes), np.inf)
    dist[0] = 0
    previous = {}
    queue = [(0.0, 0)]
    goal_blocked = bool(np.any(np.linalg.norm(centers - goal, axis=1) < radii))
    while queue:
        cost, i = heapq.heappop(queue)
        if cost > dist[i] + 1e-12:
            continue
        if i == 1:
            indices = [1]
            while indices[-1] != 0:
                indices.append(previous[indices[-1]])
            return nodes[indices[::-1]]
        for j, d in graph[i]:
            candidate = cost + d
            if candidate < dist[j] - 1e-12:
                dist[j] = candidate
                previous[j] = i
                heapq.heappush(queue, (candidate, j))
    if allow_goal_proxy and goal_blocked:
        candidates = np.flatnonzero(np.isfinite(dist) & (np.arange(len(nodes)) >= 2))
        if len(candidates):
            # The final goal may be temporarily occupied by an approaching
            # person. Guide to a reachable point around it instead of reverting
            # to a straight line toward that person. The actual goal is unchanged.
            scores = (
                np.sum((nodes[candidates] - goal) ** 2, axis=1)
                + 0.05 * dist[candidates]
            )
            i = int(candidates[np.argmin(scores)])
            indices = [i]
            while indices[-1] != 0:
                indices.append(previous[indices[-1]])
            return nodes[indices[::-1]]
    return None


def sample_path(path, distances, goal_heading=0.0):
    path = np.asarray(path, dtype=float)
    delta = np.diff(path, axis=0)
    length = np.linalg.norm(delta, axis=1)
    keep = np.r_[True, length > 1e-09]
    path = path[keep]
    if len(path) == 1:
        return np.tile([*path[0], goal_heading], (len(np.atleast_1d(distances)), 1))
    delta = np.diff(path, axis=0)
    length = np.linalg.norm(delta, axis=1)
    s = np.r_[0.0, np.cumsum(length)]
    out = []
    for distance in np.atleast_1d(distances):
        if distance >= s[-1]:
            out.append([*path[-1], goal_heading])
            continue
        i = min(
            int(np.searchsorted(s, max(0.0, distance), side="right") - 1),
            len(length) - 1,
        )
        ratio = (max(0.0, distance) - s[i]) / length[i]
        out.append([*path[i] + ratio * delta[i], math.atan2(delta[i, 1], delta[i, 0])])
    return np.asarray(out)


def guide_controls(
    current,
    path,
    N,
    Ts,
    speed,
    omega_max,
    lookahead=0.6,
    acceleration_limits=None,
    u_prev=None,
    first_step_dt=None,
):
    """A bounded steering seed, integrated by the same Euler robot model."""
    state = np.asarray(current, dtype=float).copy()
    controls = []
    previous = np.zeros(2) if u_prev is None else np.asarray(u_prev, dtype=float).copy()
    limits = (
        None
        if acceleration_limits is None
        else np.asarray(acceleration_limits, dtype=float)
    )
    if limits is not None:
        if (
            limits.shape != (2,)
            or not np.all(np.isfinite(limits))
            or np.any(limits <= 0)
            or previous.shape != (2,)
            or not np.all(np.isfinite(previous))
        ):
            raise ValueError("Invalid guide acceleration limits or previous command")
        first_dt = Ts if first_step_dt is None else float(first_step_dt)
        if not np.isfinite(first_dt) or not 0 <= first_dt <= Ts + 1e-12:
            raise ValueError("Invalid first guide interval")
    p = np.asarray(path, dtype=float)
    delta = np.diff(p, axis=0)
    length = np.linalg.norm(delta, axis=1)
    cumulative = np.r_[0.0, np.cumsum(length)]
    for _ in range(N):
        frac = np.clip(
            np.sum((state[:2] - p[:-1]) * delta, axis=1) / np.maximum(length**2, 1e-16),
            0,
            1,
        )
        nearest = p[:-1] + frac[:, None] * delta
        i = int(np.argmin(np.linalg.norm(nearest - state[:2], axis=1)))
        progress = cumulative[i] + frac[i] * length[i]
        target = sample_path(p, [progress + lookahead])[0, :2]
        error = wrap_to_pi(math.atan2(*(target - state[:2])[::-1]) - state[2])
        v = float(speed) * max(0.0, math.cos(error))
        if abs(error) > 1.0:
            v = 0.0
        omega = float(np.clip(2.5 * error, -omega_max, omega_max))
        u = np.array([v, omega])
        if limits is not None:
            dt = first_dt if not controls else Ts
            u = np.clip(u, previous - limits * dt, previous + limits * dt)
        previous = u.copy()
        controls.append(u)
        state = unicycle_step(state, u, Ts)
    return np.asarray(controls)


class RouteReference:
    """Keep a still-clear bypass so small detection changes do not flip sides."""

    def __init__(self, padding=0.12, memory_sec=10.0):
        if not np.isfinite(padding) or padding < 0:
            raise ValueError("Route padding cannot be negative")
        if not np.isfinite(memory_sec) or memory_sec <= 0:
            raise ValueError("Route observation memory must be positive")
        self.padding = padding
        self.memory_sec = float(memory_sec)
        self.path = None
        self.last_plan_time = -np.inf
        self.observations = {}
        self.history = {}

    def build(self, current, goal, humans, N, Ts, speed, now):
        centers = []
        radii = []
        active_ids = set()
        for index, human in enumerate(humans):
            key = human.get("id", index)
            active_ids.add(key)
            seen = float(human.get("last_seen_time", now))
            point = human.get("observed_position")
            if point is not None:
                observations = self.observations.get(key, [])
                if not observations or seen > observations[-1][0] + 1e-06:
                    observations.append(
                        (
                            seen,
                            np.asarray(point, dtype=float),
                            float(np.asarray(human["d_safe"]).flat[0]),
                        )
                    )
                self.observations[key] = observations
            hx, hy, ds = (
                np.broadcast_to(np.asarray(human[k], dtype=float), (N + 1,))
                for k in ("x", "y", "d_safe")
            )
            # The planner guides around the currently occupied and near-future
            # swept region. MPC constrains all humans at each horizon step.
            for i in sorted(set([0, N])):
                center = np.array([hx[i], hy[i]])
                if i > 0 and np.linalg.norm(center - np.asarray(current)[:2]) < ds[i]:
                    # the time-indexed MPC, not a static guide, handles this
                    continue
                centers.append(center)
                radii.append(ds[i])
            history = [
                (t, p) for (t, p) in self.history.get(key, []) if 0 <= now - t <= 2.0
            ]
            history.append((now, np.array([hx[0], hy[0]])))
            self.history[key] = history
            past = max(
                (p for (_, p) in history),
                key=lambda p: np.linalg.norm(p - [hx[0], hy[0]]),
            )
            if (
                np.linalg.norm(past - [hx[0], hy[0]]) > 0.25
                and np.linalg.norm(past - np.asarray(current)[:2]) > ds[0]
                and (np.linalg.norm(past - np.asarray(goal)[:2]) > ds[0])
            ):
                centers.append(past)
                radii.append(ds[0])
        self.history = {k: v for (k, v) in self.history.items() if k in active_ids}
        # A retired image/world ID is not evidence that its occupied location
        # is clear. Keep a short geometric map of actual observations instead
        # of letting an unobserved constant-velocity mean erase the bypass.
        for key, observations in list(self.observations.items()):
            observations = [
                r for r in observations if 0 <= now - r[0] <= self.memory_sec
            ]
            if not observations:
                del self.observations[key]
                continue
            self.observations[key] = observations
            points = np.array([p for (_, p, _) in observations])
            indices = set(
                [
                    int(np.argmin(points[:, 0])),
                    int(np.argmax(points[:, 0])),
                    int(np.argmin(points[:, 1])),
                    int(np.argmax(points[:, 1])),
                ]
            )
            for index in indices:
                seen, point, radius = observations[index]
                if (
                    now - seen > 1.0
                    and np.linalg.norm(point - np.asarray(goal)[:2]) < radius
                ):
                    continue
                if any(
                    (
                        np.linalg.norm(point - c) < 0.25 and radius <= r + 0.05
                        for (c, r) in zip(centers, radii)
                    )
                ):
                    continue
                centers.append(point)
                radii.append(radius)
        centers = np.asarray(centers, dtype=float).reshape(-1, 2)
        radii = np.asarray(radii)
        current = np.asarray(current)
        goal = np.asarray(goal)
        path = None
        if self.path is not None and np.linalg.norm(self.path[-1] - goal[:2]) < 1e-06:
            # Join the current pose to the furthest safely visible remaining
            # vertex; preserve the existing detour while it remains valid.
            for i in range(len(self.path) - 1, 0, -1):
                candidate = np.vstack([current[:2], self.path[i:]])
                if all(
                    (
                        segment_clear(a, b, centers, radii)
                        for (a, b) in zip(candidate[:-1], candidate[1:])
                    )
                ):
                    path = candidate
                    break
        if path is None:
            path = visibility_route(
                current[:2],
                goal[:2],
                centers,
                radii,
                self.padding,
                allow_goal_proxy=True,
                allow_start_recovery=True,
            )
            self.last_plan_time = now
        self.path = path
        if path is None:
            return (None, None)
        reference = sample_path(path, np.arange(N + 1) * Ts * speed, goal[2])
        reference[0] = current
        return (path, reference)

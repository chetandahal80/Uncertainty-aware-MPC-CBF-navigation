"""Pure helpers for interpreting MPC safety slack and predicted clearance."""

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SafetyAssessment:
    """Safety diagnostics for the executed portion and complete horizon."""

    near_slack: float
    horizon_slack: float
    worst_slack_step: int
    minimum_near_clearance: float


def _horizon_values(value, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(length, float(array))
    array = array.reshape(-1)
    if len(array) < length:
        raise ValueError("Human prediction is shorter than the MPC horizon")
    return array[:length]


def assess_solution_safety(
    x_opt,
    zeta_opt,
    humans: Iterable[dict],
    guard_steps: int,
) -> SafetyAssessment:
    """Assess the near-term command separately from far-horizon relaxation.

    Slack row ``i`` protects the transition from state ``i`` to ``i+1``.
    Therefore ``guard_steps`` slack rows correspond to predicted states
    ``1..guard_steps`` in the direct clearance calculation.
    """
    humans = list(humans)
    states = np.asarray(x_opt, dtype=float)
    if states.ndim != 2 or states.shape[1] < 2 or len(states) < 2:
        raise ValueError("MPC state solution has an invalid shape")
    if not np.all(np.isfinite(states)):
        raise ValueError("MPC state solution must be finite")

    usable_steps = min(max(int(guard_steps), 1), len(states) - 1)
    if not humans:
        return SafetyAssessment(0.0, 0.0, -1, math.inf)

    if zeta_opt is None:
        near_slack = 0.0
        horizon_slack = 0.0
        worst_slack_step = -1
    else:
        slack = np.asarray(zeta_opt, dtype=float)
        if slack.ndim != 2 or slack.shape[1] < len(humans):
            raise ValueError("MPC safety-slack solution has an invalid shape")
        relevant = slack[:, : len(humans)]
        if not np.all(np.isfinite(relevant)):
            raise ValueError("MPC safety slack must be finite")
        near_slack = float(np.max(relevant[:usable_steps]))
        horizon_slack = float(np.max(relevant))
        worst_slack_step = int(np.unravel_index(np.argmax(relevant), relevant.shape)[0])

    minimum_clearance = math.inf
    required_length = usable_steps + 1
    for human in humans:
        hx = _horizon_values(human["x"], required_length)
        hy = _horizon_values(human["y"], required_length)
        d_safe = _horizon_values(human["d_safe"], required_length)
        distances = np.hypot(
            states[1:required_length, 0] - hx[1:required_length],
            states[1:required_length, 1] - hy[1:required_length],
        )
        clearance = distances - d_safe[1:required_length]
        minimum_clearance = min(minimum_clearance, float(np.min(clearance)))

    return SafetyAssessment(
        near_slack=near_slack,
        horizon_slack=horizon_slack,
        worst_slack_step=worst_slack_step,
        minimum_near_clearance=minimum_clearance,
    )

"""Display-only interpolation; no extrapolation or controller feedback."""

from collections import deque
from copy import deepcopy
import bisect
import math
import numpy as np


class TimedBuffer:
    def __init__(self, maxlen=600):
        self.rows = deque(maxlen=maxlen)

    def add(self, t, value):
        t = float(t)
        if self.rows and t < self.rows[-1][0] - 1e-6:
            self.rows.clear()
        if self.rows and abs(t - self.rows[-1][0]) < 1e-6:
            self.rows[-1] = (t, value)
        else:
            self.rows.append((t, value))

    def bracket(self, t):
        if not self.rows:
            return None
        i = bisect.bisect_right([x[0] for x in self.rows], t)
        if i == 0:
            return self.rows[0][1], self.rows[0][1], 0.0
        if i == len(self.rows):
            return self.rows[-1][1], self.rows[-1][1], 0.0
        ta, a = self.rows[i - 1]
        tb, b = self.rows[i]
        if tb - ta > 0.6:
            return a, a, 0.0
        return a, b, (t - ta) / (tb - ta)

    def clear(self):
        self.rows.clear()


def blend_pose(a, b, weight):
    out = (1 - weight) * np.asarray(a, dtype=float) + weight * np.asarray(
        b, dtype=float
    )
    if len(out) > 2:
        out[2] = a[2] + weight * math.atan2(
            math.sin(b[2] - a[2]), math.cos(b[2] - a[2])
        )
    return out.tolist()


def interpolate_state(buffer, t):
    bracket = buffer.bracket(t)
    if bracket is None:
        return None
    a, b, w = bracket
    if w == 0:
        return deepcopy(a)
    s = deepcopy(a)
    s["t"] = float(t)
    if a["robot"] is not None and b["robot"] is not None:
        s["robot"] = blend_pose(a["robot"], b["robot"], w)
    others = {h["id"]: h for h in b["humans"]}
    for h in s["humans"]:
        other = others.get(h["id"])
        if other:
            for key in ["x", "y", "radius", "trace"]:
                if len(h[key]) == len(other[key]):
                    h[key] = (
                        (1 - w) * np.asarray(h[key]) + w * np.asarray(other[key])
                    ).tolist()
    return s


def interpolate_truth(buffer, t):
    bracket = buffer.bracket(t)
    if bracket is None:
        return {}
    a, b, w = bracket
    return {
        name: blend_pose(p, b[name], w) if name in b else list(p)
        for name, p in a.items()
    }

"""Fixed-radius obstacle-avoidance MPC baseline.

This comparison solver uses linearized per-sample keep-out constraints instead
of time-coupled discrete control barrier constraints. It shares the unicycle
model and tracking objective with the main solver but does not propagate
uncertainty-dependent safety radii. The main controller does not use this solver.
"""

import numpy as np
import cvxpy as cp

from .mpc_core import (
    NU,
    NX,
    linearize_unicycle,
    safety_function,
    wrap_to_pi,
)


def solve_mpc_naive(
    x_current,
    x_goal,
    x_ref_traj,
    u_ref_traj,
    N,
    Ts,
    Q,
    R,
    Qf,
    v_bounds,
    omega_bounds,
    humans=None,
    slack_penalty=1e4,
    u_prev=None,
    S=None,
    rate_limits=None,
    rate_slack_penalty=1e4,
):
    """Solve fixed-radius MPC with optional input-rate constraints.

    Human entries supply predicted x/y positions and a fixed d_safe radius.
    """
    humans = humans or []
    num_humans = len(humans)

    x = cp.Variable((N + 1, NX))
    u = cp.Variable((N, NU))
    zeta = cp.Variable((N, num_humans), nonneg=True) if num_humans > 0 else None
    rzeta = cp.Variable((N, NU), nonneg=True) if rate_limits is not None else None

    constraints = [x[0] == x_current]
    cost = 0

    theta_ref_unwrapped = np.unwrap(x_ref_traj[:, 2])
    goal_theta_unwrapped = x_ref_traj[0, 2] + wrap_to_pi(x_goal[2] - x_ref_traj[0, 2])
    x_goal_local = np.array([x_goal[0], x_goal[1], goal_theta_unwrapped])

    for i in range(N):
        x0_i = np.array([x_ref_traj[i, 0], x_ref_traj[i, 1], theta_ref_unwrapped[i]])
        u0_i = u_ref_traj[i]
        A, B, c = linearize_unicycle(x0_i, u0_i, Ts)

        constraints += [x[i + 1] == A @ x[i] + B @ u[i] + c]
        constraints += [u[i, 0] >= v_bounds[0], u[i, 0] <= v_bounds[1]]
        constraints += [u[i, 1] >= omega_bounds[0], u[i, 1] <= omega_bounds[1]]

        e = x[i + 1] - x_goal_local
        cost += cp.quad_form(e, Q) + cp.quad_form(u[i], R)

        # Fixed-radius keep-out constraints for each human at sample i.
        for h_idx, human in enumerate(humans):
            hx, hy = human["x"], human["y"]
            d_safe = human["d_safe"]
            hx_i = hx[i] if np.ndim(hx) > 0 else hx
            hy_i = hy[i] if np.ndim(hy) > 0 else hy
            d_safe_i = d_safe[i] if np.ndim(d_safe) > 0 else d_safe

            a_x_i = 2.0 * (x0_i[0] - hx_i)
            a_y_i = 2.0 * (x0_i[1] - hy_i)
            h0_i = safety_function(x0_i[0], x0_i[1], hx_i, hy_i, d_safe_i)

            dx_r = x[i, 0] - x0_i[0]
            dy_r = x[i, 1] - x0_i[1]

            constraints += [a_x_i * dx_r + a_y_i * dy_r + zeta[i, h_idx] >= -h0_i]
            cost += slack_penalty * zeta[i, h_idx]

    cost += cp.quad_form(x[N] - x_goal_local, Qf)

    if S is not None:
        if u_prev is not None:
            cost += cp.quad_form(u[0] - np.asarray(u_prev, dtype=float), S)
        for i in range(1, N):
            cost += cp.quad_form(u[i] - u[i - 1], S)

    if rate_limits is not None:
        dv_max, domega_max = rate_limits
        u_prev_arr = np.asarray(u_prev, dtype=float) if u_prev is not None else None
        for i in range(N):
            u_prev_i = u_prev_arr if i == 0 else u[i - 1]
            if u_prev_i is None:
                continue
            constraints += [
                cp.abs(u[i, 0] - u_prev_i[0]) <= dv_max + rzeta[i, 0],
                cp.abs(u[i, 1] - u_prev_i[1]) <= domega_max + rzeta[i, 1],
            ]
            cost += rate_slack_penalty * cp.sum(rzeta[i])

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)

    if u.value is None:
        return None, None, prob.status, None

    zeta_opt = zeta.value if zeta is not None else None
    return u.value, x.value, prob.status, zeta_opt

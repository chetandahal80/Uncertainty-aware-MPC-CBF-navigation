"""Unicycle dynamics and linearized discrete control barrier constraints.

The robot state is [x, y, theta] and its input is [v, omega]. Forward-Euler
integration defines the nonlinear prediction model. Analytic Jacobians provide
local affine dynamics around a dynamically consistent reference trajectory.

Human positions and safety radii may vary at every horizon sample. The safety
function is h = ||p_robot - p_human||^2 - d_safe^2. Linearized barrier rows
couple consecutive samples through h[k+1] >= (1-gamma)*h[k], with a
nonnegative relaxation variable for each human and horizon interval.
"""

import numpy as np
import cvxpy as cp

NX = 3  # state dimension: x, y, theta
NU = 2  # input dimension: v, omega


def wrap_to_pi(angle):
    """Wrap an angle (or array of angles) to (-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def unicycle_step(state, u, Ts):
    """True nonlinear plant propagation, used only for simulation /
    generating c(k), never inside the QP itself."""
    x, y, theta = state
    v, omega = u
    return np.array(
        [
            x + Ts * v * np.cos(theta),
            y + Ts * v * np.sin(theta),
            theta + Ts * omega,
        ]
    )


def linearize_unicycle(x0, u0, Ts):
    """
    Jacobian linearization of the unicycle plant about operating point
    (x0, u0), returning the affine model matrices A, B and c.

    Parameters
    ----------
    x0 : array_like, shape (3,)   operating-point state [x, y, theta]
    u0 : array_like, shape (2,)   operating-point input [v, omega]
    Ts : float                    sample time

    Returns
    -------
    A : (3,3) ndarray
    B : (3,2) ndarray
    c : (3,)  ndarray
    """
    theta0 = x0[2]
    v0 = u0[0]

    A = np.array(
        [
            [1.0, 0.0, -Ts * v0 * np.sin(theta0)],
            [0.0, 1.0, Ts * v0 * np.cos(theta0)],
            [0.0, 0.0, 1.0],
        ]
    )
    B = np.array(
        [
            [Ts * np.cos(theta0), 0.0],
            [Ts * np.sin(theta0), 0.0],
            [0.0, Ts],
        ]
    )
    f0 = unicycle_step(x0, u0, Ts)
    c = f0 - A @ np.asarray(x0) - B @ np.asarray(u0)
    return A, B, c


def safety_function(x_pos, y_pos, hx, hy, d_safe):
    """
    h_j = (x-x_hj)^2 + (y-y_hj)^2 - d_safe^2, evaluated at a single
    (possibly numeric, possibly symbolic-free) point. Used both to
    precompute h_j0 along the reference trajectory (for the DCBF
    right-hand side) and, purely numerically, to verify the TRUE
    realized trajectory never actually violates safety.
    """
    return (x_pos - hx) ** 2 + (y_pos - hy) ** 2 - d_safe**2


def dcbf_row_coefficients(
    x0_i, x0_ip1, u0_i, Ts, hx_i, hy_i, hx_ip1, hy_ip1, gamma, d_safe_i, d_safe_ip1
):
    """
    Compute the affine DCBF row coefficients (p_x, p_y, p_theta,
    beta_j) and the right-hand side (1-gamma)*h_j0(k+i) - h_j0(k+i+1)
    for one horizon step and one human.

    x0_i, x0_ip1 : (3,) reference states at steps i and i+1
    u0_i         : (2,) reference input at step i (only v0 is used)
    hx_i, hy_i   : human position at step i; may differ from step i+1
    hx_ip1, hy_ip1 : human position at step i+1
    d_safe_i, d_safe_ip1 : safety radius at steps i and i+1
    """
    theta0 = x0_i[2]
    v0 = u0_i[0]

    a_x_i = 2.0 * (x0_i[0] - hx_i)
    a_y_i = 2.0 * (x0_i[1] - hy_i)
    a_x_ip1 = 2.0 * (x0_ip1[0] - hx_ip1)
    a_y_ip1 = 2.0 * (x0_ip1[1] - hy_ip1)

    p_x = a_x_ip1 - (1.0 - gamma) * a_x_i
    p_y = a_y_ip1 - (1.0 - gamma) * a_y_i
    p_theta = Ts * v0 * (a_y_ip1 * np.cos(theta0) - a_x_ip1 * np.sin(theta0))
    beta_j = Ts * (a_x_ip1 * np.cos(theta0) + a_y_ip1 * np.sin(theta0))

    h0_i = safety_function(x0_i[0], x0_i[1], hx_i, hy_i, d_safe_i)
    h0_ip1 = safety_function(x0_ip1[0], x0_ip1[1], hx_ip1, hy_ip1, d_safe_ip1)
    rhs = (1.0 - gamma) * h0_i - h0_ip1

    return p_x, p_y, p_theta, beta_j, rhs, h0_i


def rollout_reference(x_current, u_reference, Ts):
    """Dynamically consistent Euler reference, anchored at the measurement.

    Homogeneous deviation dynamics require x0[i+1]=f(x0[i],u0[i]).
    Re-roll shifted controls instead of perturbing reference positions directly.
    """
    controls = np.asarray(u_reference, dtype=float)
    states = np.empty((len(controls) + 1, NX))
    states[0] = np.asarray(x_current, dtype=float)
    for i, control in enumerate(controls):
        states[i + 1] = unicycle_step(states[i], control, Ts)
    return states


def initial_reference_guess(x_current, N, humans=None, bias=0.15):
    """Stationary, dynamically consistent initial reference (zero controls)."""
    return np.tile(np.asarray(x_current, dtype=float), (N + 1, 1))


def shift_trajectory(x_traj, u_traj):
    """Shift the previous solution; re-roll controls at the next measured pose."""
    return (np.vstack([x_traj[1:], x_traj[-1:]]), np.vstack([u_traj[1:], u_traj[-1:]]))


def solve_mpc(
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
    gamma=0.5,
    slack_penalty=1e4,
    acceleration_limits=None,
    u_prev=None,
    first_step_dt=None,
):
    """Solve deviation-form MPC with optional hard acceleration bounds."""
    from .mpc_core_dpp import DPPMPCSolver

    solver = DPPMPCSolver(
        N,
        Ts,
        Q,
        R,
        Qf,
        v_bounds,
        omega_bounds,
        max_humans=max(1, len(humans or [])),
        slack_penalty=slack_penalty,
        acceleration_limits=acceleration_limits,
    )
    return solver.solve(
        x_current,
        x_goal,
        x_ref_traj,
        u_ref_traj,
        N,
        Ts,
        humans=humans,
        gamma=gamma,
        u_prev=u_prev,
        first_step_dt=first_step_dt,
    )

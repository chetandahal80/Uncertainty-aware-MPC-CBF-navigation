"""Cached parameterized deviation-form MPC quadratic program.

Task and linearization references are separate. The objective includes stage
state error, absolute control effort, terminal error and linear safety-slack
penalties. Constraints include dynamics, input bounds, optional acceleration
bounds and one discrete barrier row per human per horizon interval.
"""

import numpy as np
import cvxpy as cp
from .mpc_core import (
    NX,
    NU,
    linearize_unicycle,
    dcbf_row_coefficients,
    rollout_reference,
    wrap_to_pi,
)


class DPPMPCSolver:
    def __init__(
        self,
        N,
        Ts,
        Q,
        R,
        Qf,
        v_bounds,
        omega_bounds,
        max_humans,
        slack_penalty=1e4,
        solver=None,
        acceleration_limits=None,
    ):
        self.N, self.Ts = int(N), float(Ts)
        self.max_humans = int(max_humans)
        if self.N < 1 or not np.isfinite(self.Ts) or self.Ts <= 0:
            raise ValueError("Invalid horizon")
        if self.max_humans < 1 or not np.isfinite(slack_penalty) or slack_penalty <= 0:
            raise ValueError("Invalid capacity or slack penalty")
        weights = []
        for value, shape, strict in (
            (Q, (3, 3), False),
            (R, (2, 2), True),
            (Qf, (3, 3), False),
        ):
            value = np.asarray(value, dtype=float)
            if (
                value.shape != shape
                or not np.all(np.isfinite(value))
                or not np.allclose(value, value.T)
            ):
                raise ValueError("Weights must be finite symmetric matrices")
            eigenvalues = np.linalg.eigvalsh(value)
            if eigenvalues[0] < -1e-10 or (strict and eigenvalues[0] <= 0):
                raise ValueError("Q/P must be PSD and R positive definite")
            weights.append(value)
        Q, R, Qf = weights
        self.v_bounds = np.asarray(v_bounds, dtype=float)
        self.omega_bounds = np.asarray(omega_bounds, dtype=float)
        for bounds in (self.v_bounds, self.omega_bounds):
            if (
                bounds.shape != (2,)
                or not np.all(np.isfinite(bounds))
                or bounds[0] > bounds[1]
            ):
                raise ValueError("Invalid actuator bounds")
        self.acceleration_limits = None
        if acceleration_limits is not None:
            limits = np.asarray(acceleration_limits, dtype=float)
            if (
                limits.shape != (NU,)
                or not np.all(np.isfinite(limits))
                or np.any(limits <= 0)
            ):
                raise ValueError(
                    "Acceleration limits must be two finite positive values"
                )
            self.acceleration_limits = limits.copy()
        self.installed_solvers = set(cp.installed_solvers())
        self.solver = solver or (
            "CLARABEL" if "CLARABEL" in self.installed_solvers else cp.OSQP
        )
        self.last_solver_used = None
        self.slack_penalty = float(slack_penalty)
        # The initial deviation delta_x(0)=0 is fixed.
        self.dx = cp.Variable((self.N, NX), name="delta_x_1_to_N")
        self.du = cp.Variable((self.N, NU), name="delta_u")
        self.zeta = cp.Variable((self.N, self.max_humans), nonneg=True, name="zeta")
        self.A_p = [cp.Parameter((NX, NX)) for _ in range(self.N)]
        self.B_p = [cp.Parameter((NX, NU)) for _ in range(self.N)]
        self.state_linear_p = cp.Parameter((self.N + 1, NX))
        self.control_linear_p = cp.Parameter((self.N, NU))
        self.lower_p = cp.Parameter((self.N, NU))
        self.upper_p = cp.Parameter((self.N, NU))
        self.row_p = [cp.Parameter((self.max_humans, NX)) for _ in range(self.N)]
        self.beta_p = cp.Parameter((self.N, self.max_humans))
        self.rhs_p = cp.Parameter((self.N, self.max_humans))
        self.rate_offsets_p = self.rate_steps_p = None
        if self.acceleration_limits is not None:
            self.rate_offsets_p = cp.Parameter((self.N, NU))
            self.rate_steps_p = cp.Parameter((self.N, NU), nonneg=True)
        constraints = []
        cost = self.slack_penalty * cp.sum(self.zeta)
        for i in range(self.N):
            dx_i = np.zeros(NX) if i == 0 else self.dx[i - 1]
            constraints += [
                self.dx[i] == self.A_p[i] @ dx_i + self.B_p[i] @ self.du[i],
                self.du[i] >= self.lower_p[i],
                self.du[i] <= self.upper_p[i],
                self.row_p[i] @ dx_i
                + cp.multiply(self.beta_p[i], self.du[i, 0])
                + self.zeta[i]
                >= self.rhs_p[i],
            ]
            if self.acceleration_limits is not None:
                # Apply rate bounds to absolute controls, including reference changes.
                change = (
                    self.du[i] - (self.du[i - 1] if i else 0) + self.rate_offsets_p[i]
                )
                constraints += [
                    change >= -self.rate_steps_p[i],
                    change <= self.rate_steps_p[i],
                ]
            # Stage costs cover i=0,...,N-1; P weights the terminal state.
            cost += (
                cp.quad_form(dx_i, Q)
                + self.state_linear_p[i] @ dx_i
                + cp.quad_form(self.du[i], R)
                + self.control_linear_p[i] @ self.du[i]
            )
        cost += cp.quad_form(self.dx[-1], Qf) + self.state_linear_p[-1] @ self.dx[-1]
        self.problem = cp.Problem(cp.Minimize(cost), constraints)
        assert self.problem.is_dpp()
        self.Q, self.R, self.Qf = Q, R, Qf
        self.reference_states = self.reference_controls = None

    def solve(
        self,
        x_current,
        x_goal,
        x_ref_traj,
        u_ref_traj,
        N,
        Ts,
        humans=None,
        gamma=0.5,
        u_prev=None,
        warm_start=True,
        first_step_dt=None,
        linear_velocity_profile=None,
    ):
        if N != self.N or not np.isclose(Ts, self.Ts, rtol=0, atol=1e-12):
            raise ValueError("Solver horizon mismatch")
        if not np.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("gamma must lie in (0,1]")
        humans = humans or []
        if len(humans) > self.max_humans:
            raise ValueError("Increase solver capacity; humans must not be dropped")
        current = np.asarray(x_current, dtype=float)
        controls = np.asarray(u_ref_traj, dtype=float)
        supplied = np.asarray(x_ref_traj, dtype=float)
        if (
            current.shape != (NX,)
            or controls.shape != (self.N, NU)
            or supplied.shape != (self.N + 1, NX)
            or not all(np.all(np.isfinite(a)) for a in (current, controls, supplied))
        ):
            raise ValueError("Invalid linearization reference")
        if self.acceleration_limits is not None:
            previous = (
                np.zeros(NU) if u_prev is None else np.asarray(u_prev, dtype=float)
            )
            if previous.shape != (NU,) or not np.all(np.isfinite(previous)):
                raise ValueError("Previous command must be a finite two-vector")
            dt = self.Ts if first_step_dt is None else float(first_step_dt)
            if not np.isfinite(dt) or dt < 0 or dt > self.Ts + 1e-12:
                raise ValueError("First acceleration interval must lie in [0, Ts]")
            self.rate_offsets_p.value = np.vstack(
                [controls[0] - previous, np.diff(controls, axis=0)]
            )
            steps = np.tile(self.acceleration_limits * self.Ts, (self.N, 1))
            steps[0] = self.acceleration_limits * dt
            self.rate_steps_p.value = steps
        # Reintegrate shifted controls from the measurement so the affine offset
        # cancels exactly. A shifted, nudged or SQP-predicted state path alone
        # need not satisfy the nonlinear Euler model.
        current = current.copy()
        current[2] = supplied[0, 2] + wrap_to_pi(current[2] - supplied[0, 2])
        states = rollout_reference(current, controls, self.Ts)
        goal = np.asarray(x_goal, dtype=float)
        if goal.shape == (NX,):
            goal = np.tile(goal, (self.N + 1, 1))
        if goal.shape != states.shape or not np.all(np.isfinite(goal)):
            raise ValueError("Task reference must be a pose or N+1 poses")
        goal = goal.copy()
        goal[:, 2] = np.unwrap(goal[:, 2])
        goal[:, 2] += current[2] + wrap_to_pi(goal[0, 2] - current[2]) - goal[0, 2]
        linear = 2 * (states - goal) @ self.Q
        linear[-1] = 2 * self.Qf @ (states[-1] - goal[-1])
        self.state_linear_p.value = linear
        self.control_linear_p.value = 2 * controls @ self.R
        lower = np.array([self.v_bounds[0], self.omega_bounds[0]])
        upper = np.array([self.v_bounds[1], self.omega_bounds[1]])
        lower_values, upper_values = lower - controls, upper - controls
        if linear_velocity_profile is not None:
            profile = np.asarray(linear_velocity_profile, dtype=float)
            if (
                profile.shape != (self.N,)
                or not np.all(np.isfinite(profile))
                or np.any(profile < self.v_bounds[0] - 1e-9)
                or np.any(profile > self.v_bounds[1] + 1e-9)
            ):
                raise ValueError("Invalid docking velocity profile")
            lower_values[:, 0] = upper_values[:, 0] = profile - controls[:, 0]
        self.lower_p.value, self.upper_p.value = lower_values, upper_values
        positions = []
        for human in humans:
            values = []
            for key in ("x", "y", "d_safe"):
                value = np.asarray(human[key], dtype=float)
                if value.ndim == 0:
                    value = np.full(self.N + 1, float(value))
                if value.shape != (self.N + 1,) or not np.all(np.isfinite(value)):
                    raise ValueError("Invalid human horizon")
                values.append(value)
            if np.any(values[2] <= 0):
                raise ValueError("Safety radii must be positive")
            positions.append(values)
        beta = np.zeros((self.N, self.max_humans))
        rhs = np.zeros_like(beta)
        for i in range(self.N):
            A, B, _ = linearize_unicycle(states[i], controls[i], self.Ts)
            self.A_p[i].value, self.B_p[i].value = A, B
            rows = np.zeros((self.max_humans, NX))
            for j, (hx, hy, ds) in enumerate(positions):
                px, py, pth, b, r, _ = dcbf_row_coefficients(
                    states[i],
                    states[i + 1],
                    controls[i],
                    self.Ts,
                    hx[i],
                    hy[i],
                    hx[i + 1],
                    hy[i + 1],
                    gamma,
                    ds[i],
                    ds[i + 1],
                )
                rows[j] = (px, py, pth)
                beta[i, j], rhs[i, j] = b, r
            self.row_p[i].value = rows
        self.beta_p.value, self.rhs_p.value = beta, rhs
        self.reference_states, self.reference_controls = states.copy(), controls.copy()
        self._solve_with_fallback(warm_start)
        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return None, None, self.problem.status, None
        if self.du.value is None or self.dx.value is None:
            return None, None, self.problem.status, None
        x = states + np.vstack([np.zeros(NX), self.dx.value])
        u = controls + self.du.value
        return u, x, self.problem.status, self.zeta.value[:, : len(humans)]

    @staticmethod
    def _solver_name(solver) -> str:
        return str(solver).upper()

    def _solver_options(self, solver):
        name = self._solver_name(solver)
        if name == "OSQP":
            # The safety slack penalties are intentionally much
            # larger than the tracking weights. Loose residual tolerances can
            # otherwise produce an inaccurate steering command even when OSQP
            # reports ``optimal``.
            return dict(max_iter=20000, eps_abs=1e-6, eps_rel=1e-6)
        if name == "CLARABEL":
            return dict(max_iter=500)
        return {}

    def _solve_once(self, solver, warm_start: bool) -> None:
        self.problem.solve(
            solver=solver,
            warm_start=warm_start,
            **self._solver_options(solver),
        )
        self.last_solver_used = solver

    def _solve_with_fallback(self, warm_start: bool) -> None:
        acceptable = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        candidates = [self.solver]
        candidate_names = {self._solver_name(self.solver)}
        clarabel = getattr(cp, "CLARABEL", "CLARABEL")
        for candidate in (clarabel, cp.OSQP):
            name = self._solver_name(candidate)
            if candidate in self.installed_solvers and name not in candidate_names:
                candidates.append(candidate)
                candidate_names.add(name)

        last_error = None
        for index, candidate in enumerate(candidates):
            try:
                self._solve_once(
                    candidate,
                    warm_start=warm_start if index == 0 else False,
                )
            except cp.error.SolverError as exc:
                last_error = exc
                continue
            if self.problem.status in acceptable:
                return

        if last_error is not None and self.problem.status is None:
            raise last_error

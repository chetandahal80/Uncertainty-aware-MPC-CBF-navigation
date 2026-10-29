"""Fixed-size, single-map live view. Simulator truth is DISPLAY ONLY."""

import json
import math
import time
from collections import deque
import numpy as np
from .plot_interpolation import TimedBuffer, interpolate_state, interpolate_truth


class SafetyPlot:
    """Persistent artists and a delayed interpolated display, with fixed bounds."""

    def __init__(self, bounds=(-4.0, 14.0, -4.5, 4.5), display_delay=0.25):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyArrowPatch

        bounds = np.asarray(bounds, dtype=float)
        if (
            bounds.shape != (4,)
            or not np.all(np.isfinite(bounds))
            or bounds[0] >= bounds[1]
            or bounds[2] >= bounds[3]
        ):
            raise ValueError("Invalid fixed plot bounds")
        self.figure = plt.figure(figsize=(13, 7))
        self.map_ax = self.figure.add_axes([0.07, 0.10, 0.86, 0.80])
        self.figure.canvas.manager.set_window_title("MPC-CBF live safety margins")
        ax = self.map_ax
        ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        self.header = self.figure.text(
            0.5,
            0.97,
            "Waiting for controller snapshots …",
            ha="center",
            va="top",
            fontsize=10,
        )
        self.figure.text(
            0.5,
            0.015,
            "Red = simulator truth (display only) • circles = controller beliefs • display interpolation does not change control",
            ha="center",
            fontsize=8,
            color="dimgray",
        )
        (self.robot_line,) = ax.plot([], [], color="royalblue", lw=2)
        (self.robot_marker,) = ax.plot([], [], "o", color="royalblue", markersize=6)
        self.heading = FancyArrowPatch(
            (0, 0),
            (0, 0),
            arrowstyle="-|>",
            mutation_scale=14,
            color="royalblue",
            lw=1.7,
        )
        ax.add_patch(self.heading)
        (self.goal_marker,) = ax.plot([], [], "*", color="green", markersize=12)
        (self.route_line,) = ax.plot([], [], "--", color="gray", alpha=0.6, lw=1)
        handles = [
            Line2D(
                [], [], color="royalblue", marker="o", label="Robot / path followed"
            ),
            Line2D(
                [],
                [],
                color="crimson",
                marker="o",
                ls="--",
                label="Human: Gazebo truth",
            ),
            Line2D(
                [], [], color="black", marker="x", ls="none", label="Tracked estimate"
            ),
            Line2D([], [], color="black", label="Current safety circle"),
            Line2D([], [], color="gray", ls="--", label="Horizon-end circle"),
            Line2D([], [], color="gray", ls="--", alpha=0.6, label="Route guide"),
            Line2D([], [], color="green", marker="*", ls="none", label="Goal"),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=8)
        self.dynamic = [
            self.robot_line,
            self.robot_marker,
            self.heading,
            self.goal_marker,
            self.route_line,
            self.header,
        ]
        self.estimate_artists = {}
        self.truth_artists = {}
        self.state = None
        self.states = TimedBuffer()
        self.truth = TimedBuffer()
        self.robot_trail = deque(maxlen=10000)
        self.truth_trails = {}
        self.history = {}
        self.last_t = None
        self.received_at = None
        self.truth_received_at = None
        self.display_delay = float(display_delay)
        if not 0 <= self.display_delay <= 1:
            raise ValueError("Display delay must lie in [0,1] seconds")
        self._background = None
        self._anchor = None
        self._rate = 1.0
        self._last_present_t = None
        self.figure.canvas.mpl_connect("draw_event", self._invalidate)
        self.figure.canvas.mpl_connect("resize_event", self._invalidate)

    def _invalidate(self, event=None):
        self._background = None

    def _arrival(self, t):
        wall = time.monotonic()
        if self._anchor is None or t < self._anchor[0] - 1:
            self._anchor = (t, wall)
        elif t > self._anchor[0] + 1e-6:
            dt = wall - self._anchor[1]
            if dt > 0.001:
                self._rate = 0.8 * self._rate + 0.2 * float(
                    np.clip((t - self._anchor[0]) / dt, 0.0, 2.0)
                )
            self._anchor = (t, wall)

    def presentation_time(self):
        if self._anchor is None:
            return 0.0
        t, wall = self._anchor
        elapsed = time.monotonic() - wall
        target = min(
            t, t + (elapsed * self._rate if elapsed < 0.5 else 0.0) - self.display_delay
        )
        if self._last_present_t is not None and self._last_present_t <= t:
            target = max(target, self._last_present_t)
        self._last_present_t = target
        return target

    def update(self, state):
        t = float(state["t"])
        if self.last_t is not None and t < self.last_t - 1e-6:
            self.robot_trail.clear()
            self.history.clear()
            self.states.clear()
            self.truth.clear()
            self.truth_trails.clear()
            self._anchor = None
            self._last_present_t = None
            self.last_t = None
        new_time = self.last_t is None or t > self.last_t + 1e-6
        if new_time and state.get("robot") is not None:
            self.robot_trail.append((t, *state["robot"][:2]))
        self.states.add(t, state)
        self.state = state
        self.last_t = t
        self.received_at = time.monotonic()
        self._arrival(t)

    def update_truth(self, t, humans):
        t = float(t)
        if self.truth.rows and t < self.truth.rows[-1][0] - 1e-6:
            self.truth.clear()
            self.truth_trails.clear()
            self._anchor = None
            self._last_present_t = None
        self.truth.add(t, humans)
        self.truth_received_at = time.monotonic()
        self._arrival(t)
        for name, p in humans.items():
            rows = self.truth_trails.setdefault(name, deque(maxlen=2000))
            if rows and abs(rows[-1][0] - t) < 1e-6:
                rows[-1] = (t, *p[:2])
            else:
                rows.append((t, *p[:2]))

    def _estimate(self, key):
        from matplotlib.patches import Circle
        import matplotlib.pyplot as plt

        if key not in self.estimate_artists:
            color = plt.get_cmap("tab10")(int(key) % 10)
            ax = self.map_ax
            (point,) = ax.plot([], [], "x", color=color, markersize=7)
            (forecast,) = ax.plot([], [], ":", color=color, lw=1, alpha=0.6)
            now = Circle((0, 0), 1, fill=False, color=color, lw=2)
            end = Circle((0, 0), 1, fill=False, color=color, ls="--", lw=1, alpha=0.5)
            ax.add_patch(now)
            ax.add_patch(end)
            label = ax.text(0, 0, "", color=color, fontsize=8)
            self.estimate_artists[key] = (point, forecast, now, end, label)
            self.dynamic.extend(self.estimate_artists[key])
            self._invalidate()
        return self.estimate_artists[key]

    def _truth_artist(self, name):
        if name not in self.truth_artists:
            ax = self.map_ax
            (point,) = ax.plot(
                [], [], "o", mfc="white", mec="crimson", mew=1.8, markersize=7
            )
            (trail,) = ax.plot([], [], "--", color="crimson", alpha=0.35, lw=1)
            label = ax.text(0, 0, "", color="crimson", fontsize=8)
            self.truth_artists[name] = (point, trail, label)
            self.dynamic.extend(self.truth_artists[name])
            self._invalidate()
        return self.truth_artists[name]

    def draw(self, t=None, fast=False):
        t = self.last_t if t is None else float(t)
        state = interpolate_state(self.states, t) if t is not None else None
        humans = interpolate_truth(self.truth, t) if t is not None else {}
        for artists in list(self.estimate_artists.values()) + list(
            self.truth_artists.values()
        ):
            for a in artists:
                a.set_visible(False)
        if state is not None:
            robot = state.get("robot")
            trail = np.asarray([row[1:] for row in self.robot_trail if row[0] <= t])
            if robot is not None:
                if len(trail):
                    trail = np.vstack([trail, robot[:2]])
                else:
                    trail = np.asarray([robot[:2]])
                self.robot_marker.set_data([robot[0]], [robot[1]])
                self.heading.set_positions(
                    robot[:2],
                    (
                        robot[0] + 0.45 * math.cos(robot[2]),
                        robot[1] + 0.45 * math.sin(robot[2]),
                    ),
                )
            self.robot_marker.set_visible(robot is not None)
            self.heading.set_visible(robot is not None)
            self.robot_line.set_data(
                trail[:, 0] if len(trail) else [], trail[:, 1] if len(trail) else []
            )
            goal = state.get("goal")
            self.goal_marker.set_data(
                [goal[0]] if goal else [], [goal[1]] if goal else []
            )
            route = np.asarray(state.get("route", []))
            self.route_line.set_data(
                route[:, 0] if len(route) else [], route[:, 1] if len(route) else []
            )
            for h in state["humans"]:
                point, pred, now, end, label = self._estimate(h["id"])
                for a in (point, pred, now, end, label):
                    a.set_visible(True)
                x, y, r = h["x"], h["y"], h["radius"]
                point.set_data([x[0]], [y[0]])
                pred.set_data(x, y)
                now.center = (x[0], y[0])
                now.set_radius(r[0])
                end.center = (x[-1], y[-1])
                end.set_radius(r[-1])
                label.set_position((x[0] + 0.12, y[0] + 0.18))
                label.set_text(f"Est {h['id']}: r={r[0]:.2f} m")
            age = state.get("perception_age")
            age = "waiting" if age is None else f"{age:.2f}s"
            stale = self.received_at is None or time.monotonic() - self.received_at > 2
            truth_status = "Gazebo truth" if humans else "no simulator human data"
            self.header.set_text(
                f"d_safe={state['d_min']:.2f}+{state['kappa']:.2f}×tr(P_pos) | t={t:.2f}s | v={state['command'][0]:.2f} m/s, ω={state['command'][1]:.2f} rad/s\n{state['status']} | perception age {age} | {truth_status} | tracked: {len(state['humans'])}"
                + (" | NO NEW CONTROL DATA / PAUSED" if stale else "")
            )
            # Frame label is changed only when the frame itself changes.
            xlabel = f"x (m), frame: {state['frame_id']}"
            if self.map_ax.get_xlabel() != xlabel:
                self.map_ax.set_xlabel(xlabel)
                self._invalidate()
        else:
            self.robot_line.set_data([], [])
            self.robot_marker.set_visible(False)
            self.heading.set_visible(False)
        for name, p in humans.items():
            point, trail, label = self._truth_artist(name)
            for a in (point, trail, label):
                a.set_visible(True)
            point.set_data([p[0]], [p[1]])
            rows = np.asarray(
                [r[1:] for r in self.truth_trails.get(name, []) if t - 10 <= r[0] <= t]
            )
            rows = np.vstack([rows, p[:2]]) if len(rows) else np.asarray([p[:2]])
            trail.set_data(rows[:, 0], rows[:, 1])
            label.set_position((p[0] + 0.12, p[1] - 0.25))
            label.set_text(name + " (truth)")
        canvas = self.figure.canvas
        if fast and canvas.supports_blit:
            if self._background is None:
                visible = [a.get_visible() for a in self.dynamic]
                for a in self.dynamic:
                    a.set_visible(False)
                canvas.draw()
                self._background = canvas.copy_from_bbox(self.figure.bbox)
                for a, v in zip(self.dynamic, visible):
                    a.set_visible(v)
            canvas.restore_region(self._background)
            for a in self.dynamic:
                if a.get_visible():
                    if a.axes:
                        a.axes.draw_artist(a)
                    else:
                        self.figure.draw_artist(a)
            canvas.blit(self.figure.bbox)
        else:
            canvas.draw()


class TruthAlignment:
    """Initialize a fixed world->controller-frame transform from robot poses."""

    def __init__(self):
        self.rotation = None
        self.translation = None

    def align(self, world_robot, odom_robot):
        angle = odom_robot[2] - world_robot[2]
        c, s = math.cos(angle), math.sin(angle)
        self.rotation = np.array([[c, -s], [s, c]])
        self.translation = np.asarray(odom_robot[:2]) - self.rotation @ world_robot[:2]

    def project(self, point):
        if self.rotation is None:
            return None
        return (self.rotation @ np.asarray(point[:2]) + self.translation).tolist()


def main(args=None):
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import String
    from gazebo_msgs.msg import ModelStates

    rclpy.init(args=args)
    node = Node("live_safety_plot")
    for key, value in [
        ("visualization_topic", "/mpc_cbf/visualization"),
        ("ground_truth_topic", "/gazebo/model_states"),
        ("show_simulator_truth", True),
        ("robot_model_name", "my_bot"),
        ("human_model_pattern", "actor|human|person|pedestrian"),
        ("display_delay_sec", 0.25),
        ("plot_x_min", -4.0),
        ("plot_x_max", 14.0),
        ("plot_y_min", -4.5),
        ("plot_y_max", 4.5),
    ]:
        node.declare_parameter(key, value)
    gp = lambda k: node.get_parameter(k).value
    plot = SafetyPlot(
        (gp("plot_x_min"), gp("plot_x_max"), gp("plot_y_min"), gp("plot_y_max")),
        gp("display_delay_sec"),
    )
    alignment = TruthAlignment()
    world_rows = deque(maxlen=100)
    last_clock = [None]
    import re

    pattern = re.compile(str(gp("human_model_pattern")), re.I)

    def align_and_show():
        if not world_rows:
            return
        if alignment.rotation is None and plot.state and plot.state.get("robot"):
            stamp, row = min(world_rows, key=lambda r: abs(r[0] - plot.state["t"]))
            robot = next(
                (
                    row[n]
                    for n in [str(gp("robot_model_name")), "calibration_bot", "my_bot"]
                    if n in row
                ),
                None,
            )
            if robot is not None and abs(stamp - plot.state["t"]) <= 0.2:
                alignment.align(robot, plot.state["robot"])
        if alignment.rotation is not None:
            stamp, row = world_rows[-1]
            plot.update_truth(
                stamp,
                {n: alignment.project(p) for n, p in row.items() if pattern.search(n)},
            )

    def receive(msg):
        try:
            state = json.loads(msg.data)
            if state.get("schema_version") != 1:
                raise ValueError("Unsupported visualization schema")
            if plot.last_t is not None and state["t"] < plot.last_t - 1e-6:
                alignment.rotation = None
                alignment.translation = None
                world_rows.clear()
            plot.update(state)
            align_and_show()
        except (ValueError, KeyError, TypeError) as exc:
            node.get_logger().warn(f"Invalid plot snapshot: {exc}")

    def truth_receive(msg):
        t = node.get_clock().now().nanoseconds / 1e9
        if last_clock[0] is not None and t < last_clock[0] - 1e-6:
            world_rows.clear()
            alignment.rotation = None
            alignment.translation = None
        last_clock[0] = t
        row = {}
        for name, p in zip(msg.name, msg.pose):
            q = p.orientation
            yaw = math.atan2(
                2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)
            )
            row[name] = [p.position.x, p.position.y, yaw]
        world_rows.append((t, row))
        align_and_show()

    node.create_subscription(String, str(gp("visualization_topic")), receive, 1)
    if gp("show_simulator_truth"):
        node.create_subscription(
            ModelStates,
            str(gp("ground_truth_topic")),
            truth_receive,
            qos_profile_sensor_data,
        )
    timer = plot.figure.canvas.new_timer(interval=33)

    def tick():
        frame_start = time.monotonic()
        try:
            if not rclpy.ok():
                plt.close(plot.figure)
                return False
            for _ in range(6):
                rclpy.spin_once(node, timeout_sec=0.0)
            plot.draw(plot.presentation_time(), fast=True)
            timer.interval = max(1, 33 - int((time.monotonic() - frame_start) * 1000))
        except (KeyboardInterrupt, ExternalShutdownException):
            plt.close(plot.figure)
            return False

    timer.add_callback(tick)
    timer.start()
    node.get_logger().info(
        "Fixed-size live map open; red human truth is display only. Closing the figure leaves control running."
    )
    try:
        plt.show(block=True)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        timer.stop()
        plt.close(plot.figure)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

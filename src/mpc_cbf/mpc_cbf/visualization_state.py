"""Read-only snapshots of the controller's world-frame Kalman beliefs."""

import numpy as np
from .mpc_controller.kalman_tracker import make_d_safe_array


def controller_snapshot(controller):
    """Build a read-only visualization snapshot at the current control time."""
    now = controller._now_sec()
    humans = []
    for track in controller.track_manager.tracks:
        seen_age = now - track.last_seen_time
        if seen_age < 0 or seen_age > controller.track_manager.stale_timeout_sec:
            continue
        x, y, trace = track.kf.predict_horizon(
            controller.N,
            controller.Ts,
            start_time_offset=max(0.0, now - track.state_time),
        )
        radii = make_d_safe_array(trace, controller.d_safe_base, controller.kappa)
        humans.append(
            dict(
                id=track.internal_id,
                x=x.tolist(),
                y=y.tolist(),
                radius=radii.tolist(),
                trace=trace.tolist(),
                seen_age=seen_age,
            )
        )
    path = controller.route_reference.path
    return dict(
        schema_version=1,
        t=now,
        frame_id=controller.world_frame,
        robot=None if controller.robot_pose is None else controller.robot_pose.tolist(),
        goal=controller.goal.tolist(),
        command=controller.last_command.tolist(),
        status=controller.last_stop_reason or "moving / planning",
        d_min=controller.d_safe_base,
        kappa=controller.kappa,
        horizon_sec=controller.N * controller.Ts,
        humans=humans,
        route=[] if path is None else np.asarray(path).tolist(),
        perception_age=(
            None
            if controller.last_track_stamp_sec is None
            else now - controller.last_track_stamp_sec
        ),
    )

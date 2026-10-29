# MPC-CBF RGB-D Human Avoidance

ROS 2 package connecting RGB-D human observations to uncertainty-aware model
predictive control with discrete control barrier functions (DCBFs).

## Architecture

`rgbd_tracking_adapter` uses the `human_tracking` detector and image
tracker. It publishes current matched pixel/depth measurements and image IDs on
`/tracked_boxes`. Native floating-point depth clustering rejects invalid pixels
and checks for supported nearer layers within the detection box.

`human_world_pose` projects measurements into the configured world frame using
camera calibration and timestamped transforms. It publishes positions and
measurement covariances on `/tracked_humans_world`. Both topics use JSON encoded
in `std_msgs/msg/String` with timestamp and frame metadata.

`mpc_cbf_controller` associates world measurements, maintains a constant-velocity
Kalman filter per human and predicts position and covariance to the control time
and across the horizon. It generates an obstacle-aware reference and solves a
successively linearized MPC quadratic program. The controller publishes the
first accepted command on `/cmd_vel` and repeats on its control timer.

`live_safety_plot` displays controller estimates, trajectories and safety circles.
Gazebo human ground truth is used only for visualization, never as controller or
perception input. Plot interpolation affects display only.

## Requirements

The integration targets ROS 2 Humble and Python 3.10. Required ROS dependencies
are listed in `package.xml`; Python dependencies are listed in `setup.py`.
The `human_tracking` package requires Ultralytics, PyTorch and local YOLO
weights. Live plotting requires a graphical display
and Matplotlib with Tk support.

## Build and run

From the ROS workspace containing both packages:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select human_tracking mpc_cbf
source install/setup.bash
ros2 launch mpc_cbf mpc_cbf.launch.py goal_x:=6.0 goal_y:=0.0 goal_theta_deg:=0.0
```

Start the robot, camera publishers and transforms separately. Do not run the
standalone `human_tracking` detection node alongside the adapter. Set
`use_sim_time:=false` when using wall-clock sensor timestamps. The adapter can
be disabled with `start_tracker_adapter:=false` when a compatible publisher is
provided; the plot can be disabled with `live_plot:=false`.

Parameters are defined in `config/mpc_cbf.yaml`; use `config_file` to supply
an alternative configuration. Set `detector_weights` to a local YOLO weights
file. During initialization, the detector also loads
`Yolo-weights/yolov8m.pt` relative to the process working directory.

## Source map

| File | Responsibility |
| --- | --- |
| `rgbd_tracking_adapter_node.py` | RGB-D validation, detector configuration and image-track integration |
| `perception_measurements.py` | Float-depth clustering and current measurement-to-ID pairing |
| `image_tracker_noise.py` | Image-filter transition and noise matrices |
| `tracker_lifecycle.py` | Image-track confirmation and retirement settings |
| `detector_runtime.py` | Detector device and input-size wrapper |
| `human_world_pose_node.py` | Timestamped world projection and covariance construction |
| `geometry_utils.py` | Projection, transforms and covariance Jacobians |
| `tracking_io.py` | Timestamped JSON message serialization |
| `mpc_cbf_controller_node.py` | ROS integration, control scheduling and command publication |
| `mpc_controller/world_track_manager.py` | World association, deduplication and track lifecycle |
| `mpc_controller/kalman_tracker.py` | World Kalman filtering and horizon prediction |
| `mpc_controller/route_reference.py` | Obstacle-aware route and control references |
| `mpc_controller/mpc_core.py` | Unicycle dynamics and linearized barrier coefficients |
| `mpc_controller/mpc_core_dpp.py` | Cached parameterized MPC solver |
| `mpc_controller/safety_monitor.py` | Post-solve clearance and relaxation assessment |
| `mpc_controller/mpc_core_naive.py` | Fixed-radius comparison solver, inactive in the main node |
| `visualization_state.py` | Controller snapshots for visualization |
| `live_safety_plot.py`, `plot_interpolation.py` | Live map and display interpolation |

## Model and configuration

The robot uses forward-Euler unicycle dynamics with state `[x, y, theta]` and
input `[v, omega]`. Each human uses state `[x, y, vx, vy]` and continuous
white-acceleration process noise. `kf_process_noise_std` is the square root of
the spectral density q, where q has units m<sup>2</sup>/s<sup>3</sup>. Covariance updates use Joseph
form and retain position-velocity cross terms during prediction.

The safety radius at each horizon sample is
`d_safe = d_safe_base + kappa * trace(P_pos)`. Kappa has units 1/m. No radius
cap is applied. The selected configuration uses a 1 m base radius, kappa 0.8/m,
12 prediction intervals and a 0.1 s control period. 

The QP uses deviation variables, separate task and linearization references,
velocity bounds and optional hard acceleration bounds. Safety constraints have
nonnegative, heavily penalized slack. Solver capacity expands for active tracks.
The selected command limits are 1 m/s<sup>2</sup> linear and 3 rad/s<sup>2</sup> angular acceleration.
Emergency, watchdog and safety-stop commands can override these rate limits.

Measurement covariance is propagated using the pixel/depth projection Jacobian.
The configured depth bias (+0.10 m), world-position noise floor (0.15 m), pixel
noise and depth noise were selected for the current simulated camera and human
assets. Validate these values when changing sensors or models. Robot localization
uncertainty is not explicitly propagated. World association uses positional and
uncertainty gates with image IDs as hints; it is not semantic re-identification.

Freshness checks, an emergency-distance stop and near-term predicted-clearance
and slack checks supplement the QP. These checks do not establish a physical
collision-avoidance guarantee. Linearization error, nonzero slack, occlusion,
identity errors and unobserved direction changes can invalidate predictions.
Expired tracks leave the active constraints; expiration does not establish that
an unseen human has left the area. Route memory does not create measurements.

## Visualization

```bash
ros2 run mpc_cbf live_safety_plot --ros-args -p use_sim_time:=true
```

Solid circles show current estimated safety radii; dashed circles show the
horizon-end prediction. Ground-truth markers require `/gazebo/model_states`.
The plot toolbar supports saving still images.

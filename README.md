# Human-aware navigation

RGB-D human tracking and uncertainty-aware MPC with discrete control barrier
functions for ROS 2.

## Packages

- `human_tracking`: RGB-D detection, image tracking and robot simulation.
- `mpc_cbf`: world-coordinate tracking, MPC-DCBF control and live visualization.

## Requirements

ROS 2 Humble, Python 3.10, Gazebo Classic, Ultralytics and PyTorch.
Package dependencies are listed in `package.xml` and `setup.py`.

Provide `Yolo-weights/yolov8m.pt` relative to the working directory used to
launch detection. Set `detector_weights` in `src/mpc_cbf/config/mpc_cbf.yaml`
to the local model file used by the adapter. Gazebo models referenced by the
selected world must be available in the model search path.

## Build

From the workspace root:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select human_tracking mpc_cbf
source install/setup.bash
```

## Run

Launch the robot simulation in one terminal:

```bash
ros2 launch human_tracking launch_sim.launch.py world:=worlds/detects_with_walking_humans.world

```

Launch perception and control in a second terminal with the workspace sourced:

```bash
ros2 launch mpc_cbf mpc_cbf.launch.py goal_x:=6.0 goal_y:=0.0 goal_theta_deg:=0.0
```

The controller launch starts the RGB-D detector, world tracking and live plot.
Run a single detector instance. For physical hardware, provide camera topics,
odometry and transforms, and set `use_sim_time:=false`.

See [MPC-CBF documentation](src/mpc_cbf/README.md) for architecture,
configuration and source organization.

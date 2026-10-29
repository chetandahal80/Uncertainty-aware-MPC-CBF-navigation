"""Launch the tracking adapter, coordinate integration, and safety controller."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    goal_x = LaunchConfiguration("goal_x")
    goal_y = LaunchConfiguration("goal_y")
    goal_theta_deg = LaunchConfiguration("goal_theta_deg")
    start_tracker_adapter = LaunchConfiguration("start_tracker_adapter")

    declarations = [
        DeclareLaunchArgument(
            "config_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("mpc_cbf"), "config", "mpc_cbf.yaml"]
            ),
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("goal_x", default_value="5.0"),
        DeclareLaunchArgument("goal_y", default_value="0.0"),
        DeclareLaunchArgument("goal_theta_deg", default_value="0.0"),
        DeclareLaunchArgument("start_tracker_adapter", default_value="true"),
        DeclareLaunchArgument(
            "live_plot",
            default_value="true",
            description="Open the live robot/human safety plot",
        ),
    ]

    tracker_adapter = Node(
        package="mpc_cbf",
        executable="rgbd_tracking_adapter",
        name="rgbd_subscriber",
        output="screen",
        condition=IfCondition(start_tracker_adapter),
        parameters=[config_file, {"use_sim_time": use_sim_time}],
    )

    world_pose = Node(
        package="mpc_cbf",
        executable="human_world_pose",
        name="human_world_pose_node",
        output="screen",
        parameters=[config_file, {"use_sim_time": use_sim_time}],
    )

    controller = Node(
        package="mpc_cbf",
        executable="mpc_cbf_controller",
        name="mpc_cbf_controller_node",
        output="screen",
        parameters=[
            config_file,
            {
                "use_sim_time": use_sim_time,
                "goal_x": goal_x,
                "goal_y": goal_y,
                "goal_theta_deg": goal_theta_deg,
            },
        ],
    )

    plot = Node(
        package="mpc_cbf",
        executable="live_safety_plot",
        name="live_safety_plot",
        output="screen",
        condition=IfCondition(LaunchConfiguration("live_plot")),
        parameters=[config_file, {"use_sim_time": use_sim_time}],
    )
    return LaunchDescription(
        declarations + [tracker_adapter, world_pose, controller, plot]
    )

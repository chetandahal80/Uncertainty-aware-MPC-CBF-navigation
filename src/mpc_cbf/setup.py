from glob import glob
import os

from setuptools import find_packages, setup

package_name = "mpc_cbf"


setup(
    name=package_name,
    version="0.2.2",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (
            "share/" + package_name,
            ["package.xml", "README.md"],
        ),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "matplotlib",
        "scipy",
        "cvxpy",
        "clarabel",
        "osqp",
    ],
    zip_safe=False,
    maintainer="Chetan Dahal",
    maintainer_email="chetandahal80@users.noreply.github.com",
    description=(
        "Timestamped RGB-D human-track integration with world-frame identity recovery "
        "and robust MPC-DCBF velocity control."
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "rgbd_tracking_adapter = mpc_cbf.rgbd_tracking_adapter_node:main",
            "human_world_pose = mpc_cbf.human_world_pose_node:main",
            "mpc_cbf_controller = mpc_cbf.mpc_cbf_controller_node:main",
            "live_safety_plot = mpc_cbf.live_safety_plot:main",
        ],
    },
)

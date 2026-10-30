from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = Path(get_package_share_directory('bead_metrology'))
    default_config = pkg_share / 'config' / 'params.yaml'

    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=str(default_config),
        description='Path to ROS 2 YAML parameter file for bead_metrology_node'
    )

    metrology_node = Node(
        package='bead_metrology',
        executable='bead_metrology_node',
        name='bead_metrology_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
        emulate_tty=True
    )

    return LaunchDescription([
        config_arg,
        metrology_node
    ])

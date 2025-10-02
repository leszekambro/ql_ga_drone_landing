from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

# Launch nodes required for joystick operation


def generate_launch_description():
    Q_learn_GA_drone_path = get_package_share_directory('Q_learn_GA_drone')
    map_path = os.path.join(Q_learn_GA_drone_path, 'worlds', 'drone_world.yaml')
    namespace = "drone1"
    suffix = "_1"
    return LaunchDescription([
        #Node(package='joy', executable='joy_node', output='screen'),
        #Node(package='tello_driver', executable='tello_joy_main', output='screen'),
        Node(package='tello_driver', executable='tello_driver_main', output='screen'),
        Node(package='joy', executable='joy_node', output='screen',
             namespace=''),
        Node(package='tello_driver', executable='tello_joy_main', output='screen',
             namespace=''),
        #Node(package='Q_learn_GA_drone', executable='Estymator_pozycji_drona', output='screen',
        #     namespace=''),
        Node(package='fiducial_vlam', executable='vmap_main', output='screen',
             name='vmap_main', parameters=[{
                'publish_tfs': 1,  # Publish marker /tf
                'marker_length': 0.1778,  # Marker length
                'marker_map_load_full_filename': map_path,  # Load a pre-built map from disk
                'make_not_use_map': 0}]),  # Don't save a map to disk
        Node(package='fiducial_vlam', executable='vloc_main', output='screen',
                 name='vloc_main', namespace=namespace, parameters=[{
                    'publish_tfs': 1,
                    'base_frame_id': 'base_link'+suffix,
                    't_camera_base_z': -0.035,
                    'camera_frame_id': 'camera_link'+suffix}]),
    ])

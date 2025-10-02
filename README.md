# ql_ga_drone_landing
Comparative study of vision-based precision landing for UAVs using Q-learning and GA-tuned PID. Includes ROS2/Gazebo simulation, real-world tests, and full parameter specs.

# This package was developed using **ROS2 Galactic**.  

It is recommended to install the package in:  
`/home/[your_username]/tello_ros_ws/`  
(You may also install it elsewhere, but in that case remember to update the file paths accordingly.)

---

## Post-installation adjustments
After installation, you need to replace `[your_username]` with your actual username in the following source files:

- `src/tello-ros2-gazebo/tello_ros/tello_driver/src/tello_driver_node.cpp`  
- `src/Q_learn_GA_drone/Q_learn_GA_drone/dron_qlearnV2.py`  
- `src/Q_learn_GA_drone/Q_learn_GA_drone/PyGadV6.py`  
- `src/Q_learn_GA_drone/Q_learn_GA_drone/TestPyGadV6.py`  
- `src/Q_learn_GA_drone/Q_learn_GA_drone/test_qlearnV6.py`  
After making these changes, recompile the packages. 
---
## The simulation environment is launched via launch files.
To run the environment, type in the terminal "ros2 launch Q_learn_GA_drone drone_world_launch_V2.py or another available file.
List of available files:
- drone_world_launch_V2.py - runs the world to learning at an accelerated time
- drone_world_launch_V2.1.py - runs the world to learning with normal time
- drone_world_launch_V3.py - runs the world to learning with poor lighting

drone_launch.py is a launch file to set up a system with a real UAV.

After launching environment, run learning node by "ros2 run Q_learn_GA_drone drone_qlearn_test” or another file.
List of available nodes:
- drone_qlearn_test - runs a node to test the Q-learning algorithm with a real UAV,
- PyGadV6 - runs a node to simulate learning with a GA algorithm,
- TestPyGadV6 - runs a node to test the GA algorithm with a real UAV,
- qlearnV6 - runs a node to simulate learning with the Q-learning algorithm. 

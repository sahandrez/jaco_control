## jaco_control
This package implements motion planning and control utilities for Kinova Jaco 2 arm. This code has been
primarily implemented on Jaco 2 6DOF arm but it also works on 7DOF arm (some of the MoveIt! and Gazebo
functionalities have not been thoroughly tested on the 7DOF arm yet.)

This ROS package implements the following utilities (work in progress):
* Trajectory planning 
* Controllers: 
    * Velocity control: Joint level velocity control (real robot)
    * Feedforward torque control: Model based joint torque control (real and Gazebo robot)
    * Impedance control: Joint level torque control (real and Gazebo robot)

### Dependencies
* Ubuntu 18.04, ROS Melodic, Python 2.7
* Melodic branch of the modified [kinova-ros](https://github.com/sahandrez/kinova-ros/tree/melodic) 
* PyBullet
* MoveIt!
* SciPy

### Installation 
Clone this repo in your `/catkin_ws/src` directory and make the rospackage using `catkin_make` from `/catkin`. 

### Running the Code on Jaco 2 in Gazebo
Bring up the robot in Gazebo with joint effort controllers (Note that for now only torque 
controllers are supported in Gazebo): <br />
`roslaunch kinova_gazebo robot_launch.launch kinova_robotType:=j2n6s300 use_effort_controller:=true`

Launch MoveIt! and RViz if you want to use MoveIt! for planning: <br />
`roslaunch j2n6s300_moveit_config j2n6s300_jaco_lfd.launch`

### Running the Code on Jaco 2 Robot
Bring up the robot: <br />
`roslaunch kinova_bringup kinova_robot.launch kinova_robotType:=j2n6s300`

Launch MoveIt! and RViz if you want to use MoveIt! for planning: <br />
`roslaunch j2n6s300_moveit_config j2n6s300_jaco_lfd.launch`

To use the software E-stop in torque control mode, run the following node. Use `ctrl-C` to stop 
the motion whenever the robot was approaching a dangerous position. Shutting down this node
switches the robot from torque control mode to position control mode: <br />
`rosrun kinova_driver kinova_emergency_stop`

To launch the demo use: <br />
`roslaunch jaco_control jaco_control.launch robot_name:=[j2n6s300 or j2s7s300] gazebo_robot:=[true or false] exp:=[exp_type] control:=[control_type]`

*Note:* In order to be able to run the robot in torque control mode, the state publishing rate 
must be set to at least 50 Hz. You can do that by changing the "status_interval_seconds" 
parameter in `kinova_driver/kinova_arm.cpp`.

Available exp types: ****
* test: Joint space motion planning 
* moveit_test: Cartesian space planning using MoveIt! 
* cutting: Performs cutting motion with a simple cutting force simulation

Available control types:
* velocity (Real robot only)
* fftorque (Gazebo and real robot)
* impedance (Gazebo and real robot)

### Debugging or Gain Tuning
Dynamic configure has been set up for gain tuning of all controllers. To access it run the 
following command: <br />
`rosrun rqt_reconfigure rqt_reconfigure`

To see the actual vs desired joint positions on the real robot use the following command: <br />
`rqt_plot /j2n6s300_driver/out/joint_state/position[0]:position[1]:position[2]:position[3]:position[4]:position[5] /jaco_control/desired_joint_position/position[0]:position[1]:position[2]:position[3]:position[4]:position[5]`

To do so in Gazebo, replace `/j2n6s300_driver/out/joint_state` with `/j2n6s300/joint_states`.

#!/usr/bin/env bash

# *******************************************************************
# Author: Sahand Rezaei-Shoshtari
# Oct. 2019
# Copyright 2019, Sahand Rezaei-Shoshtari, All rights reserved.
# *******************************************************************

cd ~/jaco/bagfiles/

rosbag record /j2n6s300_driver/in/cartesian_velocity \
/j2n6s300_driver/out/actual_joint_torques \
/j2n6s300_driver/out/cartesian_command \
/j2n6s300_driver/out/compensated_joint_torques \
/j2n6s300_driver/out/finger_position \
/j2n6s300_driver/out/joint_angles \
/j2n6s300_driver/out/joint_command \
/j2n6s300_driver/out/joint_state \
/j2n6s300_driver/out/tool_pose \
/j2n6s300_driver/out/tool_wrench

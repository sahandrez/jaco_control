#!/usr/bin/env bash

# *******************************************************************
# Author: Sahand Rezaei-Shoshtari
# Oct. 2019
# Copyright 2019, Sahand Rezaei-Shoshtari, All rights reserved.
# *******************************************************************

cd ~/jaco/bagfiles/

rosbag record /j2n6s300_driver/out/joint_state \
/j2n7s300_driver/out/joint_state \
/j2n6s300/joint_states \
/j2n7s300/joint_states \
/jaco_lfd/end_effector_pose \
/jaco_lfd/end_effector_twist \
/jaco_lfd/desired_joint_position \
/jaco_lfd/interaction_force

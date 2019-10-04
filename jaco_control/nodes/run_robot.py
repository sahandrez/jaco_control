#! /usr/bin/env python

# *******************************************************************
# Author: Sahand Rezaei-Shoshtari
# Oct. 2019
# Copyright 2019, Sahand Rezaei-Shoshtari, All rights reserved.
# *******************************************************************

import sys
import rospy
import numpy as np

from jaco_control.utils import robot, planner


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: move_robot_direct --robot_name --gazebo_robot --exp --control")
    else:
        # init the robot
        jaco = robot.Robot(sys.argv[1], sys.argv[2])
        jaco.init_controller()
        jaco.connect_to_robot()

        # init the planner
        planner = planner.Planner(sys.argv[1])

        if sys.argv[3] == 'test':
            # plan the test trajectory
            if sys.argv[1] == 'j2n6s300':
                starting_position = [4.7, 3.5, 2., 4.7, 0., 1.57]
                # traj_pos = np.array([[0.4, 0.4, 0.4, 0.5, 0.5, 0.5]])
                traj_pos = np.array([[0., 0., 0., 0., 0., 0.]])
                test_freq = np.array([[0.2, 0.2, 0.2, 0.2, 0.2, 0.2]])
            else:
                starting_position = [4.7, 3.5, 0., 2., 0., 3.5, 0.]
                # traj_pos = np.array([[0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5]])
                traj_pos = np.array([[0., 0., 0., 0., 0., 0., 0.]])
                test_freq = np.array([[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]])

            test_traj = planner.create_test_traj(traj_pos, test_freq, starting_position, 100, time_step=0.001)

            if sys.argv[4] == 'velocity':
                jaco.velocity_control(test_traj, sleep_time=6)

            elif sys.argv[4] == 'fftorque':
                jaco.fftorque_control(test_traj, sleep_time=1, control_rate=50)

            elif sys.argv[4] == 'impedance':
                jaco.impedance_control(test_traj, sleep_time=1, control_rate=50)

            else:
                rospy.logerr("Invalid controller. Allowed types are 'velocity', 'fftorque', and 'impedance'.")

        elif sys.argv[3] == 'moveit_test':
            # plan the trajectory using MoveIt!
            planner.init_moveit()
            traj_1_pos, traj_1_orientation = [0., -0.4, 0.5], [0.7071, 0., 0., .7071]
            test_traj = planner.plan_moveit(traj_1_pos, traj_1_orientation, euler_flag=False, time_scale=1.)

            if sys.argv[4] == 'velocity':
                jaco.velocity_control(test_traj, sleep_time=10)

            elif sys.argv[4] == 'fftorque':
                jaco.fftorque_control(test_traj, sleep_time=1, control_rate=50)

            elif sys.argv[4] == 'impedance':
                jaco.impedance_control(test_traj, sleep_time=1, control_rate=50)

            else:
                rospy.logerr("Invalid controller. Allowed types are 'velocity', 'fftorque', and 'impedance'.")

            planner.shut_down_moveit()

        elif sys.argv[3] == 'cutting':
            # plan the cutting trajectory
            planner.init_moveit()
            traj = planner.plan_cutting(rod_radius=0.05, cut_force_k=150., cut_force_d=25., count=15)

            if sys.argv[4] == 'velocity':
                jaco.velocity_control(traj, sleep_time=6)

            elif sys.argv[4] == 'fftorque':
                jaco.fftorque_control(traj, sleep_time=1, control_rate=50)

            elif sys.argv[4] == 'impedance':
                jaco.impedance_control(traj, sleep_time=1, control_rate=50)

            else:
                rospy.logerr("Invalid controller. Allowed types are 'velocity', 'fftorque', and 'impedance'.")

            planner.shut_down_moveit()

        else:
            rospy.logerr("Invalid experiment. Allowed types are 'test', 'moveit_test', 'cutting'. ")

        # shutdown
        jaco.shutdown_controller()

        # prevents the node from shutting down
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down aml_controller node")

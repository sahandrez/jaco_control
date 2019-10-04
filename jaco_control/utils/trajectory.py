#! /usr/bin/env python

# *******************************************************************
# Author: Sahand Rezaei-Shoshtari
# Oct. 2019
# Copyright 2019, Sahand Rezaei-Shoshtari, All rights reserved.
# *******************************************************************

import math
import numpy as np
from scipy import interpolate


class Trajectory:
    """
    This class represents a trajectory in joint space.
    """

    def __init__(self, n_joints, waypoints=None, t_waypoints=None, start_pos=None, step_size=0.01, raw_waypoints=None,
                 t_raw_waypoints=None, time_scale=1.0):
        """
        :param n_joints: number of joints on the robot
        :param waypoints: a 2D [M x n_joints] array of waypoints to control the arm
        :param t_waypoints: a 1D [M] array of waypoint times
        :param start_pos: starting position of the trajectory
        :param step_size: time between each waypoint
        :param raw_waypoints: a 2D [m x n_joints] array of raw waypoints (not usable to control the arm)
        :param t_raw_waypoints: a 1D [M] array of raw waypoint times
        :param time_scale: scales time for the interpolation of raw waypoints
        """
        self.waypoints = waypoints
        self.joint_vel = None
        self.joint_acc = None
        self.t_waypoints = t_waypoints
        self.total_t = None
        self.start_pos = start_pos
        self.step_size = step_size
        self.raw_waypoints = raw_waypoints
        self.t_raw_waypoints = t_raw_waypoints

        # Force interaction modeling in Gazebo
        self.rod_center = None
        self.rod_radius = None
        self.cut_force_k = None
        self.cut_force_d = None
        self.cut_direction = None
        self.cut_plane = None

        # general parameters
        self.n_joints = n_joints

        # for numerical differentiation
        self.vel_threshold = 2.
        self.acc_threshold = 5.

        # interpolate the trajectory when the raw waypoints are provided (joint vel and acc is computed inside this)
        if raw_waypoints is not None and t_raw_waypoints is not None:
            self.start_pos = np.copy(raw_waypoints[0, :])
            # Kinova JACO starting angles are all positive
            # self.start_pos[self.start_pos < 0] += 2 * np.pi
            self.total_t = t_raw_waypoints[-1]
            self.parametrize_traj(time_scale=time_scale)

        # create joint vel and acc if waypoints are provided
        elif waypoints is not None and t_waypoints is not None:
            self.start_pos = np.copy(waypoints[0, :])
            # Kinova JACO starting angles are all positive
            # self.start_pos[self.start_pos < 0] += 2 * np.pi
            self.total_t = t_waypoints[-1]
            self.diff_traj()

    def set_interaction_force_param(self, rod_center, rod_radius, cut_force_k, cut_force_d, cut_direction, cut_plane):
        """
        Sets the parameters related to interaction force modeling implemented in Gazebo.
        :param rod_center: (x, y, z) position of the wooden rod, 0 means infinite in that dimension (Gazebo only)
        :type rod_center: list or np.array
        :param rod_radius: radius of the wooden rod (Gazebo only)
        :type rod_radius: float
        :param cut_force_k: K of cutting force
        :type cut_force_k: float
        :param cut_force_d: D of cutting force
        :type cut_force_d: float
        :param cut_direction: (x, y, z) of cut direction. For example (0, -1, 0) means the cut is in -y direction
        :type cut_direction: list or np.array
        :param cut_plane: (x, y, z) of cut direction, only use 0 and 1s. For example (1, 1, 0) means the cut plane is XY
        :type cut_plane: list or np.array
        :return: None
        """
        self.rod_center = rod_center
        self.rod_radius = rod_radius
        self.cut_force_k = cut_force_k
        self.cut_force_d = cut_force_d
        self.cut_direction = cut_direction
        self.cut_plane = cut_plane        

    def get_next_waypoint(self, elapsed_time):
        """
        Gets the elapsed time and returns the index of the next waypoint.
        :param elapsed_time: elapsed time of the trajectory
        :type elapsed_time: float
        :return: index of the next waypoint for the robot to reach
        :rtype: int
        """
        return int(math.ceil((elapsed_time + 10 ** -8) / self.step_size))

    def diff_traj(self):
        """
        Differentiates the trajectory to obtain joint velocities and accelerations. These are required for using the
        inverse dynamics model.
        :return: None
        """
        self.joint_vel = np.gradient(self.waypoints, self.step_size, axis=0)
        self.joint_vel[abs(self.joint_vel) > self.vel_threshold] = self.vel_threshold

        self.joint_acc = np.gradient(self.joint_vel, self.step_size, axis=0)
        self.joint_acc[self.joint_acc > self.acc_threshold] = self.acc_threshold

    def parametrize_traj(self, time_scale=1.):
        """
        Time parametrizes the trajectory based off the raw waypoints. For now, this method fits a spline to the raw
        waypoints and interpolates all the intermediary waypoints. This method also computes joint vel and acc.
        :param time_scale: scales time for the interpolation of raw waypoints
        :type time_scale: float
        :return: None
        """
        # scale the time
        self.t_raw_waypoints = self.t_raw_waypoints / time_scale
        self.total_t = self.t_raw_waypoints[-1]

        # fit a cubic spline with clamped boundary conditions
        spline = interpolate.CubicSpline(self.t_raw_waypoints, self.raw_waypoints, bc_type='clamped')

        # interpolate and differentiate the spline
        self.t_waypoints = np.arange(0., self.total_t, self.step_size)
        self.t_waypoints = self.t_waypoints.reshape((-1, 1))

        # interpolate
        self.waypoints = spline(self.t_waypoints)
        self.waypoints = self.waypoints.reshape((-1, self.n_joints))

        self.joint_vel = spline(self.t_waypoints, 1)
        self.joint_vel = self.joint_vel.reshape((-1, self.n_joints))
        self.joint_vel[self.joint_vel > self.vel_threshold] = self.vel_threshold

        self.joint_acc = spline(self.t_waypoints, 2)
        self.joint_acc = self.joint_acc.reshape((-1, self.n_joints))
        self.joint_acc[self.joint_acc > self.acc_threshold] = self.acc_threshold

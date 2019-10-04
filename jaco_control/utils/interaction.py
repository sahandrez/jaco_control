#! /usr/bin/env python

# *******************************************************************
# Author: Sahand Rezaei-Shoshtari
# Oct. 2019
# Copyright 2019, Sahand Rezaei-Shoshtari, All rights reserved.
# *******************************************************************

import numpy as np

# ROS libs
import rospy
import dynamic_reconfigure.server
from jaco_control.cfg import cutting_paramsConfig

# ROS messages and services
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, Wrench, Vector3, WrenchStamped, PoseStamped, TwistStamped
from gazebo_msgs.srv import ApplyBodyWrench, BodyRequest
from jaco_control.msg import InteractionParams


class Interaction:
    """
    This class handles the interaction of the robot in Gazebo by using a simple cutting model.
    """

    def __init__(self):
        """
        Initializes the interaction class.
        """
        # initialize and clean up the ROS node
        rospy.init_node('jaco_interaction', anonymous=True)

        # Robot parameters
        self.n_joints = 6
        self.prefix = '/j2n6s300'

        # Cut parameters
        # F ~ K * cut section
        self.cutting_force_K = 100.
        # F ~ C * forward speed
        self.cutting_force_D = 10.
        # Forcing a hard constraint on robot motion by applying a force
        self.constraint_force_K = 10.
        self.constraint_force_D = 2.

        # a flag that indicates one pass of cut has been done
        self.cut_done = False

        # PUBLISHERS
        # Publisher for interaction force
        self.interaction_force_publisher = None

        # SUBSCRIBERS
        # InteractionParams.msg command subscriber to receive the command from Robot object
        self.interaction_command_subscriber = None
        # State subscribers
        self.state_subscriber = None
        # End effector pose and twist subscribers (used in Gazebo)
        self.end_effector_pose_subscriber = None
        self.end_effector_twist_subscriber = None

        # SERVICES
        # Apply body wrench to the model in Gazebo
        self.apply_body_wrench_service = None
        # Clear body wrench on the model in Gazebo
        self.clear_body_wrenches_service = None

        # Dynamic reconfigure server
        self.reconfigure_server = dynamic_reconfigure.server.Server(cutting_paramsConfig, self.reconfigure_callback)

        # Callback data holders
        self.robot_joint_states = JointState()
        self.end_effector_pose = PoseStamped()
        self.end_effector_twist = TwistStamped()

        # Init all services, publishers and subscribers
        self.init_interaction_unit()

        rospy.loginfo("The interaction node is initialized and ready to start force interaction.")

    def init_interaction_unit(self):
        """
        Initializes all the publishers, subscribers and services.
        :return: None
        """
        # init services
        rospy.wait_for_service('/gazebo/apply_body_wrench')
        self.apply_body_wrench_service = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

        rospy.wait_for_service('/gazebo/clear_body_wrenches')
        self.clear_body_wrenches_service = rospy.ServiceProxy('/gazebo/clear_body_wrenches', BodyRequest)

        # init publishers
        self.interaction_force_publisher = rospy.Publisher("/jaco_control/interaction_force", WrenchStamped, queue_size=50)

        # init subscribers
        self.interaction_command_subscriber = rospy.Subscriber("/jaco_control/interaction_force_params", InteractionParams, self.force_interaction_callback, queue_size=50)
        self.state_subscriber = rospy.Subscriber("/j2n6s300/joint_states", JointState, self.receive_joint_state, queue_size=50)
        self.end_effector_pose_subscriber = rospy.Subscriber("/jaco_control/end_effector_pose", PoseStamped, self.receive_end_effector_pose, queue_size=50)
        self.end_effector_twist_subscriber = rospy.Subscriber("/jaco_control/end_effector_twist", TwistStamped, self.receive_end_effector_twist, queue_size=50)

    def receive_joint_state(self, robot_joint_state):
        """
        Callback for '/prefix_driver/out/joint_state'.
        :param robot_joint_state: data from topic
        :type robot_joint_state JointState
        :return None
        """
        self.robot_joint_states = robot_joint_state

    def receive_end_effector_pose(self, end_effector_pose):
        """
        Callback for '/jaco_control/end_effector_pose'.
        :param end_effector_pose: msg
        :type end_effector_pose: PoseStamped
        :return: None
        """
        self.end_effector_pose = end_effector_pose

    def receive_end_effector_twist(self, end_effector_twist):
        """
        Callback for '/jaco_control/end_effector_twist'.
        :param end_effector_twist: msg
        :type end_effector_twist: PoseStamped
        :return: None
        """
        self.end_effector_twist = end_effector_twist

    def reconfigure_callback(self, config, level):
        """
        Callback function for cutting dynamic reconfigure server. Logs info if a change has been made to the
        parameters, returns the updated parameters and updates the controllers.
        :param config: new configuration of the dynamic parameters
        :type config: dynamic_reconfigure.encoding.Config
        :param level: an argument for reconfigure
        :type level: int
        :return: None
        """
        # cutting parameters
        rospy.loginfo("""Reconfigure Request for cutting parameters: 
                cutting force gains: {cutting_force_K}, {cutting_force_D}
                constraint force gains: {constraint_force_K}, {constraint_force_D}""".format(**config))

        self.cutting_force_K = config['cutting_force_K']
        self.cutting_force_D = config['cutting_force_D']

        self.constraint_force_K = config['constraint_force_K']
        self.constraint_force_D = config['constraint_force_D']

        return config

    def publish_interaction_force(self, wrench):
        """
        Publisher interaction force for later data analysis.
        :param wrench: published wrench
        :type wrench: Wrench
        :return: None
        """
        header = Header(stamp=rospy.Time.now())
        wrench_stamped = WrenchStamped(header=header, wrench=wrench)

        self.interaction_force_publisher.publish(wrench_stamped)

    def apply_body_wrench(self, body_name, reference_name, reference_point, wrench, duration=1., start_time=0):
        """
        A general method to apply a wrench to a body in Gazebo through '/gazebo/apply_body_wrench' service.
        :param body_name: body name in Gazebo (for example j2n6s300::j2n6s300_link_6 is the end-effector)
        :type body_name: str
        :param reference_name: reference name in Gazebo (for example j2n6s300::j2n6s300_link_6 is the end-effector)
        :type reference_name: str
        :param reference_point: point of action of the wrench in the reference frame
        :type reference_point: Point
        :param wrench: wrench to be applied to the body
        :type wrench: Wrench
        :param duration: duration that the wrench is applied to the body
        :type duration: float
        :param start_time: start time of the wrench in seconds, if set to 0 it starts right away
        :type start_time: int
        :return: None
        """
        rospy.loginfo("Applying wrench %s to body %s for %s seconds", wrench, body_name, duration)

        # convert start_time and duration
        start_time = rospy.Time(start_time)
        duration = rospy.Duration.from_sec(duration)

        self.apply_body_wrench_service(body_name, reference_name, reference_point, wrench, start_time, duration)

    def apply_end_effector_wrench(self, wrench, duration=1., start_time=0):
        """
        Apply a body wrench to the end-effector link.
        :param wrench: wrench to be applied to the end-effector
        :type wrench: Wrench
        :param duration: duration that the wrench is applied to the end-effector
        :type duration: float
        :param start_time: start time of the wrench in seconds, if set to 0. it starts right away
        :type start_time: int
        :return: None
        """
        # init the message
        body_name = 'j2n6s300::j2n6s300_link_6'
        reference_name = 'world'
        reference_point = Point(x=0., y=0., z=0.)
        self.apply_body_wrench(body_name, reference_name, reference_point, wrench, duration=duration, start_time=start_time)

    def force_interaction_callback(self, msg):
        """
        Model and apply the interaction force created by cutting.
        :param msg: interaction params
        :type msg: InteractionParams
        :return: None
        """
        # a simple cutting force model that starts with a ramp, stays constant with a noise and lowers to zero
        rod_centre = np.array([msg.center.x, msg.center.y, msg.center.z])
        rod_radius = msg.radius
        self.cutting_force_K = msg.cut_force_k
        self.cutting_force_D = msg.cut_force_d
        cut_direction = np.array([msg.direction.x, msg.direction.y, msg.direction.z])
        cut_plane = np.array([msg.plane.x, msg.plane.y, msg.plane.z])

        x = np.array([self.end_effector_pose.pose.position.x,
                      self.end_effector_pose.pose.position.y,
                      self.end_effector_pose.pose.position.z])
        x_dot = np.array([self.end_effector_twist.twist.linear.x,
                          self.end_effector_twist.twist.linear.y,
                          self.end_effector_twist.twist.linear.z])

        # mask the variables according to the cutting plane, its normal vector and cutting direction
        plane_mask = cut_plane.astype(bool)
        direction_mask = cut_direction.astype(bool)

        rod_centre_2d, cut_normal = rod_centre[plane_mask], rod_centre[~plane_mask]
        x_plane, x_normal = x[plane_mask], x[~plane_mask]
        x_dot_normal = x_dot[~plane_mask]

        # approximate the cutting section by a straight line
        dist_to_center = np.linalg.norm(x_plane - rod_centre_2d)
        try:
            cut_section = 2. * np.sqrt(rod_radius ** 2 - dist_to_center ** 2)
        except RuntimeWarning:
            cut_section = 0.

        # determine the cut speed and check if it is currently in the direction of cut or not
        cut_speed = cut_direction * x_dot
        cut_speed = cut_speed[cut_speed.nonzero()]

        if dist_to_center < rod_radius and not self.cut_done:
            # apply force to keep the end effector motion in the cutting plane
            constraint_force = -self.constraint_force_K * (x_normal - cut_normal) - self.constraint_force_D * x_dot_normal

            # cutting force
            cut_force = self.cutting_force_K * cut_section + self.cutting_force_D * (max(np.zeros(1), cut_speed))**2

            # add the forces
            total_force = np.zeros(3)
            total_force[direction_mask] = cut_force[0]
            total_force[~plane_mask] = constraint_force[0]
            total_wrench = np.zeros(3)

            # add Gaussian noise to the force and wrench
            mean, force_std, wrench_std = 0., .5, .1
            total_force += np.random.normal(mean, force_std, 3)
            total_wrench += np.random.normal(mean, wrench_std, 3)

            force = Vector3(x=total_force[0], y=total_force[1], z=total_force[2])
            torque = Vector3(x=total_wrench[0], y=total_wrench[1], z=total_wrench[2])
            wrench = Wrench(force=force, torque=torque)

            self.apply_end_effector_wrench(wrench)
            self.publish_interaction_force(wrench)

        else:
            if abs(x[direction_mask]) > abs(rod_centre[direction_mask]) + rod_radius:
                self.cut_done = True
            elif abs(x[direction_mask]) < abs(rod_centre[direction_mask]) - rod_radius:
                self.cut_done = False

            wrench = Wrench()
            self.clear_body_wrenches()
            self.publish_interaction_force(wrench)

    def clear_body_wrenches(self, body_name='j2ns6300::j2n6s300_link_6'):
        """
        Clears the body wrench applied to the body.
        :param body_name: name of the body
        :type body_name: str
        :return: None
        """
        # rospy.loginfo("Clearing all the wrenches applying to body %s.", body_name)
        self.clear_body_wrenches_service(body_name)

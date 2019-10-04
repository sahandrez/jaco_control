#! /usr/bin/env python

# *******************************************************************
# Author: Sahand Rezaei-Shoshtari
# Oct. 2019
# Copyright 2019, Sahand Rezaei-Shoshtari, All rights reserved.
# *******************************************************************

import os
import numpy as np
import pybullet
import pid
import trajectory

# ROS libs
import rospy
import rospkg
import actionlib
import tf
import tf.transformations
import tf2_ros
import dynamic_reconfigure.server
from jaco_control.cfg import controller_gainsConfig

# ROS messages and services
from std_msgs.msg import Float64, Header
from geometry_msgs.msg import Pose, PoseStamped, Twist, TwistStamped, Vector3, Point, Quaternion, Wrench, WrenchStamped
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkState, LinkStates
from gazebo_msgs.srv import SetModelConfiguration
from kinova_msgs.msg import JointVelocity, JointTorque, JointAngles
from kinova_msgs.msg import ArmJointAnglesGoal, ArmJointAnglesAction, SetFingersPositionAction, SetFingersPositionGoal
from kinova_msgs.srv import HomeArm, SetTorqueControlMode, SetTorqueControlParameters
from jaco_control.msg import InteractionParams


class Robot:
    """
    This class handles the control of the robot.
    """

    def __init__(self, robot_name, gazebo_robot):
        """
        Initializes the robot class.
        :param robot_name: 'j2n6s300' or 'j2s7s300'
        :type robot_name: str
        :param gazebo_robot: 'true' or 'false' but not bool
        :type gazebo_robot: str
        """
        # initialize and clean up the ROS node
        rospy.init_node('jaco_controller', anonymous=True)

        # Robot parameters
        self.n_joints = int(robot_name[3])
        self.prefix = '/' + robot_name
        self.MAX_FINGER_TURNS = 6800

        # FLAGS and STATE DESCRIBERS
        self.active_controller = None
        self.GAZEBO_ROBOT = (gazebo_robot == 'true')

        # SUBSCRIBERS
        # State subscribers
        self.state_subscriber = None
        # Torque subscribers
        self.actual_joint_torque_subscriber = None
        self.compensated_joint_torque_subscriber = None
        # Link states subscriber (used in Gazebo)
        self.link_states_subscribers = None
        # End-effector state subscriber (used on the real robot)
        self.end_effector_state_subscriber = None
        # End-effector wrench
        self.end_effector_wrench_subscriber = None

        # PUBLISHERS
        # Joint velocity command publisher
        self.joint_velocity_publisher = None
        # Joint torque command publisher
        self.joint_torque_publisher = None
        # Desired joint position publisher (useful for gain tuning and debugging)
        self.desired_joint_position_publisher = None
        # Joint torque publisher for JointEffortController in Gazebo
        self.joint_1_torque_publisher = None
        self.joint_2_torque_publisher = None
        self.joint_3_torque_publisher = None
        self.joint_4_torque_publisher = None
        self.joint_5_torque_publisher = None
        self.joint_6_torque_publisher = None
        self.joint_7_torque_publisher = None
        # Send interaction params to the force interaction node
        self.force_interaction_params_publisher = None
        # End-effector state publisher (used in Gazebo for easier rosbag records)
        self.end_effector_pose_publisher = None
        self.end_effector_twist_publisher = None

        # TF Buffer and Listener
        self.tf_Buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_Buffer)

        # SERVICES and ACTIONS
        # Joint angle client
        self.joint_angle_client = None
        # Gripper client
        self.gripper_client = None
        # Robot homing service
        self.home_robot_service = None
        # Controller switcher service
        self.set_torque_control_mode_service = None
        # Torque controller parameters updater
        self.set_torque_control_parameters_service = None
        # Set robot configuration in Gazebo
        self.set_gazebo_config_service = None

        # Controllers
        self.velocity_controller = pid.PID(0., 0., 0., self.n_joints)
        self.fftorque_controller = pid.PID(0., 0., 0., self.n_joints)
        self.impedance_controller = pid.PID(0., 0., 0., self.n_joints)
        self.P_task, self.I_task, self.D_task = None, None, None

        # ROS Parameters
        self.payload_parameter_server = None

        # Dynamic reconfigure server
        self.reconfigure_server = dynamic_reconfigure.server.Server(controller_gainsConfig, self.reconfigure_callback)

        # Callback data holders
        self.robot_joint_states = JointState()
        self.end_effector_state = LinkState()
        self.actual_joint_torques = JointAngles()
        self.compensated_joint_torques = JointAngles()
        self.end_effector_wrench = Wrench()

        # PyBullet object
        self.pybullet_robot = None

        # Directories
        self.rospack = rospkg.RosPack()
        self.rospack_path = self.rospack.get_path('jaco_control')
        self.description_directory = 'description'

        rospy.loginfo("Robot init successful.")

    def init_controller(self):
        """
        Initializes all the publishers and services to control the robot.
        :return: None
        """
        if not self.GAZEBO_ROBOT:
            # init service clients
            self.joint_angle_client = actionlib.SimpleActionClient(self.prefix + '_driver/joints_action/joint_angles', ArmJointAnglesAction)
            self.joint_angle_client.wait_for_server()
            self.gripper_client = actionlib.SimpleActionClient(self.prefix + '_driver/fingers_action/finger_positions', SetFingersPositionAction)
            self.gripper_client.wait_for_server()

            # init publishers
            self.joint_velocity_publisher = rospy.Publisher(self.prefix + '_driver/in/joint_velocity', JointVelocity, queue_size=50)
            self.joint_torque_publisher = rospy.Publisher(self.prefix + '_driver/in/joint_torque', JointTorque, queue_size=50)
            self.desired_joint_position_publisher = rospy.Publisher('/jaco_control/desired_joint_position', JointState, queue_size=50)

            # init services
            rospy.wait_for_service(self.prefix + '_driver/in/set_torque_control_mode')
            self.set_torque_control_mode_service = rospy.ServiceProxy(self.prefix + '_driver/in/set_torque_control_mode', SetTorqueControlMode)

            rospy.wait_for_service(self.prefix + '_driver/in/set_torque_control_parameters')
            self.set_torque_control_parameters_service = rospy.ServiceProxy(self.prefix + '_driver/in/set_torque_control_parameters', SetTorqueControlParameters)

            rospy.wait_for_service(self.prefix + '_driver/in/home_arm')
            self.home_robot_service = rospy.ServiceProxy(self.prefix + '_driver/in/home_arm', HomeArm)

            # init payload parameter server
            self.payload_parameter_server = self.prefix + '_driver/payload'

            # init pybullet for inverse dynamics and Jacobian computation
            self.init_pybullet()

            # init the controllers
            self.init_velocity_controller()
            self.init_fftorque_controller()
            self.init_impedance_controller()

            rospy.loginfo("Jaco controller init successful.")

        else:
            # init publishers
            self.desired_joint_position_publisher = rospy.Publisher('/jaco_control/desired_joint_position', JointState, queue_size=50)
            self.joint_1_torque_publisher = rospy.Publisher(self.prefix + '/joint_1_effort_controller/command', Float64, queue_size=50)
            self.joint_2_torque_publisher = rospy.Publisher(self.prefix + '/joint_2_effort_controller/command', Float64, queue_size=50)
            self.joint_3_torque_publisher = rospy.Publisher(self.prefix + '/joint_3_effort_controller/command', Float64, queue_size=50)
            self.joint_4_torque_publisher = rospy.Publisher(self.prefix + '/joint_4_effort_controller/command', Float64, queue_size=50)
            self.joint_5_torque_publisher = rospy.Publisher(self.prefix + '/joint_5_effort_controller/command', Float64, queue_size=50)
            self.joint_6_torque_publisher = rospy.Publisher(self.prefix + '/joint_6_effort_controller/command', Float64, queue_size=50)
            self.joint_7_torque_publisher = rospy.Publisher(self.prefix + '/joint_7_effort_controller/command', Float64, queue_size=50)
            self.force_interaction_params_publisher = rospy.Publisher('/jaco_control/interaction_force_params', InteractionParams, queue_size=50)
            self.end_effector_pose_publisher = rospy.Publisher('/jaco_control/end_effector_pose', PoseStamped, queue_size=50)
            self.end_effector_twist_publisher = rospy.Publisher('/jaco_control/end_effector_twist', TwistStamped, queue_size=50)

            # init services
            rospy.wait_for_service('/gazebo/set_model_configuration')
            self.set_gazebo_config_service = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)

            # init pybullet for inverse dynamics and Jacobian computation
            self.init_pybullet()

            # init the controllers
            self.init_fftorque_controller()
            self.init_impedance_controller()

            rospy.loginfo("Jaco controller init successful.")

    def init_pybullet(self):
        """
        Initializes the robot in pybullet for inverse dynamics and Jacobian calculations.
        :return: None
        """
        # set the URDF path
        urdf_path = os.path.join(self.rospack_path, self.description_directory)

        # init pybullet
        pybullet.connect(pybullet.DIRECT)
        pybullet.setAdditionalSearchPath(urdf_path)

        # load the robot
        pybullet.setGravity(0, 0, -9.80)
        self.pybullet_robot = pybullet.loadURDF(self.prefix[1:] + ".urdf")

    def connect_to_robot(self):
        """
        Connects to the robot.
        :return: None
        """
        topics = [name for (name, _) in rospy.get_published_topics()]

        if not self.GAZEBO_ROBOT:
            if self.prefix + '_driver/out/joint_state' in topics:
                self.state_subscriber = rospy.Subscriber(self.prefix + "_driver/out/joint_state", JointState, self.receive_joint_state, queue_size=50)
                self.actual_joint_torque_subscriber = rospy.Subscriber(self.prefix + "_driver/out/actual_joint_torques",
                                                                       JointAngles, self.receive_actual_joint_torque,
                                                                       queue_size=50)
                self.compensated_joint_torque_subscriber = rospy.Subscriber(self.prefix + "_driver/out/compensated_joint_torques",
                                                                            JointAngles, self.receive_compensated_joint_torque,
                                                                            queue_size=50)
                self.end_effector_state_subscriber = rospy.Subscriber(self.prefix + "_driver/out/tool_pose", PoseStamped,
                                                                      self.receive_end_effector_state, queue_size=50)
                self.end_effector_wrench_subscriber = rospy.Subscriber(self.prefix + "_driver/out/tool_wrench",
                                                                       WrenchStamped, self.receive_end_effector_wrench,
                                                                       queue_size=50)
                rospy.loginfo("Connected to the robot")

            else:
                rospy.logerr("COULD NOT connect to the robot.")

        else:
            if self.prefix + '/joint_states' in topics:
                self.state_subscriber = rospy.Subscriber(self.prefix + "/joint_states", JointState, self.receive_joint_state, queue_size=50)
                self.link_states_subscribers = rospy.Subscriber("/gazebo/link_states", LinkStates, self.receive_link_states, queue_size=50)
                rospy.loginfo("Connected to the robot in Gazebo.")

            else:
                rospy.logerr("COULD NOT connect to the robot in Gazebo.")

    def receive_joint_state(self, robot_joint_state):
        """
        Callback for '/prefix_driver/out/joint_state'.
        :param robot_joint_state: data from topic
        :type robot_joint_state JointState
        :return None
        """
        self.robot_joint_states = robot_joint_state

    def receive_link_states(self, robot_link_states):
        """
        Callback for '/gazebo/link_states' topic. This method used to get the pose of link 6, but now it has changed to
        get the pose of end-effector based on the published transforms. Still the twist of the end-effector is assumed
        to be equal to the twist of link 6.
        :param robot_link_states: data from the topic
        :type robot_link_states: LinkStates
        :return: None
        """
        try:
            # lookup the transform
            transform_object = self.tf_Buffer.lookup_transform(self.prefix[1:] + "_link_base",
                                                               self.prefix[1:] + "_end_effector",
                                                               rospy.Time(0), rospy.Duration(1))
            translation = (transform_object.transform.translation.x,
                           transform_object.transform.translation.y,
                           transform_object.transform.translation.z)
            rotation = (transform_object.transform.rotation.x,
                        transform_object.transform.rotation.y,
                        transform_object.transform.rotation.y,
                        transform_object.transform.rotation.z)

            # create a transform matrix
            transformer_ros = tf.TransformerROS()
            transform_matrix = np.array(transformer_ros.fromTranslationRotation(translation, rotation))

            # compute end effector position and orientation
            end_effector_position = np.matmul(transform_matrix, [[0.], [0.], [0.], [1.]])
            end_effector_position = Point(x=end_effector_position[0], y=end_effector_position[1], z=end_effector_position[2])

            end_effector_orientation = tf.transformations.quaternion_from_matrix(transform_matrix)
            end_effector_orientation = Quaternion(x=end_effector_orientation[0], y=end_effector_orientation[1],
                                                  z=end_effector_orientation[2], w=end_effector_orientation[3])

            # set up the end_effector states, note that the twist is used from the Gazebo topic
            end_effector_index = 7
            self.end_effector_state.link_name = self.prefix[1:] + '_end_effector'
            self.end_effector_state.pose = Pose(position=end_effector_position, orientation=end_effector_orientation)
            self.end_effector_state.twist = robot_link_states.twist[end_effector_index]
            self.end_effector_state.reference_frame = 'world'

            # publish end effector pose and twist
            self.publish_end_effector_pose()
            self.publish_end_effector_twist()

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

    def receive_end_effector_state(self, pose):
        """
        Callback for '/j2n6s300_driver/out/tool_pose'. Sets the pose of the end effector of the real robot. This method
        does not compute the twist.
        :param pose: end-effector pose
        :type pose: PoseStamped
        :return: None
        """
        self.end_effector_state.link_name = self.prefix[1:] + '_end_effector'
        self.end_effector_state.pose = pose
        self.end_effector_state.reference_frame = 'world'

    def receive_end_effector_wrench(self, msg):
        """
        Callback for '/j2n6s300_driver/out/end/tool_wrench'.
        :param msg: end-effector wrench
        :return: None
        """
        self.end_effector_wrench = msg.wrench

    def receive_actual_joint_torque(self, actual_trq):
        """
        Callback for '/prefix_driver/out/actual_joint_torques'.
        :param actual_trq: data from the topic
        :type actual_trq: JointAngles
        :return: None
        """
        self.actual_joint_torques = actual_trq

    def receive_compensated_joint_torque(self, compensated_trq):
        """
        Callback for '/prefix_driver/out/compensated_joint_torques'.
        :param compensated_trq: data from the topic
        :return: None
        """
        self.compensated_joint_torques = compensated_trq

    def reconfigure_callback(self, config, level):
        """
        Callback function for controller dynamic reconfigure server. Logs info if a change has been made to the
        parameters, returns the updated parameters and updates the controllers.
        :param config: new configuration of the dynamic parameters
        :type config: dynamic_reconfigure.encoding.Config
        :param level: an argument for reconfigure
        :type level: int
        :return: None
        """
        if self.active_controller == 'velocity':
            # velocity controller gains
            rospy.loginfo("""Reconfigure Request for velocity controller: 
                    K_p gains: {velocity_K_p_1}, {velocity_K_p_2}, {velocity_K_p_3}, {velocity_K_p_4}, {velocity_K_p_5}, {velocity_K_p_6}, {velocity_K_p_7}
                    K_d gains: {velocity_K_d_1}, {velocity_K_d_2}, {velocity_K_d_3}, {velocity_K_d_4}, {velocity_K_d_5}, {velocity_K_d_6}, {velocity_K_d_7}""".format(**config))

            if self.n_joints == 6:
                P = np.diag([config['velocity_K_p_1'], config['velocity_K_p_2'], config['velocity_K_p_3'],
                             config['velocity_K_p_4'], config['velocity_K_p_5'], config['velocity_K_p_6']])

                D = np.diag([config['velocity_K_d_1'], config['velocity_K_d_2'], config['velocity_K_d_3'],
                             config['velocity_K_d_4'], config['velocity_K_d_5'], config['velocity_K_d_6']])
            else:
                P = np.diag([config['velocity_K_p_1'], config['velocity_K_p_2'], config['velocity_K_p_3'],
                             config['velocity_K_p_4'], config['velocity_K_p_5'], config['velocity_K_p_6'],
                             config['velocity_K_p_7']])

                D = np.diag([config['velocity_K_d_1'], config['velocity_K_d_2'], config['velocity_K_d_3'],
                             config['velocity_K_d_4'], config['velocity_K_d_5'], config['velocity_K_d_6'],
                             config['velocity_K_d_7']])
            I = 0.0 * np.eye(self.n_joints)
            self.velocity_controller.set_gains(P, I, D)

        elif self.active_controller == 'fftorque':
            # feedforward torque controller gains
            rospy.loginfo("""Reconfigure Request for feedforward torque controller: 
                    K_p gains: {fftorque_K_p_1}, {fftorque_K_p_2}, {fftorque_K_p_3}, {fftorque_K_p_4}, {fftorque_K_p_5}, {fftorque_K_p_6}, {fftorque_K_p_7}
                    K_d gains: {fftorque_K_d_1}, {fftorque_K_d_2}, {fftorque_K_d_3}, {fftorque_K_d_4}, {fftorque_K_d_5}, {fftorque_K_d_6}, {fftorque_K_d_7}""".format(**config))

            if self.n_joints == 6:
                P = np.diag([config['fftorque_K_p_1'], config['fftorque_K_p_2'], config['fftorque_K_p_3'],
                             config['fftorque_K_p_4'], config['fftorque_K_p_5'], config['fftorque_K_p_6']])

                D = np.diag([config['fftorque_K_d_1'], config['fftorque_K_d_2'], config['fftorque_K_d_3'],
                             config['fftorque_K_d_4'], config['fftorque_K_d_5'], config['fftorque_K_d_6']])
            else:
                P = np.diag([config['fftorque_K_p_1'], config['fftorque_K_p_2'], config['fftorque_K_p_3'],
                             config['fftorque_K_p_4'], config['fftorque_K_p_5'], config['fftorque_K_p_6'],
                             config['fftorque_K_p_7']])

                D = np.diag([config['fftorque_K_d_1'], config['fftorque_K_d_2'], config['fftorque_K_d_3'],
                             config['fftorque_K_d_4'], config['fftorque_K_d_5'], config['fftorque_K_d_6'],
                             config['fftorque_K_d_7']])
            I = 0.0 * np.eye(self.n_joints)
            self.fftorque_controller.set_gains(P, I, D)

        elif self.active_controller == 'impedance':
            # impedance controller gains
            rospy.loginfo("""Reconfigure Request for impedance controller: 
                    K_p gains: {impedance_K_p_x}, {impedance_K_p_y}, {impedance_K_p_z}, {impedance_K_p_w_x}, {impedance_K_p_w_y}, {impedance_K_p_w_z}
                    K_d gains: {impedance_K_d_x}, {impedance_K_d_y}, {impedance_K_d_z}, {impedance_K_d_w_x}, {impedance_K_d_w_y}, {impedance_K_d_w_z}""".format(**config))

            self.P_task = np.diag([config['impedance_K_p_x'], config['impedance_K_p_y'], config['impedance_K_p_z'],
                                   config['impedance_K_p_w_x'], config['impedance_K_p_w_y'], config['impedance_K_p_w_z']])

            self.D_task = np.diag([config['impedance_K_d_x'], config['impedance_K_d_y'], config['impedance_K_d_z'],
                                   config['impedance_K_d_w_x'], config['impedance_K_d_w_y'], config['impedance_K_d_w_z']])
            self.I_task = 0.0 * np.eye(self.n_joints)

            P_joint = self.task_space_to_joint_space(self.P_task)
            I_joint = self.task_space_to_joint_space(self.I_task)
            D_joint = self.task_space_to_joint_space(self.D_task)

            self.impedance_controller.set_gains(P_joint, I_joint, D_joint)

        return config

    def compute_idyn_torque(self, q, qdot, qddot):
        """
        Computes the inverse dynamics torques using pybullet library. This function is very fast and can be used in an
        online scheme. Note that this method only computes a single entry.
        :param q: joint positions
        :type q: np.array
        :param qdot: joint velocities
        :type qdot: np.array
        :param qddot: joint accelerations
        :type qddot: np.array
        :return: required torques to do the motion
        :rtype: np.array
        """
        # concatenate finger data because the model in PyBullet requires data of fingers as the robot states
        finger_data = np.zeros(3)
        q = np.concatenate((q, finger_data))
        qdot = np.concatenate((qdot, finger_data))
        qddot = np.concatenate((qddot, finger_data))

        # convert states to list
        q = list(q)
        qdot = list(qdot)
        qddot = list(qddot)

        # compute id torques (returns a [n_joints + 3] list)
        torque = pybullet.calculateInverseDynamics(self.pybullet_robot, q, qdot, qddot)

        # convert the torque to np.array and remove finger data
        torque = np.array(torque)[0:self.n_joints]

        return torque

    def compute_ik(self, position, orientation, euler_flag=False):
        """
        Quick IK solver implemented in PyBullet. This implementation is not working very well. Use MoveIt! instead!
        :param position: target position of the end-effector in world frame
        :type position: list or np.array
        :param orientation: target orientation of the end-effector in world frame
        :type orientation: list or np.array
        :param euler_flag: a flag to indicate whether the orientation is in euler or quaternions
        :type euler_flag: bool
        :return: joint positions
        :rtype: np.array
        """
        # convert orientation euler angles to quaternions if the optional flag is set to True
        if euler_flag:
            orientation = tf.transformations.quaternion_from_euler(*orientation)

        position = list(position)
        orientation = list(orientation)

        if self.n_joints == 6:
            end_effector_link_id = 8
        else:
            end_effector_link_id = 9
        q = pybullet.calculateInverseKinematics(self.pybullet_robot, end_effector_link_id, position, targetOrientation=orientation)

        q = np.array(q)

        return q[:self.n_joints]

    def compute_jacobian(self, q):
        """
        Computes the geometric Jacobian of the arm with respect to the end effector. Note that this method only computes
        a single entry.
        :param q: joint positions
        :type q: np.array
        :return: Jacobian [linear jacobian; angular jacobian] shape: [6, n_joints + 3]
        :rtype: np.array
        """
        # concatenate finger data because the model in PyBullet requires data of fingers as the robot states
        finger_data = np.zeros(3)
        q = np.concatenate((q, finger_data))
        qdot = np.zeros(q.shape)
        qddot = np.zeros(q.shape)

        # convert states to list
        q = list(q)
        qdot = list(qdot)
        qddot = list(qddot)

        # compute Jacobian
        if self.n_joints == 6:
            end_effector_link_id = 8
        else:
            end_effector_link_id = 9
        local_position = [0., 0., 0.]

        linear_jacobian, angular_jacobian = pybullet.calculateJacobian(self.pybullet_robot, end_effector_link_id,
                                                                       local_position, q, qdot, qddot)

        jacobian = np.concatenate((np.array(linear_jacobian), np.array(angular_jacobian)), axis=0)

        # remove fingers from the jacobian
        jacobian = jacobian[:, 0:self.n_joints]

        return jacobian

    def task_space_to_joint_space(self, matrix, q=None):
        """
        Converts a matrix from task space to joint space. if q is provided, the Jacobian is calculated at the specified
        configuration, otherwise, the Jacobian is computed at the current state of the robot.
        :param matrix: the matrix in the task space
        :type matrix: np.array
        :param q: state of the robot to compute the Jacobian (optional)
        :type q: list or np.array
        :return: converted matrix in joint space
        :rtype: np.array
        """
        # compute the jacobian
        if q is None:
            q = np.array(self.robot_joint_states.position[0:self.n_joints])
        else:
            q = np.array(q)
        jacobian = self.compute_jacobian(q)

        return np.matmul(np.matmul(jacobian.transpose(), matrix), jacobian)

    def compensate_gravity(self, torque):
        """
        This method subtracts the torques required for gravity compensations. These torques are computed by the API
        function GetGravityCompensatedTorques and published to a topic.
        :param torque: actual joint torques including the torques required for gravity compensation
        :type torque: JointTorque
        :return: joint torques required for moving the arm
        :rtype: JointTorque
        """
        torque.joint1 = torque.joint1 - (self.actual_joint_torques.joint1 - self.compensated_joint_torques.joint1)
        torque.joint2 = torque.joint2 - (self.actual_joint_torques.joint2 - self.compensated_joint_torques.joint2)
        torque.joint3 = torque.joint3 - (self.actual_joint_torques.joint3 - self.compensated_joint_torques.joint3)
        torque.joint4 = torque.joint4 - (self.actual_joint_torques.joint4 - self.compensated_joint_torques.joint4)
        torque.joint5 = torque.joint5 - (self.actual_joint_torques.joint5 - self.compensated_joint_torques.joint5)
        torque.joint6 = torque.joint6 - (self.actual_joint_torques.joint6 - self.compensated_joint_torques.joint6)
        torque.joint7 = torque.joint7 - (self.actual_joint_torques.joint7 - self.compensated_joint_torques.joint7)

        return torque

    def set_active_controller(self, controller):
        """
        Sets the active controller
        :param controller: a string containing the name of the controller ('velocity', 'fftorque', 'impedance')
        :type controller: str
        :return: None
        """
        self.active_controller = controller

    def init_velocity_controller(self):
        """
        Initializes the velocity controller.
        :return: None
        """
        p_gain = 5.0
        i_gain = 0.0
        d_gain = 1.0
        P = p_gain * np.eye(self.n_joints)
        I = i_gain * np.eye(self.n_joints)
        D = d_gain * np.eye(self.n_joints)
        self.velocity_controller = pid.PID(P, I, D, self.n_joints)

    def velocity_control(self, traj, sleep_time=10.):
        """
        NOTE: Only works on the real robot.
        Controls the robot through the waypoints with velocity controller. Note that this actually controls the joint
        position but through velocity commands. The velocity commands are sent to low level controllers on the robot.
        :param traj: trajectory generated by the Planner class
        :type traj: trajectory.Trajectory
        :param sleep_time: time to wait for the robot to reach the starting position of the trajectory
        :type sleep_time: float or int
        :return: None
        """
        self.set_active_controller('velocity')

        # the publish rate MUST be 100 Hz (Kinova API documentation)
        publish_rate = rospy.Rate(100)

        # send the robot to starting position
        rospy.loginfo("Sending robot to the starting position.")
        print(traj.start_pos)
        self.set_joint_angle(traj.start_pos)
        rospy.sleep(sleep_time)

        # tracking time
        start_time = rospy.get_time()
        elapsed_time = 0.0

        rospy.loginfo("Starting velocity controller.")
        while elapsed_time < rospy.Duration(traj.total_t).to_sec():
            # get the index of the next waypoint
            elapsed_time = rospy.get_time() - start_time
            index = traj.get_next_waypoint(elapsed_time)
            if index >= len(traj.waypoints):
                break

            # compute the error term and update the PID controller (also deals with angle wraparound problem)
            pos = np.array(self.robot_joint_states.position[0:self.n_joints])
            error = self.wrap_to_pi(pos) - self.wrap_to_pi(traj.waypoints[index][:])
            error = error.reshape((-1, 1))

            # only the diagonal elements of the control command matrix is required
            cmd = -np.diag(self.velocity_controller.update_PID(error))

            # send the joint velocity command to the robot
            joint_command = self.create_joint_velocity_cmd(cmd)
            self.send_joint_velocity_cmd(joint_command)

            # publish desired joint position
            self.publish_desired_joint_position(traj.waypoints[index][:])

            # maintain the publish rate of 100 Hz
            publish_rate.sleep()

    def init_fftorque_controller(self):
        """
        Initializes the feedforward torque controller.
        :return: None
        """
        if not self.GAZEBO_ROBOT:
            if self.n_joints == 6:
                P = np.diag([45., 45., 45., 10., 10., 10.])
                D = np.diag([4., 4., 4., 1.5, 1.5, 1.5])
            else:
                P = np.diag([45., 45., 45., 45., 10., 10., 10.])
                D = np.diag([4., 4., 4., 4., 1.5, 1.5, 1.5])
            I = 0.0 * np.eye(self.n_joints)
            self.fftorque_controller = pid.PID(P, I, D, self.n_joints)

        else:
            if self.n_joints == 6:
                P = np.diag([15., 30., 15., 5., 5., 5.])
                D = np.diag([1.5, 3., 1.5, .1, .1, .1])
            else:
                P = np.diag([15., 30., 15., 15, 5., 5., 5.])
                D = np.diag([1.5, 3., 1.5, 1.5, .1, .1, .1])
            I = 0.0 * np.eye(self.n_joints)
            self.fftorque_controller = pid.PID(P, I, D, self.n_joints)

    def fftorque_control(self, traj, control_rate=50, sleep_time=10.):
        """
        NOTE: Works on both the real and Gazebo robot.
        Controls the robot through waypoints with feedforward torque controller. The inverse dynamics is computed using
        PyBullet.
        Note that if the robot is carrying anything, make sure the payload parameter is set up, otherwise the robot will
        not switch to torque control mode.
        :param traj: trajectory generated by the Planner class
        :type traj: traj.Trajectory
        :param control_rate: controller rate
        :type control_rate: int
        :param sleep_time: time to wait for the robot to reach the starting position of the trajectory
        :type sleep_time: float or int
        :return: None
        """
        self.set_active_controller('fftorque')

        # maintain a constant publish rate
        publish_rate = rospy.Rate(control_rate)

        # send the robot to starting position
        rospy.loginfo("Sending robot to the starting position.")
        if not self.GAZEBO_ROBOT:
            self.set_joint_angle(traj.start_pos)
            rospy.sleep(sleep_time)
            # switch to torque control mode
            self.set_torque_control_mode(True)
        else:
            # this makes the Gazebo model unstable (keep it commented)
            # self.set_gazebo_config(traj.start_pos)
            pass

        # tracking time
        start_time = rospy.get_time()
        elapsed_time = 0.0

        rospy.loginfo("Starting feedforward torque controller.")
        while elapsed_time < rospy.Duration(traj.total_t).to_sec():
            # get the index of the next waypoint
            elapsed_time = rospy.get_time() - start_time
            index = traj.get_next_waypoint(elapsed_time)
            if index > len(traj.waypoints):
                break

            # compute the error term and update the PID controller
            pos = np.array(self.robot_joint_states.position[0:self.n_joints])

            # on the real robot, preferably we should do a wrap-around check
            if not self.GAZEBO_ROBOT:
                error = self.wrap_to_pi(pos) - self.wrap_to_pi(traj.waypoints[index][:])
            else:
                error = pos - traj.waypoints[index][:]
            error = error.reshape((-1, 1))

            # only the diagonal elements of the control command matrix is required
            feedback_torque = -np.diag(self.fftorque_controller.update_PID(error))

            # feedforward torque
            feedforward_torque = self.compute_idyn_torque(traj.waypoints[index][:],
                                                          traj.joint_vel[index][:],
                                                          traj.joint_acc[index][:])
            cmd = feedback_torque + feedforward_torque

            # compensate for gravity and send the joint torque command to the robot
            joint_command = self.create_joint_torque_cmd(cmd)
            if not self.GAZEBO_ROBOT:
                joint_command = self.compensate_gravity(joint_command)
            self.send_joint_torque_cmd(joint_command)

            # publish desired joint position
            self.publish_desired_joint_position(traj.waypoints[index][:])

            # maintain the publish rate
            publish_rate.sleep()

        # switch to position control mode
        if not self.GAZEBO_ROBOT:
            self.set_torque_control_mode(False)

    def init_impedance_controller(self):
        """
        Initializes the feedforward torque controller. Note that the impedance controller is first initialized with the
        Jacobian of the home state. But then at each iteration of the controller loop these values are updated.
        :return: None
        """
        if not self.GAZEBO_ROBOT:
            self.P_task = np.diag([20., 20., 20., 10., 10., 10.])
            self.I_task = np.diag([0., 0., 0., 0., 0., 0.])
            self.D_task = np.diag([2., 2., 2., 1., 1., 1.])
        else:
            self.P_task = np.diag([100., 100., 100., 10., 10., 10.])
            self.I_task = np.diag([0., 0., 0., 0., 0., 0.])
            self.D_task = np.diag([8., 8., 8., .5, .5, .5])

        if self.n_joints == 6:
            home_config = [4.9, 2.9, 1., 4.2, 1.4, 1.3]
        else:
            home_config = [4.9, 2.9, 0., 0.75, 4.6, 4.5, 5.]

        P_joint = self.task_space_to_joint_space(self.P_task, home_config)
        I_joint = self.task_space_to_joint_space(self.I_task, home_config)
        D_joint = self.task_space_to_joint_space(self.D_task, home_config)

        self.impedance_controller = pid.PID(P_joint, I_joint, D_joint, self.n_joints)

    def update_joint_space_impedance_gains(self):
        """
        Impedance controller needs to be constantly updated because as the robot moves, the joint space gains need to
        be recomputed with the new Jacobian.
        :return: None
        """
        P_joint = self.task_space_to_joint_space(self.P_task)
        I_joint = self.task_space_to_joint_space(self.I_task)
        D_joint = self.task_space_to_joint_space(self.D_task)

        self.impedance_controller.set_gains(P_joint, I_joint, D_joint)

    def impedance_control(self, traj, use_idyn=True, control_rate=50, sleep_time=10.):
        """
        NOTE: Works on both the real and Gazebo robot.
        Controls the robot through waypoints with impedance controller. If use_idyn is set to true, the inverse dynamics
        is computed using PyBullet, otherwise, the feedforward term is obtained from demonstrations.
        :param traj: trajectory generated by the Planner class
        :type traj: traj.Trajectory
        :param use_idyn: if true, inverse dynamics model is used to generate the feedforward term
        :type use_idyn: bool
        :param control_rate: controller rate
        :type control_rate: int
        :param sleep_time: time to wait for the robot to reach the starting position of the trajectory
        :type sleep_time: float or int
        :return:
        """
        self.set_active_controller('impedance')

        # maintain a constant publish rate
        publish_rate = rospy.Rate(control_rate)

        # send the robot to starting position
        rospy.loginfo("Sending robot to the starting position.")
        if not self.GAZEBO_ROBOT:
            self.set_joint_angle(traj.start_pos)
            rospy.sleep(sleep_time)
            # switch to torque control mode
            self.set_torque_control_mode(True)
        else:
            # this makes the Gazebo model unstable (keep it commented)
            # self.set_gazebo_config(traj.start_pos)
            pass

        # tracking time
        start_time = rospy.get_time()
        elapsed_time = 0.0

        rospy.loginfo("Starting impedance controller.")
        while elapsed_time < rospy.Duration(traj.total_t).to_sec():
            # start the force interaction
            if self.GAZEBO_ROBOT and traj.rod_center and 5. < elapsed_time:
                self.publish_interaction_params(traj)

            # get the index of the next waypoint
            elapsed_time = rospy.get_time() - start_time
            index = traj.get_next_waypoint(elapsed_time)
            if index > len(traj.waypoints):
                break

            # compute the error term and update the PID controller
            pos = np.array(self.robot_joint_states.position[0:self.n_joints])

            # on the real robot, preferably we should do a wrap-around check
            if not self.GAZEBO_ROBOT:
                error = self.wrap_to_pi(pos) - self.wrap_to_pi(traj.waypoints[index][:])
            else:
                error = pos - traj.waypoints[index][:]
            error = error.reshape((-1, 1))

            # update impedance controller gains
            self.update_joint_space_impedance_gains()

            # only the diagonal elements of the control command matrix is required
            feedback_torque = -np.diag(self.impedance_controller.update_PID(error))

            # feedforward torque
            if not use_idyn:
                # implement this later on based on the demonstrations
                feedforward_torque = 0.
            else:
                feedforward_torque = self.compute_idyn_torque(traj.waypoints[index][:],
                                                              traj.joint_vel[index][:],
                                                              traj.joint_acc[index][:])

            cmd = feedback_torque + feedforward_torque

            # compensate for gravity and send the joint torque command to the robot
            joint_command = self.create_joint_torque_cmd(cmd)
            if not self.GAZEBO_ROBOT:
                joint_command = self.compensate_gravity(joint_command)
            self.send_joint_torque_cmd(joint_command)

            # publish desired joint position
            self.publish_desired_joint_position(traj.waypoints[index][:])

            # maintain the publish rate
            publish_rate.sleep()

        # switch to position control mode
        if not self.GAZEBO_ROBOT:
            self.set_torque_control_mode(False)

    def set_joint_angle(self, joint_angles):
        """
        Set the joint positions on the real robot. Planning is done in the robot base.
        :param joint_angles: desired joint positions
        :type joint_angles: list or np.array
        :return: None
        """
        # create the joint command
        joint_command = self.create_joint_angle_cmd(joint_angles)

        # send the joint command to the real robot
        self.send_joint_angle_cmd(joint_command)

    def set_finger_position(self, finger_positions):
        """
        Sets the finger positions; the values are in percentage: Fully closed is 100 and fully open is 0.
        :param finger_positions: list of the finger positions
        :type finger_positions: list
        :return: None
        """
        # convert percentage to thread turn
        finger_turns = [x/100.0 * self.MAX_FINGER_TURNS for x in finger_positions]
        turns_temp = [max(0.0, x) for x in finger_turns]
        finger_turns = [min(x, self.MAX_FINGER_TURNS) for x in turns_temp]

        # create and send the goal message
        goal = SetFingersPositionGoal()
        goal.fingers.finger1 = finger_turns[0]
        goal.fingers.finger2 = finger_turns[1]
        goal.fingers.finger3 = finger_turns[2]

        self.gripper_client.send_goal(goal)

        if self.gripper_client.wait_for_server(rospy.Duration(5)):
            return self.gripper_client.get_result()
        else:
            self.gripper_client.cancel_all_goals()
            rospy.logwarn("The gripper action time-out.")
            return None

    def set_torque_control_mode(self, state):
        """
        Switches the active controller from position to torque mode.
        :param state: True enables the torque controller and False disables it
        :type state: bool
        :return: None
        """
        if state:
            rospy.loginfo("Switched to torque control mode.")
        else:
            rospy.loginfo("Switched to position control mode.")

        try:
            self.set_torque_control_mode_service(int(state))
        except rospy.service.ServiceException:
            # After the upgrade to ROS Melodic, this service returns an error but is still able to switch the robot
            # control scheme. We just need to catch the exception.
            pass

    def set_torque_control_parameters(self):
        """
        Updates the torque controller parameters.
        :return: None
        """
        rospy.loginfo("Updated the torque controller parameters")

        self.set_torque_control_parameters_service()

    def set_payload(self, payload):
        """
        Sets the payload of the arm, not to update the parameters using set_torque_control_parameters method.
        :param payload: the payload of the robot (M, COM_x, COM_y, COM_z)
        :type payload: list
        :return: None
        """
        rospy.set_param(self.payload_parameter_server, payload)

    def set_gazebo_config(self, joint_angles):
        """
        Resets the robot config in Gazebo through gazebo/set_model_configuration service.
        :param joint_angles: joint_angles of the new config
        :type joint_angles: list
        :return: None
        """
        # init the message
        model_name = self.prefix[1:]
        urdf_param_name = 'robot_description'
        if self.n_joints == 6:
            joint_names = ['j2n6s300_joint_1', 'j2n6s300_joint_2', 'j2n6s300_joint_3',
                           'j2n6s300_joint_4', 'j2n6s300_joint_5', 'j2n6s300_joint_6',
                           'j2n6s300_joint_finger_1', 'j2n6s300_joint_finger_2', 'j2n6s300_joint_finger_3']
        else:
            joint_names = ['j2n6s300_joint_1', 'j2n6s300_joint_2', 'j2n6s300_joint_3', 'j2n6s300_joint_4',
                           'j2n6s300_joint_5', 'j2n6s300_joint_6', 'j2n6s300_joint_7',
                           'j2n6s300_joint_finger_1', 'j2n6s300_joint_finger_2', 'j2n6s300_joint_finger_3']
        joint_positions = joint_angles

        # call the service
        self.set_gazebo_config_service(model_name, urdf_param_name, joint_names, list(joint_positions) + [0., 0., 0.])

    def home_robot(self):
        """
        Homes the robot by calling the home robot service
        :return: None
        """
        # call the homing service
        self.home_robot_service()

    def create_joint_angle_cmd(self, angle):
        """
        Creates a joint angle command with the target joint angles. Planning is done in the base of the robot.
        :param angle: goal position of the waypoint, angles are in radians
        :type angle: list
        :return: joint angle command
        :rtype: ArmJointAnglesGoal
        """
        # initialize the command
        joint_cmd = ArmJointAnglesGoal()

        joint_cmd.angles.joint1 = self.convert_to_degree(angle[0])
        joint_cmd.angles.joint2 = self.convert_to_degree(angle[1])
        joint_cmd.angles.joint3 = self.convert_to_degree(angle[2])
        joint_cmd.angles.joint4 = self.convert_to_degree(angle[3])
        joint_cmd.angles.joint5 = self.convert_to_degree(angle[4])
        joint_cmd.angles.joint6 = self.convert_to_degree(angle[5])
        if self.n_joints == 6:
            joint_cmd.angles.joint7 = 0.
        else:
            joint_cmd.angles.joint7 = self.convert_to_degree(angle[6])

        return joint_cmd

    def create_joint_velocity_cmd(self, velocity):
        """
        Creates a joint velocity command with the target velocity for each joint.
        :param velocity: velocity of each joint in radians/s
        :type velocity: np.array
        :return: joint velocity command
        :rtype: JointVelocity
        """
        # init
        velocity = velocity.reshape(-1)
        joint_cmd = JointVelocity()

        joint_cmd.joint1 = self.convert_to_degree(velocity[0])
        joint_cmd.joint2 = self.convert_to_degree(velocity[1])
        joint_cmd.joint3 = self.convert_to_degree(velocity[2])
        joint_cmd.joint4 = self.convert_to_degree(velocity[3])
        joint_cmd.joint5 = self.convert_to_degree(velocity[4])
        joint_cmd.joint6 = self.convert_to_degree(velocity[5])
        if self.n_joints == 6:
            joint_cmd.joint7 = 0.
        else:
            joint_cmd.joint7 = self.convert_to_degree(velocity[6])

        return joint_cmd

    def create_joint_torque_cmd(self, torque):
        """
        Creates a joint torque command with the target torque for each joint.
        :param torque: torque of each joint in N.m
        :type torque: np.array
        :return: joint torque command
        :rtype: JointTorque or list
        """
        if not self.GAZEBO_ROBOT:
            # initialize the command
            joint_cmd = JointTorque()

            joint_cmd.joint1 = torque[0]
            joint_cmd.joint2 = torque[1]
            joint_cmd.joint3 = torque[2]
            joint_cmd.joint4 = torque[3]
            joint_cmd.joint5 = torque[4]
            joint_cmd.joint6 = torque[5]
            if self.n_joints == 6:
                joint_cmd.joint7 = 0.
            else:
                joint_cmd.joint7 = torque[6]

        else:
            joint_cmd = []
            for i in range(self.n_joints):
                joint_cmd.append(Float64(data=torque[i]))

        return joint_cmd

    def send_joint_angle_cmd(self, joint_cmd):
        """
        Sends the joint angle command to the action server and waits for its execution. Note that the planning is done
        in the robot base.
        :param joint_cmd: joint angle command
        :type joint_cmd: ArmJointAnglesGoal
        :return: None
        """
        self.joint_angle_client.send_goal(joint_cmd)
        self.joint_angle_client.wait_for_result()

    def send_joint_velocity_cmd(self, joint_cmd):
        """
        Publishes the joint velocity command to the robot.
        :param joint_cmd: desired joint velocities
        :type joint_cmd: JointVelocity
        :return: None
        """
        self.joint_velocity_publisher.publish(joint_cmd)

    def send_joint_torque_cmd(self, joint_cmd):
        """
        Publishes the joint torque command to the robot.
        :param joint_cmd: desired joint velocities
        :type joint_cmd: JointTorque or list
        :return: None
        """
        if not self.GAZEBO_ROBOT:
            self.joint_torque_publisher.publish(joint_cmd)

        else:
            self.joint_1_torque_publisher.publish(joint_cmd[0])
            self.joint_2_torque_publisher.publish(joint_cmd[1])
            self.joint_3_torque_publisher.publish(joint_cmd[2])
            self.joint_4_torque_publisher.publish(joint_cmd[3])
            self.joint_5_torque_publisher.publish(joint_cmd[4])
            self.joint_6_torque_publisher.publish(joint_cmd[5])
            if self.n_joints == 7:
                self.joint_7_torque_publisher.publish(joint_cmd[6])

    def publish_desired_joint_position(self, q_desired):
        """
        Publishes the desired joint position. Useful for gain tuning or debugging.
        :param q_desired: desired joint position, in a form of [1 x n_joints] 2D array.
        :type: np.array
        :return: None
        """
        msg = JointState()
        msg.header = self.robot_joint_states.header
        msg.name = self.robot_joint_states.name
        msg.position = q_desired.tolist()

        self.desired_joint_position_publisher.publish(msg)

    def publish_interaction_params(self, traj):
        """
        Publishes the interaction parameters to '/jaco_control/force_interaction' topic.
        :param traj: the trajectory that is being run (it also contains information abot interaction params)
        :type traj: trajectory.Trajectory
        :return: None
        """
        center = Point(x=traj.rod_center[0], y=traj.rod_center[1], z=traj.rod_center[2])
        radius = traj.rod_radius
        cut_force_k = traj.cut_force_k
        cut_force_d = traj.cut_force_d
        direction = Vector3(x=traj.cut_direction[0], y=traj.cut_direction[1], z=traj.cut_direction[2])
        plane = Vector3(x=traj.cut_plane[0], y=traj.cut_plane[1], z=traj.cut_plane[2])

        msg = InteractionParams(center=center, radius=radius, cut_force_k=cut_force_k, cut_force_d=cut_force_d,
                                direction=direction, plane=plane)

        self.force_interaction_params_publisher.publish(msg)

    def publish_end_effector_pose(self):
        """
        Publishes the end-effector pose. Useful for Gazebo.
        :return: None
        """
        header = Header(stamp=rospy.Time.now(), frame_id='world')
        pose = Pose(position=self.end_effector_state.pose.position, orientation=self.end_effector_state.pose.orientation)
        msg = PoseStamped(header=header, pose=pose)

        self.end_effector_pose_publisher.publish(msg)

    def publish_end_effector_twist(self):
        """
        Publishes the end-effector twist. Useful for Gazebo.
        :return: None
        """
        header = Header(stamp=rospy.Time.now(), frame_id='world')
        twist = Twist(linear=self.end_effector_state.twist.linear, angular=self.end_effector_state.twist.angular)
        msg = TwistStamped(header=header, twist=twist)

        self.end_effector_twist_publisher.publish(msg)

    @staticmethod
    def convert_to_degree(angle):
        """
        Converts the input angle to degree.
        :param angle: input angle in radian
        :type angle: np.array or float
        :return: converted angle to degrees
        :rtype: np.array or float
        """
        return 180. * angle / np.pi

    @staticmethod
    def wrap_to_pi(angles):
        """
        Wraps a list of angles to [-pi, pi].
        :param angles: list of angles to be wrapped
        :type angles: list or np.array
        :return: list of wrapped angles
        :rtype: np.array
        """
        # convert angles to numpy array
        angles = np.array(angles)

        # wrap the angles to [-pi, pi] phases
        phases = (angles + np.pi) % (2 * np.pi) - np.pi

        return phases

    @staticmethod
    def shutdown_pybullet():
        """
        Disconnects and shuts down pybullet server.
        :return: None
        """
        pybullet.disconnect()

    @staticmethod
    def shutdown_controller():
        """
        Shuts down the controller.
        :return: None
        """
        rospy.loginfo("Shutting Down Controller.")
        rospy.signal_shutdown('Done')
        return exit(0)

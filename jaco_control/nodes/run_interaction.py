#! /usr/bin/env python

# *******************************************************************
# Author: Sahand Rezaei-Shoshtari
# Oct. 2019
# Copyright 2019, Sahand Rezaei-Shoshtari, All rights reserved.
# *******************************************************************

import rospy

from jaco_control.utils import interaction

if __name__ == '__main__':
    interaction_model = interaction.Interaction()

    # this node keeps spinning to keep the interaction unit up and running
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down aml_controller node")

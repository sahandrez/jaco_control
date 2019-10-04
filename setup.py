# ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

# *******************************************************************
# Author: Sahand Rezaei-Shoshtari
# Oct. 2019
# Copyright 2019, Sahand Rezaei-Shoshtari, All rights reserved.
# *******************************************************************

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    name='jaco_control',
    packages=['jaco_control',
              'jaco_control.models',
              'jaco_control.utils'],
    package_dir={'jaco_control': 'jaco_control',
                 'jaco_control.models': 'jaco_control/models',
                 'jaco_control.utils': 'jaco_control/utils'},
)

setup(**setup_args)

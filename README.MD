# Project Name
TODO: Write a project description

## Installation

##### Packages required:
pip3:
munkres matplotlib numpy pyyaml rospkg catkin_pkg
apt-get:
python3-tk
ros-kinetic-teleop-twist-keyboard
    - To using in gazebo to move robot around with keyboard
ros-kinetic-ros-control
ros-kinetic-ros-controller

If using Pioneer Model in ROS Kinetic
change the following
<box size="0 0 0"/>
to
<box size="0.0 0.0 0.0"/>


If using this within a VM
echo "export SVGA_VGPU10=0" >> ~/.bashrc

## Usage

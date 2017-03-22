cmake_minimum_required(VERSION 2.8.3)
project(iterative_closest_point)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  tf
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_package()

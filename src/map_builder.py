#!/usr/bin/python3

from icp import IterativeClosestPoint
import sys
import numpy
import math
from munkres import Munkres
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import copy
import os
import time
# for kdtree nearest neighbor
import scipy.spatial
# Ros specific
import rospy
from roslib import message
import sensor_msgs.point_cloud2 as PCL
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
from roslib import message
import time


def main():
    #test_icp()
    m = MapBuilder()
    m.build_map()

def get_subset_of_points(map, odom, radius):
    subset = []
    r2 = radius**2
    for p in map:
        if ((p[0] - odom[0])**2 + (p[1] - odom[1])**2 + (p[2] - odom[2])**2) <= r2:
            subset.append(p)
    return subset

class MapBuilder:
    def __init__(self):
        self.icp = IterativeClosestPoint(zero_threshold=0.1, convergence_threshold=0.0001, nearest_neighbor_upperbound=1)
        self.map = []
        rospy.init_node('listen', anonymous=True)
        self.publisher = rospy.Publisher('/icp_map', PointCloud2, queue_size=10)

        self.new_odom = None
        self.new_scan = None
        self.cloud_out_subscriber = None
        self.odom_subscriber = None
        self.frame = None

    def laser_listen_once(self):
        self.cloud_out_subscriber = rospy.Subscriber('/cloud_out', PointCloud2, self.cloud_out_callback)

    def odom_listen_once(self):
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.odom_callback)

    def cloud_out_callback(self, data):
        #rospy.loginfo('Received new sensor data.')
        self.cloud_out_subscriber.unregister()
        self.frame = data.header.frame_id
        self.new_scan = self.pointcloud2_to_list(data)

    def odom_callback(self, data):
        #rospy.loginfo('Received new odom information.')
        self.odom_subscriber.unregister()
        self.new_odom = data

    def pointcloud2_to_list(self, cloud):
        gen = PCL.read_points(cloud, skip_nans=True, field_names=('x', 'y', 'z'))
        list_of_tuples = list(gen)
        #list_of_lists = [list(elem) for elem in list_of_tuples]
        return list_of_tuples

    def list_to_pointcloud2(self, points):
        pcloud = PointCloud2()
        pcloud = PCL.create_cloud_xyz32(pcloud.header, points)
        pcloud.header.stamp = rospy.Time.now()
        pcloud.header.frame_id = self.frame
        return pcloud

    def publish_map(self):
        #rospy.loginfo('Publishing ICP Map')
        self.publisher.publish(self.list_to_pointcloud2(self.map))

    def build_map(self):
        rospy.loginfo('Starting Map Building.')
        self.laser_listen_once()
        self.odom_listen_once()
        itr = 0
        oldtime = time.time()
        while not rospy.is_shutdown():
            if self.new_scan is not None and self.new_odom is not None:
                if len(self.map) != 0:
                    m = self.map
                    #pos = self.new_odom.pose.pose.position
                    #m = get_subset_of_points(map=self.map, odom=[pos.x, pos.y, pos.z], radius=20)
                    new_scan_transformed = self.icp.iterative_closest_point(reference=m, source=self.new_scan, initial=[1, 0, 0, 0, 0, 0, 0],
                                                     number_of_iterations=10)
                    self.map.extend(new_scan_transformed)
                else:
                    self.map = self.new_scan
                self.new_scan = None
                self.new_odom = None
                self.laser_listen_once()
                self.odom_listen_once()
            if oldtime + 5 < time.time():
                self.publish_map()
                oldtime = time.time()


if __name__ == "__main__":
    main()

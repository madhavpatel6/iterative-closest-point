#!/usr/bin/env python
import rospy
import sys
import sensor_msgs.point_cloud2 as PCL
from sensor_msgs.msg import PointCloud2, PointField
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 2:
        sys.exit('Error: Please specify a file name.')
    rospy.init_node('publisher', anonymous=True)
    rospy.loginfo('Publishing ' + sys.argv[1] + ' to /icp_map')
    f = open(sys.argv[1], 'r')
    map = []
    for l in f.readlines():
        ls = l.split(' ')
        x, y, z = float(ls[0]), float(ls[1]), float(ls[2])
        map.append([x, y, z])
    pcloud = list_to_pointcloud2(map)
    publisher = rospy.Publisher('/icp_map', PointCloud2, queue_size=10)
    while not rospy.is_shutdown():
        publisher.publish(pcloud)
        rospy.sleep(5)

def list_to_pointcloud2(points):
    pcloud = PointCloud2()
    pcloud = PCL.create_cloud_xyz32(pcloud.header, points)
    pcloud.header.stamp = rospy.Time.now()
    pcloud.header.frame_id = 'odom'
    return pcloud

if __name__ == "__main__":
    main()
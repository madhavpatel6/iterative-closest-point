#!/usr/bin/env python
import rospy
import sensor_msgs.point_cloud2 as PCL
from sensor_msgs.msg import PointCloud2, PointField
from PIL import Image
import numpy
import matplotlib.pyplot as plt

def main():
    rospy.init_node('saver', anonymous=True)
    rospy.loginfo('Saving Image as map.png')
    data = rospy.wait_for_message(topic='/icp_map', topic_type=PointCloud2, timeout=None)
    gen = PCL.read_points(data, skip_nans=True, field_names=('x', 'y', 'z'))
    map = list(gen)
    list_map = [list(elem) for elem in map]
    plot_points_2d(list_map)

def plot_points_2d(map):
    plt.clf()
    for point in map:
        del point[2]
    plt.plot(*zip(*map), marker=',', color='w', ls='')
    plt.axis('scaled')
    plt.gca().set_axis_bgcolor('black')
    plt.savefig('map.png', bbox_inches='tight', dpi=1000)

if __name__ == "__main__":
    main()

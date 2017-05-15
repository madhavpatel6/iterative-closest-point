#!/usr/bin/env python
from __future__ import division
import sys
import rospy
from nav_msgs.msg import Odometry
import math
import tf
import tf.msg
import geometry_msgs.msg
import numpy
from sensor_msgs.msg import PointCloud2, PointField
from namedlist import namedlist
import message_filters
Point = namedlist('Point', 'x y z')


class OdometryCorrector:
    def __init__(self):
        self.publisher = rospy.Publisher('/odomc', Odometry, queue_size=10)
        #self.tf_publisher = rospy.Publisher('/tf', tf.msg.tfMessage, queue_size=10)
        self.odom_data = None
        self.odom_data_prev = None
        self.icp_data = None
        self.odomc = None
        self.icp_subscriber = None
        self.odom_subscriber = None
        self.odomc_translation = Point(x=0, y=0, z=0)
        self.odomc_quaternion = [1, 0, 0, 0]
        self.stamp = None
        self.pcl_scan = None

    def callback_odom(self, data):
        #rospy.loginfo('Received Odom')
        self.odom_data = data
        self.stamp = data.header.stamp

    def callback_icp(self, data):
        self.icp_data = data

    def subscribe_once_odom(self):
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.callback_odom, queue_size=10)

    def subscribe_once_icp(self):
        self.icp_subscriber = rospy.Subscriber('/icp/transform', Odometry, self.callback_icp, queue_size=10)

    def sync_callback(self, odom, scan):
        self.odom_data = odom
        self.pcl_scan = scan

    def run(self):
        self.subscribe_once_icp()
        self.subscribe_once_odom()

        while not rospy.is_shutdown():
            if self.odomc is not None:
                if self.odom_data is not None:
                    # apply odom rotation and translation to odomc
                    dx = self.odom_data.pose.pose.position.x - self.odom_data_prev.pose.pose.position.x
                    dy = self.odom_data.pose.pose.position.y - self.odom_data_prev.pose.pose.position.y
                    dz = self.odom_data.pose.pose.position.z - self.odom_data_prev.pose.pose.position.z
                    q1 = [self.odom_data_prev.pose.pose.orientation.w, self.odom_data_prev.pose.pose.orientation.x,
                          self.odom_data_prev.pose.pose.orientation.y, self.odom_data_prev.pose.pose.orientation.z]
                    q2 = [self.odom_data.pose.pose.orientation.w, self.odom_data.pose.pose.orientation.x,
                          self.odom_data.pose.pose.orientation.y, self.odom_data.pose.pose.orientation.z]

                    diff = self.multiply_quaternion(q2, self.inverse_quaternion(q1))
                    old = [self.odomc.pose.pose.orientation.w, self.odomc.pose.pose.orientation.x, self.odomc.pose.pose.orientation.y, self.odomc.pose.pose.orientation.z]
                    new_orientation = self.multiply_quaternion(diff, old)
                    self.odomc.pose.pose.position.x += dx
                    self.odomc.pose.pose.position.y += dy
                    self.odomc.pose.pose.position.z += dz
                    self.odomc.pose.pose.orientation.x = new_orientation[1]
                    self.odomc.pose.pose.orientation.y = new_orientation[2]
                    self.odomc.pose.pose.orientation.z = new_orientation[3]
                    self.odomc.pose.pose.orientation.w = new_orientation[0]

                    tm = rospy.Time.now()
                    self.odomc.header.stamp = self.stamp
                    self.publisher.publish(self.odomc)
                    self.odom_data_prev = self.odom_data
                    self.odom_data = None
                if self.icp_data is not None:
                    # apply icp transform
                    self.odomc.pose.pose.position.x += self.icp_data.pose.pose.position.x
                    self.odomc.pose.pose.position.y += self.icp_data.pose.pose.position.y
                    self.odomc.pose.pose.position.z += self.icp_data.pose.pose.position.z
                    # Update the stored total translation
                    self.odomc_translation[0] += self.icp_data.pose.pose.position.x
                    self.odomc_translation[1] += self.icp_data.pose.pose.position.y
                    self.odomc_translation[2] += self.icp_data.pose.pose.position.z
                    r = [self.odomc.pose.pose.orientation.w, self.odomc.pose.pose.orientation.x,
                         self.odomc.pose.pose.orientation.y, self.odomc.pose.pose.orientation.z]
                    q = [self.icp_data.pose.pose.orientation.w, self.icp_data.pose.pose.orientation.x,
                         self.icp_data.pose.pose.orientation.y, self.icp_data.pose.pose.orientation.z]

                    ret = self.multiply_quaternion(q, r)
                    self.odomc.pose.pose.orientation.w = ret[0]
                    self.odomc.pose.pose.orientation.x = ret[1]
                    self.odomc.pose.pose.orientation.y = ret[2]
                    self.odomc.pose.pose.orientation.z = ret[3]

                    # Update the stored total rotation
                    self.odomc_quaternion = self.multiply_quaternion(q, self.odomc_quaternion)
                    self.icp_data = None

                    self.odomc.header.stamp = self.stamp
                    self.publisher.publish(self.odomc)
            elif self.odom_data is not None:
                rospy.loginfo('Setting initial data')
                self.odomc = Odometry()
                self.odomc.pose.pose.position.x = self.odom_data.pose.pose.position.x
                self.odomc.pose.pose.position.y = self.odom_data.pose.pose.position.y
                self.odomc.pose.pose.position.z = self.odom_data.pose.pose.position.z
                self.odomc.pose.pose.orientation.w = self.odom_data.pose.pose.orientation.w
                self.odomc.pose.pose.orientation.x = self.odom_data.pose.pose.orientation.x
                self.odomc.pose.pose.orientation.y = self.odom_data.pose.pose.orientation.y
                self.odomc.pose.pose.orientation.z = self.odom_data.pose.pose.orientation.z
                self.odomc.header.frame_id = 'map'
                self.odomc.child_frame_id = 'odom'
                tm = rospy.Time.now()
                self.odomc.header.stamp = self.stamp
                self.publisher.publish(self.odomc)
                self.odom_data_prev = self.odom_data
                self.odom_data = None



    def multiply_quaternion(self, q, r):
        result = [0, 0, 0, 0]
        result[0] = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
        result[1] = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
        result[2] = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
        result[3] = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]
        return result

    def inverse_quaternion(self, q):
        mag = q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2
        inverse = [
            q[0] / mag,
            -1 * q[1] / mag,
            -1 * q[2] / mag,
            -1 * q[3] / mag
        ]
        return inverse

    def compute_rotation_matrix(self, q):
        return numpy.matrix([
            [q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2, 2 * (q[1] * q[2] - q[0] * q[3]),2 * (q[1] * q[3] + q[0] * q[2])],
            [2 * (q[1] * q[2] + q[0] * q[3]), q[0] ** 2 + q[2] ** 2 - q[1] ** 2 - q[3] ** 2,2 * (q[2] * q[3] - q[0] * q[1])],
            [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]),q[0] ** 2 + q[3] ** 2 - q[1] ** 2 - q[2] ** 2]
        ])

    def compute_quaternion(self, r):
        trace = r[0, 0] + r[1, 1] + r[2, 2]
        q = [1, 0, 0, 0]
        if trace > 0:
            k = 0.5 / math.sqrt(1 + trace)
            q[0] = k * (r[1, 2] - r[2, 1])
            q[1] = k * (r[2, 0] - r[0, 2])
            q[2] = k * (r[0, 1] - r[1, 0])
            q[3] = 0.25 / k
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            k = 0.5 / math.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2])
            q[0] = 0.25 / k
            q[1] = k * (r[1, 0] + r[0, 1])
            q[2] = k * (r[2, 0] + r[0, 2])
            q[3] = k * (r[1, 2] - r[2, 1])
        elif r[1, 1] > r[2, 2]:
            k = 0.5 / math.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2])
            q[0] = k * (r[1, 0] + r[0, 1])
            q[1] = 0.25 / k
            q[2] = k * (r[2, 1] + r[1, 2])
            q[3] = k * (r[2, 0] - r[0, 2])
        else:
            k = 0.5/math.sqrt(1 + r[2,2] - r[0,0] - r[1,1])
            q[0] = k * (r[2, 0] + r[0, 2])
            q[1] = k * (r[2, 1] + r[1, 2])
            q[2] = 0.25 / k
            q[3] = k * (r[0, 1] - r[1, 0])
        return [q[3], q[0], q[1], q[2]]
    def publish_tf(self):
        t = geometry_msgs.msg.TransformStamped()
        t.header.frame_id = 'map'
        t.header.stamp = rospy.Time.now()
        t.child_frame_id = 'odom'
        quaternion = self.inverse_quaternion(self.odomc_quaternion)
        translation = list((self.compute_rotation_matrix(quaternion) * numpy.matrix(
            [[self.odomc_translation[0]], [self.odomc_translation[1]], [self.odomc_translation[2]]])).flat)
        #translation = self.odomc_translation
        print 'Total', numpy.around(translation,3), ' ', numpy.around(quaternion, 3)
        '''t.transform.translation.x = -1*translation[0]
        t.transform.translation.y = -1*translation[1]
        t.transform.translation.z = -1*translation[2]


        t.transform.rotation.w = quaternion[0]
        t.transform.rotation.x = quaternion[1]
        t.transform.rotation.y = quaternion[2]
        t.transform.rotation.z = quaternion[3]'''

        rodom = self.compute_rotation_matrix([self.odom_data.pose.pose.orientation.w, self.odom_data.pose.pose.orientation.x, self.odom_data.pose.pose.orientation.y,
             self.odom_data.pose.pose.orientation.z])
        rodomc = self.compute_rotation_matrix([self.odomc.pose.pose.orientation.w, self.odomc.pose.pose.orientation.x, self.odomc.pose.pose.orientation.y,
             self.odomc.pose.pose.orientation.z])
        rt = rodomc.getT() * rodom
        v = [self.odom_data.pose.pose.position.x - self.odomc.pose.pose.position.x,
             self.odom_data.pose.pose.position.y - self.odomc.pose.pose.position.y,
             self.odom_data.pose.pose.position.z - self.odomc.pose.pose.position.z]

        tr = list((numpy.matrix(v) * rodomc).flat)
        q = self.compute_quaternion(rt)
        '''
        rt = rodom.getT() * rodomc
        v = [self.odomc.pose.pose.position.x - self.odom_data.pose.pose.position.x,
             self.odomc.pose.pose.position.y - self.odom_data.pose.pose.position.y,
             self.odomc.pose.pose.position.z - self.odom_data.pose.pose.position.z]
        tr = list((numpy.matrix(v)*rodom).flat)
        q = self.compute_quaternion(rt)
        '''
        print 'Rotal', numpy.around(tr, 3), ' ', numpy.around(q, 3)
        t.transform.translation.x = tr[0]
        t.transform.translation.y = tr[1]
        t.transform.translation.z = tr[2]
        t.transform.rotation.w = q[0]
        t.transform.rotation.x = q[1]
        t.transform.rotation.y = q[2]
        t.transform.rotation.z = q[3]
        tfm = tf.msg.tfMessage([t])
        self.tf_publisher.publish(tfm)

    def normalize(self, q):
        mag = math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
        return [q[0]/mag, q[1]/mag, q[2]/mag, q[3]/mag]
def main():
    rospy.init_node('OdometryCorrector')
    oc = OdometryCorrector()
    oc.run()


if __name__ == "__main__":
    main()
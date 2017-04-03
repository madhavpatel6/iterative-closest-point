#!/usr/bin/env python
import sys
import rospy
from nav_msgs.msg import Odometry
import math
import tf
import tf.msg
import geometry_msgs.msg
from namedlist import namedlist
Point = namedlist('Point', 'x y z')


class OdometryCorrector:
    def __init__(self):
        self.publisher = rospy.Publisher('/odomc', Odometry, queue_size=10)
        self.tf_publisher = rospy.Publisher('/tf', tf.msg.tfMessage, queue_size=10)
        self.odom_data = None
        self.odom_data_prev = None
        self.icp_data = None
        self.odomc = None
        self.icp_subscriber = None
        self.odom_subscriber = None
        self.odomc_to_odom = Point(x=0, y=0, z=0)
        self.odomc_quaternion = [1, 0, 0, 0]

    def callback_odom(self, data):
        #rospy.loginfo('Received Odom')
        self.odom_data = data

    def callback_icp(self, data):
        self.icp_data = data

    def subscribe_once_odom(self):
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.callback_odom, queue_size=10)

    def subscribe_once_icp(self):
        self.icp_subscriber = rospy.Subscriber('/icp_transform', Odometry, self.callback_icp, queue_size=10)

    def run(self):
        self.subscribe_once_icp()
        self.subscribe_once_odom()
        while not rospy.is_shutdown():
            if self.odomc is not None:
                if self.odom_data is not None:
                    # apply odom rotation and translation to odomc
                    self.odomc.pose.pose.position.x += self.odom_data.pose.pose.position.x - self.odom_data_prev.pose.pose.position.x
                    self.odomc.pose.pose.position.y += self.odom_data.pose.pose.position.y - self.odom_data_prev.pose.pose.position.y
                    self.odomc.pose.pose.position.z += self.odom_data.pose.pose.position.z - self.odom_data_prev.pose.pose.position.z
                    self.odomc.pose.pose.orientation.x += self.odom_data.pose.pose.orientation.x - self.odom_data_prev.pose.pose.orientation.x
                    self.odomc.pose.pose.orientation.y += self.odom_data.pose.pose.orientation.y - self.odom_data_prev.pose.pose.orientation.y
                    self.odomc.pose.pose.orientation.z += self.odom_data.pose.pose.orientation.z - self.odom_data_prev.pose.pose.orientation.z
                    self.odomc.pose.pose.orientation.w += self.odom_data.pose.pose.orientation.w - self.odom_data_prev.pose.pose.orientation.w
                    self.odom_data_prev = self.odom_data
                    self.odom_data = None

                    self.odomc.header.stamp = rospy.Time.now()
                    self.publisher.publish(self.odomc)
                    self.publish_tf()

                if self.icp_data is not None:
                    # apply icp transform
                    self.odomc.pose.pose.position.x += self.icp_data.pose.pose.position.x
                    self.odomc.pose.pose.position.y += self.icp_data.pose.pose.position.y
                    self.odomc.pose.pose.position.z += self.icp_data.pose.pose.position.z
                    # Update the stored total translation
                    self.odomc_to_odom[0] += self.icp_data.pose.pose.position.x
                    self.odomc_to_odom[1] += self.icp_data.pose.pose.position.y
                    self.odomc_to_odom[2] += self.icp_data.pose.pose.position.z
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

                    self.odomc.header.stamp = rospy.Time.now()
                    self.publisher.publish(self.odomc)
                    self.publish_tf()

            elif self.odom_data is not None:
                rospy.loginfo('Setting initial data')
                self.odomc = self.odom_data
                self.odomc.header.frame_id = '/odomc'
                self.odomc.child_frame_id = ''
                self.odomc.header.stamp = rospy.Time.now()
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

    def publish_tf(self):
        t = geometry_msgs.msg.TransformStamped()
        t.header.frame_id = '/odomc'
        t.header.stamp = rospy.Time.now()
        t.child_frame_id = '/odom'
        transform = tf.transformations.concatenate_matrices(tf.transformations.translation_matrix(self.odomc_to_odom), tf.transformations.quaternion_matrix(self.odomc_quaternion))
        inversed_transform = tf.transformations.inverse_matrix(transform)
        translation = tf.transformations.translation_from_matrix(inversed_transform)
        quaternion = tf.transformations.quaternion_from_matrix(inversed_transform)
        print('Org', self.odomc_to_odom)
        print('Inv', translation)
        print('Qrg', self.odomc_quaternion)
        print('Qnv', quaternion)

        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        t.transform.rotation.w = quaternion[0]
        t.transform.rotation.x = quaternion[1]
        t.transform.rotation.y = quaternion[2]
        t.transform.rotation.z = quaternion[3]


        tfm = tf.msg.tfMessage([t])
        self.tf_publisher.publish(tfm)

def main():
    rospy.init_node('odomc')
    oc = OdometryCorrector()
    oc.run()


if __name__ == "__main__":
    main()
#!/usr/bin/python
from __future__ import division
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
from geometry_msgs.msg import Quaternion
from roslib import message
warnings.filterwarnings("ignore")
import tf

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
        self.icp = IterativeClosestPoint(zero_threshold=0.1, convergence_threshold=0.0001, nearest_neighbor_upperbound=0.5)
        self.map = []
        self.translation_threshold = 0.2
        self.rotation_threshold = 2.5
        rospy.init_node('listen', anonymous=True)
        self.publisher = rospy.Publisher('/icp_map', PointCloud2, queue_size=10)
        self.transform_publisher = rospy.Publisher('/icp_transform', Odometry, queue_size=10)
        self.new_odom = None
        self.new_scan = None
        self.cloud_out_subscriber = None
        self.odom_subscriber = None
        self.frame = None
        self.icp_completed = True
        self.old_odom = None

    def laser_listen_once(self):
        self.cloud_out_subscriber = rospy.Subscriber('/cloud_out_global', PointCloud2, self.cloud_out_callback)

    def odom_listen_once(self):
        self.odom_subscriber = rospy.Subscriber('/odomc', Odometry, self.odom_callback)

    def cloud_out_callback(self, data):
        plotscan = False
        if plotscan:
            plt.ion()
            plt.clf()
        self.frame = data.header.frame_id
        self.seq = data.header.seq
        self.new_scan = self.pointcloud2_to_list(data)
        if plotscan:
            for p in self.new_scan:
                plt.scatter(p[0], p[1], s=5)
            if self.new_odom is not None:
                plt.scatter(self.new_odom.pose.pose.position.x, self.new_odom.pose.pose.position.y, s=10, color='blue')
            plt.axis('equal')
            plt.waitforbuttonpress()

    def odom_callback(self, data):
        #self.odom_subscriber.unregister()
        self.new_odom = data


    def pointcloud2_to_list(self, cloud):
        gen = PCL.read_points(cloud, skip_nans=True, field_names=('x', 'y', 'z'))
        list_of_tuples = list(gen)
        return list_of_tuples

    def list_to_pointcloud2(self, points):
        pcloud = PointCloud2()
        pcloud = PCL.create_cloud_xyz32(pcloud.header, points)
        pcloud.header.stamp = rospy.Time.now()
        pcloud.header.frame_id = '/odomc'
        return pcloud

    def publish_map(self):
        self.publisher.publish(self.list_to_pointcloud2(self.map))

    def check_movement(self):
        if self.new_odom is None:
            return False
        if self.old_odom is None:
            return True
        else:
            dx = self.new_odom.pose.pose.position.x - self.old_odom.pose.pose.position.x
            dy = self.new_odom.pose.pose.position.y - self.old_odom.pose.pose.position.y

            if (dx**2 + dy**2) >= self.translation_threshold**2:
                return True
            q = [self.new_odom.pose.pose.orientation.x, self.new_odom.pose.pose.orientation.y,
                 self.new_odom.pose.pose.orientation.z, self.new_odom.pose.pose.orientation.w]
            new_yaw = tf.transformations.euler_from_quaternion(q)[2]
            if new_yaw < 0:
                new_yaw += 2*math.pi
            q = [self.old_odom.pose.pose.orientation.x, self.old_odom.pose.pose.orientation.y,
                 self.old_odom.pose.pose.orientation.z, self.old_odom.pose.pose.orientation.w]
            old_yaw = tf.transformations.euler_from_quaternion(q)[2]
            if old_yaw < 0:
                old_yaw += 2*math.pi

            #print('yaw', new_yaw*180/3.14)
            dyaw = abs(new_yaw - old_yaw)*180/math.pi
            if dyaw > 180:
                dyaw = 360 - dyaw
            if abs(dyaw) >= self.rotation_threshold:
                return True
        return False

    def build_map(self):
        rospy.loginfo('Starting Map Building.')
        self.laser_listen_once()
        self.odom_listen_once()
        while not rospy.is_shutdown():
            if self.new_scan is not None and self.check_movement():
                print('Updated old odom')
                self.old_odom = self.new_odom
                if len(self.map) != 0:
                    m = self.map
                    #pos = self.new_odom.pose.pose.position
                    #m = get_subset_of_points(map=self.map, odom=[pos.x, pos.y, pos.z], radius=20)
                    new_scan_transformed, total_t, total_q, array_t, array_q= self.icp.iterative_closest_point(reference=m, source=self.new_scan, initial=[1, 0, 0, 0, 0, 0, 0],
                                                     number_of_iterations=10)
                    self.map.extend(new_scan_transformed)
                    '''transform = Odometry()
                    transform.header.frame_id = '/odomc'
                    transform.pose.pose.position.x = total_t[0]
                    transform.pose.pose.position.y = total_t[1]
                    transform.pose.pose.position.z = total_t[2]
                    transform.pose.pose.orientation.x = total_q[1]
                    transform.pose.pose.orientation.y = total_q[2]
                    transform.pose.pose.orientation.z = total_q[3]
                    transform.pose.pose.orientation.w = total_q[0]'''
                    rospy.loginfo('Total Q: ' + str(numpy.around(total_q, 3)))
                    for i in range(len(total_q)):
                        transform = Odometry()
                        transform.header.frame_id = '/odomc'
                        transform.pose.pose.position.x = array_t[i][0]
                        transform.pose.pose.position.y = array_t[i][1]
                        transform.pose.pose.position.z = array_t[i][2]
                        transform.pose.pose.orientation.x = array_q[i][1]
                        transform.pose.pose.orientation.y = array_q[i][2]
                        transform.pose.pose.orientation.z = array_q[i][3]
                        transform.pose.pose.orientation.w = array_q[i][0]
                        self.transform_publisher.publish(transform)
                else:
                    for i in range(len(self.new_scan)):
                        self.new_scan[i] = (self.new_scan[i][0], self.new_scan[i][1], 0)
                    self.map = self.new_scan

                self.new_scan = None
                self.new_odom = None
                self.publish_map()
                #self.laser_listen_once()
                #self.odom_listen_once()
            #if oldtime + 5 < time.time():
            #    rospy.loginfo('Publishing the map.')
            #    self.publish_map()
            #    oldtime = time.time()


class IterativeClosestPoint:
    def __init__(self, zero_threshold, convergence_threshold, nearest_neighbor_upperbound):
        self.zero_threshold = zero_threshold
        self.convergence_threshold = convergence_threshold
        self.total_distances = []
        self.nearest_neighbor_upperbound = nearest_neighbor_upperbound

    def iterative_closest_point(self, reference, source, initial, number_of_iterations):
        reference_n, source_n = reference, source

        # Perform initial translation
        q = initial[0:4]
        rotation_matrix = self.compute_rotation_matrix(q)
        translation_vector = numpy.matrix(initial[4:7]).getT()
        total_t, total_q = translation_vector, q
        array_t, array_q = [list(translation_vector.flat)], [q]
        for i in range(len(source_n)):
            source_n[i] = list((rotation_matrix * numpy.matrix(source_n[i]).getT() + translation_vector).flat)
            source_n[i][2] = 0
        #rospy.loginfo('Starting ICP')
        for itr in range(number_of_iterations):
            #reference_n, source_n, dist,  not_matched_reference, not_matched_source, multi_matched_index = self.nearest_neighbor_munkres(reference_n, source_n)
            reference_n, source_n, dist, not_matched_reference, not_matched_source, multi_matched_index = self.nearest_neighbor_kdtree(reference_n, source_n)
            self.total_distances.append(dist)

            reference_mean = numpy.matrix(numpy.mean(reference_n, axis=0)).getT()
            source_mean = numpy.matrix(numpy.mean(source_n, axis=0)).getT()

            try:
                covariance_matrix = [[0 for i in range(len(reference_n[0]))] for j in range(len(reference_n[0]))]
            except:
                plot_points_2d(reference, source)
                plt.plot()

            for itr in range(len(reference_n)):
                source_std = numpy.matrix(source_n[itr]).getT() - source_mean
                reference_std = numpy.matrix(reference_n[itr]) - reference_mean.getT()
                covariance_matrix += (source_std * reference_std)

            covariance_matrix *= (1 / len(reference_n))
            anti_symmetric_matrix = covariance_matrix - covariance_matrix.getT()

            column_vector = numpy.matrix(
                [anti_symmetric_matrix[1, 2], anti_symmetric_matrix[2, 0], anti_symmetric_matrix[0, 1]]).getT()

            covariance_entry = covariance_matrix + covariance_matrix.getT() - numpy.trace(
                covariance_matrix) * numpy.identity(3)

            symmetric_matrix = numpy.matrix([
                [numpy.trace(covariance_matrix), column_vector[0, 0], column_vector[1, 0], column_vector[2, 0]],
                [column_vector[0, 0], covariance_entry[0, 0], covariance_entry[0, 1], covariance_entry[0, 2]],
                [column_vector[1, 0], covariance_entry[1, 0], covariance_entry[1, 1], covariance_entry[1, 2]],
                [column_vector[2, 0], covariance_entry[2, 0], covariance_entry[2, 1], covariance_entry[2, 2]]
            ])

            e, v = numpy.linalg.eig(symmetric_matrix)

            max_eigenvalue_index = numpy.argmax(e)

            q = list(numpy.matrix(v[:, max_eigenvalue_index]).flat)

            rotation_matrix = self.compute_rotation_matrix(q)
            #print('q = ', q)
            translation_vector = reference_mean - rotation_matrix * source_mean


            total_t = numpy.add(total_t, translation_vector)
            #rospy.loginfo('T: ' + str(numpy.around(total_t, decimals=3)))
            total_q = self.multiply_quaternion(q, total_q)
            array_t.append(list(translation_vector.flat))
            array_q.append(q)

            # Add in the not matched points
            for i in sorted(multi_matched_index, reverse=True):
                del reference_n[i]
            for p in not_matched_reference:
                reference_n.append(p)
            for p in not_matched_source:
                source_n.append(p)
            for i in range(len(source_n)):
                source_n[i] = list((rotation_matrix * numpy.matrix(source_n[i]).getT() + translation_vector).flat)

            # Check convergence tolerance
            if dist < self.zero_threshold:
                break
            elif len(self.total_distances) >= 3:
                percent_difference1 = math.fabs(
                    (self.total_distances[-1] - self.total_distances[-2]) / self.total_distances[-1])
                percent_difference2 = math.fabs(
                    (self.total_distances[-1] - self.total_distances[-3]) / self.total_distances[-1])
                percent_difference3 = math.fabs(
                    (self.total_distances[-2] - self.total_distances[-3]) / self.total_distances[-2])
                if percent_difference1 < self.convergence_threshold and percent_difference2 < self.convergence_threshold and percent_difference3 < self.convergence_threshold:
                    break
        del self.total_distances[:]
        return source_n, total_t, total_q, array_t, array_q

    def compute_rotation_matrix(self, q):
        return numpy.matrix([
            [q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2, 2 * (q[1] * q[2] - q[0] * q[3]),
             2 * (q[1] * q[3] + q[0] * q[2])],
            [2 * (q[1] * q[2] + q[0] * q[3]), q[0] ** 2 + q[2] ** 2 - q[1] ** 2 - q[3] ** 2,
             2 * (q[2] * q[3] - q[0] * q[1])],
            [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]),
             q[0] ** 2 + q[3] ** 2 - q[1] ** 2 - q[2] ** 2]
        ])

    '''
    performs multiplication q*r
    Where r is the first rotation applied
    and q is the second rotation applied
    '''
    def multiply_quaternion(self, q, r):
        result = [0, 0, 0, 0]
        result[0] = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
        result[1] = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
        result[2] = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
        result[3] = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]
        return result

    def nearest_neighbor_munkres(self, a, b):
        #print('Length: ', len(a), len(b))
        rospy.loginfo('Matrix Dimensions: %d by %d', len(a), len(b))
        matrix = [[self.compute_euclidean_distance_2d(a[i], b[j]) for j in range(len(b))] for i in range(len(a))]

#        for i in range(len(a)):
#            for j in range(len(b)):
#                matrix[i][j] = self.compute_euclidean_distance_2d(a[i], b[j])):

        m = Munkres()
        start = time.time()
        indexes = m.compute(matrix)
        end = time.time()
        rospy.loginfo('Munkres computation time: %0.3f', end - start)
        total = 0
        new_a = []
        new_b = []
        # Create a set of indices, remove from the corresponding set when munkres has found a nearest neighbor
        # In the end the set will contain the index of any unmatched points
        not_matched_a_index = set(range(len(a)))
        not_matched_b_index = set(range(len(b)))

        for row, column in indexes:
            value = matrix[row][column]
            total += value
            print('(%d, %d)   -> %0.3f' % (row, column, value))
            new_a.append(a[row])
            new_b.append(b[column])

            not_matched_a_index.remove(row)
            not_matched_b_index.remove(column)
            #plt.annotate(s='', xy=new_a[-1][0:2], xytext=new_b[-1][0:2], arrowprops=dict(arrowstyle='<->'))
            plt.waitforbuttonpress()
            #print('total cost: %0.3f' % total)

        not_matched_a = []
        not_matched_b = []
        for itr in not_matched_a_index:
            not_matched_a.append(a[itr])
        for itr in not_matched_b_index:
            not_matched_b.append(b[itr])
        #print('Not matched A', not_matched_a)
        #print('Not matched B', not_matched_b)

        #rospy.loginfo("Time elapsed: " + str(end - start))
        #Since munkres guarntees that it is matching without replacement we do not have to worry about multimatched indices
        multi_matched_index = set()
        return new_a, new_b, total, not_matched_a, not_matched_b, multi_matched_index

    def nearest_neighbor_kdtree(self, a, b):
        kdtree = scipy.spatial.cKDTree(a, leafsize=100)
        cost = 0
        new_a = []
        new_b = []
        not_matched_a_index = set(range(len(a)))
        not_matched_b_index = set(range(len(b)))
        multi_matched_index = set()
        for i in range(len(b)):
            new_cost, index = kdtree.query(b[i], k=1, distance_upper_bound=self.nearest_neighbor_upperbound)
            # query will give a cost of infinite and an index = len(a) if no matches found within distance upper bound
            if new_cost == float('inf') and index == len(a):
                continue
            cost += new_cost
            new_a.append(a[index])
            new_b.append(b[i])
            if index in not_matched_a_index:
                not_matched_a_index.remove(index)
            else:
                multi_matched_index.add(len(new_a) - 1)
            not_matched_b_index.remove(i)

        not_matched_a = []
        not_matched_b = []
        for itr in not_matched_a_index:
            not_matched_a.append(a[itr])
        for itr in not_matched_b_index:
            not_matched_b.append(b[itr])

        return new_a, new_b, cost, not_matched_a, not_matched_b, multi_matched_index

    def compute_euclidean_distance_3d(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

    def compute_euclidean_distance_2d(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def plot_total_error(self):
        plt.figure(2)
        l = len(self.total_distances)
        if len(self.total_distances) > 1:
            plt.plot([l, l - 1], [self.total_distances[l - 1], self.total_distances[l - 2]], color='b')
        plt.plot(math.floor(l), self.total_distances[l - 1], 'bo-', marker='o', color='b')
        plt.pause(0.01)


def test_icp():
    #print('Iteration Cloest Point')
    inst = IterativeClosestPoint(zero_threshold=0.1, convergence_threshold=0.00000000001, nearest_neighbor_upperbound=float('inf'))
    p = read_coordinate_file('l_shape_3d.txt')

    x = read_coordinate_file('l_shape_3d.txt')
    x = rotate_2d((0, 0), x, 10)
    x = translate_2d(x, 0, 0)
    plot_points_2d(p, x)
    new_scan, total_t, total_q, array_t, array_q = inst.iterative_closest_point(p, x, [1, 0, 0, 0, 0, 0, 0], 100)
    plt.figure(2)
    plot_points_2d(p, new_scan)
    input()



def read_coordinate_file(filename):
    points = []
    try:
        f = open(str(filename), 'r')
    except:
        sys.exit('Check the file name.')
    for line in f:
        if len(line.split()) == 3:
            x, y, z = float(line.split()[0]), float(line.split()[1]), float(line.split()[2])
            points.append([x, y, z])
        elif len(line.split()) == 2:
            x, y = float(line.split()[0]), float(line.split()[1])
            points.append([x, y])
        else:
            sys.exit('File format is incorrect.')
    #random.shuffle(points)
    return points


def create_random_3d(length):
    ret = []
    for i in range(length):
        ret.append((random.uniform(0, 5), random.uniform(0, 5)))
    return ret


def translate_2d(points, x, y):
    for i in range(len(points)):
        points[i][0] += x
        points[i][1] += y
    return points


def rotate_2d(origin, points, angle):
    ox, oy = origin
    rad_angle = angle*math.pi/180.0

    for i in range(len(points)):
        qx = ox + math.cos(rad_angle) * (points[i][0] - ox) - math.sin(rad_angle) * (points[i][1] - oy)
        qy = oy + math.sin(rad_angle) * (points[i][0] - ox) + math.cos(rad_angle) * (points[i][1] - oy)
        points[i] = [qx, qy, points[i][2]]
    return points


def plot_points_2d(a, b):
    plt.ion()
    plt.clf()
    a_t = copy.deepcopy(a)
    b_t = copy.deepcopy(b)
    if len(a_t[0]) == 3:
        for row in a_t:
            del row[2]
        for row in b_t:
            del row[2]
    plt.plot(*zip(*a_t), marker='o', color='r', ls='')
    plt.plot(*zip(*b_t), marker='x', color='b', ls='')
    red = mpatches.Patch(color='red', label='Model Set')
    blue = mpatches.Patch(color='blue', label='Measured Set')
    plt.legend(handles=[red, blue])



if __name__ == '__main__':
    main()

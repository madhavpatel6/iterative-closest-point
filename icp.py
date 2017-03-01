#!/usr/bin/python3
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



warnings.filterwarnings("ignore")


def main():
    test_icp()

def test_icp():
    print('Iteration Cloest Point')
    plt.ion()
    inst = IterativeClosestPoint(0.1, 0.01)
    p = read_coordinate_file('l_shape_3d.txt')

    x = read_coordinate_file('l_shape_3d.txt')
    x = rotate_2d((0, 0), x, 75)
    x = translate_2d(x, 5, 5)
    del x[0]
    inst.iterative_closest_point(p, x, [1, 0, 0, 0, 0, 0, 0], 10)

class IterativeClosestPoint:
    def __init__(self, zero_threshold, convergence_threshold):
        self.zero_threshold = zero_threshold
        self.convergence_threshold = convergence_threshold
        self.total_distances = []
        self.debug_print = True

    def iterative_closest_point(self, reference, source, initial, number_of_iterations):
        if self.debug_print is False:
            sys.stdout = open(os.devnull, 'w')
        reference_n, source_n = reference, source
        # Perform initial translation
        q = initial[0:4]
        rotation_matrix = self.compute_rotation_matrix(q)
        translation_vector = numpy.matrix(initial[4:7]).getT()
        print('Performing initial transformation')
        print('R =\n', rotation_matrix)
        print('T =\n', translation_vector)
        plt.figure(2)
        plt.ylabel('Total Distance')
        plt.xlabel('Iteration')
        plt.grid()
        plt.figure(1)
        plot_points_2d(reference_n, source_n)
        plt.pause(.5)
        for i in range(len(source_n)):
            source_n[i] = list((rotation_matrix * numpy.matrix(source_n[i]).getT() + translation_vector).flat)

        for itr in range(number_of_iterations):
            plt.figure(1)
            plt.clf()
            print('--------------------------------- Iteration # %d ---------------------------------' % (itr + 1))
            reference_n, source_n, dist,  not_matched_reference, not_matched_source = self.nearest_neighbor_munkres(reference_n, source_n)
            self.total_distances.append(dist)
            print(reference_n)
            print(source_n)

            reference_mean = numpy.matrix(numpy.mean(reference_n, axis=0)).getT()
            source_mean = numpy.matrix(numpy.mean(source_n, axis=0)).getT()

            print('Reference Center of Mass:\n', reference_mean)
            print('Source Center of Mass   :\n', source_mean)

            covariance_matrix = [[0 for i in range(len(reference_n[0]))] for j in range(len(reference_n[0]))]

            for itr in range(len(reference_n)):
                source_std = numpy.matrix(source_n[itr]).getT() - source_mean
                reference_std = numpy.matrix(reference_n[itr]) - reference_mean.getT()
                covariance_matrix += (source_std * reference_std)

            covariance_matrix *= (1 / len(reference_n))
            print('Covariance Matrix:\n', covariance_matrix)

            anti_symmetric_matrix = covariance_matrix - covariance_matrix.getT()
            print('Anti-symmetric matrix:\n', anti_symmetric_matrix)

            column_vector = numpy.matrix([anti_symmetric_matrix[1, 2], anti_symmetric_matrix[2, 0], anti_symmetric_matrix[0, 1]]).getT()
            print('Column vector:\n', column_vector)

            covariance_entry = covariance_matrix + covariance_matrix.getT() - numpy.trace(covariance_matrix) * numpy.identity(3)
            print('covariance entry\n', covariance_entry)

            symmetric_matrix = numpy.matrix([
                [numpy.trace(covariance_matrix), column_vector[0, 0], column_vector[1, 0], column_vector[2, 0]],
                [column_vector[0, 0], covariance_entry[0, 0], covariance_entry[0, 1], covariance_entry[0, 2]],
                [column_vector[1, 0], covariance_entry[1, 0], covariance_entry[1, 1], covariance_entry[1, 2]],
                [column_vector[2, 0], covariance_entry[2, 0], covariance_entry[2, 1], covariance_entry[2, 2]]
            ])
            print('symmetric matrix\n', symmetric_matrix)

            e, v = numpy.linalg.eig(symmetric_matrix)
            print('eigen values\n', e)
            print('eigen vectors\n', v)

            max_eigenvalue_index = numpy.argmax(e)
            print('max eigen value index', max_eigenvalue_index)

            q = list(numpy.matrix(v[:, max_eigenvalue_index]).flat)
            print('max eigen vector\n', q)

            rotation_matrix = self.compute_rotation_matrix(q)
            print('rotation matrix\n', rotation_matrix)

            translation_vector = reference_mean - rotation_matrix * source_mean
            print('translation vector\n', translation_vector)
            # Add in the not matched points
            for p in not_matched_reference:
                reference_n.append(p)
            for p in not_matched_source:
                source_n.append(p)
            plot_points_2d(reference_n, source_n)
            for i in range(len(source_n)):
                source_n[i] = list((rotation_matrix*numpy.matrix(source_n[i]).getT() + translation_vector).flat)

            self.plot_total_error()
            # Check convergence tolerance
            if len(self.total_distances) >= 3:
                l = len(self.total_distances)
                percent_difference1 = math.fabs((self.total_distances[l - 1] - self.total_distances[l - 2])/self.total_distances[l - 1])
                percent_difference2 = math.fabs((self.total_distances[l - 1] - self.total_distances[l - 3])/self.total_distances[l - 1])
                percent_difference3 = math.fabs((self.total_distances[l - 2] - self.total_distances[l - 3])/self.total_distances[l - 2])
                if percent_difference1 < self.convergence_threshold and percent_difference2 < self.convergence_threshold and percent_difference3 < self.convergence_threshold:
                    print('ICP converged to a total distance', dist)
                    break
            if dist < self.zero_threshold:
                print('ICP total distance', dist, 'is below zero threshold')
                break
            plt.figure(1)
            plt.waitforbuttonpress()

        plt.figure(1)
        plt.waitforbuttonpress()
        del self.total_distances[:]
        if self.debug_print is False:
            sys.stdout = sys.__stdout__

    def compute_rotation_matrix(self, q):
        return numpy.matrix([
            [q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2, 2 * (q[1] * q[2] - q[0] * q[3]),
             2 * (q[1] * q[3] + q[0] * q[2])],
            [2 * (q[1] * q[2] + q[0] * q[3]), q[0] ** 2 + q[2] ** 2 - q[1] ** 2 - q[3] ** 2,
             2 * (q[2] * q[3] - q[0] * q[1])],
            [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]),
             q[0] ** 2 + q[3] ** 2 - q[1] ** 2 - q[2] ** 2]
        ])

    def nearest_neighbor_munkres(self, a, b):
        matrix = [[0 for i in range(len(b))] for j in range(len(a))]

        for i in range(len(a)):
            for j in range(len(b)):
                distance = self.compute_euclidean_distance_2d(a[i], b[j])
                matrix[i][j] = distance

        print('First Point Set :', a)
        print('Second Point Set:', b)
        print(numpy.matrix(matrix))
        m = Munkres()
        indexes = m.compute(matrix)
        total = 0
        new_a = []
        new_b = []
        plt.figure(1)
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
            print('total cost: %0.3f' % total)
            plt.annotate(s='', xy=new_a[-1][0:2], xytext=new_b[-1][0:2], arrowprops=dict(arrowstyle='<->'))

        not_matched_a = []
        not_matched_b = []
        for itr in not_matched_a_index:
            not_matched_a.append(a[itr])
        for itr in not_matched_b_index:
            not_matched_b.append(b[itr])
        print('Not matched A', not_matched_a)
        print('Not matched B', not_matched_b)
        return new_a, new_b, total, not_matched_a, not_matched_b

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
    random.shuffle(points)
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

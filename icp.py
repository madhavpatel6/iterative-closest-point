import sys
import numpy
import math
from munkres import Munkres
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import copy

warnings.filterwarnings("ignore")


def main():
    print('Iteration Cloest Point')
    plt.ion()
    p = read_coordinate_file('l_shape_3d.txt')

    x = read_coordinate_file('l_shape_3d.txt')
    x = rotate_2d((0,0), x, 45)
    x = translate_2d(x, 5, 5)
    iterative_closest_point(p, x, 0, 10)

    #p_n, x_n = nearest_neighbor_munkres(p, x)


def iterative_closest_point(reference, source, initial, number_of_iterations):
    itr = 0
    reference_n, source_n = reference, source
    for itr in range(number_of_iterations):
        print('--------------------------------- Iteration # %d ---------------------------------' % (itr + 1))
        reference_n, source_n, dist = nearest_neighbor_munkres(reference_n, source_n)
        if dist < 0.1:
            break
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

        rotation_matrix = numpy.matrix([
            [q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2,    2*(q[1]*q[2] - q[0]*q[3]),                2*(q[1]*q[3] + q[0]*q[2])],
            [2*(q[1]*q[2] + q[0]*q[3]),                q[0]**2 + q[2]**2 - q[1]**2 - q[3]**2,    2*(q[2]*q[3] - q[0]*q[1])],
            [2*(q[1]*q[3] - q[0]*q[2]),                2*(q[2]*q[3] + q[0]*q[1]),                q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2]
        ])
        print('rotation matrix\n', rotation_matrix)

        translation_vector = reference_mean - rotation_matrix * source_mean
        print('translation vector\n', translation_vector)

        for i in range(len(source_n)):
            source_n[i] = list((rotation_matrix*(numpy.matrix(source_n[i]).getT() + translation_vector)).flat)
        print(reference_n)
        print(source_n)


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


def nearest_neighbor_munkres(a, b):
    if len(a) != len(b):
        length = max(len(a), len(b))
    else:
        length = len(a)
    matrix = [[0 for i in range(length)] for j in range(length)]

    for i in range(length):
        for j in range(length):
            if i < len(a) and j < len(b):
                distance = compute_euclidean_distance_2d(a[i], b[j])
            else:
                distance = 0
            matrix[i][j] = distance
    print('First Point Set :', a)
    print('Second Point Set:', b)

    m = Munkres()
    indexes = m.compute(matrix)
    total = 0
    new_b = [0 for i in range(len(b))]
    a_t, b_t = plot_points_2d(a, b)
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print('(%d, %d)   -> %0.3f' % (row, column, value))
        new_b[row] = b[column]
        print('total cost: %0.3f' % total)
        plt.annotate(s='', xy=a_t[row], xytext=b_t[column], arrowprops=dict(arrowstyle='<->'))
    plt.pause(.005)
    plt.waitforbuttonpress()
    return a, new_b, total


def plot_points_2d(a, b):
    plt.clf()
    a_t = copy.deepcopy(a)
    b_t = copy.deepcopy(b)
    for row in a_t:
        del row[2]
    for row in b_t:
        del row[2]
    plt.plot(*zip(*a_t), marker='o', color='r', ls='')
    plt.plot(*zip(*b_t), marker='x', color='b', ls='')
    red = mpatches.Patch(color='red', label='Model Set')
    blue = mpatches.Patch(color='blue', label='Measured Set')
    plt.legend(handles=[red, blue])

    return a_t, b_t


def compute_euclidean_distance_3d(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)


def compute_euclidean_distance_2d(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

if __name__ == '__main__':
    main()

import getopt
import random
import sys
import math

from tsplib95 import distances
import numpy as np


def matrix(coord, distance):
    n = len(coord)
    distance_matrix = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            distance_matrix[i][j] = int(distance((coord[i][0], coord[i][1]), (coord[j][0], coord[j][1])))
            distance_matrix[i][j] = int(distance((coord[i][0], coord[i][1]), (coord[j][0], coord[j][1])))
    return n, distance_matrix


def symetric_random_instance(number_of_cities, min_distance, max_distance):
    np.random.seed(2021)
    rand_matrix = np.random.randint(min_distance, max_distance + 1, size=(number_of_cities, number_of_cities))
    for i in range(number_of_cities):
        rand_matrix[i][i] = 0
        for j in range(number_of_cities):
            rand_matrix[j][i] = rand_matrix[i][j]
    return number_of_cities, rand_matrix


def asymetric_random_instance(number_of_cities, min_distance, max_distance):
    np.random.seed(2021)
    rand_matrix = np.random.randint(min_distance, max_distance + 1, size=(number_of_cities, number_of_cities))
    for i in range(number_of_cities):
        rand_matrix[i][i] = 0
    return number_of_cities, rand_matrix


def read_file(filename):
    f = open('instance/' + filename, 'r')

    name = f.readline().strip().split()[1]
    filetype = f.readline().strip().split()[1]
    comment = f.readline().strip().split()[1]
    dimension = f.readline().strip().split()[1]

    line = f.readline()

    while line.find("EDGE_WEIGHT_TYPE") == -1:
        line = f.readline()

    if line.find("EUC_2D") != -1:
        distance = distances.euclidean
    elif line.find("MAN_2D") != -1:
        distance = distances.manhattan
    elif line.find("MAX_2D") != -1:
        distance = distances.maximum
    elif line.find("GEO") != -1:
        distance = distances.geographical
    elif line.find("CEIL_2D") != -1:
        distance = distances.functools.partial(distances.euclidean, round=math.ceil)
    elif line.find("EXPLICIT") != -1:
        line = f.readline()
    
    else:
        raise Exception

    while line.find("NODE_COORD_SECTION") == -1:
        line = f.readline()

    n = int(dimension)
    coord = []
    for i in range(0, n):
        x, y = f.readline().strip().split()[1:]
        coord.append([float(x), float(y)])

    number_of_cities, distance_matrix = matrix(coord, distance)
    return number_of_cities, distance_matrix


def rand_tour(n):
    tour = list(range(n))
    random.shuffle(tour)
    return tour


def swap_positions(tour, i, j):
    tour[i], tour[j] = tour[j], tour[i]
    return tour


def get_weight(tour, distance_matrix):
    weight = distance_matrix[tour[0]][tour[-1]]
    for i in range(1, len(tour)):
        weight += distance_matrix[tour[i]][tour[i - 1]]
    return weight


def nearest(last, unvisited, distance_matrix):
    near = unvisited[0]
    min_distance = distance_matrix[last][near]
    for i in unvisited[1:]:
        if distance_matrix[last][i] < min_distance:
            near = i
            min_distance = distance_matrix[last][near]
    return near


def nearest_neighbor(n, i, distance_matrix):
    unvisited = list(range(n))
    unvisited.remove(i)
    last = i
    tour = [i]
    while unvisited:
        nexxt = nearest(last, unvisited, distance_matrix)
        tour.append(nexxt)
        unvisited.remove(nexxt)
        last = nexxt
    return tour


def nearest_swap_neighbor(tour, distance_matrix):
    length = len(tour)
    min_weight = get_weight(tour, distance_matrix)
    best_tour = tour.copy()
    current_tour = tour.copy()
    for i in range(int(length / 2 + 1)):
        for j in range(length):
            current_tour = tour.copy()
            current_tour = swap_positions(current_tour, i, j)
            current_weight = get_weight(current_tour, distance_matrix)
            if current_weight < min_weight:
                best_tour = current_tour
                min_weight = current_weight
    return best_tour


def krandom(n, k, distance_matrix):
    results = [None] * k
    random_tour = rand_tour(n)
    min_weight = get_weight(random_tour, distance_matrix)
    for i in range(k):
        new_tour = rand_tour(n)
        new_weight = get_weight(new_tour, distance_matrix)
        if new_weight < min_weight:
            min_weight = new_weight
            random_tour = new_tour
        results[i] = min_weight
    return results


def invert(tour, i, j):
    length_sub_list = j - i
    current_tour = tour.copy()
    for k in range(int(length_sub_list / 2 + 1)):
        current_tour = swap_positions(current_tour, i, j - k)
        i += 1
    return current_tour


def opt(n, distance_matrix):
    best_solution = [None] * n
    current_solution = [None] * n
    results = []
    for i in range(n):
        solution = nearest_neighbor(n, i, distance_matrix)
        best_solution[i] = solution
    min_weight = get_weight(best_solution, distance_matrix)
    improved = True
    while improved:
        i = 0
        while i < n - 1:
            j = i + 1
            while j < n:
                current_solution = best_solution.copy()
                current_solution = invert(current_solution, i, j)
                current_distance = get_weight(current_solution, distance_matrix)
                if current_distance.any() < min_weight.any():
                    best_solution = current_solution
                    min_weight = current_distance
                    i = 0
                    j = 0
                results.append(min_weight)
                j += 1
            i += 1
        improved = False
    return results


def prd1(solution, best_known):
    return 100 * (solution - best_known) / best_known


def prd2(tour, best_known, distance_matrix):
    return 100 * (get_weight(tour, distance_matrix) - best_known) / best_known


def print_tour(tour):
    length = len(tour)
    for i in range(length - 1):
        print(tour[i], '->', tour[i + 1], '\n')


def main(param):
    try:
        opt, args = getopt.getopt(param, "", ['option=', 'type=', ])
    except getopt.GetoptError as err:
        sys.exit(2)
    try:
        if opt[0][1] == 'load':
            if opt[1][1] == 'symetric':
                sys.stdout.write("Enter name of file: ")
                filename = input().strip()
                n, distance_matrix = read_file(filename)
                sys.stdout.write("1. Calculate k-random\n")
                sys.stdout.write("2. Calculate nearest neighbor\n")
                sys.stdout.write("3. Calculate weight of tour\n")
                sys.stdout.write("4. Print tour\n")
                option = int(input().strip())
                if option == 1:
                    sys.stdout.write("Enter numer of iterations:\n")
                    k = int(input().strip())
                    results = krandom(n, k, distance_matrix)
                    results.sort()
                    print(results)
                elif option == 2:
                    sys.stdout.write("Enter number of first city:\n")
                    i = int(input().strip())
                    tour = nearest_neighbor(n, i, distance_matrix)
                    print(tour)
                    weight = get_weight(tour, distance_matrix)
                    print(weight)
                elif option == 3:
                    tour = rand_tour(n)
                    weight = get_weight(tour, distance_matrix)
                    print(weight)
                elif option == 4:
                    tour = rand_tour(n)
                    print_tour(tour)

            elif opt[1][1] == 'asymetric':
                sys.stdout.write("Enter name of file:")
                filename = input().strip()
                n, distance_matrix = read_file(filename)
                sys.stdout.write("1. Calculate k-random\n")
                sys.stdout.write("2. Calculate nearest neighbor\n")
                sys.stdout.write("3. Calculate weight of tour\n")
                sys.stdout.write("4. Print tour\n")
                option = int(input().strip())
                if option == 1:
                    sys.stdout.write("Enter numer of iterations:\n")
                    k = int(input().strip())
                    results = krandom(n, k, distance_matrix)
                    results.sort()
                    print(results)
                elif option == 2:
                    sys.stdout.write("Enter number of first city:\n")
                    i = int(input().strip())
                    tour = nearest_neighbor(n, i, distance_matrix)
                    print(tour)
                    weight = get_weight(tour, distance_matrix)
                    print(weight)
                elif option == 3:
                    tour = rand_tour(n)
                    weight = get_weight(tour, distance_matrix)
                    print(weight)
                elif option == 4:
                    tour = rand_tour(n)
                    print_tour(tour)

            elif opt[1][1] == 'randsymetric':
                sys.stdout.write("Enter the number of cities:\n")
                number_of_cities = int(input().strip())
                sys.stdout.write("Enter min distance:\n")
                min_distance = int(input().strip())
                sys.stdout.write("Enter max distance:\n")
                max_distance = int(input().strip())
                n, random_matrix = symetric_random_instance(number_of_cities, min_distance, max_distance)
                sys.stdout.write("1. Calculate k-random\n")
                sys.stdout.write("2. Calculate nearest neighbor\n")
                sys.stdout.write("3. Calculate weight of tour\n")
                sys.stdout.write("4. Print tour\n")
                option = int(input().strip())
                if option == 1:
                    sys.stdout.write("Enter numer of iterations:\n")
                    k = int(input().strip())
                    results = krandom(n, k, random_matrix)
                    results.sort()
                    print(results)
                elif option == 2:
                    sys.stdout.write("Enter number of first city:\n")
                    i = int(input().strip())
                    tour = nearest_neighbor(n, i, random_matrix)
                    print(tour)
                    weight = get_weight(tour, random_matrix)
                    print(weight)
                elif option == 3:
                    tour = rand_tour(n)
                    weight = get_weight(tour, random_matrix)
                    print(weight)
                elif option == 4:
                    tour = rand_tour(n)
                    print_tour(tour)

            elif opt[1][1] == 'randasymetric':
                sys.stdout.write("Enter the number of cities:")
                number_of_cities = int(input().strip())
                sys.stdout.write("Enter min distance:")
                min_distance = int(input().strip())
                sys.stdout.write("Enter max distance:")
                max_distance = int(input().strip())
                random_instance = asymetric_random_instance(number_of_cities, min_distance, max_distance)
                sys.stdout.write("1. Calculate k-random\n")
                sys.stdout.write("2. Calculate nearest neighbor\n")
                sys.stdout.write("3. Calculate weight of tour\n")
                sys.stdout.write("4. Print tour\n")
                option = int(input().strip())
                if option == 1:
                    sys.stdout.write("Enter numer of iterations:\n")
                    k = int(input().strip())
                    results = krandom(n, k, random_matrix)
                    results.sort()
                    print(results)
                elif option == 2:
                    sys.stdout.write("Enter number of first city:\n")
                    i = int(input().strip())
                    tour = nearest_neighbor(n, i, random_matrix)
                    print(tour)
                    weight = get_weight(tour, random_matrix)
                    print(weight)
                elif option == 3:
                    tour = rand_tour(n)
                    weight = get_weight(tour, random_matrix)
                    print(weight)
                elif option == 4:
                    tour = rand_tour(n)
                    print_tour(tour)

    except IndexError as err:
        print(err)
        sys.exit(2)


if __name__ == '__main__':

    main(sys.argv[1:])

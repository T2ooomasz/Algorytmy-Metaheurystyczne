import math
import random
import tsplib95
from tsplib95 import distances
import numpy as np
import matplotlib.pyplot as plt
import time

def matrix(coord, distance):
    n = len(coord)
    distance_matrix = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            distance_matrix[i][j] = int(distance((coord[i][0], coord[i][1]), (coord[j][0], coord[j][1])))
    return n, distance_matrix

def symetric_random_instance(number_of_cities, min_distance, max_distance, seed):
    np.random.seed(seed)
    rand_matrix = np.random.randint(min_distance, max_distance + 1, size=(number_of_cities, number_of_cities))
    for i in range(number_of_cities):
        rand_matrix[i][i] = 0
        for j in range(number_of_cities):
            rand_matrix[j][i] = rand_matrix[i][j]
    return number_of_cities, rand_matrix

def asymetric_random_instance(number_of_cities, min_distance, max_distance, seed):
    np.random.seed(seed)
    rand_matrix = np.random.randint(min_distance, max_distance + 1, size=(number_of_cities, number_of_cities))
    for i in range(number_of_cities):
        rand_matrix[i][i] = 0
    return number_of_cities, rand_matrix

def euclidean_random_instance(number_of_cities, max_x, max_y, seed):
    distance = distances.euclidean
    np.random.seed(seed)
    coord = np.array([(np.random.randint(0, max_x, dtype=int), np.random.randint(0, max_y, dtype=int)) for _ in range(number_of_cities)])
    distance_matrix = np.zeros((number_of_cities, number_of_cities))
    for i in range(0, number_of_cities):
        for j in range(0, number_of_cities):
            distance_matrix[i][j] = int(distance((coord[i][0], coord[i][1]), (coord[j][0], coord[j][1])))
    return number_of_cities, distance_matrix, coord

def explicit_instance(filename, type):
    if type == 'tsp':
        instance = tsplib95.load('instance/' +filename)
    elif type =='atsp':
        instance = tsplib95.load('instance/' +filename) 
    edges = list(instance.get_edges())
    nodes = len(list(instance.get_nodes()))
    length = len(edges)
    weight = [None]*length
    matrix = np.zeros((nodes, nodes))
    for i in range(0, length):
        weight[i] = instance.get_weight(edges[i][0], edges[i][1])
        matrix[edges[i][0]-1][edges[i][1]-1] = weight[i]

    return nodes, matrix

def euclidean_instance(filename):
    instance = tsplib95.load('instance/' +filename)
    nodes =  len(list(instance.get_nodes()))
    edges =  list(instance.get_edges())
    length = len(edges)
    weight = [None]*length
    matrix = np.zeros((nodes, nodes))
    for i in range(0, length):
        weight[i] = instance.get_weight(edges[i][0], edges[i][1])
        matrix[edges[i][0]-1][edges[i][1]-1] = weight[i]

    return nodes, matrix

def read_file(filename, type):

    if type == 'tsp':
        f = open('instance/' +filename, 'r')
    elif type == 'atsp':
        f = open('instance/' +filename, 'r')

    name = f.readline().strip().split()[1]
    filetype = f.readline().strip().split()[1]
    comment = f.readline().strip().split()[1]
    dimension = f.readline().strip().split()[1] 

    line = f.readline()

    while line.find("EDGE_WEIGHT_TYPE") == -1:
        line = f.readline()

    if line.find("EUC_2D") != -1:
        number_of_cities, distance_matrix = euclidean_instance(filename)
    elif line.find("MAN_2D") != -1:
        distance = distances.manhattan
    elif line.find("MAX_2D") != -1:
        distance = distances.maximum
    elif line.find("GEO") != -1:
        distance = distances.geographical
    elif line.find("CEIL_2D") != -1:
        distance = distances.functools.partial(distances.euclidean, round=math.ceil)
    elif line.find("EXPLICIT") != -1:
        number_of_cities, distance_matrix = explicit_instance(filename, type)
    else:
        raise Exception
    
    #while line.find("NODE_COORD_SECTION") == -1:
    #    line = f.readline()

    #n = int(dimension)
    #coord = []
    #for i in range(0, n):
    #    x, y = f.readline().strip().split()[1:]
    #    coord.append([float(x), float(y)])

    #number_of_cities, distance_matrix = matrix(coord, distance)
    #number_of_cities, distance_matrix
    return number_of_cities, distance_matrix


def rand_tour(n):
    tour = list(range(n))
    random.shuffle(tour)
    return tour

#algorytm k-rando
def krandom(n, k, distance_matrix):
    start_time = time.time()
    results = [None] * k
    best_tour = rand_tour(n)
    min_weight = get_weight(best_tour, distance_matrix)
    for i in range(k):
        new_tour = rand_tour(n)
        new_weight = get_weight(new_tour, distance_matrix)
        if new_weight < min_weight:
            min_weight = new_weight
            best_tour = new_tour
        results[i] = min_weight

    end_time = time.time()
    t = (end_time - start_time)
    return min_weight, t

def average_krandom(number_of_cities, k, distance_matrix):
    min_weight = 0
    TIME = 0
    repeats = 5 # how many times to get average from
    for i in range(repeats):
        min_weight_new, TIMEnew = krandom(number_of_cities, k, distance_matrix)
        min_weight += min_weight_new
        TIME = np.add(TIME, TIMEnew) 
    min_weight /= repeats
    TIME /= repeats
    return min_weight, TIME

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

#algorytm nearest neighbour
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
    return get_weight(tour)

#algorytm nearest neighbour extended
def nearest_neighbor_extended(n, distance_matrix):
    best_tour = nearest_neighbor(n, 0, distance_matrix)
    for i in range(1,n):
        current_tour = nearest_neighbor(n, i, distance_matrix)
        if get_weight(current_tour, distance_matrix) < get_weight(best_tour, distance_matrix):
            best_tour = current_tour
    return best_tour

#algorytm nearest swal neighbour
def nearest_swap_neighbor(number_of_cities, distance_matrix):
    tour = nearest_neighbor(number_of_cities, 0, distance_matrix)
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

def invert(tour, i, j):
    length_sub_list = j - i
    current_tour = tour.copy()
    for k in range(int(length_sub_list / 2 + 1)):
        current_tour = swap_positions(current_tour, i, j - k)
        i += 1
    return current_tour

#algorytm 2opt
def opt2(n, distance_matrix):
    best_solution = [None] * n
    current_solution = [None] * n
    solution = nearest_neighbor(n, 0, distance_matrix)
    best_solution = solution
    min_weight = get_weight(best_solution, distance_matrix)
    i = 0
    while i < n - 1:
        j = i + 1
        while j < n:
            current_solution = best_solution.copy()
            current_solution = invert(current_solution, i, j)
            current_distance = get_weight(current_solution, distance_matrix)
            if current_distance < min_weight:
                best_solution = current_solution
                min_weight = current_distance
                i = 0
                j = 0
            j += 1
        i += 1
    return best_solution

#algorytm 2opt pełny
def opt2_full(n, distance_matrix):
    best_solution = [None] * n
    current_solution = [None] * n
    solution = nearest_neighbor(n, 0, distance_matrix)
    best_solution = solution
    min_weight = get_weight(best_solution, distance_matrix)
    i = 0
    improved = True
    potential_best_solution = best_solution
    potential_min_weight = min_weight
    while improved:
        while i < n - 1:
            j = i + 1
            while j < n:
                current_solution = best_solution.copy()
                current_solution = invert(current_solution, i, j)
                current_distance = get_weight(current_solution, distance_matrix)
                if current_distance < potential_min_weight:
                    potential_best_solution = current_solution
                    potential_min_weight = current_distance

                j += 1
            i += 1
        if potential_best_solution >= best_solution:
            
            improved = False
        else:
            best_solution = potential_best_solution
            min_weight = potential_min_weight
            
    return best_solution


def prd1(solution, best_known):
    return 100 * ((solution - best_known) / best_known)

def prd2(tour, best_known, distance_matrix):
    return 100 * ((get_weight(tour, distance_matrix) - best_known) / best_known)

def print_tour(tour):
    length = len(tour)
    for i in range(length - 1):
        print(tour[i], '->', tour[i + 1], '\n')

def print_data(best_tour, best_known, distance_matrix):
    print("\n----------------------------------------------------------\n")
    # cykl
    print("Cykl: ", best_tour)
    # dł ścieżki
    print("Dlugosc sciezki: ", get_weight(best_tour, distance_matrix))
    # PRD

def scatter_cities(cities_list):
    X, Y = cities_list.T
    plt.scatter(X, Y, marker='x')
    plt.show()

'''most important function - simulation'''
def simulations_alg_tsp(filename, k, X_k, TIME_k, X_nn, TIME_nn, X_nne, TIME_nne, X_nsn, TIME_nsn, X_2opt, TIME_2opt, X_2optf, TIME_2optf):
    number_of_cities, distance_matrix = read_file(filename, type='tsp')
    # k-random
    min_weight, t = average_krandom(number_of_cities, k, distance_matrix)
    X_k.append(min_weight)
    TIME_k.append(t)

    # NN
    best_tour = nearest_neighbor(number_of_cities, 0, distance_matrix)

    # NNE
    #print("Nearest Neighbour Extended:")
    best_tour = nearest_neighbor_extended(number_of_cities, distance_matrix)
    #print_data(best_tour, best_known, distance_matrix)
    #print("\n####################################################\n")
    # NN swap
    #print("Nearest SWAP Neighbour")
    best_tour = nearest_swap_neighbor(number_of_cities, distance_matrix)
    #print_data(best_tour, best_known, distance_matrix)
    #print("\n######################################################\n")
    # 2-opt
    #print("2-opt:")
    best_tour = opt2(number_of_cities, distance_matrix)
    #print_data(best_tour, best_known, distance_matrix)
    #print("\n######################################################\n")
    # eventualy other

#symulacja k-random
def simulation_krandom():
    pass
#symulacja nearest neighour

#symulacja nearest neighbour extended

#symulacja nearest swap neighbour

#symulacja 2opt

#symulacja 2opt-full

# krandom
def save_data_k_random_atsp(X, index):
    string = "simulation-k-random/atsp-k-random"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time_k_random_atsp(X, index):
    string = "simulation-k-random/atsp-k-random-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_k_random_tsp(X, index):
    string = "simulation-k-random/tsp-k-random"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time_k_random_tsp(X, index):
    string = "simulation-k-random/tsp-k-random-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

# nearest neighbour
def save_data_nearest_neighbour_atsp(X, index):
    string = "simulation-nearest_neighbour/atsp-nearest_neighbour"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time_nearest_neighbour_atsp(X, index):
    string = "simulation-nearest_neighbour/atsp-nearest_neighbour-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_nearest_neighbour_tsp(X, index):
    string = "simulation-nearest_neighbour/tsp-nearest_neighbour"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time_nearest_neighbour_tsp(X, index):
    string = "simulation-nearest_neighbour/tsp-nearest_neighbour-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

#nearest neighbour extended
def save_data_nearest_neighbour_extended_atsp(X, index):
    string = "simulation-nearest_neighbour_extended/atsp-nearest_neighbour_extended"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time_nearest_neighbour_extended_atsp(X, index):
    string = "simulation-nearest_neighbour_extended/atsp-nearest_neighbour_extended-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_nearest_neighbour_extended_tsp(X, index):
    string = "simulation-nearest_neighbour_extended/tsp-nearest_neighbour_extended"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time_nearest_neighbour_extended_tsp(X, index):
    string = "simulation-nearest_neighbour_extended/tsp-nearest_neighbour_extended-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

# nearest swap neighbour
def save_data_nearest_swap_neighbour_atsp(X, index):
    string = "simulation-nearest_swap_neighbour/atsp-nearest_swap_neighbour"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time_nearest_swap_neighbour_atsp(X, index):
    string = "simulation-nearest_swap_neighbour/atspnearest_swap_neighbour-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_nearest_swap_neighbour_tsp(X, index):
    string = "simulation-nearest_swap_neighbour/tsp-nearest_swap_neighbour"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time_nearest_swap_neighbour_tsp(X, index):
    string = "simulation-nearest_swap_neighbour/tsp-nearest_swap_neighbour-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

# 2opt
def save_data_2opt_atsp(X, index):
    string = "simulation-2-opt/atsp-2-opt"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time_2opt_atsp(X, index):
    string = "simulation-2-opt/atsp-2-opt-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_2opt_tsp(X, index):
    string = "simulation-2-opt/tsp-2-opt"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time_2opt_tsp(X, index):
    string = "simulation-2-opt/tsp-2-opt-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

# 2opt full
def save_data_2_opt_full_atsp(X, index):
    string = "simulation-2_opt_full/atsp-2_opt_full"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time_2_opt_full_atsp(X, index):
    string = "simulation-2_opt_full/atsp-2_opt_full-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_2_opt_full_tsp(X, index):
    string = "simulation-2_opt_full/tsp-2_opt_full"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time_2_opt_full_tsp(X, index):
    string = "simulation-2_opt_full/tsp-2_opt_full-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")


def main():
    print("main")
    k = 20000 # for krandom algorithm
    X_k = []
    TIME_k = []
    X_nn = []
    TIME_nn = []
    X_nne = []
    TIME_nne = []
    X_nsn = []
    TIME_nsn = []
    X_2opt = []
    TIME_2opt = []
    X_2optf = []
    TIME_2optf = []
    instance_list_tsp = ['berlin52.tsp', 'ch130.tsp', 'gr120.tsp']
    for x in instance_list_tsp:
        filename = x
        #number_of_cities, distance_matrix = read_file(filename, type='tsp')
        simulations_alg_tsp(filename, k, X_k, TIME_k, X_nn, TIME_nn, X_nne, TIME_nne, X_nsn, TIME_nsn, X_2opt, TIME_2opt, X_2optf, TIME_2optf)

    #save datas
    save_data_k_random_tsp(X_k, 0)
    save_data_time_k_random_tsp(TIME_k, 0)

if __name__ == '__main__':
    main()
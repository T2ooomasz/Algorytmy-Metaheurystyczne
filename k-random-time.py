import math
from tsplib95 import distances
import numpy as np
import random
import matplotlib.pyplot as plt
import time


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


def get_weight(cities_list, Distance_Matrix):
    sum = 0
    n = np.shape(Distance_Matrix)[0]
    for i in range(n-2):
        sum = sum + Distance_Matrix[cities_list[i]][cities_list[i+1]]
    # back to the start city
    sum = sum + Distance_Matrix[n-1][0]
    return sum

def save_figure(X, index):
    plt.plot(X, '-b')
    plt.xlabel('iteration')
    plt.ylabel('best solution distance')
    plt.title('Distance to iteration')
    plt.grid(True)
    string = "obrazki/k-random"
    string += str(index)
    string += ".png"
    plt.savefig(string)

def save_data(X, index):
    string = "data-k-random/k-random"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time(X, index):
    string = "data-k-random/k-random-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def k_random(N, k, Distance_Matrix):
    X =  np.array([])
    TIME = np.array([])
    #initial step
    trace = random.sample(range(N),N)
    min_weight  = get_weight(trace, Distance_Matrix)# = funkcja liczaca wage

    start_time = time.time()
    # Wielka pentla powtarzajaca sie k-razy
    for i in range(k):
        new_trace = random.sample(range(N),N)
        new_weight = get_weight(new_trace, Distance_Matrix)# = funkcja liczaca wage
        if (new_weight < min_weight):
            min_weight = new_weight
            trace = new_trace
            #number_of_better_solution += 
        end_time = time.time()
        t = (end_time - start_time)
        X = np.append(X, np.array([min_weight]))
        TIME = np.append(TIME, np.array([t]))

    # return best_solutin [iteration] for given N and k
    return X, TIME

def average_for_given(repeats, N, k, rang, Distance_Matrix):
    X = np.zeros((k), dtype=np.dtype(int))
    TIME = np.zeros((k), dtype=np.dtype(float))
    for i in range(repeats):
        Xnew, TIMEnew = k_random(N, k, Distance_Matrix)
        X = np.add(X, Xnew)
        TIME = np.add(TIME, TIMEnew) 
    X = X / repeats
    TIME = TIME / repeats
    return X, TIME
    
def simulation(repeats, N, k, Distance_Matrix):
    # Distance_Matrix = set_Matrix()
    rang = 1
    for i in range(rang):
        X, TIME = average_for_given(repeats, N, k, rang, Distance_Matrix)
        save_figure(X, k)
        save_data(X, k)
        save_data_time(TIME, k)
        k *= 10

def main():
    # N = 10
    k = 100000
    repeats = 1000
    max_distance = 500
    # Distance_Matrix = symetric_random_instance(N, 1, max_distance, 2100) # distances from 1 to 100
    Distance_Matrix = set_Matrix()
    N = np.shape(Distance_Matrix)[0]
    
    simulation(repeats, N, k, Distance_Matrix)
    
if __name__ == '__main__':
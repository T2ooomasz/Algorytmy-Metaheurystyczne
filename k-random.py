import math
from tsplib95 import distances
import numpy as np
import random
import matplotlib.pyplot as plt
import time

def get_weight(cities_list, Distance_Matrix):
    sum = 0
    n = np.shape(Distance_Matrix)[0]
    for i in range(n-2):
        sum = sum + Distance_Matrix[cities_list[i]][cities_list[i+1]]
    # back to the start city
    sum = sum + Distance_Matrix[n-1][0]
    return sum

def asymetric_random_instance(number_of_cities, min_distance, max_distance):
    np.random.seed(2021)
    rand_matrix =np.random.randint(min_distance, max_distance + 1, size=(number_of_cities,number_of_cities))
    # np.random.random_integers(min_distance, max_distance, size=(number_of_cities,number_of_cities))
    for i in range(number_of_cities):
        rand_matrix[i][i] = 0
    return rand_matrix

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
    string = "data-k-random/asym-k-random"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time(X, index):
    string = "data-k-random/asym-k-random-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")


def k_random(N, k, Distance_Matrix):
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

    # return best_solutin [iteration] for given N and k
    return min_weight, t

def average_for_given(repeats, N, k, Distance_Matrix):
    min_weight = 0
    TIME = 0
    for i in range(repeats):
        min_weight_new, TIMEnew = k_random(N, k, Distance_Matrix)
        min_weight += min_weight_new
        TIME = np.add(TIME, TIMEnew) 
    min_weight /= repeats
    TIME /= repeats
    return min_weight, TIME
    

def simulation(nodi, repeats, k):
    # Distance_Matrix = set_Matrix()
    number_of_cities = 640
    X = []
    TIME = []
    Y = []
    for i in range(nodi):
        Distance_Matrix = asymetric_random_instance(number_of_cities, 1, 100)
        X_new, TIME_new = average_for_given(repeats, number_of_cities, k, Distance_Matrix)
        X.append(X_new)
        TIME.append(TIME_new)
        Y.append(number_of_cities)
        number_of_cities *= 2

    save_data(X, 1)
    save_data(Y, 2)
    save_data_time(TIME, 1)

def main():
    k = 100000
    repeats = 10
    nodi = 1 #nodi = number of diverent instances

    simulation(nodi, repeats, k)
    
if __name__ == '__main__':
    main()
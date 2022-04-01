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

def symetric_random_instance(number_of_cities, min_distance, max_distance):
    np.random.seed(2021)
    rand_matrix =np.random.randint(min_distance, max_distance + 1, size=(number_of_cities,number_of_cities))
    # np.random.random_integers(min_distance, max_distance, size=(number_of_cities,number_of_cities))
    for i in range(number_of_cities):
        rand_matrix[i][i] = 0
        for j in range(number_of_cities):
            rand_matrix[j][i] = rand_matrix[i][j]
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

def save_data_k_random_asym(X, index):
    string = "data-k-random/asym-k-random"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time_k_random_asym(X, index):
    string = "data-k-random/asym-k-random-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_k_random_sym(X, index):
    string = "data-k-random/sym-k-random"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time_k_random_sym(X, index):
    string = "data-k-random/sym-k-random-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_2opt_asym(X, index):
    string = "data-2-opt/asym-2-opt"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time_2opt_asym(X, index):
    string = "data-2-opt/asym-2-opt-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_2opt_sym(X, index):
    string = "data-2-opt/sym-2-opt"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time_2opt_sym(X, index):
    string = "data-2-opt/sym-2-opt-time"
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
    

def simulation_k_random_asym(nodi, repeats, k):
    number_of_cities = 10
    X = []
    TIME = []
    for i in range(nodi):
        Distance_Matrix = asymetric_random_instance(number_of_cities, 100, 200)
        X_new, TIME_new = average_for_given(repeats, number_of_cities, k, Distance_Matrix)
        X.append(X_new)
        TIME.append(TIME_new)
        number_of_cities *= 2

    save_data_k_random_asym(X, 320)
    save_data_time_k_random_asym(TIME, 320)

def simulation_k_random_sym(nodi, repeats, k):
    number_of_cities = 10
    X = []
    TIME = []
    for i in range(nodi):
        Distance_Matrix = symetric_random_instance(number_of_cities, 100, 200)
        X_new, TIME_new = average_for_given(repeats, number_of_cities, k, Distance_Matrix)
        X.append(X_new)
        TIME.append(TIME_new)
        number_of_cities *= 2

    save_data_k_random_sym(X, 320)
    save_data_time_k_random_sym(TIME, 320)

def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

def nearest_neighbour(aval_city, city_index, Distance_Matrix):
    n = len(aval_city)
    min = Distance_Matrix[city_index][0]
    min_ind = aval_city[0]
    for i in range(1, n):
        next_city = aval_city[i]
        if(Distance_Matrix[city_index][next_city] < min):
                min = Distance_Matrix[city_index][next_city]
                min_ind = aval_city[i]
    aval_city.remove(min_ind)
    return min_ind

def invert(city_list, i, j):
    leng = len(city_list)
    leng_sub_list = j - i 
    current_trace = city_list.copy()
    for k in range(int(leng_sub_list/2 + 1)):
        current_trace = swapPositions(current_trace, i, j - k)
        #print(leng_sub_list)
        i += 1
        # print(current_trace)

    return current_trace


def simulation_2opt_asym(N, Distance_Matrix):
    best_solution = [None]*N
    current_solution = [None]*N
    X = []
    aval_city = list(range(0,N))
    # trace = random.sample(range(N),N)
    for i in range(N):
        go_to = nearest_neighbour(aval_city, i, Distance_Matrix)
        best_solution[i] = go_to
    min_weight  = get_weight(best_solution, Distance_Matrix)# = funkcja liczaca wage

    improved = True
    start_time = time.time()
    while(improved):
        i = 0
        while i < N-1:
            j = i + 1
            while j < N:
                current_solution = best_solution.copy()
                current_solution = invert(current_solution, i, j)
                current_distance = get_weight(current_solution, Distance_Matrix)
                if(current_distance < min_weight):
                    best_solution = current_solution
                    min_weight = current_distance
                    i = 0
                    j = 0
                
                j += 1
            i += 1

        improved = False
    end_time = time.time()
    t = end_time - start_time
    return min_weight, t

def simulation_2opt_sym(N, Distance_Matrix):
    best_solution = [None]*N
    current_solution = [None]*N
    X = []
    aval_city = list(range(0,N))
    # trace = random.sample(range(N),N)
    for i in range(N):
        go_to = nearest_neighbour(aval_city, i, Distance_Matrix)
        best_solution[i] = go_to
    min_weight  = get_weight(best_solution, Distance_Matrix)# = funkcja liczaca wage

    improved = True
    start_time = time.time()
    while(improved):
        i = 0
        while i < N-1:
            j = i + 1
            while j < N:
                current_solution = best_solution.copy()
                current_solution = invert(current_solution, i, j)
                current_distance = get_weight(current_solution, Distance_Matrix)
                if(current_distance < min_weight):
                    best_solution = current_solution
                    min_weight = current_distance
                    i = 0
                    j = 0
                
                j += 1
            i += 1

        improved = False
    end_time = time.time()
    t = end_time - start_time
    return min_weight, t

def simulation_2_opt_asym(nodi):
    number_of_cities = 10
    X = []
    TIME = []
    for i in range(nodi):
        Distance_Matrix = symetric_random_instance(number_of_cities, 100, 200)
        X_new, TIME_new = simulation_2opt_asym(number_of_cities, Distance_Matrix)
        X.append(X_new)
        TIME.append(TIME_new)
        number_of_cities *= 2

    save_data_2opt_asym(X, 320)
    save_data_time_2opt_asym(TIME, 320)

def simulation_2_opt_sym(nodi):
    number_of_cities = 10
    X = []
    TIME = []
    for i in range(nodi):
        Distance_Matrix = symetric_random_instance(number_of_cities, 100, 200)
        X_new, TIME_new = simulation_2opt_sym(number_of_cities, Distance_Matrix)
        X.append(X_new)
        TIME.append(TIME_new)
        number_of_cities *= 2

    save_data_2opt_sym(X, 320)
    save_data_time_2opt_sym(TIME, 320)

def main():
    k = 100
    repeats = 5
    nodi = 6 #nodi = number of diverent instances

    simulation_k_random_asym(nodi, repeats, k)
    simulation_k_random_sym(nodi, repeats, k)
    simulation_2_opt_asym(nodi)
    simulation_2_opt_sym(nodi)


if __name__ == '__main__':
    main()
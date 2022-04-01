import math
from tsplib95 import distances
import numpy as np
import random
import matplotlib.pyplot as plt
import time

def set_Matrix():
    # Open input file
    infile = open('berlin52.tsp', 'r')

    # Read instance header
    Name = infile.readline().strip().split()[1] # NAME
    FileType = infile.readline().strip().split()[1] # TYPE
    Comment = infile.readline().strip().split()[1] # COMMENT
    Dimension = infile.readline().strip().split()[1] # DIMENSION
    EdgeWeightType = infile.readline().strip().split()[1] # EDGE_WEIGHT_TYPE
    infile.readline()

    # Read node list
    N = int(Dimension)
    nodelist = []
    distances = np.zeros((N, N))
    for i in range(0, N):
        x,y = infile.readline().strip().split()[1:]
        nodelist.append([float(x), float(y)])

    for i in range(0,N):
        for j in range(0, N):
            # xd = nodelist[i][i]
            distances[i][j] = int(math.sqrt((nodelist[i][0] - nodelist[j][0])**2 + (nodelist[i][1] - nodelist[j][1])**2))
            distances[j][i] = distances[i][j]

    # print(nodelist)
    # Close input file
    infile.close()
    return distances

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
    string = "data-2-opt/2-opt"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time(X, index):
    string = "data-2-opt/2-opt-time"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

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

def simulation(N, Distance_Matrix, best_solution, current_solution, min_weight, X):
    improved = True
    t =[]
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
                
                X.append(min_weight)
                j += 1
                end_time = time.time()
                t.append(end_time - start_time)
            i += 1

        improved = False
    return best_solution, X, t

def average_simulation(N, Distance_Matrix, best_solution, current_solution, min_weight, X):
    N = np.shape(Distance_Matrix)[0]
    best_solution = [None]*N
    current_solution = [None]*N
    aval_city = list(range(0,N))
    X = []
    BEST = []
    for i in range(N):
        go_to = nearest_neighbour(aval_city, i, Distance_Matrix)
        best_solution[i] = go_to
    best_distance = get_weight(best_solution, Distance_Matrix)
    # SYMULACJA DLA PIERWSZEJ INSTANCJI POCZĄTKOWEJ
    BEST_new, AVERAGE, TIME = simulation(N, Distance_Matrix, best_solution, current_solution, min_weight, X)
    BEST.append(BEST_new)
    X.append(best_distance)
    print(best_distance)

    for i in range(1, N):
        aval_city = list(range(0,N))
        for j in range(N):
            go_to = nearest_neighbour(aval_city, i, Distance_Matrix)
            current_solution[j] = go_to
        current_distance = get_weight(current_solution, Distance_Matrix)
        if(current_distance < best_distance):
            best_solution = current_solution
            best_distance = current_distance
        X.append(current_distance)
        # SYMULACJA DLA KAZDEJ KOLEJNEJ INSTANCJI POCZATKOWEJ

def main():
    # N = 10
    repeats = 1000
    max_distance = 500
    # Distance_Matrix = symetric_random_instance(N, 1, max_distance, 2100) # distances from 1 to 100
    Distance_Matrix = set_Matrix()
    N = np.shape(Distance_Matrix)[0]

    best_solution = [None]*N
    current_solution = [None]*N
    X = []
    aval_city = list(range(0,N))
    # trace = random.sample(range(N),N)
    for i in range(N):
        go_to = nearest_neighbour(aval_city, i, Distance_Matrix)
        best_solution[i] = go_to
    min_weight  = get_weight(best_solution, Distance_Matrix)# = funkcja liczaca wage
    
    simulation(N, Distance_Matrix, best_solution, current_solution, min_weight, X)
    
if __name__ == '__main__':
    main()
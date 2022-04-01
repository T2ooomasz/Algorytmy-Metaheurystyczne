import math
# from tsplib95 import distances
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
    string = "data-2-opt/asym-2-opt"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def save_data_time(X, index):
    string = "data-2-opt/asym-2-opt-time"
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

def simulation(N, Distance_Matrix, best_solution, current_solution, min_weight):
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
    t =[]
    t.append(end_time - start_time)
    save_data(min_weight, 0)
    save_data_time(t, 0)

def main():
    N = 320
    min_distance = 100
    max_distance = 200
    Distance_Matrix = asymetric_random_instance(N, min_distance, max_distance)

    best_solution = [None]*N
    current_solution = [None]*N

    aval_city = list(range(0,N))
    # trace = random.sample(range(N),N)
    for i in range(N):
        go_to = nearest_neighbour(aval_city, i, Distance_Matrix)
        best_solution[i] = go_to
    min_weight  = get_weight(best_solution, Distance_Matrix)# = funkcja liczaca wage
    
    simulation(N, Distance_Matrix, best_solution, current_solution, min_weight)
    
if __name__ == '__main__':
    main()
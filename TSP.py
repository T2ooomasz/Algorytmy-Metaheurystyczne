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

def take_figure(X, index):
    plt.plot(X, '-b')
    plt.xlabel('iteration')
    plt.ylabel('best solution distance')
    plt.title('Distance to iteration')
    plt.grid(True)
    string = "obrazki/figure"
    string += str(index)
    string += ".png"
    plt.savefig(string)

def take_data(X, index):
    string = "data/set"
    string += str(index)
    string += ".csv"
    np.savetxt(string, X, delimiter=",")

def k_random(N, k, Distance_Matrix):
    X = [None]*k

    #initial step
    trace = random.sample(range(N),N)
    min_weight_sym  = get_weight(trace, Distance_Matrix)# = funkcja liczaca wage

    # Wielka pentla powtarzajaca sie k-razy
    for i in range(k):
        new_trace = random.sample(range(N),N)
        new_weight = get_weight(new_trace, Distance_Matrix)# = funkcja liczaca wage
        if (new_weight < min_weight_sym):
            min_weight_sym = new_weight
            trace = new_trace
            #number_of_better_solution += 1
        X[i] = min_weight_sym

    # return best_solutin [iteration] for given N and k
    return X

def average_for_given(repeats, N, k, rang, Distance_Matrix):
    X = np.zeros((k), dtype=np.dtype(int))
    for i in range(repeats):
        Xnew = k_random(N, k, Distance_Matrix)
        X = np.add(X, Xnew)    
    X = X / repeats
    return X
    
def simulation(repeats, N, k, Distance_Matrix):
    # Distance_Matrix = set_Matrix()
    rang = 1
    for i in range(rang):
        X = average_for_given(repeats, N, k, rang, Distance_Matrix)
        take_figure(X, i)
        take_data(X, i)
        k *= 10

def main():
    # N = 10
    k = 10000
    repeats = 100
    max_distance = 500
    # Distance_Matrix = symetric_random_instance(N, 1, max_distance, 2100) # distances from 1 to 100
    Distance_Matrix = set_Matrix()
    N = np.shape(Distance_Matrix)[0]
    start_time = time.time()
    simulation(repeats, N, k, Distance_Matrix)
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))

if __name__ == '__main__':
    main()
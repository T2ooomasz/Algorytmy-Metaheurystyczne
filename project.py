import math
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
nodelist = []
distances = []
N = int(Dimension)
for i in range(0, N):
    x,y = infile.readline().strip().split()[1:]
    nodelist.append([float(x), float(y)])
'''   
for i in range(0,N):
    for j in range(0, N):
        xd = nodelist[i][i]
        distances[i][j] = distances[j][i] = int(math.sqrt())
'''
print(nodelist[0][0])
# Close input file
infile.close()


import numpy as np
number_of_cities = 100

def symetric_random_instance(number_of_cities):
    rand_matrix = np.random.random_integers(1, 100, size=10)
    for i in range(Dimension(rand_matrix)):
        rand_matrix[i][i] = 0


def print_matrix(Distance_Matrix):
    print(Distance_Matrix)


def print_solution(solution):
    for i in range(solution):
        print(solution[i], ' -> ', solution[i+1], '\n')


def get_weight(cities_list, Distance_Matrix):
    sum = 0
    n = Dimension(Distance_Matrix)
    for i in range(n-1):
        sum += Distance_Matrix[cities_list(i)][cities_list(i+1)]
    # back to the start city
    sum =+ Distance_Matrix[n][0]
    return 0


import random 

N = 120 # number of cities
k = 1000000 # number of sumples

#initial step
trace = random.sample(range(N),N)
min_weight # = funkcja liczaca wage
min_permutation = trace
# Wielka pentla powtarzajaca sie k-razy
for i in (k):
    new_trace = random.sample(range(N),N)
    new_weight # = funkcja liczaca wage
    if (new_weight < min_weight):
        min_weight = new_weight
        trace = new_trace

print(min_weight)
print(trace)


def nearest_neighbour(city_index, Distance_Matrix):
    n = Dimension(Distance_Matrix)
    if(city_index != 0):
        min = Distance_Matrix[city_index][0]
        min_ind = 0
    else:
        min = Distance_Matrix[city_index][1]
        min_ind = 1
    for i in range(n):
        if(city_index != i):
            if(Distance_Matrix[city_index][i] < min>):
                min = Distance_Matrix[city_index][i]
    return min_ind
import math
import random
import tsplib95
from tsplib95 import distances
import numpy as np
import matplotlib.pyplot as plt

'''
Initialize TSP problem with set arguments
return nesesary variables for further calculation
'''
def initialize_problem():
    number_of_cities = 7
    min_distance = 10
    max_distance = 100
    seed = 0
    np.random.seed(seed)
    rand_matrix = np.random.randint(min_distance, max_distance + 1, size=(number_of_cities, number_of_cities))
    for i in range(number_of_cities):
        rand_matrix[i][i] = 0
        for j in range(number_of_cities):
            rand_matrix[j][i] = rand_matrix[i][j]
    return number_of_cities, rand_matrix

'''
Initialize random solution for TSP problem
return order of visited citis
'''
def random_solution(n):
    tour = list(range(n))
    random.shuffle(tour)
    return tour


def get_weight(tour, distance_matrix):
    weight = distance_matrix[tour[0]][tour[-1]]
    for i in range(1, len(tour)):
        weight += distance_matrix[tour[i]][tour[i - 1]]
    return weight

def swap_positions(tour, i, j):
    tour[i], tour[j] = tour[j], tour[i]
    return tour

def invert(tour, i, j):
    length_sub_list = j - i
    current_tour = tour.copy()
    for k in range(int(length_sub_list / 2 + 1)):
        current_tour = swap_positions(current_tour, i, j - k)
        i += 1
    return current_tour

'''
Generate Neighborhood - set of all neighbours 
return set of neighbours with made move to get specific neigbour
'''
# something loke this but better!
def get_neighbors(best_candidate, number_of_cities, distance_matrix):
    Neighborhood = []
    i = 0
    while i < number_of_cities:
        j = i+1
        while j < number_of_cities:
            best_candidate_copy = best_candidate.copy()
            best_candidate_copy = invert(best_candidate_copy, i, j)
            Neighborhood.append([best_candidate_copy,[i,j], [get_weight(best_candidate_copy, distance_matrix)]])
            j += 1
        i += 1
    Neighborhood_np = np.array(Neighborhood, dtype=object)
    return Neighborhood_np

'''
Using Neighborhood and TabuList choose best neighbour
return best neighbour
'''
def choose_best(Neighborhood, TabuList):
    a = Neighborhood.shape[0]
    best_candidate = Neighborhood[0][2]
    index_best = 0
    for i in range(1, a):
        if((Neighborhood[i][2] < best_candidate) and (Neighborhood[i][1] not in TabuList)):
            best_candidate = Neighborhood[i][2]
            index_best = i
    return Neighborhood[index_best]

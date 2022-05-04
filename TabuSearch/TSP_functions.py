import math
import random
import tsplib95
from tsplib95 import distances
import numpy as np
import matplotlib.pyplot as plt

import TSP_initialize as init

'''
Initialize TSP problem with set arguments
return nesesary variables for further calculation
'''
def initialize_problem():
    x, y = init.initialization()
    return x, y

def initialize_problem2():
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


'''
Initialize nearest neighbour solution for TSP problem
return order of visited citis
'''

def nearest_neighbor_extended_solution(n, distance_matrix):
    best_tour = nearest_neighbor_solution(n, 0, distance_matrix)
    for i in range(1,n):
        current_tour = nearest_neighbor_solution(n, i, distance_matrix)
        if get_weight(current_tour, distance_matrix) < get_weight(best_tour, distance_matrix):
            best_tour = current_tour
    return best_tour

def nearest_neighbor_solution(n, i, distance_matrix):
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

def nearest(last, unvisited, distance_matrix):
    near = unvisited[0]
    min_distance = distance_matrix[last][near]
    for i in unvisited[1:]:
        if distance_matrix[last][i] < min_distance:
            near = i
            min_distance = distance_matrix[last][near]
    return near


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
Generate Neighborhood type INVERT - set of all neighbours 
return set of neighbours with made move to get specific neigbour
'''
def get_neighbors_invert(best_candidate, number_of_cities, distance_matrix):
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
Generate Neighborhood type SWAP - set of all neighbours 
return set of neighbours with made move to get specific neigbour

'''
def get_neighbors_swap(best_candidate, number_of_cities, distance_matrix):
    pass

'''
Generate Neighborhood type INSERT - set of all neighbours 
return set of neighbours with made move to get specific neigbour

'''
def get_neighbors_insert(best_candidate, number_of_cities, distance_matrix):
    pass

'''
Generate Neighborhood type INVERT - set of all neighbours 
return set of neighbours with made move to get specific neigbour
'''
def get_neighbors_invert_N(best_candidate, number_of_cities, distance_matrix, number_of_neighbours):
    Neighborhood = []
    i = 0
    # tablica - lista dostepnych ruchow [0,1], [0,2] ... [k,n] ... [n-1,n]
    while i < number_of_neighbours:
        # wybierz losowe i,j
        # wykonaj ruch
        # dodaj do sąsiedztwa
        # usuń [i,j] [j,i] z listy dostepnych ruchow
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
Generate Neighborhood type SWAP - set of all neighbours 
return set of neighbours with made move to get specific neigbour

'''
def get_neighbors_swap_N(best_candidate, number_of_cities, distance_matrix, number_of_neighbours):
    pass

'''
Generate Neighborhood type INSERT - set of all neighbours 
return set of neighbours with made move to get specific neigbour

'''
def get_neighbors_insert_N(best_candidate, number_of_cities, distance_matrix, number_of_neighbours):
    pass






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

'''
Using Neighborhood and TabuList choose best neighbour
return best neighbour
WITH ASPIRATION CRITERIUM - if move is on tabu but cost is lower 
than best then this neighbour can be taken

ALSO if all move are restricted then take off tke last one on tabu list

ALSO if still not move possible make move
'''
def choose_best_aspiration(Neighborhood, TabuList, best):
    a = Neighborhood.shape[0]
    best_candidate = Neighborhood[0][2]
    index_best = 0
    move_taken = False
    for i in range(1, a):
        if((Neighborhood[i][2] < best_candidate) and (Neighborhood[i][1] not in TabuList)):
            best_candidate = Neighborhood[i][2]
            index_best = i
            move_taken = True
        # Aspiration: if cost is lowest then take move (even if move is on tabu list)
        elif(Neighborhood[i][2] < best):
            best_candidate = Neighborhood[i][2]
            best = Neighborhood[i][2]
            index_best = i
            move_taken = True
    if(move_taken):
        return Neighborhood[index_best]
    else: # take last move out of tabu list
        if(TabuList[len(TabuList) - 1] != -1):
            TabuList.appendleft(-1)
            return choose_best_aspiration(Neighborhood, TabuList, best)
    # if still did't take move make random move
    return Neighborhood[random.randint(0,a)]

import math
import random
import tsplib95
from tsplib95 import distances
import numpy as np
import matplotlib.pyplot as plt

def initialize_problem():
    number_of_cities = 10
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

# something loke this but better!
def get_neighbors(best_candidate, number_of_cities):
    Neighborhood = np.array([0,0,0,0,0,0,0,0,0,0])
    i = 0
    while i < number_of_cities:
        j = i+1
        while j < number_of_cities:
            best_candidate_copy = best_candidate.copy()
            best_candidate_copy = invert(best_candidate_copy, i, j)
            Neighborhood = np.vstack([Neighborhood, best_candidate_copy])
            j += 1
        i += 1
    return Neighborhood

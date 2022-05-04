#import numpy as np
import collections

import TSP_functions as tsp

def main():
    
    size_of_problem, matrix_tsp = tsp.initialize_problem()
    TabuList_length = 7
    TabuList = collections.deque(maxlen=TabuList_length)
    number_of_iteration_without_update = 0
    MAX_number_of_iteration_without_update = 50
    stop_condition = False
    i = 0

    # Step 1: (first solution)
    #best_solution = tsp.random_solution(size_of_problem)
    #start_city = 0
    #best_solution = tsp.nearest_neighbor_solution(size_of_problem, start_city, matrix_tsp)
    best_solution = tsp.nearest_neighbor_extended_solution(size_of_problem, matrix_tsp)
    best_candidate = best_solution
    
    print(size_of_problem, "\n", matrix_tsp)
    
    while(not stop_condition):
        #Step 2: (generate full Neighborhood)
        Neighborhood = tsp.get_neighbors_invert(best_candidate, size_of_problem, matrix_tsp)
        print(Neighborhood)

        # Step 3: (choose best candidate from Neighborhood)
        best_candidate_array = tsp.choose_best_aspiration(Neighborhood, TabuList, best_solution)
        best_candidate = best_candidate_array[0]
        print("--------------------------------------------------")
        print(best_candidate_array)

        # Step 4: (update Tabu List)
        TabuList.appendleft(best_candidate_array[1])
        print(TabuList)

        # Step 5: (stop condition)
        if (number_of_iteration_without_update >= MAX_number_of_iteration_without_update):
            stop_condition = True
        else:
            if (tsp.get_weight(best_solution, matrix_tsp) <= tsp.get_weight(best_candidate, matrix_tsp)):
                number_of_iteration_without_update += 1
            
            else:
                number_of_iteration_without_update = 0

        # update best solution
        if (tsp.get_weight(best_solution, matrix_tsp) >= tsp.get_weight(best_candidate, matrix_tsp)):
                best_solution = best_candidate

        i += 1  
        print(tsp.get_weight(best_solution, matrix_tsp), tsp.get_weight(best_candidate, matrix_tsp))      
        print(i, "=================================================")

if __name__ == '__main__':
    main()
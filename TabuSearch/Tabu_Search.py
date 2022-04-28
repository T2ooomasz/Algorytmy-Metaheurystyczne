#import numpy as np
import collections

import TSP_functions as tsp

def main():
    
    size_of_problem, matrix_tsp = tsp.initialize_problem()
    TabuList = collections.deque([0]*3,maxlen=3)
    stop_condition = False
    i = 0

    # Step 1: (first solution)
    best_solution = tsp.random_solution(size_of_problem)
    best_candidate = best_solution
    
    print(size_of_problem, matrix_tsp)
    
    while(not stop_condition):
        #Step 2: (generate full Neighborhood)
        Neighborhood = tsp.get_neighbors(best_candidate, size_of_problem, matrix_tsp)
        print(Neighborhood)

        # Step 3: (choose best candidate from Neighborhood)
        best_candidate_array = tsp.choose_best(Neighborhood, TabuList)
        best_candidate = best_candidate_array[0]
        print("--------------------------------------------------")
        print(best_candidate_array)

        # Step 4: (update Tabu List)
        TabuList.appendleft(best_candidate_array[1])
        print(TabuList)

        # Step 5: (stop condition)
        if i > 2:
            stop_condition = True
        i += 1
        print("=================================================")

if __name__ == '__main__':
    main()
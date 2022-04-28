#import numpy as np
import collections

import TSP_functions as tsp

def main():
    # Step 1:
    size_of_problem, matrix_tsp = tsp.initialize_problem()
    best_solution = tsp.random_solution(size_of_problem)
    best_candidate = best_solution
    TabuList = collections.deque([0]*3,maxlen=3)
    
    print(size_of_problem, matrix_tsp)
    # Step 2:
    stop_condition = False
    i = 0
    while(not stop_condition):
        Neighborhood = tsp.get_neighbors(best_candidate, size_of_problem, matrix_tsp)
        print(Neighborhood)
        best_candidate_array = tsp.choose_best(Neighborhood, TabuList)
        best_candidate = best_candidate_array[0]
        print("--------------------------------------------------")
        print(best_candidate_array)
        #for candidate in Neighborhood:
            #if ( (not tabuList.contains(sCandidate)) and (fitness(sCandidate) > fitness(bestCandidate)) )
            #bestCandidate ← sCandidate

        # Step 4 - TabuList
        TabuList.appendleft(best_candidate_array[1])
        print(TabuList)
        if i > 2:
            stop_condition = True
        i += 1
        print("=================================================")

if __name__ == '__main__':
    main()
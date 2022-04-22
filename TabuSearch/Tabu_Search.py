#import numpy as np

import TSP_functions as tsp

def main():
    # Step 1:
    size_of_problem, matrix_tsp = tsp.initialize_problem()
    best_solution = tsp.random_solution(size_of_problem)
    best_candidate = best_solution
    #TabuList = np.array([])
    
    # Step 2:
    stop_condition = False
    i = 0
    while(not stop_condition):
        Neighborhood = tsp.get_neighbors(best_candidate, size_of_problem)
        print(Neighborhood)
        best_candidate = Neighborhood[0]
        print("=================================================")
        print(best_candidate)
        #for candidate in Neighborhood:
            #if ( (not tabuList.contains(sCandidate)) and (fitness(sCandidate) > fitness(bestCandidate)) )
            #bestCandidate ← sCandidate
        if i > 2:
            stop_condition = True
        i += 1

if __name__ == '__main__':
    main()
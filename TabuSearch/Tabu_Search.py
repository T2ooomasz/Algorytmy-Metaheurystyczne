import collections

import TSP_functions as tsp

def main():
    
    size_of_problem, matrix_tsp, best_know_cost = tsp.initialize_problem()
    TabuList_length = 7
    TabuList = collections.deque(maxlen=TabuList_length)
    number_of_iteration_without_update = 0
    MAX_number_of_iteration_without_update = 50
    stop_condition = False
    i = 0

    # Step 1: (first solution)
    best_solution = tsp.random_solution(size_of_problem)                                    # to start from random solution uncoment this line
    #start_city = 0                                                                         # to start from nearest neighbour solution uncoment this and following line
    #best_solution = tsp.nearest_neighbor_solution(size_of_problem, start_city, matrix_tsp)
    #best_solution = tsp.nearest_neighbor_extended_solution(size_of_problem, matrix_tsp)    # to start from nearest neigbour extended solution uncoment this line
    best_candidate = best_solution
    
    print(0, tsp.get_weight(best_solution, matrix_tsp), (tsp.get_weight(best_solution, matrix_tsp)/best_know_cost - 1) * 100, "%")
    while(not stop_condition):
        #Step 2: (generate full Neighborhood)
        Neighborhood = tsp.get_neighbors_invert(best_candidate, size_of_problem, matrix_tsp)

        # Step 3: (choose best candidate from Neighborhood)
        best_candidate_array = tsp.choose_best_aspiration(Neighborhood, TabuList, best_solution) # looking for best neighbour with aspiration criterium
        best_candidate = best_candidate_array[0]

        # Step 4: (update Tabu List)
        TabuList.appendleft(best_candidate_array[1])

        # Step 5: (stop condition)
        if (number_of_iteration_without_update >= MAX_number_of_iteration_without_update):
            stop_condition = True
        else:
            if (tsp.get_weight(best_solution, matrix_tsp) <= tsp.get_weight(best_candidate, matrix_tsp)):
                number_of_iteration_without_update += 1
            
            else:
                number_of_iteration_without_update = 0
                
        # print data if beter than previous
        if (i>0 and tsp.get_weight(best_solution, matrix_tsp) > tsp.get_weight(best_candidate, matrix_tsp)):
            print(i, tsp.get_weight(best_solution, matrix_tsp), (tsp.get_weight(best_solution, matrix_tsp)/best_know_cost - 1) * 100, "%")

        # update best solution
        if (tsp.get_weight(best_solution, matrix_tsp) >= tsp.get_weight(best_candidate, matrix_tsp)):
            best_solution = best_candidate

        i += 1  

if __name__ == '__main__':
    main()
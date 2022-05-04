'''
o - Rozwiązania początkowe
o - Długość listy Tabu
o - Od wielkości instancji
o - Zależność od rozwiązania początkowego
x - Wielkość sąsiedztwa
o - Zaleznosc prd od dlugosci tabu list oraz wielkosci instancji (wykres 3D)
'''
import numpy as np
import collections
import TSP_functions as tsp
import TSP_initialize as init

def initialization(file_name, TabuList_length, type):
    size_of_problem, matrix_tsp = init.read_file(file_name, type='atsp')
    #TabuList_length = 7
    TabuList = collections.deque(maxlen=TabuList_length)
    number_of_iteration_without_update = 0
    stop_condition = False
    i = 0

    # Step 1: (first solution)
    if type == 'rand':
        best_solution = tsp.random_solution(size_of_problem)
    elif type == 'nn':
        start_city = 0
        best_solution = tsp.nearest_neighbor_solution(size_of_problem, start_city, matrix_tsp)
    elif type == 'nne':
        best_solution = tsp.nearest_neighbor_extended_solution(size_of_problem, matrix_tsp)
    return size_of_problem, matrix_tsp, TabuList, number_of_iteration_without_update,  stop_condition, i, best_solution, best_solution

def TabuSearch(file_name, TabuList_length, MAX_number_of_iteration_without_update, type):
    size_of_problem, matrix_tsp, TabuList, number_of_iteration_without_update, stop_condition, i, best_solution, best_candidate = initialization(file_name, TabuList_length, type)

    while(not stop_condition):
        #Step 2: (generate full Neighborhood)
        Neighborhood = tsp.get_neighbors_invert(best_candidate, size_of_problem, matrix_tsp)

        # Step 3: (choose best candidate from Neighborhood)
        best_candidate_array = tsp.choose_best_aspiration(Neighborhood, TabuList, best_solution)
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

        # update best solution
        if (tsp.get_weight(best_solution, matrix_tsp) >= tsp.get_weight(best_candidate, matrix_tsp)):
                best_solution = best_candidate
        i += 1  

def start_random_simulation(instance_list, tabu, MAX_noiwu):
    for instance in instance_list:
        for tabu_length in tabu:
            for num_iter_w_update in MAX_noiwu:
                TabuSearch(instance, tabu_length, num_iter_w_update, 'rand')
                
        
def start_nne_simulation(instance_list, tabu, MAX_noiwu):
    for instance in instance_list:
        for tabu_length in tabu:
            for num_iter_w_update in MAX_noiwu:
                TabuSearch(instance, tabu_length, num_iter_w_update, 'nne')

def main():

    instance_list = ['ftv33.atsp', 'ftv55.atsp', 'ftv100.atsp', 'ftv170.atsp', 'rbg323.atsp', 'rbg443.atsp']
    min_tabu_lenght = 2
    max_tabu_lenght = 20
    step_tabu = 5
    tabu = np.arange(min_tabu_lenght, max_tabu_lenght + step_tabu, step_tabu).tolist()
    MAX_number_of_iteration_without_update = 100
    MAX_noiwu = np.arange(25, MAX_number_of_iteration_without_update + 1, 25).tolist()


    start_random_simulation(instance_list, tabu, MAX_noiwu)
    start_nne_simulation(instance_list, tabu, MAX_noiwu)

if __name__ == '__main__':
    main()
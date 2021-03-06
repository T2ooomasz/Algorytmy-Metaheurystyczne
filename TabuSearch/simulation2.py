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
import time
import TSP_functions as tsp
import TSP_initialize as init

def saveDATA(arr):
    # reshaping the array from 4D
    # matrice to 2D matrice.
    arr_reshaped = arr.reshape(arr.shape[0], -2)
    
    # saving reshaped array to file.
    np.savetxt("simulation/atsp_data_nne_3.csv", arr_reshaped)

def loadDATA(DATA, arr):
    loaded_arr = np.loadtxt("simulation/atsp_data_nne_2.csv")
  
    # This loadedArr is a 2D array, therefore
    # we need to convert it to the original
    # array shape.reshaping to get original
    # matrice with original shape.
    load_original_arr = np.reshape(loaded_arr, arr)
    
    # check the shapes:
    print("shape of arr: ", DATA.shape)
    print("shape of load_original_arr: ", load_original_arr.shape)
    
    # check if both arrays are same or not:
    if (load_original_arr == DATA).all():
        print("Yes, both the arrays are same")
    else:
        print("No, both the arrays are not same")

def initialization(file_name, TabuList_length, type, instance_type):
    size_of_problem, matrix_tsp = init.read_file(file_name, type=instance_type)
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

def TabuSearch(file_name, TabuList_length, MAX_number_of_iteration_without_update, type, instance_type):
    size_of_problem, matrix_tsp, TabuList, number_of_iteration_without_update, stop_condition, i, best_solution, best_candidate = initialization(file_name, TabuList_length, type, instance_type)

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
    return tsp.get_weight(best_candidate, matrix_tsp)

def start_random_simulation(instance_list, tabu, MAX_noiwu, DATA, instance_type):
    i = -1
    for instance in instance_list:
        i += 1
        j = -1
        for tabu_length in tabu:
            j += 1
            k = -1
            for num_iter_w_update in MAX_noiwu:
                k += 1
                start_time = time.time()
                best_candidate_cost = TabuSearch(instance, tabu_length, num_iter_w_update, 'rand', instance_type)
                end_time = time.time()
                DATA[i,j,k, 0] = best_candidate_cost
                DATA[i,j,k, 1] = end_time-start_time                
        
def start_nne_simulation(instance_list, tabu, MAX_noiwu, DATA, instance_type):
    i = -1
    for instance in instance_list:
        i += 1
        j = -1
        for tabu_length in tabu:
            j += 1
            k = -1
            for num_iter_w_update in MAX_noiwu:
                k += 1
                start_time = time.time()
                best_candidate_cost = TabuSearch(instance, tabu_length, num_iter_w_update, 'nne', instance_type)
                end_time = time.time()
                DATA[i,j,k, 0] = best_candidate_cost
                DATA[i,j,k, 1] = end_time-start_time 

def main():
    instance_type = 'atsp'
    instance_list = ['br17.atsp', 'ftv33.atsp', 'ftv44.atsp', 'ftv55.atsp', 'ftv64.atsp', 'ftv70.atsp']
    #instance_list = ['br17.atsp', 'ry48p.atsp', 'ftv70.atsp', 'kro124p.atsp', 'ftv170.atsp', 'rbg323.atsp', 'rbg403.atsp']
    #instance_list = ['gr24.tsp', 'gr48.tsp', 'pr76.tsp', 'pr107.tsp', 'gr120.tsp', 'pr136.tsp', 'pr152.tsp']
    min_tabu_lenght = 2
    max_tabu_lenght = 37
    step_tabu = 5
    tabu = np.arange(min_tabu_lenght, max_tabu_lenght + step_tabu, step_tabu).tolist()
    #MAX_number_of_iteration_without_update = 25
    #MAX_noiwu = np.arange(25, MAX_number_of_iteration_without_update + 1, 25).tolist()
    MAX_noiwu = [33,100]
    x = len(instance_list)
    y = len(tabu)
    z = len(MAX_noiwu)
    k = 2
    arr = (x,y,z,k)     #(6,5,4,2)
    DATA_rand = np.zeros(arr)   # 4D - instance x tabu_length x MAXiterations x (best, time)
    DATA_nne = np.zeros(arr)

    start_random_simulation(instance_list, tabu, MAX_noiwu, DATA_rand, instance_type)
    start_nne_simulation(instance_list, tabu, MAX_noiwu, DATA_nne, instance_type)

    saveDATA(DATA_rand)
    saveDATA(DATA_nne)
    #loadDATA(DATA_nne, arr)

if __name__ == '__main__':
    main()
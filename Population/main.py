'''
Main - where it all begins

takes all parameters required for simulation
run class algorithm
'''

from alg import algorithm as alg

def main():
    # algorithm attributs
    number_of_populations = 1
    type_of_parents_selection = "-1"
    type_of_crossing = "KLM"
    type_of_mutation = "0"
    type_of_population_selection = "-1"
    
    alg.algorithm(number_of_populations, type_of_parents_selection, type_of_crossing, type_of_mutation, type_of_population_selection)

    print("\nPopulation algorithm END")

if __name__ == '__main__':
    main()
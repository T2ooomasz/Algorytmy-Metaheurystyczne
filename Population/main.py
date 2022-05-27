'''
Main - where it all begins

takes all parameters required for simulation
run class algorithm
'''

from alg import algorithm as alg

def main():
    # algorithm attributs

    '''
    set number of populations (island)
        must be positive integer
    '''
    number_of_populations = 1   # not implemented - there is no population class

    '''
    set type of parents selection:
        random - every individuum have equal chance to be a parent 
        roulette - beter individuum have larges chance to be a parent
        tournament - best individuum will be a parent and rest of turnament "winners"
    '''
    type_of_parents_selection = "random"

    '''
    set type of crossing:
        HX - Half Crossover
        OX - Order Crossover
        PMX - Partially Mapped Crossover
    '''
    type_of_crossing = "KLM"

    '''
    set type of mutation:
        0 - 
        1 - 
    '''
    type_of_mutation = "0"

    '''
    set type of population selection: (from old population and childers - 'kill some individuums')
        random - every individuum have equal chance to be a parent 
        roulette - beter individuum have larges chance to be a parent
        tournament - best individuum will be a parent and rest of turnament "winners"
    '''
    type_of_population_selection = "random"

    '''
   set type of stop condition:
        0 - iteration without update # for now just iterations
        1 - iterations
        2 - time (how many second algorithm will run)
    '''
    type_of_stop_condition = "2"

    '''
    set number - for diferent stop condition it's mean:
        0 - for iteration without update it's number of iteration without update
        1 - for iteration is number of iterations
        2 - for time it's seconds
    '''
    number_for_stop = 1

    # run algorithm
    alg.algorithm (
         number_of_populations,
         type_of_parents_selection,
         type_of_crossing,
         type_of_mutation,
         type_of_population_selection,
         type_of_stop_condition,
         number_for_stop)

    # print info about end of algorithm
    print("\nPopulation algorithm END")

if __name__ == '__main__':
    main()
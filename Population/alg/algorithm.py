'''
Class algorithm.

Takes all parameteres required for whole simulation.
return the best solution
'''
from alg import parents_selection as ps
from alg import crossing as c
from alg import mutation as m
from alg import population_selection as pops
from alg import stop_condition as sc

class algorithm():
    # atribute
    name = "algorithm"


    def __init__(self,
         number_of_populations,
         type_of_parents_selection,
         type_of_crossing,
         type_of_mutation,
         type_of_population_selection,
         type_of_stop_condition,
         number_for_stop):

        self.number_of_populations = number_of_populations
        self.type_of_parents_selection = type_of_parents_selection
        self.type_of_crossing = type_of_crossing
        self.type_of_mutation = type_of_mutation
        self.type_of_population_selection = type_of_population_selection
        self.type_of_stop_condition = type_of_stop_condition
        self.number_for_stop = number_for_stop

        # run algorithm
        self.alg()

    def alg(self):
        population = "syntetic population"
        stop = sc.stop_condition(self.type_of_stop_condition, self.number_for_stop)
        while(stop.stop_condition == False):
            parents = ps.parents_selection(population, self.type_of_parents_selection)
            childrens = c.crossing(parents, self.type_of_crossing)
            childrens = m.mutation(childrens, self.type_of_mutation)
            population = pops.population_selection(population, childrens, self.type_of_population_selection)
            #best_individuum = update_best_individuum()
            stop.update_stop_condition()

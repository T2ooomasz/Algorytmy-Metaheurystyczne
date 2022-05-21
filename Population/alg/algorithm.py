'''
Class algorithm.

Takes all parameteres required for whole simulation.
return the best solution
'''
from alg import parents_selection as ps
from alg import crossing as c
from alg import mutation as m
from alg import population_selection as pops

class algorithm():
    # atribute
    name = "algorithm"


    def __init__(self, number_of_populations, type_of_parents_selection, type_of_crossing, type_of_mutation, type_of_population_selection):
        self.number_of_populations = number_of_populations
        self.type_of_parents_selection = type_of_parents_selection
        self.type_of_crossing = type_of_crossing
        self.type_of_mutation = type_of_mutation
        self.type_of_population_selection = type_of_population_selection
        # run algorithm
        self.alg()

    def alg(self):
        population = "syntetic population"
        stop_condition = False
        i=0
        while(stop_condition == False):
            parents = ps.parents_selection(population, self.type_of_parents_selection)
            childrens = c.crossing(parents, self.type_of_crossing)
            childrens = m.mutation(childrens, self.type_of_mutation)
            population = pops.population_selection(population, childrens, self.type_of_population_selection)
            stop_condition = self.update_stop_condition(i)
            i += 1

    def update_stop_condition(self, i):
        if i >= 0:
            stop_condition = True
        else:
            stop_condition = False
        return stop_condition
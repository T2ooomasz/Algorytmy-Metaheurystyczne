from random import choice, randint, uniform
import time
from pandas import array
import tsp_initialize as init
import tsp_functions as func
import population
import genetic_algorithm

if __name__ == '__main__':
    pop = population(200)
    parents = pop.initialize_population()
    phenotypes = []
    for i in range(200):
        phenotypes.append(parents[i].phenotype)
    print(min(phenotypes))   
    gen = genetic_algorithm('roulette', 'roulette', 'PMX', parents)
    new_population = gen.parents
    phenotypes1 = []
    for i in range(len(new_population)):
        phenotypes1.append(new_population[i].phenotype)
    print(min(phenotypes1))
    #pop = population(200) 
    #set_of_individuals = pop.initialize_population()
    #sel = selection('random', set_of_individuals)
    #fathers = sel.selected_fathers
    #mothers = sel.selected_mothers
    #cro = crossing('PMX', fathers, mothers)
    #children = cro.children
    #sel_pop = selection_population('roulette', set_of_individuals, children)
    #new_population = sel_pop.new_population

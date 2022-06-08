from random import choice, randint, uniform
import time
import numpy as np
from pandas import array
import tsp_initialize as init
import tsp_functions as func

class individual:

    def __init__(self, genotype, phenotype, fitness):
        self.genotype = genotype
        self.phenotype = phenotype
        self.fitness = fitness

class population:

    def __init__(self, size_of_population):
        self.size_of_population = size_of_population
        self.size_of_individual = 0
        self.distance_matrix = []
        self.phenotype = []
        self.genotype = []
        self.set_of_individuals = []
        self.best_solution = []
        self.initialize_population()

    def initialize_2opt_genotype(self):
        for _ in range(self.size_of_population):
            self.genotype.append(func.opt2(self.size_of_individual, self.distance_matrix))

    def initialize_genotype(self):
        for _ in range(self.size_of_population):
            self.genotype.append(func.random_solution(self.size_of_individual))

    def initialize_phenotype(self):
        for i in range(self.size_of_population):
            self.phenotype.append(func.get_weight(self.genotype[i], self.distance_matrix))

    def initialize_population(self):
        self.size_of_individual, self.distance_matrix = func.initialize_problem()
        self.initialize_genotype()
        self.initialize_phenotype()
        self.best_solution.append(min(self.phenotype))
        for i in range(len(self.genotype)):
            self.set_of_individuals.append(individual(self.genotype[i], self.phenotype[i], (1/self.phenotype[i])))
        return self.set_of_individuals, self.best_solution, self.size_of_population, self.distance_matrix

class selection:

    def __init__(self, type_of_selection = '', set_of_individuals = []):
        self.type_of_selection = type_of_selection
        self.set_of_individuals = set_of_individuals
        self.total_fitness = 0
        self.field = 0
        self.fitness = []
        self.roulette = []
        self.selected_fathers = []
        self.selected_mothers = []
        self.probability = []
        self.selection_algorithm()

    def selection_algorithm(self):
        if self.type_of_selection == 'random':
            self.random_selection()
        elif self.type_of_selection == 'roulette':
            self.roulette_selection()
        elif self.type_of_selection == 'tournament':
            self.tournament_selection()
        else:
            print("There is no \"", self.type_of_selection, "\" type of selection")

    def random_selection(self):
        size_of_population = len(self.set_of_individuals)
        values = list(range(size_of_population))
        for _ in range(int(size_of_population/2)):
            rand1 = choice(values)
            self.selected_fathers.append(self.set_of_individuals[rand1])
            values.remove(rand1)
            rand2 = choice(values)
            self.selected_mothers.append(self.set_of_individuals[rand2])
            values.remove(rand2)
        return self.selected_fathers, self.selected_mothers

    def roulette_selection(self):
        for i in range(len(self.set_of_individuals)):
            self.total_fitness = self.total_fitness + self.set_of_individuals[i].fitness
        for n in range(len(self.set_of_individuals)):
            self.probability.append(self.set_of_individuals[n].fitness/self.total_fitness)
            self.field = self.field + self.probability[n]
            self.roulette.append(self.field)
        for _ in range(int(len(self.set_of_individuals)/2)):
            r1 = uniform(0,1)
            for m in range(len(self.roulette)):
                if r1 <= self.roulette[m]:
                    self.selected_fathers.append(self.set_of_individuals[m])
                    break
                else:
                    continue  
            r2 = uniform(0,1)
            for n in range(len(self.roulette)):
                if r2 <= self.roulette[n]:
                    self.selected_mothers.append(self.set_of_individuals[n])
                    break
                else:
                    continue  
        return self.selected_fathers, self.selected_mothers

    def tournament_selection(self):
        pass

class selection_population:

    def __init__(self, type_of_selection_population = '', population = [], best_solutions = []):
        self.type_of_selection_population = type_of_selection_population
        self.total_probability = 0
        self.total_fitness = 0
        self.field = 0
        self.roulette = []
        self.fitness = []
        self.phenotypes = []
        self.probability = []
        self.new_population = []
        self.population = population
        self.best_solutions = best_solutions
        self.selection_population_algorithm()

    def selection_population_algorithm(self):
        if self.type_of_selection_population == 'roulette':
            self.roulette_selection_population()
        elif self.type_of_selection_population == 'tournament':
            self.tournament_selection_population()
        else:
            print("There is no \"", self.type_of_selection_population, "\" type of selection population")

    def roulette_selection_population(self):
        for i in range(len(self.population)):
            self.total_fitness = self.total_fitness + self.population[i].fitness
        for n in range(len(self.population)):
            self.probability.append(self.population[n].fitness/self.total_fitness)
            self.field = self.field + self.probability[n]
            self.roulette.append(self.field)
        for _ in range(int(len(self.population)/2)):
            r = uniform(0, 1)
            for p in range(len(self.roulette)):
                if r <= self.roulette[p]:
                    self.new_population.append(self.population[p])
                    break
                else:
                    continue
        for i in range(len(self.new_population)):
            self.phenotypes.append(self.new_population[i].phenotype)
        self.best_solutions.append(min(self.phenotypes))
        return self.new_population

    def tournament_selection_population(self):
        pass

class crossing:

    def __init__(self, type_of_crossing = '', selected_fathers = [], selected_mothers = [], distance_matrix = []):
        self.type_of_crossing = type_of_crossing
        self.selected_fathers = selected_fathers
        self.selected_mothers = selected_mothers
        self.distance_matrix = distance_matrix
        self.children = []
        self.crossing_algorithm()

    def crossing_algorithm(self):
        if self.type_of_crossing == 'HX':
            self.half_crossover()
        elif self.type_of_crossing == 'OX':
            self.order_crossover()
        elif self.type_of_crossing == 'PMX':
            self.partially_mapped_crossover()
        else:
            print("There is no \"", self.type_of_crossing, "\" type of crossing")

    def half_crossover(self):
        pass

    def order_crossover(self):
        def __ocover(parent1, parent2, start, stop):
            child = [None]*len(parent1)
            child[start:stop] = child[start:stop]
            index1 = stop
            index2 = stop
            length = len(parent1)
            while None in child:
                if parent2[index1%length] not in child:
                    child[index2%length] = parent2[index1%length]
                    index2 += 1
                index1 += 1
            return child
        
        for i in range(len(self.selected_fathers)):       
            half = len(self.selected_fathers[i].genotype) // 2
            start = randint(0, len(self.selected_fathers[i].genotype)-half)
            stop = start + half
            child1_genotype = __ocover(self.selected_fathers[i].genotype, self.selected_mothers[i].genotype, start, stop)
            child2_genotype = __ocover(self.selected_mothers[i].genotype, self.selected_fathers[i].genotype, start, stop)
            child1_phenotype = func.get_weight(child1_genotype, self.distance_matrix)
            child2_phenotype = func.get_weight(child2_genotype, self.distance_matrix)
            self.children.append(individual(child1_genotype, child1_phenotype, 1/child1_phenotype))
            self.children.append(individual(child2_genotype, child2_phenotype, 1/child2_phenotype))
        return self.children

    def partially_mapped_crossover(self):
        def __pmx(parent1, parent2, start, stop):
            child = [None]*len(parent1)
            child[start:stop] = parent1[start:stop]
            for index1, x in enumerate(parent2[start:stop]):
                index1 += start
                if x not in child:
                    while child[index1] != None:
                        index1 = parent2.index(parent1[index1])
                    child[index1] = x
            for index1, x in enumerate(child):
                if x == None:
                    child[index1] = parent2[index1]
            return child

        for i in range(len(self.selected_fathers)):
            half = len(self.selected_fathers[i].genotype) // 2
            start = randint(0, len(self.selected_fathers[i].genotype)-half)
            stop = start + half
            child1_genotype = __pmx(self.selected_fathers[i].genotype, self.selected_mothers[i].genotype, start, stop)
            child2_genotype = __pmx(self.selected_mothers[i].genotype, self.selected_fathers[i].genotype, start, stop)
            child1_phenotype = func.get_weight(child1_genotype, self.distance_matrix)
            child2_phenotype = func.get_weight(child2_genotype, self.distance_matrix)
            self.children.append(individual(child1_genotype, child1_phenotype, 1/child1_phenotype))
            self.children.append(individual(child2_genotype, child2_phenotype, 1/child2_phenotype))
        return self.children

class mutation:

    def __init__(self, size_of_population, probability_of_mutation, type_of_mutation = '', set_of_individuals = []):
        self.size_of_population = size_of_population
        self.probability_of_mutation = probability_of_mutation
        self.type_of_mutation = type_of_mutation
        self.set_of_individuals = set_of_individuals 
        self.mutation_algorithm()

    def mutation_algorithm(self):
        if self.type_of_mutation == 'swap':
            self.mutation_0()
        elif self.type_of_mutation == 'invert':
            self.mutation_1()
        else:
            print("There is no \"", self.type_of_mutation, "\" type of mutation")

    def mutation_0(self):
        r = uniform(0, 1)
        if r <= self.probability_of_mutation:
            for i in range(20):
                rand = randint(0, self.size_of_population-1)
                rand1 = randint(0, len(self.set_of_individuals[rand].genotype)-1)
                rand2 = randint(0, len(self.set_of_individuals[rand].genotype)-1)
                func.swap_positions(self.set_of_individuals[rand].genotype, rand1, rand2)
        return self.set_of_individuals

    def mutation_1(self):
        pass
        #losowy invert

class stop_condition:

    def __init__(self, type_of_stop_condition, number_for_stop):
        self.type_of_stop_condition = type_of_stop_condition
        self.number_for_stop = number_for_stop
        self.stop_condition = False
        self.start_time = time.time()
        self.iterations = 0
        self.iterations_without_update = 0

    def selection_stop_condition(self):
        if self.type_of_stop_condition == '0':
            self.stop_condition_0()
        elif self.type_of_stop_condition == '1':
            self.stop_condition_1()
        elif self.type_of_stop_condition == '2':
            self.stop_condition_2()
        else:
            print("There is no \"", self.type_of_stop_condition, "\" type of stop condition")

    def update_stop_condition(self):
        self.selection_stop_condition()

    def stop_condition_0(self):
        if self.iterations_without_update > self.number_for_stop:
            self.stop_condition = True
            self.iterations_without_update += 1
            print('stop_condition_0 works!')
        
    def stop_condition_1(self):
        self.iterations += 1
        if self.iterations >= self.number_for_stop:
            self.stop_condition = True
        print('stop_condition_1 works!')

    def stop_condition_2(self):
        end = time.time()
        if end - self.start_time > self.number_for_stop:
            self.stop_condition = True
        print('stop_condition_2 works!')

class genetic_algorithm:
    
    def __init__(self, type_of_selection, type_of_selection_population, type_of_crossing, type_of_mutation, probability_of_mutation, size_of_population, parents = [], best_solutions = [], distance_matrix = []):
        self.type_of_selection = type_of_selection
        self.type_of_selection_population = type_of_selection_population
        self.type_of_crossing = type_of_crossing
        self.type_of_mutation = type_of_mutation
        self.probability_of_mutation = probability_of_mutation
        self.size_of_population = size_of_population
        #self.type_of_stop_condition = type_of_stop_condition
        #self.number_for_stop = number_for_stop
        self.parents = parents
        self.best_solutions = best_solutions
        self.distance_matrix = distance_matrix
        self.children = []
        self.algorithm()

    def algorithm(self):
        #stop = stop_condition(self.type_of_stop_condition)
        #while(stop.stop_condition == False):
        for i in range(20000):
            sel = selection(self.type_of_selection, self.parents)
            fathers = sel.selected_fathers
            mothers = sel.selected_mothers
            cro = crossing(self.type_of_crossing, fathers, mothers, self.distance_matrix)
            self.children = cro.children
            self.population = np.concatenate((self.children, self.parents))
            sel_pop = selection_population(self.type_of_selection_population, self.population, self.best_solutions)
            self.parents = sel_pop.new_population
            mut = mutation(self.size_of_population, self.probability_of_mutation, self.type_of_mutation, self.parents)
            self.parents = mut.set_of_individuals
            #stop.update_stop_condition()
        return self.parents, self.best_solutions


def main():
    condition = True
    print()
    print("Genetic Algorithm for 'Travelling salesman problem'\n")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    while condition:
        print()
        print("To continue press '1'\nTo exit press '0")
        option = int(input().strip())
        if option == 1:
            print("Enter size of population:")
            size_of_population = int(input().strip())
            print("Enter the type of selection individuals for the crossing:")
            print("1.random\n2.roulette\n3.tournament")
            type_of_selection = str(input().strip())
            print("Enter the type of selection the new generation:")
            print("1.random\n2.roulette\n3.tournament")
            type_of_selection_population = str(input().strip())
            print("Enter the type of the crossing:")
            print("1.HX\n2.OX\n3.PMX")
            type_of_crossing = str(input().strip())
            print("Enter the type of the mutation:")
            print("1.swap\n2.invert")
            type_of_mutation = str(input().strip())
            print("Enter the probability of the mutation:")
            probability_of_mutation = float(input().strip())
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            best = []
            pop = population(size_of_population)
            parents = pop.set_of_individuals
            best_solutions = pop.best_solution
            size_of_population = pop.size_of_population
            distance_matrix = pop.distance_matrix
            gen = genetic_algorithm(str(type_of_selection), str(type_of_selection_population), str(type_of_crossing), str(type_of_mutation), probability_of_mutation, size_of_population, parents, best_solutions, distance_matrix)
            new_population = gen.parents
            best_individual = min(gen.best_solutions)
            print(best_individual)
            for i in range(len(parents)):
                best.append(new_population[i].phenotype)
            print(min(best))
        else:
            condition = False

if __name__ == '__main__':
    main()


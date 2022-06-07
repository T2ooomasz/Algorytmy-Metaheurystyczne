from random import choice, randint, uniform
import time
from pandas import array
import tsp_initialize as init
import tsp_functions as func

class individual:

    def __init__(self, genotype, phenotype):
        self.genotype = genotype
        self.phenotype = phenotype

class population:

    def __init__(self, size_of_population):
        self.size_of_population = size_of_population
        self.size_of_individual = 0
        self.distance_matrix = []
        self.phenotype = []
        self.genotype = []
        self.set_of_individuals = []

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
        for i in range(len(self.genotype)):
            self.set_of_individuals.append(individual(self.genotype[i], self.phenotype[i]))
        return self.set_of_individuals

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
            self.fitness.append(1/self.set_of_individuals[i].phenotype)
        for j in range(len(self.fitness)):
            self.total_fitness = self.total_fitness + self.fitness[j]
        for k in range(len(self.set_of_individuals)):
            self.probability.append(self.fitness[k]/self.total_fitness)
        self.field = self.probability[0]
        self.roulette.append(self.field)
        for l in range(1, len(self.probability)):
            self.field = self.field + self.probability[l]
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

    def __init__(self, type_of_selection_population = '', parents = [], children = []):
        self.type_of_selection_population = type_of_selection_population
        self.total_probability = 0
        self.total_fitness = 0
        self.field = 0
        self.roulette = []
        self.fitness = []
        self.probability = []
        self.new_population = []
        self.parents = parents
        self.children = children
        self.selection_population_algorithm()

    def selection_population_algorithm(self):
        if self.type_of_selection_population == 'roulette':
            self.roulette_selection_population()
        elif self.type_of_selection_population == 'tournament':
            self.tournament_selection_population()
        else:
            print("There is no \"", self.type_of_selection_population, "\" type of selection population")

    def roulette_selection_population(self):
        iterations = len(self.parents) + len(self.children)
        for i in range(len(self.parents)):
            self.fitness.append(1/self.parents[i].phenotype)
        for j in range(len(self.children)):
            self.fitness.append(1/self.children[j].phenotype)
        for k in range(iterations):
            self.total_fitness = self.total_fitness + self.fitness[k]
        for l in range(len(self.parents)):
            self.probability.append(self.fitness[l]/self.total_fitness)
        for m in range(len(self.children), iterations):
            self.probability.append(self.fitness[m]/self.total_fitness)
        self.field = self.probability[0]
        self.roulette.append(self.field)
        for n in range(1, len(self.probability)):
            self.field = self.field + self.probability[n]
            self.roulette.append(self.field)
        for _ in range(len(self.parents)):
            r = uniform(0, 1)
            for p in range(len(self.roulette)):
                if r <= self.roulette[p]:
                    if p < len(self.parents):
                        self.new_population.append(self.parents[p])
                        break
                    elif p >= len(self.parents) and p < len(self.roulette):
                        self.new_population.append(self.children[p - len(self.children)])
                        break
                    else:
                        break
                else:
                    continue
        return self.new_population

    def tournament_selection_population(self):
        pass
 
class crossing:

    def __init__(self, type_of_crossing = '', selected_fathers = [], selected_mothers = []):
        self.type_of_crossing = type_of_crossing
        self.selected_fathers = selected_fathers
        self.selected_mothers = selected_mothers
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
        pass

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
            child1_phenotype = func.get_weight(child1_genotype, pop.distance_matrix)
            child2_phenotype = func.get_weight(child2_genotype, pop.distance_matrix)
            self.children.append(individual(child1_genotype, child1_phenotype))
            self.children.append(individual(child2_genotype, child2_phenotype))
        return self.children

class mutation:

    def __init__(self, type_of_mutation = '', set_of_individuals = []):
        self.type_of_mutation = type_of_mutation
        self.set_of_individuals = set_of_individuals 
        self.mutation_algorithm()

    def mutation_algorithm(self):
        if self.type_of_mutation == '0':
            self.mutation_0()
        elif self.type_of_mutation == '1':
            self.mutation_1()
        else:
            print("There is no \"", self.type_of_mutation, "\" type of mutation")

    def mutation_0(self):
        pass
        #losowy swap       

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
    
    def __init__(self, type_of_selection, type_of_selection_population, type_of_crossing, parents = []):
        self.type_of_selection = type_of_selection
        self.type_of_selection_population = type_of_selection_population
        self.type_of_crossing = type_of_crossing
        #self.type_of_mutation = type_of_mutation
        #self.type_of_stop_condition = type_of_stop_condition
        #self.number_for_stop = number_for_stop
        self.parents = parents
        self.algorithm()

    def algorithm(self):
        #stop = stop_condition(self.type_of_stop_condition)
        #while(stop.stop_condition == False):
        for _ in range(40000):
            sel = selection(self.type_of_selection, self.parents)
            fathers = sel.selected_fathers
            mothers = sel.selected_mothers
            cro = crossing(self.type_of_crossing, fathers, mothers)
            children = cro.children
            sel_pop = selection_population(self.type_of_selection_population, self.parents, children)
            self.parents = sel_pop.new_population
            #stop.update_stop_condition()
        return self.parents


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

from random import choice, randint
import time
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

    def initialize_nearest_neighbor_genotype(self):
        for _ in range(self.size_of_population):
            self.genotype.append(func.nearest_neighbor_solution(self.size_of_individual, 0, self.distance_matrix))

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
        self.selected_fathers = []
        self.selected_mothers = []
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
        pass

    def tournament_selection(self):
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
            child1_phenotype = func.get_weight(child1_genotype, population.distance_matrix)
            child2_phenotype = func.get_weight(child2_genotype, population.distance_matrix)
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

    def mutation_1(self):
        pass

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
    
    def __init__(self, size_of_population, type_of_selection, type_of_crossing, type_of_mutation, type_of_stop_condition, number_for_stop):
        self.size_of_population = size_of_population
        self.type_of_selection = type_of_selection
        self.type_of_crossing = type_of_crossing
        self.type_of_mutation = type_of_mutation
        self.type_of_stop_condition = type_of_stop_condition
        self.number_for_stop = number_for_stop
        self.set_of_individuals = []
        self.algorithm()

    def algorithm(self):
        pop = population(self.size_of_population)
        stop = stop_condition(self.type_of_stop_condition)
        self.set_of_individuals = pop.initialize_population()
        while(stop.stop_condition == False):
            sel = selection(self.type_of_selection, self.set_of_individuals)
            fathers = sel.selected_fathers
            mothers = sel.selected_mothers
            cro = crossing(self.type_of_crossing, fathers, mothers)
            children = cro.children
            stop.update_stop_condition()

if __name__ == '__main__':
    genetic_algorithm(200, 'random', 'PMX', '0', '0')






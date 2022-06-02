from random import choice, randint, random
import tsp_initialize as init
import tsp_functions as func

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

class individual:

    def __init__(self, genotype, phenotype):
        self.genotype = genotype
        self.phenotype = phenotype

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
        pass

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

class genetic_algorithm:
    
    def __init__(self, type_of_selection, type_of_crossing, type_of_mutation, type_of_stop_condition, set_of_individuals = []):
        self.type_of_selection = type_of_selection
        self.type_of_crossing = type_of_crossing
        self.type_of_mutation = type_of_mutation
        self.type_of_stop_condition = type_of_stop_condition
        self.set_of_individuals = set_of_individuals

    def algorithm(self):
        pass

if __name__ == '__main__':
    pop = population(200)
    set_of_individuals = pop.initialize_population()
    sel = selection('random', set_of_individuals)
    



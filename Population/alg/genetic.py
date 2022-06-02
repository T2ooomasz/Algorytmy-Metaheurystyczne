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

    def __init__(self, set_of_individuals = []):
        self.set_of_individuals = set_of_individuals

    def random_selection(self):
        pass

    def roulette_selection(self):
        pass

    def tournament_selection(self):
        pass

class crossing:

    def __init__(self, selected_individuals = []):
        self.selected_individuals = selected_individuals

    def half_crossover(self):
        pass

    def order_crossover(self):
        pass

    def partially_mapped_crossover(self):
        pass

def genetic_algorithm():
    pass

if __name__ == '__main__':
    pop = population(200)
    set_of_individuals = pop.initialize_population()
    



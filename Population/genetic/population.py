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

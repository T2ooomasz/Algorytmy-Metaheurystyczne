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

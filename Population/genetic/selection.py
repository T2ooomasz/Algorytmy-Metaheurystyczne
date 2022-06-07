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

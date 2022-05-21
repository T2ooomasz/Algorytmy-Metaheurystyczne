
class mutation():
    def __init__(self, population, type_of_mutation):
        self.population = population
        self.type_of_mutation = type_of_mutation
        self.mutation_alg()
    
    def mutation_alg(self):
        if(self.type_of_mutation == 0):
            self.mutation_0(self.population)
        elif(self.type_of_mutation == 1):
            self.mutation_1(self.population)

    def mutation_0(self):
        pass

    def mutation_1(self):
        pass

class alg():
    mutation(populacja, 0)
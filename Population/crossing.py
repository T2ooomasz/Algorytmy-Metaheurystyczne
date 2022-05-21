
class crossing():
    def __init__(self, population, type_of_crossing):
        self.population = population
        self.type_of_crossing = type_of_crossing

    def crossing_alg(self):
        if(self.type_of_crossign == 'KLM'):
            self.crossing_KLM(self.population)
        elif(self.type_of_mutation == 'RPD'):
            self.crossing_RPD(self.population)

    def crossing_KLM(self):
        pass

    def crossing_RPD(self):
        pass
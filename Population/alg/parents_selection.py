'''
Class parent selection

take: population, type of selection
return: parents
'''

class parents_selection():
    def __init__(self, population, type_of_selection):
        self.population = population
        self.type_of_selection = type_of_selection
        self.selection_alg()

    def selection_alg(self):
        if self.type_of_selection == "random":
            self.selection_random()
        elif self.type_of_selection == "roulette":
            self.selection_roulette()
        elif self.type_of_selection == "tournament":
            self.selection_tournament()
        else:
            print("Working, but there is no \"", self.type_of_selection, "\" type of selection" )

    def selection_random(self):
        pass

    def selection_roulette(self):
        pass

    def selection_tournament(self):
        pass

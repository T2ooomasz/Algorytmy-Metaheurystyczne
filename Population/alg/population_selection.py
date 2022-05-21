'''
Class population selection

take: population, type of selection
return: valid population
'''

class population_selection():
    def __init__(self, population, childrens, type_of_selection):
        self.population = population
        self.childrens =childrens
        self.type_of_selection = type_of_selection
        self.selection_alg()

    def selection_alg(self):
        if self.type_of_selection == "0":
            self.selection_0()
        elif self.type_of_selection == "1":
            self.selection_1()
        else:
            print("Working, but there is no \"", self.type_of_selection, "\" type of selection" )

    def selection_0(self):
        print("selection 0 works")
        pass

    def selection_1(self):
        print("selection 1 works")
        pass
'''
Class (? may be class isn't the best idea) crossing

Takes parameters required for crossing:
Parents & type of crossing
return childrens 
'''
class crossing():
    def __init__(self, parents, type_of_crossing):
        self.parents = parents
        self.type_of_crossing = type_of_crossing
        self.crossing_alg()

    def crossing_alg(self):
        if(self.type_of_crossing == 'HX'):
            self.crossing_HX()
        elif(self.type_of_crossing == 'OX'):
            self.crossing_OX()
        elif(self.type_of_crossing == 'PMX'):
            self.crossing_PMX()
        else:
            print("Working, but there is no \"", self.type_of_crossing, "\" type of crossing" )

    def crossing_HX(self):
        pass

    def crossing_OX(self):
        pass

    def crossing_PMX(self):
        pass
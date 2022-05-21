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
        if(self.type_of_crossing == 'KLM'):
            self.crossing_KLM()
        elif(self.type_of_crossing == 'RPD'):
            self.crossing_RPD()
        else:
            print("Working, but there is no \"", self.type_of_crossing, "\" type of crossing" )

    def crossing_KLM(self):
        print("\"KLM\" works")

    def crossing_RPD(self):
        print("\"RPD\" works")
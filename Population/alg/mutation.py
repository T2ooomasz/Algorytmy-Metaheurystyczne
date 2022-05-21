'''
Class (? may be class isn't the best idea) mutation

Takes parameters required for mutation:
childrens & type of mutation
return childrens 
'''
class mutation():
    def __init__(self, childrens, type_of_mutation):
        self.childrens = childrens
        self.type_of_mutation = type_of_mutation
        self.mutation_alg()
    
    def mutation_alg(self):
        if(self.type_of_mutation == "0"):
            self.mutation_0()
        elif(self.type_of_mutation == "1"):
            self.mutation_1()
        else:
            print("Working, but there is no \"", self.type_of_mutation, "\" type of mutation" )

    def mutation_0(self):
        print("mutation 0 works")

    def mutation_1(self):
        print("mutation 1 works")
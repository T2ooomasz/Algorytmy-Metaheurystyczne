class genetic_algorithm:
    
    def __init__(self, type_of_selection, type_of_selection_population, type_of_crossing, parents = []):
        self.type_of_selection = type_of_selection
        self.type_of_selection_population = type_of_selection_population
        self.type_of_crossing = type_of_crossing
        #self.type_of_mutation = type_of_mutation
        #self.type_of_stop_condition = type_of_stop_condition
        #self.number_for_stop = number_for_stop
        self.parents = parents
        self.algorithm()

    def algorithm(self):
        #stop = stop_condition(self.type_of_stop_condition)
        #while(stop.stop_condition == False):
        for _ in range(40000):
            sel = selection(self.type_of_selection, self.parents)
            fathers = sel.selected_fathers
            mothers = sel.selected_mothers
            cro = crossing(self.type_of_crossing, fathers, mothers)
            children = cro.children
            sel_pop = selection_population(self.type_of_selection_population, self.parents, children)
            self.parents = sel_pop.new_population
            #stop.update_stop_condition()
        return self.parents

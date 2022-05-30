'''
Class (? may be class isn't the best idea) crossing

Takes parameters required for crossing:
Parents & type of crossing
return childrens 
'''
import numpy as np


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
            print("There is no \"", self.type_of_crossing, "\" type of crossing" )

    def crossing_HX(self):
        pass

    def crossing_OX(self):
        pass

    def crossing_PMX(self):
        '''
        for [parent1, parent2] in Parents:
            Child1, Child2 = PMX(Parent1, Parent2, cut_points)
            childrens.append(Child1, Child2)

        return childrens
        '''
        pass

    def PMX(self, Parent1, Parent2, cut_points):  # para rodziców - może być inaczej przekazana
        Child1 = np.zeros(8, dtype=int)
        Child2 = np.zeros(8, dtype=int)
        Child1[cut_points[0]:cut_points[1]] = Parent2[cut_points[0]:cut_points[1]]
        Child2[cut_points[0]:cut_points[1]] = Parent1[cut_points[0]:cut_points[1]]
        Child1 = self.PMX_cross(Parent1, Child1, cut_points)
        CHild2 = self.PMX_cross(Parent2, Child2, cut_points)
        return Child1, Child2


    def PMX_cross(self, Parent, Child, cut_points):
        Child = self.filling(Parent, Child, cut_points)
        Child = self.filling_mapping(Parent, Child, cut_points)
        return Child

    def mapping_(self, Parent, Child, numb):
        if numb in Child:
            j = 0
            while Child[j] != numb:
                j += 1
            numb = self.mapping_(Parent, Child, Parent[j])
        return numb

    def filling(self, Parent, Child, cut_points):
        for i in range(0, cut_points[0]):
            if Parent[i] not in Child:
                Child[i] = Parent[i]
        for i in range(cut_points[1], len(Parent)):
            if Parent[i] not in Child:
                Child[i] = Parent[i]
        return Child

    def filling_mapping(self, Parent, Child, cut_points):
        for i in range(cut_points[0]):
            if Parent[i] in Child and Parent[i] != Child[i]:
                j = 0
                while Child[j] != Parent[i]:
                    j += 1
                Child[i] = self.mapping_(Parent, Child, Parent[i])
        for i in range(cut_points[1], len(Parent)):
            if Parent[i] in Child and Parent[i] != Child[i]:
                j = 0
                while Child[j] != Parent[i]:
                    j += 1
                Child[i] = self.mapping_(Parent, Child, Parent[i])
        return Child

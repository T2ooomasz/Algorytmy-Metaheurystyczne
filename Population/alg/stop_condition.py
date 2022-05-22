'''
Class stop condition

take: type of stop condition
return: stop condition True of False
'''
import time

class stop_condition():
    # class atribute
    stop_condition = False
    time_start = time.time() # time of working algorithm
    iterations = 0
    iteration_without_update = 0

    def __init__(self, type_of_stop_condition, number):
        self.type_of_stop_condition = type_of_stop_condition
        self.number = number

    def update_stop_condition(self):
        self.selection_stop_cond()

    def selection_stop_cond(self):
        if self.type_of_stop_condition == "0":
            self.stop_condition_0()
        elif self.type_of_stop_condition == "1":
            self.stop_condition_1()
        elif self.type_of_stop_condition == "2":
            self.stop_condition_2()
        else:
            print("Working, but there is no \"", self.type_of_stop_condition, "\" type of stop_condition" )

    def stop_condition_0(self):
        if self.iteration_without_update > self.number:
            self.stop_condition = True
        self.iteration_without_update += 1
        print("stop_condition 0 works")
        

    def stop_condition_1(self):
        self.iterations += 1
        if self.iterations >= self.number:
            self.stop_condition = True
        print("stop_condition 1 works")

    def stop_condition_2(self):
        end = time.time()
        if (end - self.time_start) > self.number:
            self.stop_condition = True
        print("stop_condition 2 works")
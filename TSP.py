import tsplib95
import sys
import random

def fill_matrix(sizeTab, matr):

    for i in range(0, sizeTab):
        for j in range(0, sizeTab):
            edge = i, j
            matr[i][j] = problem.get_weight(*edge)


def destination(sizeTab, matr):
    weight = 0
    for i in range(0, int(sizeTab) - 1):
        weight += matr[tour[i]][tour[i + 1]]
    weight += matr[tour[int(sizeTab) - 1]][tour[0]]
    print(weight)

problem = tsplib95.load('C:/Users/piotr/Desktop/Meta/ft53.atsp')
problem2 = tsplib95.load('C:/Users/piotr/Desktop/Meta/brg180.opt.tour')
k = problem.is_full_matrix()
zmienna = list(problem.get_nodes())
sizeTab = len(zmienna)
print(sizeTab)
matr = [[0 for _ in range(sizeTab)] for _ in range(sizeTab)]
if not k:
    if not problem.is_explicit():

        tour = [0 for j in range(int(sizeTab))]
        for i in range(0, int(sizeTab)):
            tour[i] = i
        random.shuffle(tour)
        tour = list(problem2.tours[0])
        print(tour)

        for i in range(0, sizeTab):
            for j in range(0, sizeTab):
                edge = i + 1, j + 1
                matr[i][j] = problem.get_weight(*edge)
        weight = 0
        for i in range(0, int(sizeTab) - 1):
            weight += matr[tour[i] - 1][tour[i + 1] - 1]
        weight += matr[tour[int(sizeTab) - 1] - 1][tour[0] - 1]
        print(weight)
    else:

        tour = [0 for j in range(int(sizeTab))]
        for i in range(0, int(sizeTab)):
            tour[i] = i
        random.shuffle(tour)
        tour = list(problem2.tours[0])
        for i in range(0, int(sizeTab)):
            tour[i] = tour[i] - 1
        print(tour)
        fill_matrix(sizeTab, matr)
        destination(sizeTab, matr)
else:

    tour = [0 for j in range(int(sizeTab))]
    for i in range(0, int(sizeTab)):
        tour[i] = i
    random.shuffle(tour)
    # tour = list(problem2.tours[0])
    print(tour)
    fill_matrix(sizeTab, matr)
    destination(sizeTab, matr)

matrix = [[0, 45, 16, 123], [124, 0, 34, 24], [12, 90, 0, 13], [1, 234, 122, 0]]
tour = [0, 2, 1, 3]
destination(4, matrix)
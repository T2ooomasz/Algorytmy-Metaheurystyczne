import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv
from mpl_toolkits import mplot3d

def loadDATA(arr):
    loaded_arr = np.loadtxt("simulation/data_rand.csv")
    load_original_arr = np.reshape(loaded_arr, arr)
    #print("shape of arr: ", loaded_arr.shape)
    return load_original_arr

def plot(arr):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x = np.array([33,44,55,64,70,170])
    #x = np.array([17,21,24,48,52,76,99,107])
    min_tabu_lenght = 2
    max_tabu_lenght = 25
    step_tabu = 5
    y = np.arange(min_tabu_lenght, max_tabu_lenght + step_tabu, step_tabu).tolist()
    print(x)
    print('-----------')
    print(y)
    X, Y = np.meshgrid(x,y)
    z = np.zeros((arr.shape[1],arr.shape[0]))
    for i in range(len(x)):
        for j in range(len(y)):
            print('i,j = ', i,j)
            z[j][i] = arr[i,j,0,0]
            print(z)
    
    
    Z = np.array(z)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    #ax.contour3D(Xx, Yy, Zz, 50, cmap='binary')
    ax.set_title('surface')
    ax.set_xlabel('instance')
    ax.set_ylabel('tabu list size')
    ax.set_zlabel('cost')
    formatter = mticker.ScalarFormatter()
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(mticker.FixedLocator(x))
    ax.yaxis.set_major_locator(mticker.FixedLocator(y))
    plt.show()
    #plt.savefig('simulation/chart1.png')

def scatter(arr):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #arr = np.random.rand(4, 4, 2, 2)
    X = arr.shape[0]
    Y = arr.shape[1]
    Z = arr.shape[2]
    x = []
    y = []
    z = []
    for i in range(X):
        x.append(arr[i,0,0,0])
    for j in range(Y):
        y.append(arr[0,j,0,0])
    for k in range(Z):
        z.append(arr[0,0,k,0])
    Xx = np.array(x)
    Yy = np.array(y)
    Zz = np.array(z)
    ax = plt.axes(projection='3d')
    ax.scatter3D(Xx, Yy, Zz, c=Zz, cmap='Greens')
    ax.set_title('surface')
    ax.set_xlabel('instance')
    ax.set_ylabel('tabu list size')
    ax.set_zlabel('cost')
    plt.show()
    #plt.savefig('simulation/chart1.png')

def main():
    arr = [6,6,4,2]
    #arr = [8,2,1,2]
    DATA = loadDATA(arr)
    plot(DATA)

if __name__ == '__main__':
    main()
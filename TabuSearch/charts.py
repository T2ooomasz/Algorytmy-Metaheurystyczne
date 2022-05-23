import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv
from mpl_toolkits import mplot3d

def loadDATA(arr):
    loaded_arr = np.loadtxt("simulation/atsp_data_nne_3.csv")
    load_original_arr = np.reshape(loaded_arr, arr)
    #print("shape of arr: ", loaded_arr.shape)
    return load_original_arr

def plot(arr):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    #x = np.array([1286]) #['gr24.tsp', 'gr48.tsp', 'pr76.tsp', 'pr107.tsp', 'gr120.tsp', 'pr136.tsp', 'pr152.tsp'] 24,48,76,107,120,136,152
    #best_known = np.array([1272, 5046, 108159, 44303, 6942, 96772, 73682])
    #x = np.array([17,21,24,48,52,76,99,107])
    #x = np.array([33, 44, 55, 64, 70, 170])#['ftv33.atsp', 'ftv44.atsp', 'ftv55.atsp', 'ftv64.atsp', 'ftv70.atsp', 'ftv170.atsp']
    #best_known = np.array([1286, 1613, 1608, 1839, 1950, 2755])
    x = np.array([17, 33, 44, 55, 64, 70]) #['br17.atsp', 'ftv33.atsp', 'ftv44.atsp', 'ftv55.atsp', 'ftv64.atsp', 'ftv70.atsp']
    best_known = np.array([39, 1286, 1613, 1608, 1839, 1950])
    min_tabu_lenght = 2  # 2
    max_tabu_lenght = 37  # 37
    step_tabu = 5    #5
    y = np.arange(min_tabu_lenght, max_tabu_lenght + step_tabu, step_tabu).tolist()
    #print(x)
    #print('-----------')
    #print(y)
    X, Y = np.meshgrid(x,y)
    z = np.zeros((arr.shape[1],arr.shape[0]))
    for i in range(len(x)):
        for j in range(len(y)):
            #print('i,j = ', i,j)
            #z[j][i] = (arr[i,j,1,0] / best_known[i] - 1) * 100 #prd
            z[j][i] = arr[i,j,1,1] #time
            #print(z)
    
    
    Z = np.array(z)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    #ax.contour3D(Xx, Yy, Zz, 50, cmap='binary')
    ax.set_title('NNE rozwiązanie początkowe')
    ax.set_xlabel('instance size')
    ax.set_ylabel('tabu list size')
    ax.set_zlabel('time [s]')
    #ax.set_zlabel('PRD [%]')
    formatter = mticker.ScalarFormatter()
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(mticker.FixedLocator(x))
    ax.yaxis.set_major_locator(mticker.FixedLocator(y))

    '''
    xline = np.linspace(24, 152, 1000)
    yline = xline - xline
    zline = 0.0000005*np.power(xline, 4)
    ax.plot3D(xline, yline, zline, 'red')
    '''
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
    
    arr = [6,8,2,2]     # for atsp 1
    #arr = [6,6,4,2]
    #arr = [7,8,2,2]
    DATA = loadDATA(arr)
    plot(DATA)

if __name__ == '__main__':
    main()
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

with open('simulation-k-random/atsp-k-random1.csv','r') as k_random: #Opens the file in read mode
    csv_reader = csv.reader(k_random, quoting=csv.QUOTE_NONNUMERIC) # Making use of reader method for reading the file
 
    Y_k = np.array([])
    for line in csv_reader: #Iterate through the loop to read line by line
        Y_k = np.append(Y_k, np.array([line]))

with open('simulation-k-random/atsp-k-random-time1.csv','r') as k_random_t: #Opens the file in read mode
    csv_reader = csv.reader(k_random_t, quoting=csv.QUOTE_NONNUMERIC) # Making use of reader method for reading the file
 
    Y_k_t = np.array([])
    for line in csv_reader: #Iterate through the loop to read line by line
        Y_k_t = np.append(Y_k_t, np.array([line]))

with open('simulation-nearest_neighbour/atsp-nearest_neighbour1.csv','r') as nn: #Opens the file in read mode
    csv_reader = csv.reader(nn, quoting=csv.QUOTE_NONNUMERIC) # Making use of reader method for reading the file
 
    Y_nn = np.array([])
    for line in csv_reader: #Iterate through the loop to read line by line
        Y_nn = np.append(Y_nn, np.array([line]))

with open('simulation-nearest_neighbour/atsp-nearest_neighbour-time1.csv','r') as nn_t: #Opens the file in read mode
    csv_reader = csv.reader(nn_t, quoting=csv.QUOTE_NONNUMERIC) # Making use of reader method for reading the file
 
    Y_nn_t = np.array([])
    for line in csv_reader: #Iterate through the loop to read line by line
        Y_nn_t = np.append(Y_nn_t, np.array([line]))

with open('simulation-nearest_neighbour_extended/atsp-nearest_neighbour_extended1.csv','r') as nne: #Opens the file in read mode
    csv_reader = csv.reader(nne, quoting=csv.QUOTE_NONNUMERIC) # Making use of reader method for reading the file
 
    Y_nne = np.array([])
    for line in csv_reader: #Iterate through the loop to read line by line
        Y_nne = np.append(Y_nne, np.array([line]))

with open('simulation-nearest_neighbour_extended/atsp-nearest_neighbour_extended-time1.csv','r') as nne_t: #Opens the file in read mode
    csv_reader = csv.reader(nne_t, quoting=csv.QUOTE_NONNUMERIC) # Making use of reader method for reading the file
 
    Y_nne_t = np.array([])
    for line in csv_reader: #Iterate through the loop to read line by line
        Y_nne_t = np.append(Y_nne_t, np.array([line]))

with open('simulation-nearest_swap_neighbour/atsp-nearest_swap_neighbour1.csv','r') as nsw: #Opens the file in read mode
    csv_reader = csv.reader(nsw, quoting=csv.QUOTE_NONNUMERIC) # Making use of reader method for reading the file
 
    Y_nsw = np.array([])
    for line in csv_reader: #Iterate through the loop to read line by line
        Y_nsw = np.append(Y_nsw, np.array([line]))

with open('simulation-nearest_swap_neighbour/atspnearest_swap_neighbour-time1.csv','r') as nsw_t: #Opens the file in read mode
    csv_reader = csv.reader(nsw_t, quoting=csv.QUOTE_NONNUMERIC) # Making use of reader method for reading the file
 
    Y_nsw_t = np.array([])
    for line in csv_reader: #Iterate through the loop to read line by line
        Y_nsw_t = np.append(Y_nsw_t, np.array([line]))

with open('simulation-2-opt/atsp-2-opt1.csv','r') as opt2: #Opens the file in read mode
    csv_reader = csv.reader(opt2, quoting=csv.QUOTE_NONNUMERIC) # Making use of reader method for reading the file
 
    Y_opt2 = np.array([])
    for line in csv_reader: #Iterate through the loop to read line by line
        Y_opt2 = np.append(Y_opt2, np.array([line]))

with open('simulation-2-opt/atsp-2-opt-time1.csv','r') as opt2_t: #Opens the file in read mode
    csv_reader = csv.reader(opt2_t, quoting=csv.QUOTE_NONNUMERIC) # Making use of reader method for reading the file
 
    Y_opt2_t = np.array([])
    for line in csv_reader: #Iterate through the loop to read line by line
        Y_opt2_t = np.append(Y_opt2_t, np.array([line]))

with open('simulation-2_opt_full/atsp-2_opt_full1.csv','r') as opt2f: #Opens the file in read mode
    csv_reader = csv.reader(opt2f, quoting=csv.QUOTE_NONNUMERIC) # Making use of reader method for reading the file
 
    Y_opt2f = np.array([])
    for line in csv_reader: #Iterate through the loop to read line by line
        Y_opt2f = np.append(Y_opt2f, np.array([line]))

with open('simulation-2_opt_full/atsp-2_opt_full-time1.csv','r') as opt2f_t: #Opens the file in read mode
    csv_reader = csv.reader(opt2f_t, quoting=csv.QUOTE_NONNUMERIC) # Making use of reader method for reading the file
 
    Y_opt2f_t = np.array([])
    for line in csv_reader: #Iterate through the loop to read line by line
        Y_opt2f_t = np.append(Y_opt2f_t, np.array([line]))




# X =  np.array([10,20,40,80,160,320,640,1280])
X = np.array([33, 55, 100, 170, 323, 443])
#Best = np.array([1032, 2140, 4150, 8236, 16182, 32167, 64123, 128117])
Best = np.array([1286, 1608, 1788, 2755, 1326, 2720])
Y = Y_opt2f
#Y = Y_2opt
Y_k = Y_k - Best
Y_k = (Y_k / Best )* 100
Y_nn = Y_nn - Best
Y_nn = (Y_nn / Best )* 100
Y_nne = Y_nne - Best
Y_nne = (Y_nne / Best )* 100
Y_nsw = Y_nsw - Best
Y_nsw = (Y_nsw / Best )* 100
Y_opt2 = Y_opt2 - Best
Y_opt2 = (Y_opt2 / Best )* 100
Y_opt2f = Y_opt2f - Best
Y_opt2f = (Y_opt2f / Best )* 100
Y2 = Y_opt2f_t

'''
#plt.plot(X, Y_k, 'o')
plt.plot(X, Y_nn, 'x', label='Nearest Neighbour')
plt.plot(X, Y_nne, 'd', label='Nearest Neighbour Extended')
plt.plot(X, Y_nsw, 'o', label='Nearest SWAP Neighbour')
plt.plot(X, Y_opt2, 'x', label='2 opt')
plt.plot(X, Y_opt2f, 'd', label='2 opt full')
plt.xlabel("Liczba miast")
plt.ylabel("PRD [%}")
plt.legend()
string = 'obrazki/atsp-together'
plt.savefig(string)
#plt.show()

'''
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Liczba miast. [Instancja symetryczna prXXX].')
ax1.set_ylabel('PRD [%]', color=color)
ax1.plot(X, Y_opt2f, 'o', color=color)
ax1.tick_params(axis='y', labelcolor=color)
# ax1.set_xscale('log')
formatter = mticker.ScalarFormatter()
ax1.xaxis.set_major_formatter(formatter)
ax1.xaxis.set_major_locator(mticker.FixedLocator(X))

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('time [s]', color=color)  # we already handled the x-label with ax1
ax2.plot(X, Y_opt2f_t, 'x', color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('2 opt full')
string = "obrazki/atsp-2optf1"
# string += str(1)
string += ".png"
for ax in fig.get_axes():
    ax.label_outer()
plt.savefig(string)

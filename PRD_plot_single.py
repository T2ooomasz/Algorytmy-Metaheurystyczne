import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

with open('data-2-opt/2-opt999.csv','r') as csv_file1: #Opens the file in read mode
    csv_reader = csv.reader(csv_file1, quoting=csv.QUOTE_NONNUMERIC) # Making use of reader method for reading the file
 
    Y_k = np.array([])
    for line in csv_reader: #Iterate through the loop to read line by line
        Y_k = np.append(Y_k, np.array([line]))

with open('data-2-opt/sym-2-opt1280.csv','r') as csv_file2: #Opens the file in read mode
    csv_reader = csv.reader(csv_file2, quoting=csv.QUOTE_NONNUMERIC) # Making use of reader method for reading the file
 
    Y_2opt = np.array([])
    for line in csv_reader: #Iterate through the loop to read line by line
        Y_2opt = np.append(Y_2opt, np.array([line]))

with open('data-2-opt/2-opt-time999.csv','r') as csv_file3: #Opens the file in read mode
    csv_reader = csv.reader(csv_file3, quoting=csv.QUOTE_NONNUMERIC) # Making use of reader method for reading the file
 
    Y2 = np.array([])
    for line in csv_reader: #Iterate through the loop to read line by line
        Y2 = np.append(Y2, np.array([line]))

X =  np.array([10,20,40,80,160,320,640,1280])
# X = np.array([10,20,40,80,160,320])
Best = 7542
# Best = np.array([1032, 2140, 4150, 8236, 16182, 32167])
# Y = np.array([1383, 2873, 5617, 11660, 23216, 47156])
Y = Y_k
Y = Y - Best
Y = (Y / Best )* 100
# Y2 = np.array([0.00011110305786132812, 0.0007011890411376953, 0.004830121994018555, 0.037393808364868164, 0.28330016136169434, 2.2036406993865967])

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('iteracja')
ax1.set_ylabel('PRD [%]', color=color)
ax1.plot(Y, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('time [s]', color=color)  # we already handled the x-label with ax1
ax2.plot(Y2, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('2-opt')
string = "obrazki/2-opt-999"
# string += str(1)
string += ".png"
for ax in fig.get_axes():
    ax.label_outer()
plt.savefig(string)
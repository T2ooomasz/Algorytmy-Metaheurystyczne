import csv
import matplotlib.pyplot as plt
import numpy as np

with open('data-k-random/k-random100000.csv','r') as csv_file: #Opens the file in read mode
    csv_reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC) # Making use of reader method for reading the file
 
    X = np.array([])
    for line in csv_reader: #Iterate through the loop to read line by line
        X = np.append(X, np.array([line]))

best = 7542
X = ((X - best) / best )* 100
plt.plot(X, '-b')
plt.xlabel('iteration')
plt.ylabel('% PRD')
plt.title('Distance to iteration')
plt.grid(True)
string = "obrazki/figure"
string += str(1)
string += ".png"
plt.savefig(string)

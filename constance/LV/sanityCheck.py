import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

files = {1:'no_evs',2:'uncontrolled',3:'lf',4:'lm'}
plt.figure()
for i in range(1,5):
    plt.subplot(2,2,i)
    currents = []
    with open('lv test/'+files[i]+'.csv','rU') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            currents.append(float(row[1]))
    plt.plot(currents[1:])
plt.show()

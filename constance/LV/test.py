import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
#from cvxopt import matrix, spdiag, sparse, solvers

def red(f,upper,lower):
    f -= lower
    f = f/(upper-lower)
    f = int(f*255)
    f = str(hex(f)[2:])
    if len(f) == 1:
        f = '0'+f
    c = '#FF'+f+f

    #print(c)

    return c
buses = {}
with open('lv test/Buscoords.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    next(reader)
    for row in reader:
        buses[int(row[0])] = [float(row[1]),float(row[2])]


r0 = {}
with open('lv test/LineCodes.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    next(reader)
    for row in reader:
        r0[row[0]] = float(row[4])
    
lines = {}    
linesR = {}
linesL = {}
maxRL = 0
minRL = 10000
with open('lv test/Lines.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    next(reader)
    for row in reader:
        lines[row[0]] = [int(row[1]),int(row[2])]
        linesR[row[0]] = float(row[4])*r0[row[6]]
        linesL[row[0]] = float(row[4])
        if r0[row[6]] > maxRL:
            maxRL = r0[row[6]]
        if r0[row[6]] < minRL:
            minRL = r0[row[6]]

loads = []
with open('lv test/Loads.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    next(reader)
    next(reader)
    for row in reader:
        loads.append(int(row[2]))

plt.figure(figsize=(6,6))
for i in range(1,5):
    plt.subplot(2,2,i)
    for l in lines:
        a = lines[l][0]
        b = lines[l][1]
        x = [buses[a][0],buses[b][0]]
        y = [buses[a][1],buses[b][1]]
        plt.plot(x,y,c=red(linesR[l]/linesL[l],maxRL,minRL),
                 lw=linesR[l]/linesL[l])#'gray',#lw=linesR[l]/linesL[l])

    x = []
    y = []
    for l in loads:
        x.append(buses[l][0])
        y.append(buses[l][1])
    plt.scatter(x,y,c='gray')
    plt.xlim(390860,391030)
    plt.ylim(392740,392890)
    plt.xticks([390860,391030],['',''])
    plt.yticks([392740,392890],['',''])
    plt.title('Simulation '+str(i)) 
plt.tight_layout()
plt.show()

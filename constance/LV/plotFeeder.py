import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
        lines[row[0][4:]] = [int(row[1]),int(row[2])]
        linesR[row[0][4:]] = float(row[4])*r0[row[6]]
        linesL[row[0][4:]] = float(row[4])
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

fig,ax = plt.subplots(1,figsize=(5,4.5))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 9
for l in lines:
    a = lines[l][0]
    b = lines[l][1]
    x = [buses[a][0],buses[b][0]]
    y = [buses[a][1],buses[b][1]]
    plt.plot(x,y,lw=1,c='k')

x = []
y = []
for l in loads:
    x.append(buses[l][0])
    y.append(buses[l][1])
plt.scatter(x,y,c='gray',label='Households')
plt.xlim(390860,391030)
plt.ylim(392740,392890)
plt.xticks([390860,391030],['',''])
plt.yticks([392740,392890],['',''])
plt.scatter([buses[1][0]],[buses[1][1]],c='r',label='Substation')
plt.plot([390990,391000],[392760,392760],c='k')
plt.annotate('10 m',(391004,392758.5))
plt.annotate('Length Scale',(390988,392769))
plt.legend()
ax.axis('off')
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/losses/img/test_feeder.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0)
plt.show()

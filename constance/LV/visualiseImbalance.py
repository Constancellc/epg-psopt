import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib import cm
#from cvxopt import matrix, spdiag, sparse, solvers


# first get phases
lds = np.load('../../../Documents/ccModels/loadBuses/eulvLptloadBusesCc-24.npy')
lds = lds.flatten()[0]

phase = []
for i in range(len(lds)):
    bus = lds['load'+str(i+1)]
    if bus[-1] == '1':
        phase.append('A')
    elif bus[-1] == '2':
        phase.append('B')
    elif bus[-1] == '3':
        phase.append('C')

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

losses = {}
with open('lv test/branch_2.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        losses[row[0][4:]] = [float(row[1]),float(row[2]),float(row[3]),
                              float(row[4]),float(row[5])]
with open('lv test/branch_1.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        for i in range(1,6):
            if losses[row[0][4:]][i-1] < 1:
                losses[row[0][4:]][i-1] = 1
            else:
                losses[row[0][4:]][i-1] = losses[row[0][4:]][i-1]/float(row[i])


titles = ['Uncontrolled','Load Flatttening','Loss Minimising',
          'LF+Phase Balancing']
fig, ax = plt.subplots(1,figsize=(6,5))
#f, axs = plt.subplots(2, 2)

pm = {1:4,2:1,3:2,4:3}
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 9
mi = 10
#sps = [axs[0,0],axs[0,1],axs[1,0],axs[1,1]]
for i in range(4):
    sp = fig.add_subplot(2,2,pm[i+1])#sps[i]
    #plt.subplot(2,2,i)
    sp.set_title(titles[i-1])
    for l in lines:
        a = lines[l][0]
        b = lines[l][1]
        x = [buses[a][0],buses[b][0]]
        y = [buses[a][1],buses[b][1]]
        try:
            lpd = losses[l][i-1]
        except:
            continue

        if lpd < mi:
            mi = lpd
        if lpd > 0.85:
            sp.plot(x,y,c='gray',lw=1,zorder=0.5)
        elif lpd > 0.5:
            sp.plot(x,y,c=cm.viridis(0),lw=4,zorder=1)
        elif lpd < 0.1:
            sp.plot(x,y,c=cm.viridis(1),lw=4,zorder=2)
        else:
            sp.plot(x,y,c=cm.viridis(1-(lpd-0.1)*2.5),lw=4)
        
    x = {'A':[],'B':[],'C':[]}
    y = {'A':[],'B':[],'C':[]}


    for i in range(len(loads)):
        l = loads[i]
        p = phase[i]
        x[p].append(buses[l][0])
        y[p].append(buses[l][1])

    sp.scatter(x['A'],y['A'],10,c='b')
    sp.scatter(x['B'],y['B'],10,c='r')
    sp.scatter(x['C'],y['C'],10,c='g')

    sp.set_xlim(390860,391030)
    sp.set_ylim(392740,392890)
    sp.set_xticks([390860,391030],['',''])
    sp.set_yticks([392740,392890],['',''])
    sp.axis('off')
#plt.tight_layout()
print(mi)
top = 0.95
btm = 0.05

for i in range(100):
    y1 = btm+(top-btm)*(i/100)
    y2 = btm+(top-btm)*((i+1)/100)
    ax.plot([11,11],[y1,y2],lw=6,c=cm.viridis(i/100))
rect = pat.Rectangle((4.5,0.52),2,0.165,facecolor='w',edgecolor='gray',zorder=0.5)
ax.add_patch(rect)
ax.scatter([4.8],[0.65],10,c='b',zorder=2)
ax.scatter([4.8],[0.6],10,c='r',zorder=2)
ax.scatter([4.8],[0.55],10,c='g',zorder=2)
ax.annotate('Phase A',(5.1,0.64))
ax.annotate('Phase B',(5.1,0.59))
ax.annotate('Phase C',(5.1,0.54))
tcks = ['10%','20%','30%','40%','50%']
for i in range(5):
    y_ = btm+(top-btm)*(i/4)-0.01
    ax.annotate('- '+tcks[4-i],(11,y_))
#ax.plot([1,1],[1,0])
ax.set_xlim(0,11)
ax.set_ylim(0,1)
ax.axis('off')

plt.savefig('../../../Dropbox/papers/losses/img/network_loss_map.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0)

 
plt.show()       
'''

fig,ax = plt.subplots(1,figsize=(5.5,3.5))

for l in lines:
    a = lines[l][0]
    b = lines[l][1]
    x = [buses[a][0],buses[b][0]]
    y = [buses[a][1],buses[b][1]]
    lpd = 10*(losses[l][2]-losses[l][3])/losses[l][2]
    
    if lpd <= 0:
        plt.plot(x,y,lw=1,c='r')
    else:
        if lpd < 1:
            plt.plot(x,y,lw=1,c='b')
        else:
            plt.plot(x,y,lw=lpd,c='b')

x = []
y = []
for l in loads:
    x.append(buses[l][0])
    y.append(buses[l][1])
rect1 = patches.Rectangle((391040,392850),40,4.6,linewidth=0.1,edgecolor='b',facecolor='b')
rect2 = patches.Rectangle((391040,392830),40,2.4,linewidth=0.1,edgecolor='b',facecolor='b')
rect3 = patches.Rectangle((391040,392810),40,0.8,linewidth=0.1,edgecolor='b',facecolor='b')
rect4 = patches.Rectangle((391040,392790),40,0.8,linewidth=0.1,edgecolor='r',facecolor='r')

ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.axis('off')
plt.text(391085,392849,'200 W/m')
plt.text(391085,392829,'100 W/m')
plt.text(391085,392809,'50 W/m')
plt.text(391085,392789,'-50 W/m')

plt.scatter(x,y,c='gray')
plt.xlim(390860,391120)
plt.ylim(392740,392890)
plt.xticks([390860,391030],['',''])
plt.yticks([392740,392890],['',''])
plt.tight_layout()
#plt.savefig('../../../Dropbox/papers/losses/img/network_loss_map2.eps', format='eps',
#            dpi=1000, bbox_inches='tight', pad_inches=0)
plt.show()
titles = ['Load Flattening','Loss Minimizing']
plt.figure(figsize=(6,3))
for i in range(1,3):
    plt.subplot(1,2,i)
    plt.title(titles[i-1])
    for l in lines:
        a = lines[l][0]
        b = lines[l][1]
        x = [buses[a][0],buses[b][0]]
        y = [buses[a][1],buses[b][1]]
        if losses[l][1] < losses[l][i+1]:
            plt.plot(x,y,lw=0.5,c='r')
            continue
        lpd = np.log(0.1*(losses[l][1]-losses[l][i+1])/linesL[l])
        if lpd < 0.5:
            plt.plot(x,y,lw=0.5,c='b')
        else:
            plt.plot(x,y,lw=lpd,c='b')
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
plt.tight_layout()
plt.show()
'''

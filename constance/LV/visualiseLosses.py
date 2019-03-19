import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
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
with open('lv test/branch_losses.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        losses[row[0][4:]] = [float(row[1]),float(row[2]),float(row[3]),
                              float(row[4]),float(row[5])]


titles = ['No EVs','Uncontrolled']
plt.subplots(1,figsize=(6,3))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 9
for i in range(1,3):
    plt.subplot(1,2,i)
    plt.title(titles[i-1])
    for l in lines:
        a = lines[l][0]
        b = lines[l][1]
        x = [buses[a][0],buses[b][0]]
        y = [buses[a][1],buses[b][1]]
        lpd = np.log(0.05*losses[l][i-1]/linesL[l])
        if lpd < 0.1:
            plt.plot(x,y,lw=0.1,c='r')
        else:
            plt.plot(x,y,lw=lpd,c='r')

        '''
        else:
            lpd = (losses[l][1]-losses[l][i-1])/linesL[l]
        if lpd > 5:
            lpd = 5
        #lpd = np.log(0.01*losses[l][i-1]/linesL[l])
        if lpd < 0:
            if lpd < -5:
                lpd = -5
            plt.plot(x,y,lw=-lpd,c='b')
        else:
            plt.plot(x,y,lw=lpd,c='r')#c=red(losses,0,3))#,lw=linesR[l]/linesL[l])#'gray',#lw=linesR[l]/linesL[l])
        '''
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
    plt.axis('off')
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/losses/img/network_loss_map.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0)


fig,ax = plt.subplots(1,figsize=(4.8,3.5))
lpd = {}
lmax = 0
lmin = 0

for l in lines:
    lpd[l] = (losses[l][4]-losses[l][3])/linesL[l]
    if lpd[l] > lmax:
        lmax = lpd[l]
    if lpd[l] < lmin:
        lmin = lpd[l]

lower = -200
upper = 200
for l in lines:
    ls = lpd[l]
    if ls < lower:
        ls = lower
    elif ls > upper:
        ls = upper

    if ls < 0:
        ls = ls/lower
        ls = 0.5-ls*0.5
        if ls < 0:
            ls = 0
    else:
        ls = ls/upper
        ls = 0.5+ls*0.5
        if ls > 1:
            ls = 1
    
    '''
    lpd[l] = (lpd[l]+50)/100
    if lpd[l] > 1:
        lpd[l] = 1
    if lpd[l] < 0:
        lpd[l] = 0'''
    a = lines[l][0]
    b = lines[l][1]
    x = [buses[a][0],buses[b][0]]
    y = [buses[a][1],buses[b][1]]
    plt.plot(x,y,c=cm.coolwarm(ls),lw=2)
    '''lpd = (losses[l][4]-losses[l][3])/linesL[l]
    
    if lpd <= 0:
        plt.plot(x,y,lw=1,c='r')
    else:
        lpd = np.log(lpd)
        if lpd < 1:
            plt.plot(x,y,lw=1,c='b')
        else:
            plt.plot(x,y,lw=lpd,c='b')'''

x = []
y = []
for l in loads:
    x.append(buses[l][0])
    y.append(buses[l][1])
'''
rect1 = patches.Rectangle((391040,392850),40,4.6,linewidth=0.1,edgecolor='b',facecolor='b')
rect2 = patches.Rectangle((391040,392830),40,2.4,linewidth=0.1,edgecolor='b',facecolor='b')
rect3 = patches.Rectangle((391040,392810),40,0.8,linewidth=0.1,edgecolor='b',facecolor='b')
rect4 = patches.Rectangle((391040,392790),40,0.8,linewidth=0.1,edgecolor='r',facecolor='r')

ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
plt.text(391085,392849,'200 W/m')
plt.text(391085,392829,'100 W/m')
plt.text(391085,392809,'50 W/m')
plt.text(391085,392789,'-50 W/m')'''

ax.axis('off')
# now to make a colorscale

top = 392880
btm = 392750

for i in range(100):
    y1 = btm+(top-btm)*(i/100)
    y2 = btm+(top-btm)*((i+1)/100)
    plt.plot([391040,391040],[y1,y2],lw=6,c=cm.coolwarm(i/100))

tcks = [' -200',' -100','0','+100','+200']
for i in range(5):
    y_ = btm+(top-btm)*(i/4)-2
    plt.annotate('- '+tcks[i]+' W/m',(391042,y_))
plt.scatter(x,y,c='gray')
plt.xlim(390860,391080)
plt.ylim(392740,392890)
plt.xticks([390860,391030],['',''])
plt.yticks([392740,392890],['',''])
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/losses/img/network_loss_map2.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0)
plt.show()
        
'''
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
'''
plt.show()

import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random
import scipy.ndimage
#import win32com.client

# the goal here is to get a map of the losses resulting from two
# charging power combinations

step = 0.001 #
Pmax = 1.0 # max individual charging power
base = 0.0 # kW load of all unused houses


if step == 0.01:
    dec = 2
elif step == 0.001:
    dec = 3

hh_A = 1 # test households
hh_B = 54

data = '../../../Documents/simulation_results/LV/loss_model_test_map.csv'

# ok i'm now going to get a set of readings from which I need to interpolate,
# or maybe just find the nearest?

losses = {}
for i in np.arange(0.0,Pmax+step,step):
    losses[round(i,dec)] = {}

    
with open(data,'rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        if row == []:
            continue

        i = round(float(row[0]),dec)
        j = round(float(row[1]),dec)

        if i > 1 or j > 1:
            continue

        losses[i][j] = float(row[2])
        
        #losses[i][j].append([float(row[0])]+[float(row[1])]+[float(row[2])])

lowest = 100
best = None

highest = 0
worst = None

heatmap = []
for i in range(int(1.0/step)+1):
    heatmap.append([0.0]*(int(1.0/step)+1))

for i in range(len(heatmap)):
    for j in range(len(heatmap[0])):
        p11 = step*i
        p12 = step*j

        p21 = round(Pmax-p11,dec)
        p22 = round(Pmax-p12,dec)
        try:
            heatmap[i][j] = losses[p11][p12]+losses[p21][p22]

        except:
            continue

plt.figure(1)
plt.imshow(heatmap)
plt.colorbar()

empty = []
for i in range(1,len(heatmap)-1):
    if sum(heatmap[i]) == 0:
        empty.append(i)

heatmap2 = np.zeros((len(heatmap)-len(empty),len(heatmap)-len(empty)))

i2 = 0
for i in range(len(heatmap)):
    if i in empty:
        continue
    j2 = 0
    for j in range(len(heatmap[0])):
        if j in empty:
            continue
        heatmap2[i2][j2] = heatmap[i][j]/max(max(heatmap))
        j2 += 1
    i2 += 1

# now flip in x and y
new = np.zeros((len(heatmap2),len(heatmap2)))

for i in range(len(heatmap2)):
    for j in range(len(heatmap2)):
        new[len(heatmap2)-1-i][len(heatmap2)-1-j] = heatmap2[i][j]

heatmap2 = new
        
for i in range(len(heatmap2)):
    for j in range(len(heatmap2[0])):
        if heatmap2[i][j] == 0:
            x = []
            if i > 0:
                if heatmap2[i-1][j] != 0:
                    x.append(heatmap2[i-1][j])
            if j > 0:
                if heatmap2[i][j-1] != 0:
                    x.append(heatmap2[i][j-1])
            if i < len(heatmap2)-1:
                if heatmap2[i+1][j] != 0:
                    x.append(heatmap2[i+1][j])
            if j < len(heatmap2)-1:
                if heatmap2[i][j+1] != 0:
                    x.append(heatmap2[i][j+1])

            if x != []:
                heatmap2[i][j] = sum(x)/len(x)
            else:
                print(i)
                
heatmap2 = scipy.ndimage.filters.gaussian_filter(heatmap2,1)

levels = np.arange(0.55,1.05,0.1)
spc = len(heatmap2)/8





# ok great, now I want to do the same for my loss-minimisation model to compare
# this will take the form of a very small quadratic program

P_ = np.zeros((55,55))
with open('P.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        if row == []:
            continue
        for j in range(len(row)):
            P_[i][j] = float(row[j])
            j += 1
        i += 1

P0 = matrix([[P_[hh_A-1][hh_A-1],P_[hh_A-1][hh_B-1]],
             [P_[hh_B-1][hh_A-1],P_[hh_B-1][hh_B-1]]])

P0 = P0.T
P = spdiag([P0]*2)

q_ = np.zeros((55,1))
with open('q.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        q_[i] = float(row[0])
        i += 1

q0 = [q_[hh_A-1][0],q_[hh_B-1][0]]
q0 += q0
q = matrix(q0)

with open('c.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        c = float(row[0])

def model_losses(p11,p12):
    p21 = -1000*(Pmax-p11)
    p22 = -1000*(Pmax-p12)
    p11 = -1000*p11
    p12 = -1000*p12

    x = matrix([p11,p12,p21,p22])

    return x.T*P*x+q.T*x+c*2


heatmap3 = np.zeros((int(1.0/step),int(1.0/step)))
heatmap4 = np.zeros((int(1.0/step),int(1.0/step)))

for i in range(len(heatmap3)):
    for j in range(len(heatmap3)):
        p11 = i*1.0/len(heatmap3)
        p12 = j*1.0/len(heatmap3)
        heatmap3[len(heatmap3)-1-i][len(heatmap3)-1-j] = model_losses(p11,p12)[0]
        heatmap4[len(heatmap3)-1-i][len(heatmap3)-1-j] = np.power(p11+p12,2) + \
                                         np.power(2.0-p11-p12,2)


for i in range(len(heatmap3)):
    for j in range(len(heatmap3)):
        heatmap3[i][j] = heatmap3[i][j]/np.amax(heatmap3)
        heatmap4[i][j] = heatmap4[i][j]/np.amax(heatmap4)


#manual_locations = [(46,338),(103,285),(173,221),(270,168)]

x_int = len(heatmap2)/5
ax = [0,1*x_int,2*x_int,3*x_int,4*x_int,5*x_int-1]
ax_ticks = ['0.0','0.2','0.4','0.6','0.8','1.0']
ax_ticks2 = ['1.0','0.8','0.6','0.4','0.2','0.0']


x1 = np.linspace(0,len(heatmap2),num=len(heatmap3))
x2 = np.linspace(0,len(heatmap3),num=len(heatmap2))
y1 = np.linspace(len(heatmap2),0,num=len(heatmap3))
y2 = np.linspace(len(heatmap3),0,num=len(heatmap2))

plt.figure(figsize=(6,3))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 8

plt.subplot(1,2,1)
x_int = len(heatmap4)/5
ax = [0,1*x_int,2*x_int,3*x_int,4*x_int,5*x_int-1]

manual_locations = [(72,393),(148,343),(218,258),(349,205)]
plt.imshow(heatmap4,cmap='inferno')
plt.xticks(ax,ax_ticks)
plt.yticks(ax,ax_ticks)
plt.xlabel('EV2 P1 (kW)')
plt.ylabel('EV1 P1 (kW)')
plt.title('Load Flattening')
CS = plt.contour(heatmap4,colors="white", levels=levels,linewidths=1.0)
plt.xlim(1,len(heatmap4)-2)
plt.ylim(1,len(heatmap4)-2)
plt.plot([0,len(heatmap4)],[len(heatmap4),0],ls='--',color='r')


x_int = len(heatmap2)/5
ax = [0,1*x_int,2*x_int,3*x_int,4*x_int,5*x_int-1]

plt.subplot(1,2,2)
plt.imshow(heatmap2,cmap='inferno')
plt.xticks(ax,ax_ticks)
plt.yticks(ax,ax_ticks)
plt.xlabel('EV2 P1 (kW)')
plt.title('Loss Minimising')
plt.contour(x1,x1,heatmap3,colors="cyan",linestyles='dashed',levels=levels,
            linewidths=1.0)
plt.scatter([len(heatmap2)/2],[len(heatmap2)/2],marker='x',color='r')
#plt.grid()
#plt.colorbar()
CS = plt.contour(heatmap2, colors="white", levels=levels,linewidths=1.0)
plt.xlim(1,len(heatmap2)-2)
plt.ylim(1,len(heatmap2)-2)
#plt.clabel(CS, inline=1, fontsize=8,manual=manual_locations)##plt.show()

'''
x_int = len(heatmap3)/5
ax = [0,1*x_int,2*x_int,3*x_int,4*x_int,5*x_int-1]

#manual_locations = [(72,393),(148,343),(218,258),(349,205)]
plt.subplot(1,2,2)
plt.imshow(heatmap3,cmap='inferno')
plt.xticks(ax,ax_ticks)
plt.yticks(ax,ax_ticks2)
plt.xlabel('HH54 P1 (kW)')
plt.title('Model')
#levels = np.arange(0.5,1.0,0.05)
#plt.colorbar()
plt.contour(x2,x2,heatmap2,colors="cyan",linestyles='dashed',levels=levels,
            linewidths=1.0)
#plt.grid()
CS = plt.contour(heatmap3,colors="white", levels=levels,linewidths=1.0)

plt.xlim(1,len(heatmap3)-2)
plt.ylim(1,len(heatmap3)-2)
#plt.clabel(CS, inline=1, fontsize=10, manual=manual_locations)
'''
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/losses/img/2bus.eps', format='eps', dpi=1000)

plt.show()

import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random

step = 0.001 #
Pmax = 1.0 # max individual charging power

base = 0.0 # kW load of all unused houses

hh_A = 1 # test households
hh_B = 54

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
P1 = spdiag([P0]*2)

q_ = np.zeros((55,1))
with open('q.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        q_[i] = float(row[0])
        i += 1

q0 = [q_[hh_A-1][0],q_[hh_B-1][0]]
q0 += q0
q1 = matrix(q0)

with open('c.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        c1 = float(row[0])

def model_losses1(p11,p12):
    p21 = -1000*(Pmax-p11)
    p22 = -1000*(Pmax-p12)
    p11 = -1000*p11
    p12 = -1000*p12

    x = matrix([p11,p12,p21,p22])

    return x.T*P1*x+q1.T*x+c1*2

# now getting 0.6 linerization point
P_ = np.zeros((55,55))
with open('P_.csv','rU') as csvfile:
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
P2 = spdiag([P0]*2)

q_ = np.zeros((55,1))
with open('q_.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        q_[i] = float(row[0])
        i += 1

q0 = [q_[hh_A-1][0],q_[hh_B-1][0]]
q0 += q0
q2 = matrix(q0)

with open('c_.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        c2 = float(row[0])

def model_losses2(p11,p12):
    p21 = -1000*(Pmax-p11)
    p22 = -1000*(Pmax-p12)
    p11 = -1000*p11
    p12 = -1000*p12

    x = matrix([p11,p12,p21,p22])

    return x.T*P2*x+q2.T*x+c2*2

heatmap1 = np.zeros((int(0.5/step),int(0.5/step)))
heatmap2 = np.zeros((int(0.5/step),int(0.5/step)))
heatmap3 = np.zeros((int(0.5/step),int(0.5/step)))

for i in range(len(heatmap1)):
    for j in range(len(heatmap1)):
        p11 = i*0.5/len(heatmap1)
        p12 = j*0.5/len(heatmap1)
        heatmap1[len(heatmap1)-1-i][j] = model_losses1(p11,p12)[0]
        heatmap2[len(heatmap1)-1-i][j] = model_losses2(p11,p12)[0]

s1 = np.amax(heatmap1)
s2 = np.amax(heatmap2)

for i in range(len(heatmap1)):
    for j in range(len(heatmap1)):
        heatmap1[i][j] = heatmap1[i][j]/1.0#s1
        heatmap2[i][j] = heatmap2[i][j]/1.0#s2
        heatmap3[i][j] = 100*(heatmap1[i][j]-heatmap2[i][j])/heatmap2[i][j]


x_int = len(heatmap3)/5
ax = [0,1*x_int,2*x_int,3*x_int,4*x_int,5*x_int-1]
ax_ticks = ['0.0','0.1','0.2','0.3','0.4','0.5']
ax_ticks2 = ['0.5','0.4','0.3','0.2','0.1','0.0']

levels = np.arange(2,4,0.5)
plt.figure(1)
plt.subplot(2,3,1)
plt.title('1kW Lin')
plt.imshow(heatmap1)
plt.xticks(ax,ax_ticks)
plt.yticks(ax,ax_ticks2)
plt.xlabel('HH54 P1 (kW)')
plt.ylabel('HH1 P1 (kW)')
CS = plt.contour(heatmap1, colors="white", levels=levels,linewidths=1.0)
plt.clabel(CS, inline=1, fontsize=10)#, manual=manual_locations)

plt.subplot(2,3,2)
plt.title('0.6kW Lin')
plt.imshow(heatmap2)
plt.xticks(ax,ax_ticks)
plt.yticks(ax,ax_ticks2)
plt.xlabel('HH54 P1 (kW)')
plt.ylabel('HH1 P1 (kW)')
CS = plt.contour(heatmap2, colors="white", levels=levels,linewidths=1.0)
plt.clabel(CS, inline=1, fontsize=10)#, manual=manual_locations)

levels = np.arange(1.4,1.6,0.05)
plt.subplot(2,3,3)
plt.title('% Difference')
plt.imshow(heatmap3)
plt.xticks(ax,ax_ticks)
plt.yticks(ax,ax_ticks2)
plt.xlabel('HH54 P1 (kW)')
plt.ylabel('HH1 P1 (kW)')
CS = plt.contour(heatmap3, colors="white", levels=levels,linewidths=1.0)
plt.clabel(CS, inline=1, fontsize=10)#, manual=manual_locations)

for i in range(len(heatmap1)):
    for j in range(len(heatmap1)):
        heatmap1[i][j] = heatmap1[i][j]/s1
        heatmap2[i][j] = heatmap2[i][j]/s2
        heatmap3[i][j] = 100*(abs(heatmap2[i][j]-heatmap1[i][j]))/heatmap2[i][j]

levels = np.arange(0.550,0.950,0.1)
manual_locations = [(72,393),(148,343),(218,258),(349,205)]

#plt.figure(2)
#plt.subplot(1,3,1)
plt.subplot(2,3,4)
plt.title('1kW Lin')
plt.imshow(heatmap1)
plt.xticks(ax,ax_ticks)
plt.yticks(ax,ax_ticks2)
plt.xlabel('HH54 P1 (kW)')
plt.ylabel('HH1 P1 (kW)')
CS = plt.contour(heatmap1, colors="white", levels=levels,linewidths=1.0)
plt.clabel(CS, inline=1, fontsize=10, manual=manual_locations)

#plt.subplot(1,3,2)
plt.subplot(2,3,5)
plt.title('0.6kW Lin')
plt.imshow(heatmap2)
plt.xticks(ax,ax_ticks)
plt.yticks(ax,ax_ticks2)
plt.xlabel('HH54 P1 (kW)')
plt.ylabel('HH1 P1 (kW)')
CS = plt.contour(heatmap2, colors="white", levels=levels,linewidths=1.0)
plt.clabel(CS, inline=1, fontsize=10, manual=manual_locations)

levels = np.arange(-0.075,0.1,0.025)
plt.subplot(2,3,6)
#plt.subplot(1,3,3)
plt.title('% Difference')
plt.imshow(heatmap3)
plt.xticks(ax,ax_ticks)
plt.yticks(ax,ax_ticks2)
plt.xlabel('HH54 P1 (kW)')
plt.ylabel('HH1 P1 (kW)')
CS = plt.contour(heatmap3, colors="white", levels=levels,linewidths=1.0)
plt.clabel(CS, inline=1, fontsize=10)#, manual=manual_locations)
plt.show()

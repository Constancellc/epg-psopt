import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random

from lv_optimization import LVTestFeeder

Pmax = 6 # kW
pph = 60
T = 24*pph
t_int = int(60/pph)

# okay so this script is going to calculate the optimal losses for some false
# energy requirements

household_profiles = []
with open('household_demand_pool_HH.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        profile = []
        hh = []
        for cell in row:
            hh.append(float(cell))
            
        hh.append(hh[0])
        for j in range(0,1440):
            p1 = int(j/30)
            p2 = p1 + 1
            f = float(j%30)/30
            profile.append((1-f)*hh[p1] + f*hh[p2])
        household_profiles.append(profile)

chosen = []
while len(chosen) < 55:
    ran = int(random.random()*len(household_profiles))
    if ran not in chosen:
        chosen.append(ran)
        
# first I need to set up the energy demands
energy = [] # kWh
for hh in range(55):
    energy.append(5)

# then I need to acquire the losses model
P0 = matrix(0.0,(55,55))
with open('P0.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        for j in range(len(row)):
            P0[i,j] += float(row[j])
        i += 1

P = spdiag([P0]*T)

q0 = []
with open('q0.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        q0.append(float(row[0]))

q = []
for t in range(T):
    q += q0
q = matrix(q)

x_h = []

for t in range(T):
    for j in range(55):
        x_h.append(-household_profiles[chosen[j]][int(t*t_int)]*1000)

x_h = matrix(x_h)
q += (P+P.T)*x_h

# then I want to find the optimal losses
A = matrix(0.0,(55,55*T))
b = matrix(0.0,(55,1))
        
for j in range(55):
    for t in range(T):
        A[j,55*t+j] = 1.0/pph # energy requirement

    b[j] = -energy[j]*1000

G = sparse([spdiag([-1.0]*(55*T)),spdiag([1.0]*(55*T))])
h = matrix([Pmax*1000]*(55*T)+[0.0]*(55*T))

sol=solvers.qp(P*2,q,G,h,A,b)
x = sol['x']

av_lm = [0.0]*T
hh1 = [0.0]*T
hh54 = [0.0]*T

for i in range(55):
    for t in range(T):
        av_lm[t] -= x[55*t+i]/55000
        if i == 0:
            hh1[t] -= x[55*t+i]/1000
        elif i == 53:
            hh54[t] -= x[55*t+i]/1000

# adding base loads
base1 = [0.0]*T
base54 = [0.0]*T
baseAv = [0.0]*T

for t in range(T):
    for i in range(55):
        baseAv[t] += household_profiles[chosen[i]][t]/55
        if i == 0:
            base1[t] += household_profiles[chosen[i]][t]
        if i == 53:
            base54[t] += household_profiles[chosen[i]][t]

    av_lm[t] += baseAv[t]
    hh1[t] += base1[t]
    hh54[t] += base54[t]


t = np.linspace(0,24,num=1440)

plt.figure(1)
plt.plot(t,av_lm,label='Loss Minimising')
plt.title('Feeder Average')
plt.plot(t,baseAv,'k',ls=':',label='Base Load')
plt.xlim(0,24)
plt.legend()
plt.grid()
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(t,hh1)
plt.title('Household 1',y=0.85)
plt.plot(t,base1,'k',ls=':')
plt.xlim(0,24)
plt.ylim(0,int(max(hh1+hh54))+1)
plt.grid()
plt.subplot(2,1,2)
plt.plot(t,hh54)
plt.title('Household 54',y=0.85)
plt.plot(t,base54,'k',ls=':')
plt.xlim(0,24)
plt.ylim(0,int(max(hh1+hh54))+1)
plt.grid()
plt.show()

# finally I should store and / or plot the results

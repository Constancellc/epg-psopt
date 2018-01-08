import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random

Pmax = 70 # kW
x0 = 0.0 # kW
pph = 6
T = 24*pph

skipUnused = False
unused = []
'''
NOTES

Can i alter the optimisation to skip households where vehicles do not
need charging?

Also are the loads specified in kW or W?

Also is ignoring the souce bus from the matrix legitimate?

'''
# first pick the households
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
        
# then pick the vehicles
vehicle_profiles = []
with open('vehicle_demand_pool.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        profile = []
        for i in range(len(row)):
            profile.append(float(row[i]))
        if skipUnused == True:
            if sum(profile) == 0:
                continue

        vehicle_profiles.append(profile)
        
chosenV = []

while len(chosenV) < 55:
    ran = int(random.random()*len(vehicle_profiles))
    if ran not in chosenV:
        chosenV.append(ran)

energyV = [] # vector containing energy req in kWh of each vehicle
energyHH = []
avaliable = [] # contains range for which charging NOT allowed

for v in range(55):
    energyV.append(sum(vehicle_profiles[chosenV[v]])/60)
    hh_en = 0
    for t in range(0,1440,int(60/pph)):
        hh_en += household_profiles[chosen[v]][t]
    energyHH.append(hh_en/pph)

    if energyV[-1] == 0:
        i = 0
        unused.append(v)
        
    else:
        i = 0
        while vehicle_profiles[chosenV[v]][i] != 0:
            i += 1
            
        while vehicle_profiles[chosenV[v]][i] == 0 and i < 1439:
            i += 1

    # picks random start time between 7 and 11 and gets actual end time
    avaliable.append([int(6*60+random.random()*int((i/60)-6)*60),i])
'''
c = []
with open('c.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        c.append(float(row[0]))
'''
c = [1.0]*55
c = c*T
    
c = matrix(c)

# x[:55] is the charging power of all vehicles at the first time instant

A = matrix(0.0,(110,55*T))
b = matrix(0.0,(110+110*T,1))

for v in range(55):
    b[2*v] = energyV[v]+energyHH[v]-x0*24 # energy required in kW
    b[2*v+1] = -energyV[v]-energyHH[v]+x0*24
    
    for t in range(T):
        A[2*v,55*t+v] = 1.0/pph
        A[2*v+1,55*t+v] = -1.0/pph
        
A = sparse([A,spdiag([-1]*55*T),spdiag([1]*55*T)])

for i in range(55*T):
    hh = chosen[i%55]
    t = int(int(i/55)*60/pph)
    
    b[110+i] = -household_profiles[hh][t]+x0
    
    if t > avaliable[i%55][0] and t < avaliable[i%55][1]: # if unavaliable
        b[110+55*T+i,0] = household_profiles[hh][t]-x0
    else:
        b[110+55*T+i,0] = household_profiles[hh][t]-x0+Pmax

sol=solvers.lp(c,A,b)
x = sol['x']
lm = [0.0]*T
bl = [0.0]*T
for v in range(55):
    for t in range(T):
        lm[t] += x[55*t+v]/55
        bl[t] += household_profiles[chosen[v]][int(t*60/pph)]

# NOW EMBARKING ON THE LOAD FLATTENING FOR COMPARISON
n = 55-len(unused)
print(n)
b = []

for i in range(55):
    if i in unused:
        continue
    b.append(energyV[i])
    
A1 = matrix(0.0,(n,T*n))
A2 = matrix(0.0,(n,T*n))

b += [0.0]*n
b = matrix(b)

skp = 0
for j in range(55):
    if j in unused:
        skp += 1
        continue

    v = j-skp

    for i in range(T):
        A1[n*(T*v+i)+v] = 1.0/pph
        
        if i*60/pph > avaliable[j][0] and i*60/pph < avaliable[j][1]:
            A2[n*(T*v+i)+v] = 1.0

A = sparse([A1,A2])
A3 = spdiag([-1]*(T*n))
A4 = spdiag([1]*(T*n))
G = sparse([A3,A4])

h = []
for i in range(T*n):
    h.append(0.0)
for i in range(T*n):
    h.append(Pmax)
    
h = matrix(h)

q = []
for i in range(n):
    q += bl

q = matrix(q)

I = spdiag([1]*T)
P = sparse([[I]*n]*n)

print(A.size)
print(b.size)
print(P.size)
print(q.size)
print(G.size)
print(h.size)

sol = solvers.qp(P,q,G,h,A,b)
x = sol['x']


lf = [0.0]*T

v = 0
for i in range(55):
    if i in unused:
        continue

    load = []
    for t in range(T):
        lf[t] += x[v*T+t]/55
    v += 1

for i in range(len(bl)):
    bl[i] = bl[i]/55
    lf[i] += bl[i]
    
plt.figure(1)
plt.plot(bl)
plt.plot(lm)
plt.plot(lf)
plt.show()        

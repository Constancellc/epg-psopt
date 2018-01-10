import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random

Pmax = 3.5 # kW
x0 = 0.0 # kW
pph = 60
T = 24*pph

skipUnused = True
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


n = 55-len(unused)

A2 = matrix(0.0,(55,55))
Q0 = matrix(0.0,(n,n))
with open('A2.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        for j in range(len(row)):
            A2[i,j] -= float(row[j])
        i += 1

iskp = 0
for i in range(55):
    if i in unused:
        iskp += 1
        continue
    jskp = 0
    for j in range(55):
        if j in unused:
            jskp += 1
            continue
        Q0[i-iskp,j-jskp] = A2[i,j]

a = 0.5*(A2 + A2.T)
print('examining Q0')
print('size')
for i in range(Q0.size[0]):
    for j in range(Q0.size[1]):
        if Q0[i,j] > 1.0e-10:
            print('X',end='')
        else:
            print('0',end='')
    print('')


#P = spdiag([Q0]*T)
P = spdiag([1.0e-07]*n*T)

q = []
# for each time instant I need the hosuehold loads
for t in range(T):
    x_h = []
    for i in range(55):
        x_h.append(household_profiles[chosen[i]][t])

    x_h = matrix(x_h)

    new = a*x_h
    for i in range(55):
        if i in unused:
            continue
        q.append(new[i])
        
q = matrix(q)

# x[:55] is the charging power of all vehicles at the first time instant

A = matrix(0.0,(n,n*T))
b = matrix(0.0,(n,1))

skp = 0
for j in range(55):
    if j in unused:
        skp += 1
        continue
    v = j-skp
    b[v] = energyV[j]#x0*24 # energy required in kW
    
    for t in range(T):
        A[v,n*t+v] = 1.0/pph
        
G = sparse([spdiag([-1]*n*T),spdiag([1]*n*T)])
h = matrix(0.0,(2*n*T,1))

for i in range(n*T):
    t = int(int(i/n)*60/pph)
    
    h[i] = 0
    h[int(n*T+i)] = Pmax

sol=solvers.qp(P,q,G,h,A,b)
x = sol['x']

print('losses are approx:')
dx = x-matrix(x0,(n*T,1))

print(q.T*dx+dx.T*P*dx)


lm = [0.0]*T
bl = [0.0]*T
skp = 0
for j in range(55):
    if j in unused:
        skp += 1
        continue
    v = j-skp
    for t in range(T):
        lm[t] += (x[n*t+v]+x0)/55
        bl[t] += household_profiles[chosen[j]][int(t*60/pph)]

# NOW EMBARKING ON THE LOAD FLATTENING FOR COMPARISON

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
    lm[i] += bl[i]
    
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(bl,c='k',ls=':')
plt.plot(lm)
plt.plot(lf)

for i in range(T):
    lm[i] -= bl[i]
    lf[i] -= bl[i]

plt.subplot(2,1,2)
plt.plot(lm)
plt.plot(lf)

plt.show()        

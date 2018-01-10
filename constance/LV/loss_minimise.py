import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random

Pmax = 6 # kW
#x0 = 0.0 # kW
pph = 10
T = 24*pph
t_int = int(60/pph)

pf = 0.95 # power factor
alpha = round(np.sqrt(np.power(1/pf,2)-1),2)

skipUnused = True
unused = []
'''
NOTES

desisions need to be made about formulation - how is the valley filling going
to be taken into account

also need to figure out what we're doing with reactive power

need to crack the units question once and for all

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
        
    for t in range(0,1440,t_int):
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
    avaliable.append([int(5*60+random.random()*int((i/60)-5)*60),i])

n = 55-len(unused)
print(n)
# map of new to original indexs
ind_map = {}

j = 0
for i in range(55):
    if i in unused:
        continue
    ind_map[j] = i
    j += 1
    
P0 = matrix(0.0,(2*n,2*n))

with open('P.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    iskp = 0
    for row in reader:
        
        if i in unused or i-55 in unused:
            i += 1
            iskp += 1
            continue
        
        jskp = 0
        for j in range(len(row)):
            if j in unused or j-55 in unused:
                jskp += 1
                continue
            P0[i-iskp,j-jskp] += float(row[j])
        i += 1


P0_r = P0[0:n,0:n] + matrix(alpha*P0[n:2*n,0:n]) + matrix(alpha*P0[0:n,n:2*n])\
       + matrix(alpha*alpha*P0[n:2*n,n:2*n])

P = spdiag([P0_r]*T)
#P = spdiag([1.0]*(n*T))
x_h = []

for t in range(T):
    for j in range(n):
        x_h.append(household_profiles[chosen[ind_map[j]]][int(t*t_int)])

x_h = matrix(x_h)

q = 2*x_h.T*P
q = q.T

# x[:n] is the real power of all vehicles at the first time instant
# x[n:2n] is the imaginary power at the first time instant

A = matrix(0.0,(2*n,n*T))
b = matrix(0.0,(2*n,1))
# skipping avaliability constraint for now

for j in range(n):
    for t in range(T):
        A[j,n*t+j] = 1.0/pph # energy requirement

        if t > avaliable[ind_map[j]][0]/t_int and \
           t < avaliable[ind_map[j]][1]/t_int:
            A[j+n,n*t+j] = 1.0

    b[j] = energyV[ind_map[j]]

G = sparse([spdiag([1.0]*(n*T)),spdiag([-1.0]*(n*T))])
h = matrix([Pmax]*(n*T)+[0.0]*(n*T))

print(A.size)
print(b.size)
print(G.size)
print(h.size)
'''
for fp in np.arange(0.1,1,0.1):
    test = matrix([fp]*(n*T))
    print(q.T*test+test.T*P*test)
'''
sol=solvers.qp(P,q,G,h,A,b)
x = sol['x']

lm = [0.0]*T
bl = [0.0]*T
skp = 0

for j in range(n):
    for t in range(T):
        lm[t] += (x[n*t+j])/55
        
for j in  range(55):
    for t in range(T):
        bl[t] += household_profiles[chosen[j]][int(t*t_int)]
        
'''
test = matrix([sum(bl)/T]*(n*T))
print(q.T*test+test.T*P*test)
x = []
for i in range(n):
    x += bl
test = matrix(x)
print(test.size)
print(q.T*test+test.T*P*test)
'''

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

import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random
import copy 

Pmax = 6 # kW
pph = 60
T = 24*pph
t_int = int(60/pph)
nRuns = 1
skipUnused = True
constrainAvaliability = False

unused = []
# either using HH data
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

# get loss minimization matricies
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
q0 = matrix(q)

with open('c.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        c = float(row[0])


# also setting up optimization matricies which don't change here
A = matrix(0.0,(55,55*T))
for j in range(55):
    for t in range(T):
        A[j,55*t+j] = 1.0/pph
G = sparse([spdiag([-1.0]*(55*T)),spdiag([1.0]*(55*T))])
h = matrix([Pmax*1000]*(55*T)+[0.0]*(55*T))

A2 = matrix(0.0,(55,55*T))
for j in range(55):
    for t in range(T):
        A2[j,T*j+t] = 1.0/pph
        
h2 = matrix([0.0]*(55*T)+[Pmax*1000]*(55*T))
I = spdiag([1]*T)
P2 = sparse([[I]*55]*55)

# this is to store the results in:
lf_profiles = []
lm_profiles = []
lf_losses = []
lm_losses = []
chosenHH = []

# for run in mc
for mc in range(10):
    # chose hh profiles
    chosen = []
    while len(chosen) < 55:
        ran = int(random.random()*len(household_profiles))
        if ran not in chosen:
            chosen.append(ran)
    chosenHH.append(chosen)
        
    # chose vehicle energy req
    vEnergy = []
    for i in range(55):
        vEnergy.append(30*random.random())

    # loss minimize
    b = []
    for i in range(55):
        b.append(-vEnergy[i]*1000)
    b = matrix(b)

    x_h = []
    for t in range(T):
        for j in range(55):
            x_h.append(-household_profiles[chosen[j]][int(t*t_int)]*1000)

    x_h = matrix(x_h)
    q = copy.copy(q0) + (P+P.T)*x_h

    sol=solvers.qp(P,q,G,h,A,b)
    x = sol['x']
    print(sum(x))

    print(x.T*P*x+2*q.T*x)

    # estimate losses
    y = x+x_h
    lm_losses.append((y.T*P*y + q0.T*y + c*T)[0])

    # store results
    profiles = []
    for i in range(55):
        profile = []
        for t in range(T):
            profile.append(-x[55*t+i]/1000)
        profiles.append(profile)
    lm_profiles.append(profiles)

    # load flatten
    b = []
    for i in range(55):
        b.append(vEnergy[i]*1000)
    b = matrix(b)

    bl = [0.0]*T
    for t in range(T):
        for i in range(55):
            bl[t] += household_profiles[chosen[i]][int(t*t_int)]*1000

    q2 = []
    for i in range(55):
        q2 += bl
    q2 = matrix(q2)

    sol2 = solvers.qp(P2,q2,G,h2,A2,b)
    x2 = sol2['x']
    print(sum(x2))
    # estimate losses
    y2 = matrix(0.0,(55*T,1))
    for t in range(T):
        for i in range(55):
            y2[55*t+i] = -x2[i*T+t]
    print(sum(y2))
    
    print(y2.T*P*y2+2*q.T*y2)
    y2 += x_h
    
    plt.figure(1)
    plt.plot(y2,alpha=0.2)
    plt.plot(y,alpha=0.2)
    plt.show()
    lf_losses.append((y2.T*P*y2 + q0.T*y2 + c*T)[0])

    # store results
    profiles = []
    for i in range(55):
        profile = []
        for t in range(T):
            profile.append(x2[T*i+t])
        profiles.append(profile)
    lf_profiles.append(profiles)

diff = []
for i in range(len(lf_losses)):
    diff.append(lf_losses[i]-lm_losses[i])
diff = sorted(diff)

plt.figure(1)
plt.subplot(2,1,1)
plt.boxplot([lf_losses,lm_losses],0,'',whis=[0.05, 99.5])
plt.grid()
plt.subplot(2,1,2)
plt.plot(diff)
plt.show()

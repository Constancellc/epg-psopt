import csv
import numpy as np
from cvxopt import matrix, spdiag, sparse, solvers
import random

Pmax = 7 # kW
x0 = 1.0 # kW
pph = 1
T = 24*pph
'''
NOTES

I need to EITHER incorporate the household profiles into the decision variable
or I need to linearize around the actual operating point

Ahhh decisions

Also I need to decide if the negative values of c are actually wrong?

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
    energyHH.append(sum(household_profiles[chosen[v]])/60)

    if energyV[-1] == 0:
        i = 0
    else:
        i = 0
        while vehicle_profiles[chosenV[v]][i] != 0:
            i += 1
        while vehicle_profiles[chosenV[v]][i] == 0 and i < 1439:
            i += 1

    # picks random start time between 7 and 11 and gets actual end time
    avaliable.append([int(7*pph+random.random()*4*pph),int(i*pph/60)+1])

c = []
with open('c.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        c.append(float(row[0]))

c = c*T
    
c = matrix(c)

# x[:55] is the charging power of all vehicles at the first time instant

A = matrix(0.0,(110,55*T))
b = matrix(0.0,(110+110*T,1))

for v in range(55):
    b[2*v] = energyV[v]+energyHH[v]-x0*24
    b[2*v+1] = -energyV[v]-energyHH[v]+x0*24
    
    for t in range(T):
        A[2*v,55*t+v] = 1.0/pph
        A[2*v+1,55*t+v] = -1.0/pph
        
A = sparse([A,spdiag([-1]*55*T),spdiag([1]*55*T)])

for i in range(55*T):
    hh = chosen[i%55]
    t = i%T
    b[110+i] = household_profiles[hh][t]-x0
    
    if t > avaliable[i%55][0] and t < avaliable[i%55][1]: # if unavaliable
        b[110+55*T+i,0] = household_profiles[hh][t]-x0
    else:
        b[110+55*T+i,0] = household_profiles[hh][t]-x0+Pmax

with open('A.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(A.size[0]):
        row = []
        for j in range(A.size[1]):
            row.append(A[i,j])
        writer.writerow(row)
print(A.size)
print(b.size)
print(c.size)
sol=solvers.lp(c,A,b)

        

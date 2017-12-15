import csv
import numpy as np
from cvxopt import matrix, spdiag, sparse, solvers
import random

Pmax = 3.5
T = 1
'''
NOTES

I need to EITHER incorporate the household profiles into the decision variable
or I need to linearize around the actual operating point

Ahhh decisions

Also I need to decide if the negative values of c are actually wrong?

Also are the loads specified in kW or W?

Also is ignoring the souce bus from the matrix legitimate?

'''
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

energy = [] # vector containing energy req in kWh of each vehicle
avaliable = [] # contains range for which charging NOT allowed

for v in chosenV:
    energy.append(sum(vehicle_profiles[chosenV[v]]))

    if energy == 0:
        avaliable = [0,55*T]

    i = 0
    while vehicle_profiles[i] != 0:
        i += 1
    while vehicle_profiles[i] == 0:
        i += 1

    # picks random start time between 7 and 11 and gets actual end time
    avaliable.append([t_deadline = int(7*60+random.random()*4*60),i])

c = []
with open('c.csv','rU') as csvfile:
    for row in reader:
        new = []
        for i in range(len(row)):
            new.append(float(row[i]))
        c.append(row)

for i in range(T-1):
    c += c
    
c = matrix(c)

# x[:55] is the charging power of all vehicles at the first time instant



A = matrix(0.0,(220,55*T))
b = matrix(0.0,(220+110*T,1))

for v in range(55):
    for t in range(T):
        A[2*v,55*t+v] = 1.0/60
        A[2*v+1,55*t+v] = -1.0/60
        b[2*v] = energy[v]
        b[2*v+1] = -energy[v]

        if t > avaliable[v][0] and t < avaliable[v][1]:
            A[110+2*v,55*t+v] = 1.0
            A[110+2*v+1,55*t+v] = -1.0

A = sparse([A,spdiag([-1]*55*T),spdiag([1]*55*T)])

for i in range(55*T):
    b[220+55*T+i,0] = Pmax
                


        

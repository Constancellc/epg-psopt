import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random
#import win32com.client

Pmax = 3.5 # kW
x0 = 1.0 # kW
pph = 15
T = 24*pph

skipUnused = True
unused = []

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

    # picks random start time between 5 and end time and gets actual end time
    avaliable.append([int(5*60+random.random()*int((i/60)-5)*60),i])
#    avaliable.append([int(i-9*60*random.random()),i])
    
q = []
with open('p.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        q.append(float(row[0]))

q = q*T    
q = matrix(q)

Q0 = matrix(0.0,(55,55))
with open('Q.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        for j in range(len(row)):
            Q0[i,j] = float(row[j])
        i += 1

P = spdiag([Q0]*T)

# x[:55] is the charging power of all vehicles at the first time instant

A = matrix(0.0,(55,55*T))
b = matrix(0.0,(55,1))

for v in range(55):
    b[v] = energyV[v]+energyHH[v]-x0*24 # energy required in kW
    
    for t in range(T):
        A[v,55*t+v] = 1.0/pph
        
G = sparse([spdiag([-1]*55*T),spdiag([1]*55*T)])
h = matrix(0.0,(110*T,1))

for i in range(55*T):
    hh = chosen[i%55]
    t = int(int(i/55)*60/pph)
    
    h[i] = -household_profiles[hh][t]+x0
    
    if t > avaliable[i%55][0] and t < avaliable[i%55][1]: # if unavaliable
        h[55*T+i] = household_profiles[hh][t]-x0
    else:
        h[55*T+i] = household_profiles[hh][t]-x0+Pmax

sol=solvers.qp(P,q,G,h,A,b)
x = sol['x']

lm_profiles = []
bl = [0.0]*T

for v in range(55):
    profile = []
    for t in range(T):
        profile.append(x[55*t+v]+x0)
        bl[t] += household_profiles[chosen[v]][int(t*60/pph)]
    lm_profiles.append(profile)

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

sol = solvers.qp(P,q,G,h,A,b)
x = sol['x']


lf_profiles = []

skp = 0
for i in range(55):
    profile = []
    if i in unused:
        skp += 1
        for t in range(T):
            profile.append(household_profiles[chosen[i]][int(t*60/pph)])
    else:
        v = i-skp
        for t in range(T):
            profile.append(household_profiles[chosen[i]][int(t*60/pph)]+\
                           x[v*T+t])
    lf_profiles.append(profile)

# interpolating if necessary
if pph != 60:
    t_int = int(60/pph)
    for p in [lf_profiles,lm_profiles]:
        for hh in p:
            new = [0.0]*1440
            for i in range(1440):
                f = float(i%t_int)
                p1 = int(i/t_int)
                p2 = p1+1

                if p2 == T:
                    p2 -= 1

                new[i] += (1-f)*hh[p1]+f*hh[p2]

            hh = new

# NOW FOR THE OPEN DSS BIT

engine = win32com.client.Dispatch("OpenDSSEngine.DSS")
engine.Start("0")

# Load flattening first
totalLoadLF = []
lossesLF = []

for i in range(1,56):
    with open('household-profiles/'+str(i)+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        for t in range(1440):
            writer.writerow(lf_profiles[i][t])

engine.text.Command='clear'
circuit = engine.ActiveCircuit

engine.text.Command='compile master.dss'
engine.Text.Command='Export mon LINE1_PQ_vs_Time'

powerIn = [0.0]*1440
powerOut = [0.0]*1440

with open('LVTest_Mon_line1_pq_vs_time.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    i = 0
    for row in reader:
        powerIn[i] -= float(row[2])
        powerIn[i] -= float(row[4])
        powerIn[i] -= float(row[6])

        i += 1

for hh in range(1,56):
    engine.Text.Command='Export mon hh'+str(hh)+'_pQ_vs_time'

    i = 0
    with open('LVTest_Mon_hh'+str(hh)+'_pq_vs_time.csv','rU') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            powerOut[i] += float(row[2])
            i += 1

totalLoadLF.append(powerOut)

loss = 0.0
for i in range(1440):
    loss += powerIn[i]-powerOut[i]

lossesLF.append(loss)

# now loss minimising
totalLoadLM = []
lossesLM = []

for i in range(1,56):
    with open('household-profiles/'+str(i)+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        for t in range(1440):
            writer.writerow(lm_profiles[i][t])

engine.text.Command='clear'
circuit = engine.ActiveCircuit

engine.text.Command='compile master.dss'
engine.Text.Command='Export mon LINE1_PQ_vs_Time'

powerIn = [0.0]*1440
powerOut = [0.0]*1440

with open('LVTest_Mon_line1_pq_vs_time.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    i = 0
    for row in reader:
        powerIn[i] -= float(row[2])
        powerIn[i] -= float(row[4])
        powerIn[i] -= float(row[6])

        i += 1

for hh in range(1,56):
    engine.Text.Command='Export mon hh'+str(hh)+'_pQ_vs_time'

    i = 0
    with open('LVTest_Mon_hh'+str(hh)+'_pq_vs_time.csv','rU') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            powerOut[i] += float(row[2])
            i += 1

totalLoadLM.append(powerOut)

loss = 0.0
for i in range(1440):
    loss += powerIn[i]-powerOut[i]

lossesLM.append(loss)

print('load flattening: ',end='')
print(lossesLF[0])
print('loss minimising: ',end='')
print(lossesLM[0])

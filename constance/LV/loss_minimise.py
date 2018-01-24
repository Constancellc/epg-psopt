import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random

Pmax = 6 # kW
#x0 = 0.0 # kW
pph = 60
T = 24*pph
t_int = int(60/pph)
nRuns = 1

pf = 0.95 # power factor
alpha = round(np.sqrt(np.power(1/pf,2)-1),2)

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

# or using the crest profiles...

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

lfs = []
bls = []
lms = []

hh1 = {'lm':[0.0]*T,'lf':[0.0]*T,'bl':[0.0]*T}
hh54 = {'lm':[0.0]*T,'lf':[0.0]*T,'bl':[0.0]*T}

for run in range(nRuns):
    chosen = []
    while len(chosen) < 55:
        ran = int(random.random()*len(household_profiles))
        if ran not in chosen:
            chosen.append(ran)

            
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
        
    P0 = matrix(0.0,(n,n))

    with open('P.csv','rU') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        iskp = 0
        for row in reader:
            
            if i in unused:
                i += 1
                iskp += 1
                continue
            
            jskp = 0
            for j in range(len(row)):
                if j in unused:
                    jskp += 1
                    continue
                P0[i-iskp,j-jskp] += float(row[j])
            i += 1

    P = spdiag([P0]*T)

    q0 = []
    with open('q.csv','rU') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            if i in unused:
                i += 1
                continue
            q0.append(float(row[0]))
            i += 1

    q = []
    for t in range(T):
        q += q0
    q = matrix(q)

    x_h = []

    for t in range(T):
        for j in range(n):
            x_h.append(-household_profiles[chosen[ind_map[j]]][int(t*t_int)]*1000)

    x_h = matrix(x_h)

    q += (P+P.T)*x_h

    with open('c.csv','rU') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            c = float(row[0])

    # x[:n] is the real power of all vehicles at the first time instant

    if constrainAvaliability == False:
        A = matrix(0.0,(n,n*T))
        b = matrix(0.0,(n,1))
    else:
        A = matrix(0.0,(2*n,n*T))
        b = matrix(0.0,(2*n,1))
        
    # skipping avaliability constraint for now
    for j in range(n):
        for t in range(T):
            A[j,n*t+j] = 1.0/pph # energy requirement
            '''
            if t > avaliable[ind_map[j]][0]/t_int and \
               t < avaliable[ind_map[j]][1]/t_int:
                A[j+n,n*t+j] = 1.0
            '''
        b[j] = -energyV[ind_map[j]]*1000

    G = sparse([spdiag([-1.0]*(n*T)),spdiag([1.0]*(n*T))])
    h = matrix([Pmax*1000]*(n*T)+[0.0]*(n*T))

    sol=solvers.qp(P,q,G,h,A,b)
    x = sol['x']

    y = x+x_h
    '''
    print('I think total losses are:')
    print(y.T*P*y + q.T*y + c*T)
    print('per second:')
    print((y.T*P*y + q.T*y + c*T)/1440)
    '''
    Pl = P
    ql = q

    lm = [0.0]*T
    bl = [0.0]*T
    skp = 0

    for j in range(n):
        for t in range(T):
            lm[t] -= (x[n*t+j])/55000

    for t in range(T):
        hh1['lm'][t] += (household_profiles[chosen[0]][int(t*t_int)]-x[n*t]/1000)\
                       /nRuns
        hh54['lm'][t] += (household_profiles[chosen[53]][int(t*t_int)] - \
                         x[n*t+53]/1000)/nRuns
            
    for j in  range(55):
        for t in range(T):
            bl[t] += household_profiles[chosen[j]][int(t*t_int)]

    # NOW EMBARKING ON THE LOAD FLATTENING FOR COMPARISON

    b = []
    for i in range(55):
        if i in unused:
            continue
        b.append(energyV[i])
        
    A1 = matrix(0.0,(n,T*n))
    A2 = matrix(0.0,(n,T*n))

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

    if constrainAvaliability == False:
        A = A1
    else:
        A = sparse([A1,A2])
        b += [0.0]*n
        
    b = matrix(b)

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

    y2 = matrix(0.0,(n*T,1))
    for t in range(T):
        for i in range(n):
            y2[n*t+i] = -1000*x[i*T+t]

    y2 += x_h
    '''
    print('I think total losses are:')
    print(y2.T*Pl*y2 + ql.T*y2 + c*T)
    print('per second:')
    print((y2.T*Pl*y2 + ql.T*y2 + c*T)/1440)
    '''
    lf = [0.0]*T

    v = 0
    for i in range(55):
        if i in unused:
            continue

        load = []
        for t in range(T):
            lf[t] += x[v*T+t]/55
        v += 1

    for t in range(T):
        hh1['lf'][t] += (household_profiles[chosen[0]][int(t*t_int)]+x[t])/nRuns
        hh54['lf'][t] += (household_profiles[chosen[53]][int(t*t_int)] + \
                         x[53*T+t])/nRuns
        hh1['bl'][t] += household_profiles[chosen[0]][int(t*t_int)]/nRuns
        hh54['bl'][t] += household_profiles[chosen[53]][int(t*t_int)]/nRuns

    for i in range(len(bl)):
        bl[i] = bl[i]/55
        lf[i] += bl[i]
        lm[i] += bl[i]

    bls.append(bl)
    lfs.append(lf)
    lms.append(lm)

lm_av = [0.0]*T
lf_av = [0.0]*T
bl_av = [0.0]*T

for i in range(len(bls)):
    for t in range(T):
        lm_av[t] += lms[i][t]/nRuns
        lf_av[t] += lfs[i][t]/nRuns
        bl_av[t] += bls[i][t]/nRuns
        
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(bl_av,c='k',ls=':')
plt.plot(lm_av)
plt.plot(lf_av)

for i in range(T):
    lm_av[i] -= bl_av[i]
    lf_av[i] -= bl_av[i]

plt.subplot(2,1,2)
plt.plot(lm_av)
plt.plot(lf_av)

t = np.linspace(0,24,num=1440)
x = np.arange(4,24,4)
x_ticks = ['04:00','08:00','12:00','16:00','20:00']

plt.figure(2)
plt.subplot(2,1,1)
plt.title('Household 1')
plt.plot(t,hh1['bl'],c='k',ls=':')
plt.plot(t,hh1['lm'],label='loss minimise')
plt.plot(t,hh1['lf'],label='load flatten')
plt.xlim(0,24)
plt.legend(loc=[0.3,1.15],ncol=2)
plt.xticks(x,x_ticks)
plt.grid()


plt.subplot(2,1,2)
plt.title('Household 55')
plt.plot(t,hh54['bl'],c='k',ls=':')
plt.plot(t,hh54['lm'])
plt.plot(t,hh54['lf'])
plt.xlim(0,24)
plt.xticks(x,x_ticks)
plt.grid()

plt.show()        

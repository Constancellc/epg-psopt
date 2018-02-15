import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random
import copy
import win32com.client

Pmax = 3.5 # kW
pph = 60
T = 24*pph
t_int = int(60/pph)

plotIndividual = False
skipUnused = True
constrainAvaliability = False

nRuns = 100
losses = {'lm':[],'lf':[],'uc':[],'po':{}}
predictions = {'lm':[],'lf':[],'uc':[]}
totalLoads = {'lm':[],'lf':[],'uc':[],'po':[]}
if plotIndividual == True:
    individuals = {'lm':{0:[],54:[]},'lf':{0:[],54:[]},'bl':{0:[],54:[]}}

# getting national valley filling profile
std = [24385,23518,23089,22818,22492,22280,21962,21991,21856,21881,22370,23502,
       26891,29993,33385,35125,36058,35948,36440,36550,36333,36407,36364,36397,
       36172,36093,36006,35659,35491,35364,35381,36081,36998,38085,38645,38831,
       38944,39333,40428,40440,39095,37713,36167,33910,31752,29580,27250,25256]

# inverting
off = max(std)+1
for i in range(std):
    std[i] = off-std[i]

# interpolating
new = []
for j in range(0,1440):
    p1 = int(j/30)
    p2 = p1 + 1
    f = float(j%30)/30
    new.append((1-f)*hh[p1] + f*hh[p2])
std = new

household_profiles = []
vehicle_profiles = []
for i in range(0,1000):
    household_profiles.append([0.0]*1440)

# the following is using CREST
#'''
i = 0
with open('household_demand_pool.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row == []:
            continue
        for j in range(0,1000):
            household_profiles[j][i] = float(row[j])
        i += 1
'''

# and this uses smart meter data
i = 0
with open('household_demand_pool_HH.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        hh = []
        for cell in row:
            hh.append(float(cell))
        hh.append(hh[0])
        for j in range(0,1440):
            p1 = int(j/30)
            p2 = p1 + 1
            f = float(j%30)/30
            household_profiles[i][j] = (1-f)*hh[p1] + f*hh[p2]
        i += 1
'''
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
        
for mc in range(nRuns):
    
    lm_profiles = []
    lf_profiles = []
    po_profiles = []
    uc_profiles = []
    for i in range(55):
        lm_profiles.append([0.0]*1440)
        lf_profiles.append([0.0]*1440)
        po_profiles.append([0.0]*1440)
        uc_profiles.append([0.0]*1440)
        
    # pick households
    unused = []
    chosen = []
    while len(chosen) < 55:
        ran = int(random.random()*len(household_profiles))
        if ran not in chosen:
            chosen.append(ran)
            
    # then pick the vehicles    
    chosenV = []

    while len(chosenV) < 55:
        ran = int(random.random()*len(vehicle_profiles))
        if ran not in chosenV:
            chosenV.append(ran)

    energyV = [] # vector containing energy req in kWh of each vehicle
    energyHH = []
    avaliable = [] # contains range for which charging NOT allowed

    for v in range(55):
        energyV.append(float(sum(vehicle_profiles[chosenV[v]]))/60)
            
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
    # x[n:2n] is the imaginary power at the first time instant

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
    predictions['lm'].append(y.T*P*y + q.T*y + c*T)

    for j in range(n):
        p = []
        for t in range(T):
            p.append(float(-x[n*t+j])/1000)
        # now I may need to interpolate
        for t in range(1440):
            p1 = int(t/t_int)
            p2 = p1 + 1
            if p2 == T:
                p2 -= 1
            f = float(t%t_int)
            lm_profiles[ind_map[j]][t] = (1-f)*p[p1] + f*p[p2] + \
                                         household_profiles[chosen[ind_map[j]]][t]

    for j in unused:
        for t in range(1440):
            lm_profiles[j][t] = household_profiles[chosen[j]][t]
            lf_profiles[j][t] = household_profiles[chosen[j]][t]

    # NOW EMBARKING ON THE LOAD FLATTENING FOR COMPARISON

    # getting base load to be flattened
    bl = [0.0]*T
    for t in range(T):
        for j in range(55):
            bl[t] += household_profiles[chosen[j]][t*t_int]
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

    q1 = []
    for i in range(n):
        q1 += bl

    q1 = matrix(q1)
    I = spdiag([1]*T)
    P1 = sparse([[I]*n]*n)

    sol = solvers.qp(P1,q1,G,h,A,b)
    x = sol['x']

    for j in range(n):
        p = []
        for t in range(T):
            p.append(x[j*T+t])
        # now I may need to interpolate
        for t in range(1440):
            p1 = int(t/t_int)
            p2 = p1 + 1
            if p2 == T:
                p2 -= 1
            f = float(t%t_int)
            lf_profiles[ind_map[j]][t] = (1-f)*p[p1] + f*p[p2] + \
                                         household_profiles[chosen[ind_map[j]]][t]
    y2 = matrix(0.0,(n*T,1))
    for t in range(T):
        for i in range(n):
            y2[n*t+i] = -1000*x[i*T+t]

    y2 += x_h
    predictions['lf'].append(y2.T*P*y2 + q.T*y2 + c*T)
    
    if plotIndividual == True:
        for hh in [0,54]:
            individuals['lf'][hh].append(lf_profiles[hh])
            individuals['lm'][hh].append(lm_profiles[hh])
            individuals['bl'][hh].append(household_profiles[chosen[hh]])

    # for comparison let's get the uncontrolled charging results

    for i in range(55):
        for t in range(1440):
            uc_profiles[i][t] = household_profiles[chosen[i]][t] + \
                                vehicle_profiles[chosenV[i]][t]

    # and the psuedo optimal

    for i in range(55):
        req_en = sum(vehicle_profiles[chosenV[i]])
        p = copy.copy(std)

        sf = req_en/sum(p)
        for t in range(1440):
            po_profiles[i][t] = household_profiles[chosen[i]][t] + \
                                p[chosenV[i]][t]*sf
    
    # NOW FOR THE OPEN DSS PORTION OF THE EVENT
    engine = win32com.client.Dispatch("OpenDSSEngine.DSS")
    engine.Start("0")

    runs = {'lf':lf_profiles,'lm':lm_profiles,'po':uc_profiles,
            'uc':uc_profiles}

    for key in runs:
        profiles = runs[key]

        for i in range(1,56):
            with open('household-profiles/'+str(i)+'.csv','w') as csvfile:
                writer = csv.writer(csvfile)
                for t in range(1440):
                    writer.writerow([profiles[i-1][t]])

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

        totalLoads[key].append(powerOut)
        net = []
        for i in range(0,1440):
            net.append(powerIn[i]-powerOut[i])

        losses[key].append(sum(net))

for key in losses:
    with open(key+'_loads.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        for t in range(1440):
            row = []
            for i in range(len(totalLoads[key])):
                row.append(totalLoads[key][i][t])
            writer.writerow(row)
            
with open('lf_lm_losses.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['load flattening','loss minimising','pusedo optimal',
                     'uncontrolled','total Load'])#,'delta','predicted delta'])
    for i in range(nRuns):
        writer.writerow([losses['lf'][i],losses['lm'][i],losses['po'][i],
                         losses['uc'][i],sum(totalLoads['lf'][i])])
#                         losses['lf'][i]-losses['lm'][i],
#                         (predictions['lf'][i]-predictions['lm'][i])[0]/1000])

if plotIndividual == True:
    for hh in [0,54]:
        with open('lf_lm_inidividuals'+str(hh)+'.csv','w') as csvfile:
            writer = csv.writer(csvfile)
            header = ['t']
            for i in range(nRuns):
                header += ['lf','lm','bl']
            writer.writerow(header)
            for t in range(1440):
                row = [str(t)]
                for i in range(nRuns):
                    row.append(individuals['lf'][hh][i][t])
                    row.append(individuals['lm'][hh][i][t])
                    row.append(individuals['bl'][hh][i][t])
                writer.writerow(row)

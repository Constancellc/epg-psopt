import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random
#import win32com.client

# the goal here is to get a map of the losses resulting from two
# charging power combinations

step = 0.1 #
Pmax = 1.0 # max individual charging power

base = 0.0 # kW load of all unused houses

hh_A = 10 # test households
hh_B = 50

outfile = 'loss_model_test_map.csv'

poss = []
p = Pmax

appl = {hh_A:[],hh_B:[]}

while p >= 0:
    poss.append(p)
    p -= step
    p = round(p,2)

pairs = []
for i in range(len(poss)):
    for j in range(len(poss)):
        pairs.append([poss[i],poss[j]])

N_t = len(pairs) # the number of required time steps
nDays = int(N_t/1440)+1 # the number of required simulation days

while len(pairs) < nDays*1440:
    pairs.append([0.0,0.0])

# first set all empty households to the base load
for i in range(1,56):
    if i == hh_A or i == hh_B:
        continue
    with open('household-profiles/'+str(i)+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        for t in range(1440):
            writer.writerow([base])

# start DSS
engine = win32com.client.Dispatch("OpenDSSEngine.DSS")
engine.Start("0")

results = []           
for sim in range(nDays):
    # set two households in question to correct values
    with open('household-profiles/'+str(hh_A)+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        for t in range(1440):
            writer.writerow([pairs[sim*1440+t][0]])
    with open('household-profiles/'+str(hh_B)+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        for t in range(1440):
            writer.writerow([pairs[sim*1440+t][1]])

    # set up power flow
    engine.text.Command='clear'
    circuit = engine.ActiveCircuit
    engine.text.Command='compile master.dss'
    engine.Text.Command='Export mon LINE1_PQ_vs_Time'

    # get power in and power out
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

    for hh in [hh_A,hh_B]:
        engine.Text.Command='Export mon hh'+str(hh)+'_pQ_vs_time'

        i = 0
        with open('LVTest_Mon_hh'+str(hh)+'_pq_vs_time.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                powerOut[i] += float(row[2])
                appl[hh].append(float(row[2]))
                i += 1
                
    # store losses results
    for t in range(1440):
        results.append([appl[hh_A][sim*1440+t]]+[appl[hh_B][sim*1440+t]]
                        +[powerIn[t]-powerOut[t]])
        #results.append(pairs[sim*1440+t]+[powerIn[t]-powerOut[t]])

# write results to csv file
with open(outfile,'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['LoadA','LoadB','Losses'])
    for row in results:
        writer.writerow(row)

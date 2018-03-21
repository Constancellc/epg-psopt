import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random

from lv_optimization import LVTestFeeder

household_profiles = []
with open('data/household_demand_pool_HH.csv','rU') as csvfile:
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

vehicle_pool = []
with open('data/EVchargingWedJanUT.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row == []:
            continue
        if float(row[0]) != 0:
            vehicle_pool.append([float(row[0]),int(360+random.random()*360),
                                 int(720+random.random()*720)])
        else:
            vehicle_pool.append([0.01,int(float(row[1])),
                                 int(float(row[2]))])

outfile = '../../../Documents/simulation_results/LV/total_losses/'
plt.figure(figsize=(6,4))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 9

t = np.linspace(0,24,num=1440)
x = [2,6,10,14,18,22]
x_ticks = ['02:00','06:00','10:00','14:00','18:00','22:00']
for mc in range(1):
    chosen = []
    while len(chosen) < 55:
        ran = int(random.random()*len(household_profiles))
        chosen.append(household_profiles[ran])
    # set two hh to same base load
    chosen[53] = chosen[0]
            
    # for actual EV demands
    energy = []
    while len(energy) < 55:
        ran = int(random.random()*len(vehicle_pool))
        if vehicle_pool[ran] not in energy:
            energy.append([5.0]+vehicle_pool[ran][1:])

           
    feeder = LVTestFeeder()
    feeder.set_households(chosen)
    feeder.set_evs(energy)

    feeder.load_flatten(7,constrain=False)
    base1, combined1 = feeder.get_inidividual_load(0)
    base54, combined54 = feeder.get_inidividual_load(53)
    plt.subplot(2,1,1)
    plt.plot(t,base1,'k',ls=':',label='Base Load')
    plt.plot(t,combined1,'b',label='Node 1')
    plt.plot(t,combined54,'r',ls='--',label='Node 54')
    plt.xlim(0,24)
    plt.ylim(0,max(combined1)*1.5)
    plt.title('Load Flattening',y=0.8)
    plt.xticks(x,x_ticks)
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.grid()

    
    feeder.loss_minimise(7,constrain=False)
    base1, combined1 = feeder.get_inidividual_load(0)
    base54, combined54 = feeder.get_inidividual_load(53)
    
    plt.subplot(2,1,2)
    plt.plot(t,base1,'k',ls=':',label='Base Load')
    plt.plot(t,combined1,'b',label='Node 1')
    plt.plot(t,combined54,'r',ls='--',label='Node 54')
    plt.xlim(0,24)
    plt.ylim(0,max(combined1)*1.5)
    plt.title('Loss Minimising',y=0.8)
    plt.xticks(x,x_ticks)
    plt.ylabel('Power (kW)')
    plt.grid()
plt.tight_layout()
#plt.savefig('../../../Dropbox/papers/losses/individuals.eps', format='eps', dpi=1000)
plt.show()




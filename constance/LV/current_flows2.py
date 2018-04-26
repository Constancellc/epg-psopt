import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random
import copy

from lv_optimization import LVTestFeeder

Node2 = 53
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
        if household_profiles[ran] not in chosen:
            chosen.append(household_profiles[ran])
    # set two hh to same base load
    chosen[Node2] = chosen[0]
            
    # for actual EV demands
    energy = []
    while len(energy) < 55:
        ran = int(random.random()*len(vehicle_pool))
        if vehicle_pool[ran] not in energy:
            energy.append([5.0]+vehicle_pool[ran][1:])
    # also need same energy
    energy[Node2] = energy[0]

    feeder = LVTestFeeder()
    feeder.set_households(chosen)
    feeder.set_evs(energy)
    

    feeder.load_flatten(7,constrain=False)
    base1, combined1 = feeder.get_inidividual_load(0)
    base54, combined54 = feeder.get_inidividual_load(Node2)
    c2 = feeder.getLineCurrents()
    feeder.ev[46] = [0.0]*1440
    c3 = feeder.getLineCurrents()
    
    plt.subplot(2,1,1)
    plt.plot(t,base1,'k',ls=':',label='Base Load')
    plt.plot(t,combined1,'b',label='Node 1')
    plt.plot(t,combined54,'r',ls='--',label='Node '+str(Node2+1))
    plt.xlim(0,24)
    ym = max(combined1)*1.5
    plt.ylim(0,ym)
    plt.title('Load Flattening',y=0.8)
    plt.xticks(x,x_ticks)
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.grid()

    
    feeder.loss_minimise(7,constrain=False)
    base1, combined1 = feeder.get_inidividual_load(0)
    base54, combined54 = feeder.get_inidividual_load(Node2)
    c4 = feeder.getLineCurrents()
    feeder.ev[Node2] = [0.0]*1440
    c5 = feeder.getLineCurrents()
    
    plt.subplot(2,1,2)
    plt.plot(t,base1,'k',ls=':',label='Base Load')
    plt.plot(t,combined1,'b',label='Node 1')
    plt.plot(t,combined54,'r',ls='--',label='Node '+str(Node2+1))
    plt.xlim(0,24)
    plt.ylim(0,ym)
    plt.title('Loss Minimising',y=0.8)
    plt.xticks(x,x_ticks)
    plt.ylabel('Power (kW)')
    plt.grid()
plt.tight_layout()
#plt.savefig('../../../Dropbox/papers/losses/individuals2.eps', format='eps', dpi=1000)

plt.figure(figsize=(6,4))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 9

vehicleOnly = []
for tt in range(1440):
    vehicleOnly.append(combined54[tt]-base1[tt])
        
for mc in range(1):
    plt.subplot(2,1,1)
    plt.title('Vehicle Charging',y=0.8)
    plt.plot(t,vehicleOnly,'b')
    plt.ylim(0,max(vehicleOnly)*1.2)
    plt.grid()
    plt.xlim(0,24)
    plt.ylabel('Power (kW)')
    plt.xticks(x,x_ticks)
    
    for i in range(1,2):
        plt.subplot(2,1,2)
        plt.title('Line 296-A Current',y=0.8)
        plt.plot(t,c4[i],'b',label='With Vehicle')
        plt.plot(t,c5[i],'k',ls=':',label='Without Vehicle')
        plt.grid()
        plt.xlim(0,24)
        plt.ylabel('Current (A)')
        plt.xticks(x,x_ticks)
        plt.legend()
    
plt.tight_layout()



#plt.savefig('../../../Dropbox/papers/losses/branch_currents2.eps', format='eps', dpi=1000)

plt.show()

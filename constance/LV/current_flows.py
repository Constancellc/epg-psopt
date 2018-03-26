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


plt.figure(figsize=(5,4))
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
            
    # for actual EV demands
    energy = []
    while len(energy) < 55:
        ran = int(random.random()*len(vehicle_pool))
        if vehicle_pool[ran] not in energy:
            energy.append(vehicle_pool[ran])

           
    feeder = LVTestFeeder()
    feeder.set_households(chosen)
    feeder.set_evs(energy)
    
    feeder.load_flatten(7,constrain=False)
    c2 = feeder.getLineCurrents()
    
    lbls = {0:'Line 110',1:'Line 296'}
    for i in range(2):
        plt.subplot(2,1,i+1)
        plt.title(lbls[i],y=0.8)
        plt.plot(t,c2[i],'b',label='Load Flattening')
        plt.xlim(0,24)
        plt.ylim(0.5*min(c2[i]),1.15*max(c2[i]))
        
        plt.xticks(x,x_ticks)
        #plt.ylim(0.9*min(c2[3]),1.1*max(c2[1]))
        plt.grid()
        plt.ylabel('Current (A)')
            
    totallm = feeder.loss_minimise(7)
    c3 = feeder.getLineCurrents()
    
    for i in range(2):
        plt.subplot(2,1,i+1)
        plt.title(lbls[i])
        plt.plot(t,c3[i],'r',ls='--',label='Loss Minimising')
        if i == 0:
            plt.legend(loc=3)
plt.tight_layout()
#plt.savefig('../../../Dropbox/papers/losses/branch_currents.eps', format='eps', dpi=1000)

plt.show()

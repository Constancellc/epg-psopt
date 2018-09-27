import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random

from lv_optimization2 import LVTestFeeder

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
        vehicle_pool.append([float(row[0]),int(float(row[1])),
                             int(float(row[2]))])

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

    feeder.uncontrolled(3.5)
    u3 = feeder.predict_lowest_voltage()
    feeder.load_flatten(7,constrain=True)
    lf = feeder.predict_lowest_voltage()
    feeder.loss_minimise(7,constrain=True)
    lm = feeder.predict_lowest_voltage()

t = np.linspace(0,24,num=1440)
plt.figure(figsize=(6,2))
plt.rcParams["font.family"] = 'serif'
plt.rcParams["font.size"] = '9'
plt.plot(t,u3,c='g',ls=':',label='Uncontrolled')
plt.plot(t,lf,c='b',label='Load Flattening')
plt.plot(t,lm,c='r',ls='--',label='Loss Minimizing')
plt.legend(ncol=3)
plt.ylabel('Lowest Bus Voltage (V)')
plt.tight_layout()
plt.xlim(0,24)
plt.ylim(240,251)
plt.grid()
plt.xticks([2,6,10,14,18,22],['02:00','06:00','10:00','14:00','18:00','22:00'])
plt.savefig('../../../Dropbox/papers/losses/img/voltages2.eps', format='eps', dpi=1000)
plt.show()


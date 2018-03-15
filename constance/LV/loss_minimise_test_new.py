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
        vehicle_pool.append([float(row[0]),int(float(row[1])),
                             int(float(row[2]))])

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
feeder.load_flatten(6,constrain=False)
total3 = feeder.get_feeder_load()
base2, combined2 = feeder.get_inidividual_load(54)
print(sum(feeder.predict_losses()))
feeder.loss_minimise(6,constrain=False)
total2 = feeder.get_feeder_load()
base1, combined1 = feeder.get_inidividual_load(54)
print(sum(feeder.predict_losses()))

plt.figure(1)
plt.plot(total2)
plt.plot(total3)
plt.plot(feeder.base)

plt.figure(2)
plt.subplot(2,1,1)
plt.plot(base1)
plt.plot(combined1)
plt.subplot(2,1,2)
plt.plot(base2)
plt.plot(combined2)
plt.show()

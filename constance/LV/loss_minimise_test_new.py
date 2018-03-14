import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random

from lv_optimization import LVTestFeeder

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
    if household_profiles[ran] not in chosen:
        chosen.append(household_profiles[ran])
        
# first I need to set up the energy demands
energy = [] # kWh
for hh in range(55):
    energy.append([1,int(200+random.random()*320),int(random.random()*320+720)])

feeder = LVTestFeeder()
feeder.set_households(chosen)
feeder.set_evs(energy)
feeder.load_flatten(6,constrain=True)
total3 = feeder.get_feeder_load()
base2, combined2 = feeder.get_inidividual_load(54)
print(sum(feeder.predict_losses()))
feeder.regularised_loss_minimise(6,constrain=True)
total1 = feeder.get_feeder_load()
base, combined = feeder.get_inidividual_load(54)
print(sum(feeder.predict_losses()))
feeder.loss_minimise(6,constrain=True)
total2 = feeder.get_feeder_load()
base1, combined1 = feeder.get_inidividual_load(54)
print(sum(feeder.predict_losses()))

plt.figure(1)
plt.plot(total1)
plt.plot(total2)
plt.plot(total3)
plt.plot(feeder.base)

plt.figure(2)
plt.subplot(3,1,1)
plt.plot(base)
plt.plot(combined)
plt.subplot(3,1,2)
plt.plot(base1)
plt.plot(combined1)
plt.subplot(3,1,3)
plt.plot(base2)
plt.plot(combined2)
plt.show()

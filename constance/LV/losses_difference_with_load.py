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
            vehicle_pool.append([0.01,int(360+random.random()*360),
                                 int(720+random.random()*720)])

outfile = '../../../Documents/simulation_results/LV/losses_difference_with_load.csv'
results = []
for mc in range(3000):
    chosen = []
    while len(chosen) < 55:
        ran = int(random.random()*len(household_profiles))
        if household_profiles[ran] not in chosen:
            chosen.append(household_profiles[ran])
            
    # all EVs having the same energy requirement
    energy = []
    en = float(int(mc/100)+1)#random.random()*30)
    while len(energy) < 55:
        ran = int(random.random()*len(vehicle_pool))
        if vehicle_pool[ran] not in energy:
            energy.append([en]+vehicle_pool[ran][1:])

    try:           
        feeder = LVTestFeeder()
        feeder.set_households(chosen)
        feeder.set_evs(energy)

        Load0 = sum(feeder.get_feeder_load())
        feeder.load_flatten(7,constrain=False)
        lf = sum(feeder.predict_losses())/Load0
        feeder.loss_minimise(7,constrain=False)
        lm = sum(feeder.predict_losses())/Load0

        results.append([en,lf,lm])
    except:
        continue

with open(outfile,'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Demand per EV (kWh)','load flattening','loss minimising'])
    for row in results:
        writer.writerow(row)
        

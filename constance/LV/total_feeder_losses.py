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

for mc in range(200):
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

    base = feeder.predict_losses()
    Load0 = sum(feeder.get_feeder_load())
    feeder.uncontrolled(3.5)
    totalu3 = feeder.predict_losses()
    Load = sum(feeder.get_feeder_load())
    feeder.uncontrolled(7.0)
    totalu7 = feeder.predict_losses()
    feeder.load_flatten(7,constrain=False)
    totallf = feeder.predict_losses()
    feeder.loss_minimise(7,constrain=False)
    totallm = feeder.predict_losses()

    with open(outfile+str(mc+1)+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Load w/out EVs:',Load0,'Load w/ EVs:',Load,'(both in kWmin)'])
        writer.writerow(['time','no_ev','uncontrolled3.5','uncontolled7',
                         'load_flatten','loss_minimise'])
        for t in range(1440):
            writer.writerow([t+1,base[t],totalu3[t],totalu7[t],totallf[t],
                             totallm[t]])



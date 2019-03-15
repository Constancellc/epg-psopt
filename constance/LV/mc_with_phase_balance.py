import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from lv_optimization_new import LVTestFeeder

fdr = '1'
runs = 100

tr = 1
nEVs = 55

phase = ['A','B','A','A','A','B','B','C','A','B','B','C','B','A','B','C','C',
         'C','C','A','A','A','B','C','A','B','C','C','A','A','A','C','C','A',
         'B','B','B','B','C','B','B','C','C','B','B','A','C','A','A','B','A',
         'A','B','A','A']

network = LVTestFeeder('manc_models/'+fdr,t_res=tr)

lds = {'b':[],'u':[],'m':[],'f':[],'p':[]}
lss = {'b':[],'u':[],'m':[],'f':[],'p':[]}

for mc in range(runs):
    print(mc)
    network.set_households_NR('../../../Documents/netrev/TC2a/03-Dec-2013.csv')
    network.set_evs_MEA('../../../Documents/My_Electric_Avenue_Technical_Data/'+
                        'constance/ST1charges/',nEVs=nEVs)
    b = network.get_feeder_load()
    l_b = network.predict_losses()

    network.uncontrolled()
    p = network.get_feeder_load()
    l_u = network.predict_losses()

    try:
        network.load_flatten()
    except:
        continue
    
    p2 = network.get_feeder_load()
    l_f = network.predict_losses()

    try:
        network.loss_minimise()
    except:
        continue

    if network.status != 'optimal':
        print(network.status)
        continue
    p3 = network.get_feeder_load()
    l_m = network.predict_losses()

    try:
        network.balance_phase2(phase)
    except:
        continue
    
    p4 = network.get_feeder_load()
    l_p = network.predict_losses()

    lds['b'].append(b)
    lds['u'].append(p)
    lds['f'].append(p2)
    lds['m'].append(p3)
    lds['p'].append(p4)
    
    lss['b'].append(l_b)
    lss['u'].append(l_u)
    lss['f'].append(l_f)
    lss['m'].append(l_m)
    lss['p'].append(l_p)
'''
with open('../../../Documents/simulation_results/LV/varying-pen/'+str(nEVs)+\
          '-losses.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Base','Unc','LF','LM'])
    for s in range(len(lss['b'])):
        writer.writerow([lss['b'][s],lss['u'][s],lss['f'][s],lss['m'][s]])
        
'''                       
with open('../../../Documents/simulation_results/LV/manc-models/'+fdr+\
          '-loads-b.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['t','runs'])
    for t in range(int(1440/tr)):
        row = [t]
        for s in range(len(lss['b'])):
            row.append(lds['b'][s][t])
        writer.writerow(row)
        
with open('../../../Documents/simulation_results/LV/manc-models/'+fdr+\
          '-loads-u.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['t','runs'])
    for t in range(int(1440/tr)):
        row = [t]
        for s in range(len(lss['u'])):
            row.append(lds['u'][s][t])
        writer.writerow(row)
        
with open('../../../Documents/simulation_results/LV/manc-models/'+fdr+\
          '-loads-f.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['t','runs'])
    for t in range(int(1440/tr)):
        row = [t]
        for s in range(len(lss['f'])):
            row.append(lds['f'][s][t])
        writer.writerow(row)
        
with open('../../../Documents/simulation_results/LV/manc-models/'+fdr+\
          '-loads-m.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['t','runs'])
    for t in range(int(1440/tr)):
        row = [t]
        for s in range(len(lss['m'])):
            row.append(lds['m'][s][t])
        writer.writerow(row)
        
with open('../../../Documents/simulation_results/LV/manc-models/'+fdr+\
          '-loads-p.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['t','runs'])
    for t in range(int(1440/tr)):
        row = [t]
        for s in range(len(lss['p'])):
            row.append(lds['p'][s][t])
        writer.writerow(row)

with open('../../../Documents/simulation_results/LV/manc-models/'+fdr+\
          '-losses.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Base','Unc','LF','LM','LF+P'])
    for s in range(len(lss['b'])):
        writer.writerow([lss['b'][s],lss['u'][s],lss['f'][s],lss['m'][s],
                         lss['p'][s]])

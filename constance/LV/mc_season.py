import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from lv_optimization_new import LVTestFeeder
import time

fdr = '1'
runs = 1

tr = 1

phaseInfo = {'1':'eulvLptloadBusesCc-24','2':'021LptloadBusesCc-15',
             '3':'031LptloadBusesCc-15','4':'041LptloadBusesCc-82',
             '024':'024LptloadBusesCc-05','041':'041LptloadBusesCc-82',
             '074':'074LptloadBusesCc-15','162':'162LptloadBusesCc-24',
             '193':'193LptloadBusesCc-24','213':'213LptloadBusesCc-44'}

# first get phases
lds = np.load('../../../Documents/ccModels/loadBuses/'+phaseInfo[fdr]+'.npy')
lds = lds.flatten()[0]
phase = []
for i in range(len(lds)):
    bus = lds['load'+str(i+1)]
    if bus[-1] == '1':
        phase.append('A')
    elif bus[-1] == '2':
        phase.append('B')
    elif bus[-1] == '3':
        phase.append('C')

lds = {'b':[],'u':[],'m':[],'f':[],'p':[]}
lss = {'b':[],'u':[],'m':[],'f':[],'p':[]}

for nEVs in [55]:
    network = LVTestFeeder('manc_models/'+fdr,t_res=tr)
    for mc in range(runs):
        print(mc)
        #network.set_households_NR('../../../Documents/netrev/TC2a/03-Dec-2013.csv')
        network.set_households_NR('../../../Documents/netrev/TC2a/08-Jul-2013.csv')
        #network.set_households_NR('../../../Documents/netrev/TC2a/03-Mar-2014.csv')
        #network.set_households_NR('../../../Documents/netrev/TC2a/18-Oct-2013.csv')
        network.set_evs_MEA('../../../Documents/My_Electric_Avenue_Technical_Data/'+
                            'constance/ST1charges/',nEVs=nEVs)
        b = network.get_feeder_load()
        l_b = network.predict_losses()

        network.uncontrolled()
        p = network.get_feeder_load()
        l_u = network.predict_losses()
        t0 = time.time()

        try:
            network.load_flatten()
        except:
            continue
        
        t1 = time.time()
        
        #p2 = network.get_feeder_load()
        #l_f = network.predict_losses()

        try:
            network.loss_minimise()
        except:
            continue
        
        t2 = time.time()

        if network.status != 'optimal':
            print(network.status)
            continue
        #p3 = network.get_feeder_load()
        #l_m = network.predict_losses()

        network.balance_phase2(phase)

        
        t3 = time.time()

        print(t1-t0)
        print(t2-t1)
        print(t3-t2)

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

    with open('../../../Documents/simulation_results/LV/manc-models/'+fdr+\
              'jul-losses.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Base','Unc','LF','LM','PB'])
        for s in range(len(lss['b'])):
            writer.writerow([lss['b'][s],lss['u'][s],lss['f'][s],lss['m'][s],
                             lss['p'][s]])'''
        
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
'''

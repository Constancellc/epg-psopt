import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from lv_optimization_new import LVTestFeeder
import time

fdr = '1'
runs = 10

# things I have run:

tr = 1

phaseInfo = {'1':'eulvLptloadBusesCc-24','2':'021LptloadBusesCc-15',
             '3':'031LptloadBusesCc-15','4':'041LptloadBusesCc-82',
             '024':'024LptloadBusesCc-05','041':'041LptloadBusesCc-82',
             '074':'074LptloadBusesCc-15','162':'162LptloadBusesCc-24',
             '193':'193LptloadBusesCc-24','213':'213LptloadBusesCc-44'}

# first get phases
'''
lds = np.load('../../../Documents/ccModels/loadBuses/'+phaseInfo[fdr]+'.npy')
lds = lds.flatten()[0]


nEVs = len(lds)
res = []
phase = []
for i in range(len(lds)):
    bus = lds['load'+str(i+1)]
    if bus[-1] == '1':
        phase.append('A')
    elif bus[-1] == '2':
        phase.append('B')
    elif bus[-1] == '3':
        phase.append('C')
'''
ph = {0:'A',1:'B',2:'C'}
res = []

network = LVTestFeeder('manc_models/'+fdr,t_res=tr)


for mc in range(runs):
    print(mc)
    phase = []
    for i in range(55):
        phase.append(ph[int(random.random()*3)])
    '''
    network.set_households_NR('../../../Documents/netrev/TC2a/03-Dec-2013.csv')
    
    network.set_evs_MEA('../../../Documents/My_Electric_Avenue_Technical_Data/'+
                        'constance/ST1charges/',nEVs=nEVs)
    '''
    network.set_households_synthetic(4)
    network.set_evs_synthetic(5)
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

    #p4 = network.get_feeder_load()
    #l_p = network.predict_losses()

    t3 = time.time()

    res.append([t1-t0,t2-t1,t3-t2])

for i in range(3):
    t = 0
    for j in range(len(res)):
        t += res[j][i]/len(res)
    print(t)

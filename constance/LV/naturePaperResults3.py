import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from lv_optimization_new import LVTestFeeder

r_type_data = '../../../Documents/census/LA_rural_urban.csv'
sf_data = '../../../Documents/simulation_results/NTS/clustering/power/'+\
          'locationsLA/lvScaling.csv'
res_stem = '../../../Documents/simulation_results/LV/LA/'

# okay this one is going to have several stages

# there are nearly 400 smulations to do

# there are 3 feeders - I need to select the right one

# I think I should do one feeder at once
rType = {}
with open(r_type_data,'rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        rType[row[0]] = row[1]

# then I need to get the scaling factor
hh_sf = {}
v_sf = {}
with open(sf_data,'rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        hh_sf[row[0]] = float(row[1])
        v_sf[row[0]] = float(row[2])
    
# then I need to run the simulations
runs = 200
alpha = 0.328684513701
tr = 10
rt = '2' # NOTE: 3 is Most urban!!!
fdr = 'n1'

a = np.load('../../../Documents/ccModels/'+fdr+'/'+fdr+'aCc060.npy')
My = np.load('../../../Documents/ccModels/'+fdr+'/'+fdr+'MyCc060.npy')
v0 = np.load('../../../Documents/ccModels/'+fdr+'/'+fdr+'V0Cc060.npy')

las = []
for la in rType:
    if rType[la] == rt:
        las.append(la)
        
i = 0
'''
while las[i] != 'E07000031':
    i += 1
'''

for la in las:
    print(la)
    if la not in hh_sf:
        continue

    for file in ['_load','_voltages_m','_voltages_p','_voltages_a']:
        res1 = []
        res2 = []
        with open(res_stem+la+file+'.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                x = []
                for i in range(len(row)):
                    x.append(float(row[i]))
                if len(res1) < 144:
                    res1.append(x)
                else:
                    res2.append(x)
        if len(res2) == 0:
            continue

        for t in range(144):
            for i in [1,4,7]:
                if res2[t][i] < res1[t][i]:
                    res1[t][i] = res2[t][i]
            for i in [3,6,9]:
                if res2[t][i] > res1[t][i]:
                    res1[t][i] = res2[t][i]
            for i in [2,5,8]:
                res1[t][i] = 0.5*(res1[t][i]+res2[t][i])

        
        with open(res_stem+la+file+'.csv','w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['t','b-','b','b+','u-','u','u+','f-','f','f+'])
            for row in res1:
                writer.writerow(row)
            
        
        

    
    
for ii in range(i,len(las)):#la in las:
    la = las[ii]
    print(la)
    if la not in hh_sf:
        continue
    network = LVTestFeeder('manc_models/'+fdr,t_res=tr)
    #network.nH += 399
    results_p = {'b':[],'u':[],'f':[]}
    results_l = {'b':[],'u':[],'f':[]}
    results_vp = {'b':[],'u':[],'f':[]}
    results_va = {'b':[],'u':[],'f':[]}
    results_vm = {'b':[],'u':[],'f':[]}

    for mc in range(runs):
        if mc%10 == 0:
            print(mc)
        network.set_households_NR('../../../Documents/netrev/TC2a/03-Dec-2013.csv',
                                  sf=hh_sf[la])
        network.set_evs_MEA('../../../Documents/My_Electric_Avenue_Technical_Data/'+
                                'constance/ST1charges/',sf=v_sf[la])
        
        results_p['b'].append(network.get_feeder_load())
        results_l['b'].append(network.predict_losses())
        #results_l['b'].append(network.predict_losses_cheat(125,150,125))
    
        v = network.get_all_voltages_mag(My,a,alpha,v0,cut=6)
        #v = network.get_all_voltages_mag_cheat(My,a,alpha,v0,125,150,125)
        vav = []
        vm = []
        vu = []
        for t in v:
            vav.append(sum(v[t])/len(v[t]))
            vm.append(min(v[t]))
            vu.append(max(v[t]))
            
        results_vp['b'].append(vu)
        results_va['b'].append(vav)
        results_vm['b'].append(vm)

        network.uncontrolled()
        results_p['u'].append(network.get_feeder_load())
        results_l['u'].append(network.predict_losses())
        #results_l['u'].append(network.predict_losses_cheat(125,150,125))
        v = network.get_all_voltages_mag(My,a,alpha,v0)
        #v = network.get_all_voltages_mag_cheat(My,a,alpha,v0,125,150,125)
        vav = []
        vm = []
        vu = []
        for t in v:
            vav.append(sum(v[t])/len(v[t]))
            vm.append(min(v[t]))
            vu.append(max(v[t]))
            
        results_vp['u'].append(vu)
        results_va['u'].append(vav)
        results_vm['u'].append(vm)

        network.load_flatten2()

        results_p['f'].append(network.get_feeder_load())
        results_l['f'].append(network.predict_losses())
        #results_l['f'].append(network.predict_losses_cheat(125,150,125))
        v = network.get_all_voltages_mag(My,a,alpha,v0)
        #v = network.get_all_voltages_mag_cheat(My,a,alpha,v0,125,150,125)
        vav = []
        vm = []
        vu = []
        for t in v:
            vav.append(sum(v[t])/len(v[t]))
            vm.append(min(v[t]))
            vu.append(max(v[t]))
            
        results_vp['f'].append(vu)
        results_va['f'].append(vav)
        results_vm['f'].append(vm)

    if len(results_vp['f']) == 0:
        print('NOT GOOD')
        continue

    tys = ['b','u','f']
    res_map = {'_load':results_p,'_voltages_m':results_vm,
               '_voltages_p':results_vp,'_voltages_a':results_va}

    # finally, I need to store the results
    for file in ['_load','_voltages_m','_voltages_p','_voltages_a']:
        existing = []
        with open(res_stem+la+file+'.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                x = []
                for i in range(len(row)):
                    x.append(float(row[i]))
                existing.append(x)

        with open(res_stem+la+file+'.csv','w') as csvfile:
            writer = csv.writer(csvfile)
            for t in range(144):
                row = [t]
                for ty in tys
                    x = []
                    for m in range(len(res_map[file][ty])):
                        x.append(res_map[file][ty][m][t])
                    x = sorted(x)
                    row.append(x[0])
                    row.append(x[int(len(x)/2)])
                    row.append(x[-1])
                for i in [1,4,7]:
                    if existing[t][i] < row[i]:
                        row[i] = existing[t][i]
                for i in [3,6,9]:
                    if existing[t][i] > row[i]:
                        row[i] = existing[t][i]
                for i in [2,5,8]:
                    row[i] = 0.5*(row[i]+existing[t][i])
                writer.writerow(row)
                
    existing = []
    try:
        with open(res_stem+la+'_losses.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                existing.append(row)
    except:
        existing = []
        
    with open(res_stem+la+'_losses.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sim','b','u','f'])
        for row in existing:
            writer.writerow(row)
        for i in range(len(results_l['f'])):
            writer.writerow([i,results_l['b'][i],results_l['u'][i],
                             results_l['f'][i]])



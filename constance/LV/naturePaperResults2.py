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
runs = 4
alpha = 0.328684513701
tr = 10
rt = '1'
fdr = 'n26'

a = np.load('../../../Documents/ccModels/'+fdr+'/'+fdr+'aCc060.npy')
My = np.load('../../../Documents/ccModels/'+fdr+'/'+fdr+'MyCc060.npy')
v0 = np.load('../../../Documents/ccModels/'+fdr+'/'+fdr+'V0Cc060.npy')

las = []
for la in rType:
    if rType[la] == rt:
        las.append(la)

for la in las:
    print(la)
    if la not in hh_sf:
        continue
    network = LVTestFeeder('manc_models/'+fdr,t_res=tr)
    network.nH += 399
    results_p = {'b':[],'u':[],'f':[]}
    results_l = {'b':[],'u':[],'f':[]}
    results_vp = {'b':[],'u':[],'f':[]}
    results_va = {'b':[],'u':[],'f':[]}
    results_vm = {'b':[],'u':[],'f':[]}

    for mc in range(runs):
        network.set_households_NR('../../../Documents/netrev/TC2a/03-Dec-2013.csv',
                                  sf=hh_sf[la])
        network.set_evs_MEA('../../../Documents/My_Electric_Avenue_Technical_Data/'+
                                'constance/ST1charges/',sf=v_sf[la])
        
        results_p['b'].append(network.get_feeder_load())
        #results_l['b'].append(network.predict_losses())
        results_l['b'].append(network.predict_losses_cheat(125,150,125))
    
        #v = network.get_all_voltages_mag(My,a,alpha,v0,cut=6)
        v = network.get_all_voltages_mag_cheat(My,a,alpha,v0,125,150,125)
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
        #results_l['u'].append(network.predict_losses())
        results_l['u'].append(network.predict_losses_cheat(125,150,125))
        #v = network.get_all_voltages_mag(My,a,alpha,v0)
        v = network.get_all_voltages_mag_cheat(My,a,alpha,v0,125,150,125)
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

        try:
            network.load_flatten()
        except:
            continue

        if network.status != 'optimal':
            continue

        results_p['f'].append(network.get_feeder_load())
        #results_l['f'].append(network.predict_losses())
        results_l['f'].append(network.predict_losses_cheat(125,150,125))
        #v = network.get_all_voltages_mag(My,a,alpha,v0)
        v = network.get_all_voltages_mag_cheat(My,a,alpha,v0,125,150,125)
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
        continue

    # finally, I need to store the results
    with open(res_stem+la+'_load.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['t','b-','b','b+','u-','u','u+','f-','f','f+'])
        for t in range(144):
            row = [t]
            for ty in ['b','u','f']:
                x = []
                for m in range(len(results_p[ty])):
                    x.append(results_p[ty][m][t])
                x = sorted(x)
                row.append(x[0])
                row.append(x[int(len(x)/2)])
                row.append(x[-1])
            writer.writerow(row)
                
    with open(res_stem+la+'_voltages_p.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['t','b-','b','b+','u-','u','u+','f-','f','f+'])
        for t in range(144):
            row = [t]
            for ty in ['b','u','f']:
                x = []
                for m in range(len(results_vp[ty])):
                    x.append(results_vp[ty][m][t])
                x = sorted(x)
                row.append(x[0])
                row.append(x[int(len(x)/2)])
                row.append(x[-1])
            writer.writerow(row)
                
    with open(res_stem+la+'_voltages_a.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['t','b-','b','b+','u-','u','u+','f-','f','f+'])
        for t in range(144):
            row = [t]
            for ty in ['b','u','f']:
                x = []
                for m in range(len(results_va[ty])):
                    x.append(results_va[ty][m][t])
                x = sorted(x)
                row.append(x[0])
                row.append(x[int(len(x)/2)])
                row.append(x[-1])
            writer.writerow(row)
                
    with open(res_stem+la+'_voltages_m.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['t','b-','b','b+','u-','u','u+','f-','f','f+'])
        for t in range(144):
            row = [t]
            for ty in ['b','u','f']:
                x = []
                for m in range(len(results_vm[ty])):
                    x.append(results_vm[ty][m][t])
                x = sorted(x)
                row.append(x[0])
                row.append(x[int(len(x)/2)])
                row.append(x[-1])
            writer.writerow(row)
            
    with open(res_stem+la+'_losses.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sim','b','u','f'])
        for i in range(len(results_l['f'])):
            writer.writerow([i,results_l['b'][i],results_l['u'][i],
                             results_l['f'][i]])


'''
fdr = '1'
tr = 1

eulv/'
a = np.load(stem+'eulvLptaCc060.npy')
My = np.load(stem+'eulvLptMyCc060.npy')
v0 = np.load(stem+'eulvLptV0Cc060.npy')

network = LVTestFeeder('manc_models/'+fdr,t_res=tr)
results_p = {'b':[],'u':[],'f':[]}
results_l = {'b':[],'u':[],'f':[]}
results_vp = {'b':[],'u':[],'f':[]}
results_va = {'b':[],'u':[],'f':[]}
results_vm = {'b':[],'u':[],'f':[]}
for mc in range(runs):
    print(mc)
    network.set_households_NR('../../../Documents/netrev/TC2a/03-Dec-2013.csv')
    network.set_evs_MEA('../../../Documents/My_Electric_Avenue_Technical_Data/'+
                            'constance/ST1charges/')
    
    results_p['b'].append(network.get_feeder_load())
    results_l['b'].append(network.predict_losses())
    v = network.get_all_voltages_mag(My,a,alpha,v0)
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
    v = network.get_all_voltages_mag(My,a,alpha,v0)
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

    try:
        network.load_flatten()
    except:
        continue

    if network.status != 'optimal':
        continue

    results_p['f'].append(network.get_feeder_load())
    results_l['f'].append(network.predict_losses())
    v = network.get_all_voltages_mag(My,a,alpha,v0)
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

    


ty = ['b','u','f']

conf_u = 0.95
conf_l = 0.05

# this is demand
time = np.arange(0,1440,tr)
ttls = ['No Charging','Uncontrolled','Controlled']
xt = ['04:00','12:00','20:00']
xt_ = [240,720,1200]
plt.figure(figsize=(8,3.5))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 11
for i in range(3):
    plt.subplot(1,3,i+1)
    av = []
    ma = []
    mi = []

    for t in range(len(results_p[ty[i]][0])):
        x = []
        for r in range(len(results_p[ty[i]])):
            x.append(results_p[ty[i]][r][t])
        x = sorted(x)
        av.append(x[int(len(x)*0.5)])
        ma.append(x[int(len(x)*conf_u)])
        mi.append(x[int(len(x)*conf_l)])

    plt.plot(time,av,c='g')
    plt.title(ttls[i],y=0.85)
    if i == 0:
        plt.ylabel('Power Demand (kW)')
    else:
        plt.yticks([0,20,40,60,80,100,120],['','','','','','',''])
    plt.fill_between(time,mi,ma,color='#CCFFCC')
    plt.ylim(0,120)
    plt.xlim(0,1439)
    plt.grid(ls=':')
    plt.xticks(xt_,xt)
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/Nature/img/lv_power.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0)
# this is voltages
plt.figure(figsize=(8,3.5))
for i in range(3):
    plt.subplot(1,3,i+1)
    av = []
    ma = []
    mi = []

    for t in range(len(results_vp[ty[i]][0])):
        x = []
        for r in range(len(results_vp[ty[i]])):
            x.append(0.98*results_vp[ty[i]][r][t])
        x = sorted(x)
        av.append(x[int(len(x)*0.5)])
        ma.append(x[int(len(x)*conf_u)])
        mi.append(x[int(len(x)*conf_l)])

    plt.plot(time,av,c='r',label='Maximum')
    plt.fill_between(time,mi,ma,color='#FFCCCC')
    
    av = []
    ma = []
    mi = []

    for t in range(len(results_va[ty[i]][0])):
        x = []
        for r in range(len(results_va[ty[i]])):
            x.append(0.98*results_va[ty[i]][r][t])
        x = sorted(x)
        av.append(x[int(len(x)*0.5)])
        ma.append(x[int(len(x)*conf_u)])
        mi.append(x[int(len(x)*conf_l)])
        
    plt.plot(time,av,c='k',label='Average')
    #plt.fill_between(time,mi,ma,alpha=0.2)
    
    av = []
    ma = []
    mi = []

    for t in range(len(results_vm[ty[i]][0])):
        x = []
        for r in range(len(results_vm[ty[i]])):
            x.append(0.98*results_vm[ty[i]][r][t])
        x = sorted(x)
        av.append(x[int(len(x)*0.5)])
        ma.append(x[int(len(x)*conf_u)])
        mi.append(x[int(len(x)*conf_l)])

    plt.plot(time,av,c='b',label='Minimum')
    plt.fill_between(time,mi,ma,color='#CCCCFF')
    plt.xticks(xt_,xt)
    plt.xlim(0,1439)
    plt.grid(ls=':')
    plt.ylim(0.94,1.06)
    plt.title(ttls[i],y=0.85)

    if i == 0:
        plt.legend(loc=3)
        plt.ylabel('Voltage (p.u.)')
    else:
        plt.yticks([0.94,0.96,0.98,1.0,1.02,1.04,1.06],
                   ['','','','','','',''])
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/Nature/img/lv_voltages.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0)
    
    


# this is losses

plt.rcParams['font.size'] = 12
plt.figure(figsize=(5,3))

# losses first
m = []
q1 = []
q3 = []
u = []
l = []

for i in range(3):
    x = results_l[ty[i]]
    x = sorted(x)
    
    m.append(1*x[int(len(x)/2)])
    q1.append(1*x[int(len(x)*0.25)])
    q3.append(1*x[int(len(x)*0.75)])
    l.append(1*x[0])
    u.append(1*x[-1])


plt.scatter(range(1,len(m)+1),l,marker='_',c='gray')
plt.scatter(range(1,len(m)+1),u,marker='_',c='gray')
for i in range(len(m)):
    plt.plot([i+1,i+1],[l[i],q1[i]],c='gray')
    plt.plot([i+1,i+1],[q3[i],u[i]],c='gray')

# box
for i in range(len(m)):
    plt.plot([i+0.81,i+1.19],[m[i],m[i]],c='b',lw='2')
    plt.plot([i+0.8,i+1.2],[q1[i],q1[i]],c='k')
    plt.plot([i+0.8,i+1.2],[q3[i],q3[i]],c='k')
    plt.plot([i+1.2,i+1.2],[q1[i],q3[i]],c='k')
    plt.plot([i+0.8,i+0.8],[q1[i],q3[i]],c='k')    
plt.grid(ls=':')
plt.ylim(0,40)
plt.ylabel('Losses (kWh)')
plt.xticks([1,2,3],ttls)
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/Nature/img/lv_losses.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0)
plt.show()'''
        

import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from lv_optimization_new import LVTestFeeder

fdr = 'epriK1'
runs = 400
tr = 10
alpha = 0.328684513701

stem = '../../../Documents/ccModels/epriK1/'
a = np.load(stem+'epriK1a.npy')
My = np.load(stem+'epriK1My.npy')
v0 = np.load(stem+'epriK1LptV0Cc100.npy')

network = LVTestFeeder('manc_models/'+fdr,t_res=tr)
results_p = {'b':[],'u':[],'f':[]}
results_l = {'b':[],'u':[],'f':[]}
results_vp = {'b':[],'u':[],'f':[]}
results_va = {'b':[],'u':[],'f':[]}
results_vm = {'b':[],'u':[],'f':[]}
for mc in range(runs):
    print(mc)
    network.set_evs_and_hh_pecanstreet('../../../Documents/pecan-street/'+\
                                       '1min-texas/summer-18.csv')
    
    results_p['b'].append(network.get_feeder_load())
    results_l['b'].append(network.predict_losses())

    v = network.get_all_voltages_mag(My,a,alpha,v0,scale=1)
    vav = []
    vm = []
    vu = []
    for t in v:
        pu = []
        for i in range(len(v[t])):
            if v[t][i] > 1000:
                continue
            elif v[t][i] < 100:
                continue
            elif v[t][i] > 255:
                pu.append(v[t][i]/277)
            else:
                pu.append(v[t][i]/240)
                
        vav.append(sum(pu)/len(pu))
        vm.append(min(pu))
        vu.append(max(pu))
        
    results_vp['b'].append(vu)
    results_va['b'].append(vav)
    results_vm['b'].append(vm)

    network.uncontrolled()
    results_p['u'].append(network.get_feeder_load())
    results_l['u'].append(network.predict_losses())
    v = network.get_all_voltages_mag(My,a,alpha,v0,scale=1)
    vav = []
    vm = []
    vu = []
    for t in v:
        pu = []
        for i in range(len(v[t])):
            if v[t][i] > 1000:
                continue
            elif v[t][i] < 100:
                continue
            elif v[t][i] > 255:
                pu.append(v[t][i]/277)
            else:
                pu.append(v[t][i]/240)
                
        vav.append(sum(pu)/len(pu))
        vm.append(min(pu))
        vu.append(max(pu))
        
    results_vp['u'].append(vu)
    results_va['u'].append(vav)
    results_vm['u'].append(vm)
    try:
        network.load_flatten()
    except:
        print('boo')
        continue
    
    if network.status != 'optimal':
        continue

    results_p['f'].append(network.get_feeder_load())
    results_l['f'].append(network.predict_losses())
    v = network.get_all_voltages_mag(My,a,alpha,v0,scale=1)
            
    vav = []
    vm = []
    vu = []

    for t in v:
        pu = []
        for i in range(len(v[t])):
            if v[t][i] > 1000:
                continue
            elif v[t][i] < 100:
                continue
            elif v[t][i] > 255:
                pu.append(v[t][i]/277)
            else:
                pu.append(v[t][i]/240)
                
        vav.append(sum(pu)/len(pu))
        vm.append(min(pu))
        vu.append(max(pu))
        
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
        plt.yticks([-200,0,200,400,600,800,1000,1200],['','','','','','','',''])
    plt.fill_between(time,mi,ma,color='#CCFFCC')
    plt.ylim(-200,1200)
    plt.xlim(0,1439)
    plt.grid(ls=':')
    plt.xticks(xt_,xt)
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/Nature/img/texas_lv_power.eps', format='eps',
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
            x.append(results_vp[ty[i]][r][t])
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
            x.append(results_va[ty[i]][r][t])
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
            x.append(results_vm[ty[i]][r][t])
        x = sorted(x)
        av.append(x[int(len(x)*0.5)])
        ma.append(x[int(len(x)*conf_u)])
        mi.append(x[int(len(x)*conf_l)])

    plt.plot(time,av,c='b',label='Minimum')
    plt.fill_between(time,mi,ma,color='#CCCCFF')
    plt.xticks(xt_,xt)
    plt.xlim(0,1439)
    plt.grid(ls=':')
    plt.ylim(0.96,1.06)
    plt.title(ttls[i],y=0.85)

    if i == 0:
        plt.legend(loc=3)
        plt.ylabel('Voltage (p.u.)')
    else:
        plt.yticks([0.96,0.98,1.0,1.02,1.04,1.06],
                   ['','','','','',''])
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/Nature/img/texas_lv_voltages.eps', format='eps',
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
    
    m.append(0.001*x[int(len(x)/2)])
    q1.append(0.001*x[int(len(x)*0.25)])
    q3.append(0.001*x[int(len(x)*0.75)])
    l.append(0.001*x[0])
    u.append(0.001*x[-1])


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
#plt.ylim(0,40)
plt.ylabel('Losses (MWh)')
plt.xticks([1,2,3],ttls)
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/Nature/img/texas_lv_losses.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0)
plt.show()
        

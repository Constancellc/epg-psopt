import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

ss = ['mar','jul','oct','']

stem = '../../../Documents/simulation_results/LV/manc-models/1'
# losses first
m = []
q1 = []
q3 = []
u = []
l = []
_m = []
_q1 = []
_q3 = []
_u = []
_l = []

for s in ss:
    diff = []
    diff2 = []
    with open(stem+s+'-losses.csv','rU') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            diff.append(float(row[2])-float(row[3]))
            diff2.append(float(row[2])-float(row[4]))
            if diff[-1] < 0:
                diff[-1] = 0
            if diff2[-1] < 0:
                diff2[-1] = 0
    diff = sorted(diff)
    diff2 = sorted(diff2)
    
    m.append(1000*diff[int(len(diff)/2)]/55)
    q1.append(1000*diff[int(len(diff)*0.25)]/55)
    q3.append(1000*diff[int(len(diff)*0.75)]/55)
    l.append(1000*diff[9]/55)
    u.append(1000*diff[-10]/55)
    
    _m.append(1000*diff2[int(len(diff2)/2)]/55)
    _q1.append(1000*diff2[int(len(diff2)*0.25)]/55)
    _q3.append(1000*diff2[int(len(diff2)*0.75)]/55)
    _l.append(1000*diff2[9]/55)
    _u.append(1000*diff2[-10]/55)

plt.figure(figsize=(6,2))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 9
# whiskers
plt.scatter(np.arange(1,len(m)+1)-0.1,l,marker='_',c='gray')
plt.scatter(np.arange(1,len(m)+1)-0.1,u,marker='_',c='gray')
for i in range(len(m)):
    plt.plot([i+0.9,i+0.9],[l[i],q1[i]],c='gray')
    plt.plot([i+0.9,i+0.9],[q3[i],u[i]],c='gray')
# whiskers
plt.scatter(np.arange(1,len(m)+1)+0.1,_l,marker='_',c='gray')
plt.scatter(np.arange(1,len(m)+1)+0.1,_u,marker='_',c='gray')
for i in range(len(m)):
    plt.plot([i+1.1,i+1.1],[_l[i],_q1[i]],c='gray')
    plt.plot([i+1.1,i+1.1],[_q3[i],_u[i]],c='gray')

x_ticks = ['Spring','Summer','Autumn','Winter']
for i in range(len(m)):
    if i == 0:
        plt.plot([i+0.82,i+0.98],[m[i],m[i]],c='b',lw='2',
                 label='Loss Minimising')
    else:
        plt.plot([i+0.82,i+0.98],[m[i],m[i]],c='b',lw='2')
    plt.plot([i+0.8,i+1],[q1[i],q1[i]],c='k')
    plt.plot([i+0.8,i+1],[q3[i],q3[i]],c='k')
    plt.plot([i+1,i+1],[q1[i],q3[i]],c='k')
    plt.plot([i+0.8,i+0.8],[q1[i],q3[i]],c='k')
    
for i in range(len(m)):
    if i == 0:
        plt.plot([i+1.02,i+1.18],[_m[i],_m[i]],c='r',lw='2',
                 label='LF+Phase Balancing')
    else:
        plt.plot([i+1.02,i+1.18],[_m[i],_m[i]],c='r',lw='2')
    plt.plot([i+1,i+1.2],[_q1[i],_q1[i]],c='k')
    plt.plot([i+1,i+1.2],[_q3[i],_q3[i]],c='k')
    plt.plot([i+1.2,i+1.2],[_q1[i],_q3[i]],c='k')
    plt.plot([i+1,i+1],[_q1[i],_q3[i]],c='k')
plt.xticks(range(1,len(m)+1),x_ticks)
plt.ylabel('Losses reduction\n(Wh per household)')
plt.grid(linestyle=':')
plt.legend(ncol=2)
plt.tight_layout()
plt.ylim(0,60)
plt.savefig('../../../Dropbox/papers/losses/img/season_comp.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0)
plt.show()

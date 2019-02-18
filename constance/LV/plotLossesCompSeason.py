import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

ss = ['-mar','-jul','-oct','']

stem = '../../../Documents/simulation_results/LV/manc-models/1'
# losses first
m = []
q1 = []
q3 = []
u = []
l = []

for s in ss:
    diff = []
    with open(stem+s+'-losses.csv','rU') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            diff.append(float(row[2])-float(row[3]))
            if diff[-1] < 0:
                diff[-1] = 0
    diff = sorted(diff)
    
    m.append(1000*diff[int(len(diff)/2)]/55)
    q1.append(1000*diff[int(len(diff)*0.25)]/55)
    q3.append(1000*diff[int(len(diff)*0.75)]/55)
    l.append(1000*diff[0]/55)
    u.append(1000*diff[-1]/55)

plt.figure(figsize=(6,2))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 9
# whiskers
plt.scatter(range(1,len(m)+1),l,marker='_',c='gray')
plt.scatter(range(1,len(m)+1),u,marker='_',c='gray')
for i in range(len(m)):
    plt.plot([i+1,i+1],[l[i],q1[i]],c='gray')
    plt.plot([i+1,i+1],[q3[i],u[i]],c='gray')

x_ticks = ['Spring','Summer','Autumn','Winter']
for i in range(len(m)):
    plt.plot([i+0.61,i+1.39],[m[i],m[i]],c='b',lw='2')
    plt.plot([i+0.6,i+1.4],[q1[i],q1[i]],c='k')
    plt.plot([i+0.6,i+1.4],[q3[i],q3[i]],c='k')
    plt.plot([i+1.4,i+1.4],[q1[i],q3[i]],c='k')
    plt.plot([i+0.6,i+0.6],[q1[i],q3[i]],c='k')
plt.xticks(range(1,len(m)+1),x_ticks)
plt.ylabel('Losses reduction\n(Wh per household)')
plt.grid(linestyle=':')
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/losses/img/season_comp.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0)
plt.show()

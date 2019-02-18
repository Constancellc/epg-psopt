import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt



stem = '../../../Documents/simulation_results/LV/manc-models/'
# losses first
m = []
q1 = []
q3 = []
u = []
l = []



ls = {0:[],1:[],2:[],3:[]}
with open(stem+'1-losses.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        for i in range(4):
            ls[i].append(float(row[i]))
                             
for i in range(4):
    diff = sorted(ls[i])
    m.append(1000*diff[int(len(diff)/2)]/55)
    q1.append(1000*diff[int(len(diff)*0.25)]/55)
    q3.append(1000*diff[int(len(diff)*0.75)]/55)
    l.append(1000*diff[0]/55)
    u.append(1000*diff[-1]/55)

plt.figure(figsize=(6,2.2))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 9
# whiskers
plt.scatter(range(1,len(m)+1),l,marker='_',c='gray')
plt.scatter(range(1,len(m)+1),u,marker='_',c='gray')
for i in range(len(m)):
    plt.plot([i+1,i+1],[l[i],q1[i]],c='gray')
    plt.plot([i+1,i+1],[q3[i],u[i]],c='gray')

x_ticks = ['No EVs','Uncontrolled','Load Flattening','Loss Minimizing']
# box
for i in range(len(m)):
    plt.plot([i+0.61,i+1.39],[m[i],m[i]],c='b',lw='2')
    plt.plot([i+0.6,i+1.4],[q1[i],q1[i]],c='k')
    plt.plot([i+0.6,i+1.4],[q3[i],q3[i]],c='k')
    plt.plot([i+1.4,i+1.4],[q1[i],q3[i]],c='k')
    plt.plot([i+0.6,i+0.6],[q1[i],q3[i]],c='k')
plt.xticks(range(1,len(m)+1),x_ticks)
plt.ylabel('Losses (Wh per household)')
plt.grid(linestyle=':')
plt.ylim(0,600)
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/losses/img/losses.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0)
plt.show()

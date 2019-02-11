import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

fds = ['041','4','213','162','1','3','2','193','074','024']

hhs = {'1':55,'2':175,'3':94,'4':24,'024':115,'041':24,'074':186,'162':73,
       '193':65,'213':67}

stem = '../../../Documents/simulation_results/LV/manc-models/'
# losses first
m = []
q1 = []
q3 = []
u = []
l = []

for f in fds:
    diff = []
    with open(stem+f+'-losses.csv','rU') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            diff.append(float(row[2])-float(row[3]))
            if diff[-1] < 0:
                diff[-1] = 0
    diff = sorted(diff)
    
    m.append(1000*diff[int(len(diff)/2)]/hhs[f])
    q1.append(1000*diff[int(len(diff)*0.25)]/hhs[f])
    q3.append(1000*diff[int(len(diff)*0.75)]/hhs[f])
    l.append(1000*diff[0]/hhs[f])
    u.append(1000*diff[-1]/hhs[f])

plt.figure(figsize=(6,3))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 9
# whiskers
plt.scatter(range(1,len(m)+1),l,marker='_',c='gray')
plt.scatter(range(1,len(m)+1),u,marker='_',c='gray')
for i in range(len(m)):
    plt.plot([i+1,i+1],[l[i],q1[i]],c='gray')
    plt.plot([i+1,i+1],[q3[i],u[i]],c='gray')

x_ticks = []
for i in range(len(m)):
    x_ticks.append(str(i+1)+'\n('+str(hhs[fds[i]])+')')
# box
for i in range(len(m)):
    plt.plot([i+0.61,i+1.39],[m[i],m[i]],c='b',lw='2')
    plt.plot([i+0.6,i+1.4],[q1[i],q1[i]],c='k')
    plt.plot([i+0.6,i+1.4],[q3[i],q3[i]],c='k')
    plt.plot([i+1.4,i+1.4],[q1[i],q3[i]],c='k')
    plt.plot([i+0.6,i+0.6],[q1[i],q3[i]],c='k')
plt.xticks(range(1,len(m)+1),x_ticks)
plt.ylabel('Losses reduction (Wh per household)')
plt.grid(linestyle=':')
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/losses/img/losses_comp.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0)
plt.show()

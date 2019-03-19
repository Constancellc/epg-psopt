import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

fds = ['041','213','162','1','3','2','193','074','024']

hhs = {'1':55,'2':175,'3':94,'4':24,'024':115,'041':24,'074':186,'162':73,
       '193':65,'213':67}

res = {'1':1,'2':30,'3':1,'4':1,'024':2,'041':1,'074':5,'162':1,
       '193':1,'213':1}

stem = '../../../Documents/simulation_results/LV/manc-models/'
# losses first
m = []
q1 = []
q3 = []
u = []
l = []

for f in fds:
    lf = []
    lm = []
    diff = []
    for r in range(100):
        lf.append([])
        lm.append([])
    with open(stem+f+'-loads-f.csv','rU') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            for i in range(len(row)-1):
                lf[i].append(float(row[i+1]))
    with open(stem+f+'-loads-m.csv','rU') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            for i in range(len(row)-1):
                lm[i].append(float(row[i+1]))

    r = 0
    while len(lf[r]) > 0:
        r += 1
    lf30 = []
    lm30 = []
    for r_ in range(r):
        lf30.append([0.0]*48)
        lm30.append([0.0]*48)

        for t in range(int(1440/res[f])):
            lf30[r_][int(t*res[f]/30)] += lf[r_][t]*res[f]/30
            lm30[r_][int(t*res[f]/30)] += lm[r_][t]*res[f]/30

        diff.append(max(lm30[r_])-max(lf30[r_]))

        if diff[-1] < 0:
            diff[-1] = 0

    diff = sorted(diff)
    
    m.append(diff[int(len(diff)/2)]/hhs[f])
    q1.append(diff[int(len(diff)*0.25)]/hhs[f])
    q3.append(diff[int(len(diff)*0.75)]/hhs[f])
    l.append(diff[0]/hhs[f])
    u.append(diff[-1]/hhs[f])


plt.figure(figsize=(6,2))
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
plt.ylabel('30 min Peak Demand Increase\n(kW per household)')
plt.grid(linestyle=':')
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/losses/img/admd_comp.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0)
plt.show()

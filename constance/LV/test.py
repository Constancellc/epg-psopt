import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt


stem = '../../../Documents/simulation_results/LV/manc-models/'

m = []
q1 = []
q3 = []
u = []
l = []


diff1 = []
diff2 = []
with open(stem+'2-losses.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        diff1.append(1000*(float(row[2])-float(row[3]))/55)
        diff2.append(1000*(float(row[2])-float(row[4]))/55)


d = [diff1,diff2]                           
for i in range(2):
    diff = sorted(d[i])
    m.append(diff[int(len(diff)/2)])
    q1.append(diff[int(len(diff)*0.25)])
    q3.append(diff[int(len(diff)*0.75)])
    l.append(diff[0])
    u.append(diff[-1])


plt.figure()
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 12
# whiskers
plt.scatter(range(1,len(m)+1),l,marker='_',c='gray')
plt.scatter(range(1,len(m)+1),u,marker='_',c='gray')
for i in range(len(m)):
    plt.plot([i+1,i+1],[l[i],q1[i]],c='gray')
    plt.plot([i+1,i+1],[q3[i],u[i]],c='gray')

x_ticks = ['Loss\nMinimizing','Phase Balance\nRegularization']
# box
for i in range(len(m)):
    plt.plot([i+0.705,i+1.295],[m[i],m[i]],c='b',lw='2')
    plt.plot([i+0.7,i+1.3],[q1[i],q1[i]],c='k')
    plt.plot([i+0.7,i+1.3],[q3[i],q3[i]],c='k')
    plt.plot([i+1.3,i+1.3],[q1[i],q3[i]],c='k')
    plt.plot([i+0.7,i+0.7],[q1[i],q3[i]],c='k')
plt.xticks(range(1,len(m)+1),x_ticks)
plt.grid(linestyle=':')
plt.ylabel('Additional Savings\n(Wh per household)')
plt.tight_layout()
plt.show()

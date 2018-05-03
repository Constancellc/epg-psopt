import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random
import scipy.ndimage
#import win32com.client

stem = '../../../Documents/simulation_results/LV/manc'
end = '-losses_difference_with_load.csv'


plt.figure(figsize=(6,3))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 9
for i in range(1,2):#5):
    diffs = []
    with open(stem+str(i)+end,'rU') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            lf = float(row[1])
            lm = float(row[2])
            diff = 100*(lf-lm)/lm
            diffs.append([float(row[0]),diff])

    m = [0.0]*30
    v = [0.0]*30
    n = [0]*30
    for i in range(len(diffs)):
        m[int(diffs[i][0])-1] += diffs[i][1]
        n[int(diffs[i][0])-1] += 1

    print(n)

    for i in range(30):
        m[i] = m[i]/n[i]
        d = []
        for j in range(len(diffs)):
            if diffs[j][0] == i+1:
                d.append(diffs[j][1])
        d = sorted(d)
        #d = d[3:97]

        for j in range(len(d)):
            v[i] += np.power(d[j]-m[i],2)/len(d)

    print(v)
    u = [0.0]*30
    l = [0.0]*30

    for i in range(30):
        u[i] = m[i]+np.sqrt(v[i])
        l[i] = m[i]-np.sqrt(v[i])
    print(u)
    m = [0.0] + m
    u = [0.01] + u
    l = [0.0] + l
    #plt.fill_between(range(31),u,l,color='#c5d9f9')
    plt.plot(range(31),m,label=str(i))#,'b')
plt.xlim(0,30)
#plt.ylim(0,5.5)
plt.xlabel('Energy required per EV (kWh)')
plt.ylabel('% Difference in losses')
plt.grid()
plt.legend()
plt.tight_layout()
#plt.savefig('../../../Dropbox/papers/losses/loss_diff_with_load.eps', format='eps', dpi=1000)

plt.show()

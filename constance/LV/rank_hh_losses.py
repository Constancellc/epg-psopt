import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random

unused = []

P = matrix(0.0,(55,55))
with open('P.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    iskp = 0
    for row in reader:
        
        if i in unused:
            i += 1
            iskp += 1
            continue
        
        jskp = 0
        for j in range(len(row)):
            if j in unused:
                jskp += 1
                continue
            P[i-iskp,j-jskp] += float(row[j])
        i += 1


q = []
with open('q.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        if i in unused:
            i += 1
            continue
        q.append(float(row[0]))
        i += 1

q = matrix(q)


with open('c.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        c = float(row[0])

print(q.size)
print(P.size)

losses = [0.0]*55

x = matrix(0.0,(55,1))
for i in range(55):
    x[i] = -1000.0

    losses[i] = (x.T*P*x + q.T*x+c)[0]

    x[i] = 0.0


plt.figure(figsize=(6,2))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 10
plt.scatter(range(1,56),losses,marker='x')
plt.xlim(0,56)
plt.ylim(0,4)
plt.grid()
plt.xlabel('Node #')
plt.ylabel('Losses due to\n1kW Load (W)')
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/losses/img/rank_hh.eps', format='eps', dpi=1000)

plt.show()
    
    

      

import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random
import scipy.sparse as sparse

n = '024'
folder = 'manc_models/'+n+'/'

q = []
with open(folder+'q.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        q.append(float(row[0]))
        
nH = len(q)
q = np.array(q)
P = np.zeros((nH,nH))
x = np.array([-1000]*nH)

with open(folder+'P.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        for j in range(nH):
            P[i][j] = float(row[j])
        i += 1
        
with open(folder
          +'c.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        c = float(row[0])
        
losses = np.matmul(x.T,np.matmul(P,x)) + np.matmul(x.T,q) + c
print(losses)



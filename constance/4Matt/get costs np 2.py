import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random
import scipy.sparse as sparse

n = 'n1'
folder = '../../../Documents/ccModels/'+n+'/'+n
output = 'manc_models/'+n+'/'


a = np.load(folder+'aCc060.npy')
My = np.load(folder+'MyCc060.npy')
v0 = np.load(folder+'V0Cc060.npy')
xhy = np.load(folder+'xhyCc060.npy')

nn = np.load(folder+'vYNodeOrder060.npy')


xhy2 = []
for i in range(len(xhy)):
    if xhy[i] < -1e-4:
        xhy2.append(xhy[i])
xhy = np.array(xhy2)

Y = np.load(folder+'YbusCc060.npy')
Y = Y.flatten()[0]
Y = Y.conj()

a = np.concatenate([v0,a])
My = np.concatenate([np.zeros((3,len(xhy))),My])
MyC = My.conj()
P = np.matmul(My.T,Y.dot(MyC))
q = np.matmul(a.T,Y.dot(MyC))
q = np.multiply(q,2)
c = np.matmul(a.T,Y.dot(a.conj()))

print(c)

nH2 = len(xhy)
nH = int(len(xhy)/2)

alpha = xhy[nH]/xhy[0]

print(alpha)

P_r = np.zeros((nH,nH))
q_r = np.zeros((nH,1))
for i in range(nH):
    q_r[i][0] = q[i].real+alpha*q[i+nH].real
    for j in range(nH):
        P_r[i][j] = P[i][j].real+alpha*P[i][j+nH].real+\
                    alpha*P[i+nH][j].real+alpha*alpha*P[i+nH][j+nH].real

with open(output+'P.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(int(nH)):
        writer.writerow(P_r[i])
        
with open(output+'q.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(int(nH)):
        writer.writerow(q_r[i])
        
with open(output+'c.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([c.real])
'''
y = np.array([-1000]*nH)
y[0] = -200000
print(c.real+np.dot(q_r.T,y)+np.dot(y,np.dot(P_r,y)))
'''
y2 = np.array([-1000]*nH+[-300]*nH)
v = np.dot(My,y2)+a
for v_ in v:
    print(abs(v_))
print(v[:10])


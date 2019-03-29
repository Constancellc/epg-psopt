import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random
import scipy.sparse as sparse


folder = '../../../Documents/ccModels/epriK1/epriK1'
output = 'manc_models/epriK1/'


a = np.load(folder+'LptaCc100.npy')
My = np.load(folder+'LptMyCc100.npy')
v0 = np.load(folder+'LptV0Cc100.npy')
xhy = np.load(folder+'LptxhyCc100.npy')


Sord = np.load(folder+'LptSyYNodeOrder100.npy')
nT = len(Sord)
#Vord = np.load(folder+'LptvYNodeOrder100.npy')
sO = {}
c = 0
for s in Sord:
    sO[s] = c
    c += 1
vO = {}
        
lb = np.load(folder+'LptloadBusesCc100.npy')
lb = lb.flatten()[0]
toRemove = []
tc = 0
for l in lb:
    if l[:3] == 'com':
        try:
            toRemove.append(sO['LD'+lb[l][2:]])
        except:
            toRemove.append(sO['LD'+lb[l][2:12]+'1'])
            toRemove.append(sO['LD'+lb[l][2:12]+'2'])
            toRemove.append(sO['LD'+lb[l][2:12]+'3'])

nH = nT-len(toRemove)
My2 = np.empty([len(My),2*nH],dtype=complex)
a2 = np.empty([len(My)],dtype=complex)

i = 0
for j in range(nT):
    if j in toRemove:
        a2 += np.multiply(My[:,j],xhy[j])
        tc -= xhy[j]/1000
        continue
    My2[:,i] = My[:,j]
    i += 1
print(tc)
    
for j in range(nT):
    if j in toRemove:
        a2 += np.multiply(My[:,j+nT],xhy[j+nT])
        continue
    My2[:,i] = My[:,j+nT]
    i += 1

My = My2
a = np.add(a,a2)

                      

Y = np.load(folder+'LptYbusCc100.npy')
Y = Y.flatten()[0]
Y = Y.conj()


# So some of the loads are commerical, I need to find those and
# basically add them into the a

a = np.concatenate([v0,a])
My = np.concatenate([np.zeros((3,622)),My])
MyC = My.conj()
P = np.matmul(My.T,Y.dot(MyC))
q = np.matmul(a.T,Y.dot(MyC))
q = np.multiply(q,2)
c = np.matmul(a.T,Y.dot(a.conj()))

print(c)

nH2 = 622
nH = 311
print(nT)

alpha = 0.25

#print(alpha)

P_r = np.zeros((nH,nH))
q_r = np.zeros((nH,1))
for i in range(nH):
    q_r[i][0] = q[i].real+alpha*q[i+nH].real
    for j in range(nH):
        P_r[i][j] = P[i][j].real+alpha*P[i][j+nH].real+\
                    alpha*P[i+nH][j].real+alpha*alpha*P[i+nH][j+nH].real

x = np.ones(nH)
'''
print(c.real+np.dot(x,np.dot(P_r,x))+np.dot(x,q_r))
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
    writer.writerow([c.real])'''




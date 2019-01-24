import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random
from scipy.sparse import csr_matrix

n = '041'
folder = '../../../Documents/ccModels/'+n+'/'+n
output = 'manc_models/'+n+'/'

a = np.load(folder+'LptaCc060.npy')
My = np.load(folder+'LptMyCc060.npy')
v0 = np.load(folder+'LptV0Cc060.npy')
xhy = np.load(folder+'LptxhyCc060.npy')
Y = np.load(folder+'LptYbusCc060.npy')

a = np.concatenate([v0,a])
My = np.concatenate([np.zeros((3,len(xhy))),My])
Y = csr_matrix(Y)
#Y.todense()
P = np.matmul(np.matmul(np.transpose(My),Y),My)

print(type(My))

nH2 = len(xhy)
nH = int(len(xhy)/2)
print(nH)

alpha = xhy[nH]/xhy[0]

My_r = np.zeros((3,nH2))
My_i = np.zeros((3,nH2))

My_r = np.concatenate([My_r,My.real])
My_i = np.concatenate([My_i,My.imag])


a_r = np.concatenate([v0.real,a.real])
a_r = np.concatenate([v0.imag,a.imag])

My_r = My_r.astype(float)
My_i = My_i.astype(float)
#Y_r = Y_r.astype(float)
#Y_i = Y_i.astype(float)
print('tic')
m1 = np.matmul(np.matmul(np.transpose(My_r),Y_r),My_r)
m2 = np.matmul(np.matmul(np.transpose(My_r),Y_i),My_r)
m3 = np.matmul(np.matmul(np.transpose(My_i),Y_i),My_r)
m4 = np.matmul(np.matmul(np.transpose(My_i),Y_r),My_i)
P = m1+m3+m3+m4

print('toc')
with open(output+'P.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(int(nH)):
        row = []
        for j in range(int(nH)):
            row.append(P[i][j]+alpha*P[i+nH][j]+alpha*P[i][j+nH]+\
                       alpha*alpha*P[i+nH][j+nH])
        writer.writerow(row)

del P
del m1
del m2
del m3
del m4

print('done with P')

#q = 2*np.transpose(My_r


import csv
import numpy as np
from cvxopt import matrix, spdiag, sparse
# or get them from some csv file
from reduce_matricies import My_r, My_i, a, Y_r, Y_i#, vs_r, vs_i

T = 1 # number of time instants

My_r = matrix(My_r)
My_i = matrix(My_i)
Y_r = matrix(Y_r)
Y_i = matrix(Y_i)
a = matrix(a)

M = sparse([My_r, My_i])

# first get household loads
x0 = matrix(1.0,(55,1))
v0 = M*x0 + a
#v0r = v0[:2718,:]
#v0i = v0[2718:,:]
#v0 = sparse([matrix(vs_r),v0r,matrix(vs_i),v0i])

D = matrix(0.0,(1,2718))#21))

I = spdiag([1]*2718)#21)
O = spdiag([0]*2718)#21)

Cr = sparse([[I],[O]])
Ci = sparse([[O],[I]])

A1 = sparse([[Y_r],[-1*Y_i]])
A2 = sparse([[Y_i],[Y_r]])
    
K4 = matrix(0.0,(1,55))

for p in range(3):
    D[p] = 1.0

    DCr = D*Cr
    DCi = D*Ci
    DA1 = D*A1
    DA2 = D*A2

    K1 = DCr*v0*DA1 + DA1*v0*DCr + DCi*v0*DA2 + DA2*v0*DCi
    K4 += K1*M-matrix(1.0,(1,55))

with open('c.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(K4.size[1]):
        writer.writerow([K4[0,i]])

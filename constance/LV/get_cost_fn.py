import csv
import numpy as np
from cvxopt import matrix, spdiag, sparse
# or get them from some csv file
from reduce_matricies import My_r, My_i, a_r, a_i, Y_r, Y_i#, vs_r, vs_i

T = 1 # number of time instants

My_r = matrix(My_r)
My_i = matrix(My_i)
Y_r = matrix(Y_r)
Y_i = matrix(Y_i)
a_r = matrix(a_r)
a_i = matrix(a_i)
k = matrix(0.0,(1,2718))
alpha = 0.05 # 1-pf

print(My_r.size)
print(My_i.size)
print(a_r.size)
print(a_i.size)
print(Y_r.size)
print(Y_i.size)

c = [0.0]*55

#M = sparse([My_r, My_i])

# first get household loads
x0 = matrix(1.0,(55,1))

for ph in range(3):#3):
    k[0,ph] = 1.0
    
    A1 = (k*Y_r*a_r-k*Y_i*a_i)*k*My_r + alpha*k*(Y_r*a_i+Y_i*a_r)*k*My_i + \
         k*a_r*k*Y_r*My_r - alpha*k*a_r*k*Y_i*My_i + alpha*k*a_i*k*Y_r*My_i + \
         k*a_i*k*Y_i*My_r
    
    A2 = My_r.T*k.T*k*Y_r*My_r + My_r.T*Y_r.T*k.T*k*My_r + alpha*( - \
         My_r.T*k.T*k*Y_i*My_i - My_i.T*Y_i.T*k.T*k*My_r + \
         My_i.T*k.T*k*Y_i*My_r + My_r.T*Y_i.T*k.T*k*My_i) + alpha*alpha*(\
         My_i.T*k.T*k*Y_r*My_i + My_i.T*Y_r.T*k.T*k*My_i)

    print(A1.size)
    print(A2.size)
    new_c = A1.T+A2*x0
    
    for i in range(55):
        c[i] -= new_c[i,0]

    k[0,ph] = 0.0
                                                             
#v0 = M*x0 + a
#v0r = v0[:2718,:]
#v0i = v0[2718:,:]
#v0 = sparse([matrix(vs_r),v0r,matrix(vs_i),v0i])
'''
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
'''
with open('c.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(c)):
        writer.writerow([c[i]])

import csv
import numpy as np
from cvxopt import matrix, spdiag, sparse
# or get them from some csv file
from reduce_matricies import My_r, My_i, a_r, a_i, Y_r, Y_i

My_r = matrix(My_r)
My_i = matrix(My_i)
Y_r = matrix(Y_r)
Y_i = matrix(Y_i)
a_r = matrix(a_r)
a_i = matrix(a_i)
k = matrix(0.0,(1,2721))
alpha = 0.05 # 1-pf

print(My_r.size)
print(My_i.size)
print(a_r.size)
print(a_i.size)
print(Y_r.size)
print(Y_i.size)

c = [0.0]*55

# first get household loads
x0 = matrix(1.0,(55,1))

Q = matrix(0.0,(110,55))
print(Y_r)

for ph in range(3):#3):
    k[0,ph] = 1.0
    
    A1 = (k*Y_r*a_r-k*Y_i*a_i)*k*My_r + alpha*k*(Y_r*a_i+Y_i*a_r)*k*My_i + \
         k*a_r*k*Y_r*My_r - alpha*k*a_r*k*Y_i*My_i + alpha*k*a_i*k*Y_r*My_i + \
         k*a_i*k*Y_i*My_r
    '''
    A2 = My_r.T*k.T*k*Y_r*My_r + My_r.T*Y_r.T*k.T*k*My_r + alpha*( - \
         My_r.T*k.T*k*Y_i*My_i - My_i.T*Y_i.T*k.T*k*My_r + \
         My_i.T*k.T*k*Y_i*My_r + My_r.T*Y_i.T*k.T*k*My_i) + alpha*alpha*(\
         My_i.T*k.T*k*Y_r*My_i + My_i.T*Y_r.T*k.T*k*My_i)
    '''

    A2 = My_r.T*k.T*k*Y_r*My_r - alpha*My_r.T*k.T*k*Y_i*My_i + alpha*\
         My_i.T*k.T*k*Y_i*My_r + alpha*alpha*My_i.T*k.T*k*Y_r*My_i
    
    print(A2)
    
    for i in range(A2.size[0]):
        for j in range(A2.size[1]):
            Q[i,j] += A2[i,j] # not 100% about negative sign!
    '''
    print(A1.size)
    print(A2.size)
    new_c = A1.T+A2*x0
    
    for i in range(55):
        c[i] -= new_c[i,0]
    '''
    k[0,ph] = 0.0
                                                             

'''
with open('p.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(c)):
        writer.writerow([c[i]])

with open('Q.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(Q.size[0]):
        writer.writerow(Q[i,:])
'''

with open('A2.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(55):
        writer.writerow(Q[i,:])

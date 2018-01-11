import numpy as np
from reduce_matricies import a_r, a_i, My_r, My_i, Y_r, Y_i
from cvxopt import matrix
import csv
'''
M = matrix(complex(0,0),(2721,110))
a = matrix(complex(0,0),(2721,1))
'''
M = np.empty((2721,110),dtype=complex)
a = np.empty((2721,1),dtype=complex)

for i in range(2721):
    a[i] = complex(float(a_r[i]),float(a_i[i]))

    for j in range(110):
        #M[i,j] = complex(My_r[i][j],My_i[i][j])
        M[i][j] = complex(float(My_r[i][j]),float(My_i[i][j]))
''''
a_r = matrix(a_r)
a_i = matrix(a_i)
Y_r = matrix(Y_r)
Y_i = matrix(Y_i)
x = matrix([-1000.0]*55+[-330]*55)

print(M.size)
print(x.size)
print(a.size)
'''

x = np.array([-1000.0]*55+[-330]*55)

v = np.matmul(M,x) + a

'''
Vlin_r = []
Vlin_i = []
with open('../../../Documents/LV_LPF/Vlin_r.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        Vlin_r.append(float(row[0]))
with open('../../../Documents/LV_LPF/Vlin_i.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        Vlin_i.append(float(row[0]))

Vlin = matrix(complex(0,0),(2721,1))

for i in range(2721):
    Vlin[i] = complex(Vlin_r[i],Vlin_i[i])

    if Vlin[i] != v[i]:
        print(Vlin[i]-v[i])
'''                     
Y = matrix(complex(0,0),(2721,2721))


for i in range(2721):
    for j in range(2721):
        Y[i,j] = complex(Y_r[i,j],Y_i[i,j])

I = Y*v

I_ = matrix(complex(0,0),(2721,1))
for i in range(2721):
    I_[i] = complex(I[i].real,-I[i].imag)

print(I_.T*v.real)

P = []
q = []

with open('P.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        new = []
        for i in range(len(row)):
            new.append(float(row[i]))
        P.append(new)

P = np.array(P)
P = matrix(P)

with open('q.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        q.append(float(row[0]))

q = matrix(q)

losses  = x.T*P*x + q.T*x + a_r.T*Y_r*a_r - a_r.T*Y_i*a_i + a_i.T*Y_r*a_r + a_i.T*Y_i*a_i

print(losses)

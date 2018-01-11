import numpy as np
from reduce_matricies import a_r, a_i, My_r, My_i, Y_r, Y_i
from cvxopt import matrix
import csv
from dec_mat_mul import MM
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

                
            

x = np.zeros((110,1))
for i in range(55):
    x[i] = -1000.0
for i in range(55,110):
    x[i] = -330.0

v = np.matmul(M,x) + a

print(len(v))
print(len(v[0]))

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
#Y = matrix(complex(0,0),(2721,2721))
Y = np.empty((2721,2721),dtype=complex)

for i in range(2721):
    for j in range(2721):
        Y[i][j] = complex(Y_r[i][j],Y_i[i][j])

I = np.matmul(Y,v)

I_ = matrix(complex(0,0),(2721,1))
for i in range(2721):
    I_[i] = complex(I[i].real,-I[i].imag)

print(np.matmul(np.transpose(I_),v).real)

P = np.zeros((110,110))
q = []

with open('P.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        for j in range(110):
            P[i][j] = float(row[i])
        i += 1
'''
P = np.array(P)
P = matrix(P)
'''
with open('q.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        q.append(float(row[0]))

q = np.array(q)

print(a_i.shape)
losses  = MM(MM(np.transpose(x),P),x) + MM(np.transpose(q),x) + \
          MM(MM(np.transpose(a_r),Y_r),a_r) - MM(MM(np.transpose(a_r),Y_i),a_i) + \
          MM(MM(np.transpose(a_i),Y_r),a_r) + \
          MM(MM(np.transpose(a_i),Y_i),a_i)

print(losses)

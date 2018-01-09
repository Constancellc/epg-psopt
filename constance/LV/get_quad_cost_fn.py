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

print(My_r.size)
print(My_i.size)
print(a_r.size)
print(a_i.size)
print(Y_r.size)
print(Y_i.size)


# first get household loads

P = My_r.T*Y_r*My_r - My_r.T*Y_i*My_i + My_i.T*Y_r*My_r + My_i.T*Y_i*My_i

q = 2*My_r.T*Y_r*a_r - My_r.T*Y_i*a_i - My_i.T*Y_i*a_r + My_i.T*Y_r*a_r +\
    My_r.T*Y_r*a_i + 2*My_i.T*Y_i*a_i

print(P.size)
print(q.size)

with open('P.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(110):
        writer.writerow(P[i,:])
        
with open('q.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(q)):
        writer.writerow([q[i]])

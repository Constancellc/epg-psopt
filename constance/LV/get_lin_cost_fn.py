import csv
import numpy as np
from cvxopt import matrix, spdiag, sparse
# or get them from some csv file
from reduce_matricies import My_r, My_i, Y_r, Y_i, v0_r, v0_i

My_r = matrix(My_r)
My_i = matrix(My_i)
Y_r = matrix(Y_r)
Y_i = matrix(Y_i)

v0_r += [0.0]*2718
v0_i += [0.0]*2718

v0_r = matrix(v0_r)
v0_i = matrix(v0_i)

print(My_r.size)
print(My_i.size)
print(Y_r.size)
print(Y_i.size)
print(v0_r.size)
print(v0_i.size)

c = [0.0]*55

# first get household loads
x0 = matrix([-1000.0]*55+[-60.0]*55)



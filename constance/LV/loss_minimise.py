import csv
import numpy as np
from cvxopt import matrix, spdiag, sparse
# or get them from some csv file
from reduce_matricies import My_r, My_i

T = 1 # number of time instants

# first get household loads
x0 = [0.0]*(55*T)

D = matrix(0.0,(1,2718))
D[k] = 1.0

I = spdiag([1]*2718)
O = spdiag([0]*2718)

Cr = sparse([I],[O])
Ci = sparse([O],[I])


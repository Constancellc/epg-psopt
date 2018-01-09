import numpy as np
from reduce_matricies import a_r, a_i, My_r, My_i
from cvxopt import matrix

M = matrix(complex(0,0),(2721,55))
a = matrix(complex(0,0),(2721,1))

for i in range(2721):
    a[i] = complex(a_r[i],a_i[i])

    for j in range(55):
        M[i,j] = complex(My_r[i][j],My_i[i][j])

x = matrix(complex(1.0,0.06),(55,1))

print(M.size)
print(x.size)
print(a.size)

v = M*x + a

print(v)

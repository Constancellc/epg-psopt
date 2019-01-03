import numpy as np
from cvxopt import spmatrix
from scipy import random

# example from online
x0 = [2,-1,2,-2,1,4,3]
I0 = [1,2,0,2,3,2,0]
J0 = [0,0,1,1,2,3,4]
A0 = spmatrix(x0,I0,J0)
print('These two pairs work as expected')
print(A0)
print(A0.I)
print(I0)
print(A0.J)
print(J0)

# new example.
n = 4
x = np.ones(n).tolist()
I = list(range(n))
J = random.permutation(range(n)).tolist()
B0 = spmatrix(x,I,J)
print('Now, I and J are switched around?')
print(B0)
print(B0.I) 
print(I)
print(B0.J)
print(J)


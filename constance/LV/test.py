import numpy as np
from dec_mat_mul import MM
import random
from decimal import *

a = np.empty((4,4))
b = np.empty((4,1))

for i in range(4):
    for j in range(4):
        a[i][j] = Decimal(1.0)#4+random.random())
    b[i] = Decimal(2.3)#2+random.random())

print(MM(a,b))

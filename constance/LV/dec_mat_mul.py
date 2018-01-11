import numpy as np
from decimal import *

def MM(a,b):
    '''
    if type(a[0]) != list:
        a = [a]
    if type(b[0]) != list:
        b = [b]

    if len(a[0]) != len(b): # do soething better
        a = b
        b = a
    '''
    r = len(a)
    c = len(b[0])
    n = len(a[0])
    if len(b) != n:
        raise Exception
    
    out = np.empty((r,c))

    for i in range(r):
        for j in range(c):
            tot = Decimal(0.0)
            for ii in range(n):
                tot += Decimal(a[i][ii]*b[ii][j])
            out[i][j] = tot
    return out

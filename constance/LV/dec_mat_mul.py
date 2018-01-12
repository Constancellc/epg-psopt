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
    print('input:')
    print(a.shape)
    print(b.shape)
    r = a.shape[0]
    c = b.shape[1]
    if a.shape[1] != b.shape[0]:
        raise Exception
    
    out = np.empty((r,c))

    for i in range(r):
        for j in range(c):
            tot = Decimal(0.0)
            for ii in range(a.shape[1]):
                tot += Decimal(a[i][ii]*b[ii][j])
            out[i][j] = tot

    print('outputing:')
    print(out.shape)
    return out

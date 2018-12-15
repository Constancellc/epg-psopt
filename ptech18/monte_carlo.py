import time
from cmath import pi, exp
from math import sin, acos, inf
import numpy as np
import os
import win32com.client
from scipy.io import loadmat
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from cvxopt import matrix, spmatrix

# based on monte_carlo.py

print('Start.\n',time.process_time())

FD = r"C:\\Users\Matt\Documents\DPhil\pesgm19\pesgm19_paper\figures\\"
WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"

nl = np.linspace(0,1,7)
nl[0]=1e-4

np.random.seed(0)

frac_set = 0.05

modeli = 4
models = [('eulv',WD+'\LVTestCase_copy\master_z_g')]
models.append(('n1f1',WD+r'\manchester_models\\network_1\Feeder_1\master_g'))
models.append(('n2f1',WD+r'\manchester_models\\network_2\Feeder_1\master_g'))
models.append(('n3f1',WD+r'\manchester_models\\network_3\Feeder_1\master_g'))
models.append(('n4f1',WD+r'\manchester_models\\network_4\Feeder_1\master_g'))

model = models[modeli]

DSSObj = win32com.client.Dispatch("OpenDSSEngine.dss")

DSSText=DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution=DSSCircuit.Solution

LDS = DSSCircuit.loads
LM = loadmat(WD+'\\lin_models\\'+model[0])
DSSText.command='Compile ('+model[1]+')'

Nl = np.ceil(nl*LDS.count).astype(int)

# Ns = 1000
Ns = 1000
Vb = 230
vp = 1.1
vmax=Vb*vp


# My = sparse.csr_matrix(LM['My'])
# xhy0 = sparse.csr_matrix(LM['xhy0'])
# a = sparse.csr_matrix(LM['a'])
# Vh0 = (My.dot(xhy0) + a).toarray()
# MyT = My.T
# xhp0 = xhy0[0:xhy0.shape[0]//2]

# My = LM['My']
# xhy0 = LM['xhy0']
# a = LM['a']
# Vh0 = (My.dot(xhy0) + a)
# xhp0 = xhy0[0:xhy0.shape[0]//2]

# fxy0 = xhy0.nonzero()[0]
# fxp0 = xhp0.nonzero()[0]
# Vmax = vmax*np.ones(Vh0.shape)

# CVXOPT Version here =====
My = matrix(LM['My'])
xhy0_raw=LM['xhy0'].astype('d')
xhy0 = spmatrix(xhy0_raw[xhy0_raw.nonzero()],xhy0_raw.nonzero()[0],xhy0_raw.nonzero()[1],(My.size[1],1))
a = matrix(LM['a'])
Vh0 = matrix((My*xhy0) + a)
xhp0 = xhy0[0:xhy0.size[0]//2]

ang0 = np.exp(1j*np.angle(Vh0))
Va0 = (Vh0/ang0).real

fxy0 = xhy0.I
fxp0 = xhp0.I
Vmax = vmax*np.ones(Vh0.size)

b = Vmax - Va0;

gen_pf = 1.00
gen_pf = 0.95

aa = exp(1j*pi*2/3)
qgen = sin(acos(abs(gen_pf)))/gen_pf

X = np.zeros((Ns,len(Nl)))

print('Run simulations:\n',time.process_time())
for i in range(len(Nl)):
    # V = np.concatenate((np.ones(Nl[i]),np.ones(Nl[i])*qgen))
    # J = np.zeros(Nl[i]*2).astype(int)
    V = matrix(np.concatenate((np.ones(Nl[i]),np.ones(Nl[i])*qgen)))
    J = matrix(np.zeros(Nl[i]*2).astype(int))
    
    for j in range(Ns):
        rdi = np.random.permutation(LDS.count)[:Nl[i]]
        rdi_i = np.concatenate((rdi,rdi+LDS.count))
        
        # I = fxy0[rdi_i]
        I = fxy0[rdi_i.tolist()]
        
        # xhs = sparse.coo_matrix((V,(I,J)),(My.shape[1],1)).tocsr()
        # xhs = sparse.coo_matrix((V,(J,I)),(1,My.shape[1])).tocsr()
        # xhs = sparse.coo_matrix((V,(I,J)),(My.shape[1],1)).toarray()
        xhs = spmatrix(V,I,J,(My.size[1],1))
        
        # Mk = My.dot(xhs).toarray()
        # Mk = My.dot(xhs)
        # Mk = xhs.dot(MyT).toarray()
        Mk = My*xhs
        
        A = (Mk/ang0).real

        Anew = A[fxp0]
        bnew = b[fxp0]
        
        xab = bnew/Anew
        xab[xab<0] = 1e10
        
        X[j,i] = xab.min()
print(time.process_time())
# plt.boxplot(X), plt.show()
    
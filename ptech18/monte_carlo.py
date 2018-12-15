import time
from math import sin, acos, inf
import numpy as np
import win32com.client
from scipy.io import loadmat
import matplotlib.pyplot as plt
from cvxopt import matrix, spmatrix, sparse
from random import sample

# based on monte_carlo.py
print('Start.\n',time.process_time())

FD = r"C:\\Users\Matt\Documents\DPhil\pesgm19\pesgm19_paper\figures\\"
WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"

nl = np.linspace(0,1,7)
nl[0]=1e-4

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
Ns = 100
Vb = 230
vp = 1.1
vmax=Vb*vp


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

qgen = sin(acos(abs(gen_pf)))/gen_pf

X = np.zeros((Ns,len(Nl)))
LD_set = range(LDS.count)
print('Run simulations:\n',time.process_time())

for i in range(len(Nl)):
    Vp = matrix(np.ones(Nl[i])*1)
    Vq = matrix(np.ones(Nl[i])*qgen)
    J = matrix(np.zeros(Nl[i]).astype(int))    
    for j in range(Ns):
        rdi = sample(LD_set,Nl[i])
        I = fxy0[rdi]
        
        xhp = spmatrix(Vp,I,J,(My.size[1]//2,1))
        xhq = spmatrix(Vq,I,J,(My.size[1]//2,1))
        xhs = sparse([xhp,xhq])
        
        Mk = My*xhs
        
        A = (Mk/ang0).real

        Anew = A[fxp0]
        bnew = b[fxp0]
        
        xab = bnew/Anew
        xab[xab<0] = 1e10
        
        X[j,i] = xab.min()
print(time.process_time())
plt.boxplot(X*1e-3,positions=Nl), 
plt.show()

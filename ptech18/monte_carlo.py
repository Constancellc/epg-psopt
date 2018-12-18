import time
from math import sin, acos, inf
import numpy as np
import win32com.client
from scipy.io import loadmat
import matplotlib.pyplot as plt
from cvxopt import matrix, spmatrix, sparse
from cvxopt.solvers import lp, options
from random import sample
from dss_python_funcs import feeder_to_fn

# based on monte_carlo.py
print('Start.\n',time.process_time())

FD = r"C:\Users\chri3793\Documents\DPhil\malcolm_updates\wc181217\\"
# WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"

nl = np.linspace(0,1,7)
nl[0]=1e-4

frac_set = 0.05
lin_point = 1.0


fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4']
ckts = {'feeder_name':['fn_ckt','fn']}
ckts[fdrs[0]]=[WD+'\\LVTestCase_copy',WD+'\\LVTestCase_copy\\master_z']
ckts[fdrs[1]] = feeder_to_fn(WD,fdrs[1])
ckts[fdrs[2]] = feeder_to_fn(WD,fdrs[2])
ckts[fdrs[3]] = feeder_to_fn(WD,fdrs[3])
ckts[fdrs[4]] = feeder_to_fn(WD,fdrs[4])

fdr_i = 3
tp = 0.051
tm = 0.05

feeder=fdrs[fdr_i]
Ky = matrix(np.loadtxt(WD+'\\lin_models\\'+feeder+'Ky'+str(np.round(lin_point*100).astype(int)).zfill(3)+'.txt'))
Kt = matrix(np.loadtxt(WD+'\\lin_models\\'+feeder+'Kt'+str(np.round(lin_point*100).astype(int)).zfill(3)+'.txt'))
bV = matrix(np.loadtxt(WD+'\\lin_models\\'+feeder+'bV'+str(np.round(lin_point*100).astype(int)).zfill(3)+'.txt'))
sy0 = matrix(np.loadtxt(WD+'\\lin_models\\'+feeder+'xhy0'+str(np.round(lin_point*100).astype(int)).zfill(3)+'.txt'))
DFs = np.loadtxt(WD+'\\datasets\\kersting235.txt',delimiter=',')

nLds = len(sy0)//2
Nl = np.ceil(nl*nLds).astype(int)

Ns = 100
Vb = 230
vp = 1.1
vm = 0.9

MD = 5. # kW
ADMD = 1e3*MD*nLds/DFs[min(nLds-1,69)]

vmax=Vb*vp
vmin=Vb*vm

gen_pf = 0.95
qgen = sin(acos(abs(gen_pf)))/gen_pf

X = np.zeros((Ns,len(Nl)))
T = np.zeros((Ns,len(Nl)))
LD_set = range(len(sy0)//2)
print('Run simulations:\n',time.process_time())

lp_c = matrix([-1.,0.])

bb = Ky*sy0 + bV

h00 = + bb - matrix(np.ones(len(bb))*vmax)
h01 = - bb + matrix(np.ones(len(bb))*vmin)
lp_h0 = matrix([matrix(h00),matrix(h01)])
lp_h1 = matrix([tm,-tp])
lp_h2 = matrix([-ADMD])
lp_h = matrix([lp_h0,lp_h1,lp_h2])

lp_G1 = matrix([[0,0],[1,-1]])

options['show_progress']=False
# options['maxiters']=1000 # needs to be increased slightly to solve some of these.
options['maxiters']=100000 # needs to be increased slightly to solve some of these.
for i in range(len(Nl)):
    Vp = matrix(np.ones(Nl[i])*1)
    Vq = matrix(np.ones(Nl[i])*qgen)
    J = matrix(np.zeros(Nl[i]).astype(int))
    lp_G2 = matrix([[-float(Nl[i])],[0]])
    print("Nl = ",Nl[i])
    for j in range(Ns):
        I = sample(LD_set,Nl[i])
        xhp = spmatrix(Vp,I,J,(Ky.size[1]//2,1))
        xhq = spmatrix(Vq,I,J,(Ky.size[1]//2,1))
        xhs = sparse([xhp,xhq])
        Ktot = matrix([[Ky*xhs],[Kt]])
        lp_G0 = matrix([-Ktot,Ktot])
        lp_G = matrix([lp_G0,lp_G1,lp_G2])
        lp_sln = lp(lp_c,-lp_G,-lp_h)
        if lp_sln['status']=='optimal':
            X[j,i] = lp_sln['x'][0]
            T[j,i] = lp_sln['x'][1]
        else:
            print("Failed to solve, status: ",lp_sln['status'], ', I = ',I)
print("Complete.\n",time.process_time())

# plt.figure()
# plt.grid(True), plt.xlabel("Number of Houses"), plt.ylabel("Power per house (kW)")
# plt.boxplot(X*1e-3,positions=Nl)
# plt.xlim(0,plt.xlim()[1]), 
# plt.plot(np.arange(1,Nl[-1]+1),ADMD*1e-3/np.arange(1,Nl[-1]+1),'k--')
# plt.ylim(0,X.max()*1e-3*1.5)
# # plt.show()
# plt.savefig(FD+feeder+'_X_'+str(round(tp*100)).zfill(3)+'_'+str(round(-tm*100)).zfill(3))

# plt.figure()
# plt.grid(True), plt.xlabel("Number of Houses"), plt.ylabel("Total Power (kW)")
# plt.boxplot(Nl*X*1e-3,positions=Nl)
# plt.xlim(0,plt.xlim()[1])
# plt.plot(plt.xlim(),np.ones(2)*ADMD*1e-3,'kx--')
# plt.ylim(0,plt.ylim()[1])
# # plt.show()

# plt.savefig(FD+feeder+'_Nl_'+str(round(tp*100)).zfill(3)+'_'+str(round(-tm*100)).zfill(3))

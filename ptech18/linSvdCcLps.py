# Things to do.

# 1. Load linear model with everything in it. Load program to have Ky, Kd, Kt, bV.
# 2. Load stuff and run linear hosting capacity analysis with linear model.
# 3. Try again, but with explictly solving the feasibility problem with Q and T.

# 1. Load linear model.

import pickle, os, sys, time
import numpy as np
from dss_python_funcs import *
from linSvdCalcs import hcPdfs, linModel, plotCns
import matplotlib.pyplot as plt
import dss_stats_funcs as dsf

import scipy.stats as sts

from cvxopt import matrix, solvers
solvers.options['show_progress']=False

# CHOOSE NETWORK
fdr_i = 19
pdfName = 'gammaFrac'

WD = os.path.dirname(sys.argv[0])

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr']
feeder = fdrs[fdr_i]

LM = linModel(fdr_i,WD)
pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,WD=WD )

LM.runLinHc(pdf)
rslt = LM.linHcRsl
plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=feeder,lineStyle='-',pltShow=True)

LMfxd = LM.LMfxd
Ky = LMfxd['Ky']
Kd = LMfxd['Kd'] # for J1: this is empty
Kt = LMfxd['Kt']
xhyLd = LMfxd['xhyLds0']*LM.loadPointLo/LM.linPoint
xhdLd = LMfxd['xhdLds0']*LM.loadPointLo/LM.linPoint
xhyCp = LMfxd['xhyCap0']
xhdCp = LMfxd['xhdCap0']
bV0 = LMfxd['bV']/LM.dvBase

xLds = np.concatenate((xhyLd,xhdLd))
xCap = np.concatenate((xhyCp,xhdCp))

xFxdNom = xLds + xCap

Ks = dsf.vmM(1/LM.dvBase,np.concatenate((Ky,Kd),axis=1))
Kp = Ks[:,LM.pIs]
Kq = Ks[:,LM.qIs]
0.00625
bV = bV0 + Ks.dot(xFxdNom)

Vp = LM.VpMv*np.ones(LM.nV)
Vp[LM.lvIdxFxd] = LM.VpLv
Vm = LM.VmMv*np.ones(LM.nV)
Vm[LM.lvIdxFxd] = LM.VmLv

c = matrix(np.array(LM.nT*[0] + [-1.,-1.]))
Gt = matrix( np.concatenate( (Kt,-Kt),axis=0  ) )
Gvar = matrix( np.concatenate( (np.eye(LM.nT + 2),-np.eye(LM.nT + 2)), axis=0 ) )

tapMax = 16*0.00625
Qmax = 0.20 # check with pf2kq(pf)
Pmax = 0.10 # between (negative) this and zero

varMaxUp = np.concatenate( (tapMax*np.ones(LM.nT), np.array([0,Qmax])) )
varMaxLo = np.concatenate( (tapMax*np.ones(LM.nT), np.array([Pmax,Qmax])) )
varMax = np.concatenate( (varMaxUp,varMaxLo) )

N2008 = 200 # ish, for n = 10, eps/beta=0.1
loadMc = np.random.uniform(0,2.0,size=(LM.nS,N2008))
loadMc2 = np.kron(np.array([[1],[1]]),loadMc)

solvedSet = []
slnsSet = []
# # for prmI in range(25):
prmI = 30
pdfMc = 1e3*pdf.genPdfMcSet(nMc=100,Mu0=np.zeros(LM.nS),prmI=prmI)[0] # pdfMc in kW (Mu is in W)
solved = []
slns = []
print('prmI: '+str(prmI))
for i in range(pdf.pdf['nMc']):
    if (i % (pdf.pdf['nMc']//4))==0:
        print(i)
    dVp = Kp.dot(pdfMc.T[i])
    dVq = Kq.dot(pdfMc.T[i])
    
    Kp1 = matrix(dVp)
    Kq1 = matrix(dVq)
    Gs = matrix( [[Kp1,-Kp1],[Kq1,-Kq1]] )
    
    G = matrix( [[Gt,Gvar[:,:LM.nT]],[Gs,Gvar[:,LM.nT:]]] )
    
    V0 = dVp + bV
    h = matrix(np.concatenate( (Vp-V0,-(Vm-V0),varMax) ))
    sol = solvers.lp(c,G,h)
    if sol['x']==None:
        solved.append(0)
        slns.append(None)
    else:
        solved.append(1)
        slns.append(sol['x'])

print(sum(solved)/len(solved))

solvedSet.append(solved)
slnsSet.append(slns)



# NOW: same again; this time oriented on a single of generation
solvedGenSet = []
slnsGenSet = []

solvedGen = []
slnsGen = []
genI = 1
dVp = Kp.dot(pdfMc.T[genI])
dVq = Kq.dot(pdfMc.T[genI])

Kp1 = matrix(dVp)
Kq1 = matrix(dVq)
Gs = matrix( [[Kp1,-Kp1],[Kq1,-Kq1]] )
G = matrix( [[Gt,Gvar[:,:LM.nT]],[Gs,Gvar[:,LM.nT:]]] )
Gts = matrix( [[Gt],[Gs]] )
for i in range(loadMc2.shape[1]):
    if (i % (loadMc2.shape[1]//4))==0:
        print(i)
    
    loadSet = loadMc2[:,i]
    xFxd = (xLds*loadSet) + xCap
    bV = bV0 + Ks.dot(xFxd)
    V0 = dVp + bV
    h = matrix(np.concatenate( (Vp-V0,-(Vm-V0),varMax) ))
    sol = solvers.lp(c,G,h)
    if sol['x']==None:
        solvedGen.append(0)
        slnsGen.append(None)
    else:
        solvedGen.append(1)
        slnsGen.append(sol['x'])

print(sum(solvedGen)/len(solvedGen))

solvedGenSet.append(solvedGen)
slnsGenSet.append(slnsGen)

h = matrix(varMax)
G = matrix([Gvar])

for i in range(loadMc2.shape[1]):
    loadSet = loadMc2[:,i]
    xFxd = (xLds*loadSet) + xCap
    bV = bV0 + Ks.dot(xFxd)
    V0 = dVp + bV
    h = matrix([h,matrix(np.concatenate( (Vp-V0,-(Vm-V0))) )])
    G = matrix([G,Gts])

sol = solvers.lp(c,G,h,solver='mosek')
print(sol['x'])







# solvedSetPct = 100 - np.sum(np.array(solvedSet),axis=1)
# plt.plot((1+np.arange(25))*2,solvedSetPct)
# plt.plot((1+np.arange(25))*2,rslt['Vp_pct'][0:25])
# plt.grid(True)
# plt.show()


# # plt.plot(np.diff(np.array(residential)))
# plt.plot(np.linspace(0,1,len(residential)-1),np.sort(np.abs(np.diff(np.array(residential)))))
# plt.yscale('log')
# plt.grid(True)
# plt.show()

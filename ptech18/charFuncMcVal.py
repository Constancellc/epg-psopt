# based on the script from 21/1, 'fft_calcs' in matlab.
# Based on script charFuncHcAlys, deleted 30/01

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import getpass
from dss_python_funcs import loadLinMagModel
from math import gamma
import time
import dss_stats_funcs as dsf

if getpass.getuser()=='chri3793':
    WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"
elif getpass.getuser()=='Matt':
    WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
    sn = r"C:\Users\Matt\Documents\DPhil\malcolm_updates\wc190128\\charFuncHcAlys_"

# def cf(k,th,t,a,dgn):
    # charFuncNeg = ((1 + th*1j*t*a)**(-k))*dgn + (1-dgn); # negation of t: see 'definition' in CF wiki
    # return charFuncNeg

# def cf(x0,k,t):
    # charFunc = k + (1-k)*np.exp(-1j*x0*t)
    # return charFunc

pltGen = True
pltGen = False
pltPdfs = True
pltPdfs = False
pltCdfs = True
pltCdfs = False
pltBox = True
# pltBox = False

pltSave = True
pltSave = False
    
intgt = 5
intmax = 10
dgn = 1 - (intgt/intmax) # only this percentage of loads are installed.

dVpu = 1e-5; # Tmax prop. 1/dVpu. This has to be quite big (over 1e-6) to get reasonable answers for i = 0:5.
dVpu = 1.0*1e-5; # Tmax prop. 1/dVpu. This has to be quite big (over 1e-6) to get reasonable answers for i = 0:5.
DVpu = 0.15; # Nt = DVpu/dVpu.
iKtot = 12

fdr_i = 5
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod']
feeder = fdrs[fdr_i]
lin_point=0.6


Vmax = 1.075
Vmin  = 0.95

dP = 10*1e3 # W
DP = 100000*1e3 # W

ld2mean = 0.5 # ie the mean of those generators which install is 1/2 of their load
# ld2mean = 2.0 # ie the mean of those generators which install is 1/2 of their load
# ld2mean = 8.0 # ie the mean of those generators which install is 1/2 of their load

# STEPS:
# Part A: analytic solution
# 1. Load linear model.
# 2. Choose bus; distribution
# 3. Calculate distribution
# 4. Run MC analysis using linear model
#
# Part B: Run MC analysis using OpenDSS
# 1. load appropriate model
# 2. sample distibution appropriately and run load flow
# 3. Compare results.




# PART A ===========================

LM = loadLinMagModel(feeder,lin_point,WD,'Lpt')
Ky=LM['Ky'];Kd=LM['Kd'];Kt=LM['Kt'];bV=LM['bV'];xhy0=LM['xhy0'];xhd0=LM['xhd0']
vBase = LM['vKvbase']

b0 = Ky.dot(xhy0) + Kd.dot(xhd0) + bV

KyP = Ky[:,:Ky.shape[1]//2]
KdP = Kd[:,:Kd.shape[1]//2]
Ktot = np.concatenate((KyP,KdP),axis=1)
Ktot[abs(Ktot)<1e-9]=0

# NB: mean of gamma distribution is k*th.
rndI = 1e4
xhy0rnd = ld2mean*rndI*np.round(xhy0[:xhy0.shape[0]//2]/rndI)
xhd0rnd = ld2mean*rndI*np.round(xhd0[:xhd0.shape[0]//2]/rndI)
k = 2.0;  # choose the same for all
Th = -np.concatenate((xhy0rnd,xhd0rnd))/k # negative so that the pds are positive.

Nt = int(DVpu/dVpu)

Tscale = 2e4*np.linalg.norm(Ktot,axis=1)
Tpscale = 2e6

Tmax = np.pi/(Tscale*vBase*dVpu) # see WB 22-1-19
dV = np.pi/Tmax

Tpmax = np.pi/(dP) # see WB 22-1-19
Np = int(DP/dP)
tp = np.linspace(-Tpmax,Tpmax,int(Np + 1))
P = dP*np.arange(-Np//2,Np//2 + 1)

cfTot = np.ones((len(Ktot),Nt+1),dtype='complex')

vDnew = np.zeros((len(Ktot),int(Nt+1)))
Vpu = np.zeros((len(Ktot),int(Nt+1)))

for i in range(len(Ktot)):
    t = np.linspace(-Tmax[i],Tmax[i],int(Nt + 1))
    j=0
    for th in Th:
        cfJ = dsf.cf_gm_dgn(k,th,t,Ktot[i,j],dgn);
        cfTot[i,:] = cfTot[i,:]*cfJ
        j+=1
    print('Calc DFT:',time.process_time())
    vDnew[i,:] = abs(np.fft.fftshift(np.fft.ifft(cfTot[i,:])))*vBase[i]/dV[i]
    
    v0 = b0[i]/vBase[i]
    Vpu[i,:] = (dV[i]*np.arange(-Nt//2,Nt//2 + 1) + b0[i])/vBase[i]


vDnewSumEr = ((sum(vDnew.T)/(vBase/dV)) - 1)*100 # normalised (%)
print('Checksum: sum of PDFs:',vDnewSumEr)

vDnewSum = sum(vDnew.T)
vDnewCdf = np.cumsum(vDnew,axis=1).T/vDnewSum.T

vAll = np.linspace(0.85,1.15,int(1e3))
minV = []
maxV = []
for v in vAll:
    cdfSet = []
    for i in range(len(Vpu)):
        cdfSet = cdfSet + [vDnewCdf[ np.argmin(abs(Vpu[i] - v)),i]]
    minV = minV + [min(cdfSet)]
    maxV = maxV + [max(cdfSet)]
    
print('Complete.',time.process_time())    








# ================ PLOTTING FUNCTIONS FROM HERE
if pltGen:
    pDnew = np.zeros((Np+1))
    pgTot = np.ones((Np+1),dtype='complex')
    j=0
    for th in Th:
        pgJ = cf(k,th,tp,1,dgn);
        pgTot = pgTot*pgJ
        j+=1
    pDnew = abs(np.fft.fftshift(np.fft.ifft(pgTot)))/dP

    plt.plot(P/1e6,pDnew*1e6)
    plt.xlabel('x (Power, MW)')
    plt.ylabel('p(x)')
    plt.xlim((0,5))
    plt.grid(True)
    if pltSave:
        plt.savefig(sn+'pltGen.png')
    else:
        plt.show()


if pltPdfs:
    for i in range(len(Ktot)):
        plt.plot(Vpu[i,:],vDnew[i,:])
    plt.xlim((0.90,1.1))
    
    plt.ylim((-5,90))
    ylm = plt.ylim()
    plt.plot(Vmin*np.ones(2),ylm,'r:')
    plt.plot(Vmax*np.ones(2),ylm,'r:')
    plt.ylim(ylm)
    plt.grid(True)
    if pltSave:
        plt.savefig(sn+'pltPdfs.png')
    else:
        plt.show()

if pltCdfs:
    plt.plot(Vpu.T,vDnewCdf)
    plt.plot(vAll,minV,'k--',linewidth=2.0)
    plt.plot(vAll,maxV,'k--',linewidth=2.0)
    plt.xlim((0.925,1.125))
    ylm = plt.ylim()
    plt.plot([Vmax,Vmax],ylm,'r:')
    plt.plot([Vmin,Vmin],ylm,'r:')
    plt.ylim(ylm)
    plt.xlabel('x (Voltage, pu)')
    plt.ylabel('p(X <= x)')
    plt.grid(True)
    if pltSave:
        plt.savefig(sn+'pltCdfs'+str(int(ld2mean*100))+'.png')
    else:
        plt.show()


if pltBox:
    Vmn = np.zeros(len(Ktot))
    Vlo = np.zeros(len(Ktot))
    Vmd = np.zeros(len(Ktot))
    Vhi = np.zeros(len(Ktot))
    Vmx = np.zeros(len(Ktot))

    emn = 0.01
    elo = 0.25
    emd = 0.50
    ehi = 0.75
    emx = 0.99

    for i in range(len(Ktot)):
        Vmn[i]=Vpu[i,np.argmin(abs(vDnewCdf[:,i] - emn))]
        Vlo[i]=Vpu[i,np.argmin(abs(vDnewCdf[:,i] - elo))]
        Vmd[i]=Vpu[i,np.argmin(abs(vDnewCdf[:,i] - emd))]
        Vhi[i]=Vpu[i,np.argmin(abs(vDnewCdf[:,i] - ehi))]
        Vmx[i]=Vpu[i,np.argmin(abs(vDnewCdf[:,i] - emx))]
        plt.plot(i,Vmn[i],'k^'); 
        plt.plot(i,Vlo[i],'g_'); plt.plot(i,Vmd[i],'b_'); plt.plot(i,Vhi[i],'g_');
        plt.plot(i,Vmx[i],'kv'); 
        plt.plot([i,i],[Vmn[i],Vmx[i]],'k:')
        plt.plot(i,b0[i]/vBase[i],'rx')

    plt.xlabel('Bus No.')
    plt.ylabel('Voltage (pu)')
    xlm = plt.xlim()
    plt.plot(xlm,[Vmax,Vmax],'r--')
    plt.plot(xlm,[Vmin,Vmin],'r--')
    plt.xlim(xlm)
    if pltSave:
        plt.savefig(sn+'pltBox.png')
    else:
        plt.show()
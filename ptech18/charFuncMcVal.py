# based on the script from 21/1, 'fft_calcs' in matlab.
# Based on script charFuncHcAlys, deleted 30/01
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import getpass
from dss_python_funcs import *
from math import gamma
import time
import dss_stats_funcs as dsf
import win32com.client

if getpass.getuser()=='chri3793':
    WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"
    sn = r"C:\Users\chri3793\Documents\DPhil\malcolm_updates\wc190204\\charFuncHcAlys_"
elif getpass.getuser()=='Matt':
    WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
    sn = r"C:\Users\Matt\Documents\DPhil\malcolm_updates\wc190128\\charFuncHcAlys_"

# Plotting options:
pltGen = True
pltGen = False
pltPdfs = True
pltPdfs = False
pltCdfs = True
pltCdfs = False
pltBox = True
pltBox = False
pltBoxDss = True
pltBoxDss = False
pltBoxBoth = True
pltBoxBoth = False

pltSave = True
pltSave = False

ltcModel=True
ltcModel=False

intgt = 00
intmax = 10
dgn = 1 - (intgt/intmax) # only this percentage of loads are installed.

dVpu = 1e-5; # Tmax prop. 1/dVpu. This has to be quite big (over 1e-6) to get reasonable answers for i = 0:5.
dVpu = 1.0*1e-5; # Tmax prop. 1/dVpu. This has to be quite big (over 1e-6) to get reasonable answers for i = 0:5.
DVpu = 0.15; # Nt = DVpu/dVpu.
iKtot = 12

fdr_i = 8
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7']
feeder = fdrs[fdr_i]
lin_point=0.6
lp_taps='Lpt'

nMc = int(1e3)

ckt = get_ckt(WD,feeder)
fn_ckt = ckt[0]
fn = ckt[1]

Vmax = 1.05
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
if not ltcModel:
    # IF using the FIXED model:
    LM = loadLinMagModel(feeder,lin_point,WD,'Lpt')
    Ky=LM['Ky'];Kd=LM['Kd'];bV=LM['bV'];xhy0=LM['xhy0'];xhd0=LM['xhd0']
    vBase = LM['vKvbase']
    

    b0 = Ky.dot(xhy0) + Kd.dot(xhd0) + bV

    KyP = Ky[:,:Ky.shape[1]//2]
    KdP = Kd[:,:Kd.shape[1]//2]
    Ktot = np.concatenate((KyP,KdP),axis=1)
    Ktot[abs(Ktot)<1e-9]=0
elif ltcModel:
    # IF using the LTC model:
    LM = loadLtcModel(feeder,lin_point,WD,'Lpt')
    A=LM['A'];bV=LM['B'];xhy0=LM['xhy0'];xhd0=LM['xhd0']; 
    vBase = LM['Vbase']
    
    x0 = np.concatenate((xhy0,xhd0))
    b0 = A.dot(x0) + bV
    
    KyP = A[:,0:len(xhy0)//2]
    KdP = A[:,len(xhy0):len(xhy0) + (len(xhd0)//2)]
    
    Ktot = np.concatenate((KyP,KdP),axis=1)
    Ktot[abs(Ktot)<1e-9]=0    
v_idx=LM['v_idx']; 
YZp = LM['SyYNodeOrder']
YZd = LM['SdYNodeOrder']

# NB: mean of gamma distribution is k*th.
rndI = 1e4
xhy0rnd = ld2mean*rndI*np.round(xhy0[:xhy0.shape[0]//2]/rndI  - np.finfo(np.float64).eps) # latter required to make sure that this is negative
xhd0rnd = ld2mean*rndI*np.round(xhd0[:xhd0.shape[0]//2]/rndI - np.finfo(np.float64).eps)
k = 2.0;  # choose the same for all
Th = -np.concatenate((xhy0rnd,xhd0rnd))/k # negative so that the pds are positive.

Nt = round(DVpu/dVpu)

Tscale = 2e4*np.linalg.norm(Ktot,axis=1)
Tpscale = 2e6

Tmax = np.pi/(Tscale*vBase*dVpu) # see WB 22-1-19
dV = np.pi/Tmax

Tpmax = np.pi/(dP) # see WB 22-1-19
Np = int(DP/dP)
tp = np.linspace(-Tpmax,Tpmax,int(Np + 1))

P = dP*np.arange(-Np//2,Np//2 + 1)

cfTot = np.ones((len(Ktot),Nt+1),dtype='complex')
# cfTot = np.ones((len(Ktot),Nt//2+1),dtype='complex')

vDnew = np.zeros((len(Ktot),int(Nt+1)))
Vpu = np.zeros((len(Ktot),int(Nt+1)))

for i in range(len(Ktot)):
    if i%(len(Ktot)//10)==0:
        print(i,'/',len(Ktot))
    t = np.linspace(-Tmax[i],Tmax[i],int(Nt + 1))
    # t = np.linspace(0,Tmax[i],int(Nt//2 + 1))
    j=0
    for th in Th:
        cfJ = dsf.cf_gm_dgn(k,th,t,Ktot[i,j],dgn);
        cfTot[i,:] = cfTot[i,:]*cfJ
        j+=1
    vDnew[i,:] = abs(np.fft.fftshift(np.fft.ifft(cfTot[i,:])))*vBase[i]/dV[i]
    # vDnew[i,:] = abs(np.fft.irfft(cfTot[i,:]))*vBase[i]/dV[i]
    
    v0 = b0[i]/vBase[i]
    Vpu[i,:] = (dV[i]*np.arange(-Nt//2,Nt//2 + 1) + b0[i])/vBase[i]


vDnewSumEr = ((sum(vDnew.T)/(vBase/dV)) - 1)*100 # normalised (%)
print('DFT Calc complete.',time.process_time())

print('Checksum: max PDFs error',max(vDnewSumEr))
print('Checksum: mean PDFs error',np.mean(vDnewSumEr))

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

# PART B FROM HERE ==============================

# 1. load the appropriate model/DSS
DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
DSSText = DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution = DSSCircuit.Solution

DSSText.command='Compile ('+fn+'.dss)'
BB0,SS0 = cpf_get_loads(DSSCircuit)
if lp_taps=='Lpt':
    cpf_set_loads(DSSCircuit,BB0,SS0,lin_point)
    DSSSolution.Solve()

if not ltcModel:
    DSSText.command='set controlmode=off'
elif ltcModel:
    DSSText.command='set maxcontroliter=30'
    DSSText.command='set maxiterations=100'


YNodeVnom = tp_2_ar(DSSCircuit.YNodeVarray)
YZ = DSSCircuit.YNodeOrder
YZ = vecSlc(YZ[3:],v_idx)

# 2. run MC analysis, going through each generator and setting to a power.
genNamesY = add_generators(DSSObj,YZp,False)
genNamesD = add_generators(DSSObj,YZd,True)
DSSSolution.Solve()

genNames = genNamesY+genNamesD

# 2a. now draw from the correct distributions
pdfGen = np.zeros((nMc,len(genNames)))
for i in range(len(genNames)):
    pdfGen[:,i] = np.random.gamma(k,1e-3*Th[i],nMc)

vOut = np.zeros((nMc,len(v_idx)))
conv = []
print('---- Start MC ----',time.process_time())
for i in range(nMc):
    if i%(nMc//10)==0:
        print(i,'/',nMc)
    set_generators( DSSCircuit,genNames,pdfGen[i] )
    DSSSolution.Solve()
    conv = conv+[DSSSolution.Converged]
    v00 = abs(tp_2_ar(DSSCircuit.YNodeVarray))
    vOut[i,:] = v00[3:][v_idx]/vBase

print('MC complete.',time.process_time())
print('No. Converged:',sum(conv),'/',nMc)

# COMPARE RESULTS ==========
if pltBoxDss:
    plt.boxplot(vOut,whis=[1,99])
    plt.plot(range(1,len(vBase)+1),abs(YNodeVnom[3:])[v_idx]/vBase,'rx')
    plt.xlabel('Bus no.')
    plt.ylabel('Voltage (pu)')
    xlm = plt.xlim()
    plt.plot(xlm,[Vmax,Vmax],'r--')
    plt.plot(xlm,[Vmin,Vmin],'r--')
    plt.xlim(xlm)
    plt.grid(True)
    if pltSave:
        plt.savefig(sn+'pltBoxDss.png')
        plt.close()
    else:
        plt.show()

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
    plt.grid(True)
    if pltSave:
        plt.savefig(sn+'pltBox.png')
    else:
        plt.show()
if pltBoxBoth:
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
    
    plt.figure(figsize=(9,4))

    plt.subplot(122)
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
    plt.grid(True)
    
    plt.title('Linear Model')
    plt.subplot(121)
    
    Vmn = np.percentile(vOut,1,axis=0)
    Vlo = np.percentile(vOut,25,axis=0)
    Vmd = np.percentile(vOut,50,axis=0)
    Vhi = np.percentile(vOut,25,axis=0)
    Vmx = np.percentile(vOut,99,axis=0)
    
    plt.plot(Vmn,'k^')
    plt.plot(Vlo,'g_'); plt.plot(Vmd,'b_'); plt.plot(Vhi,'g_');
    plt.plot(Vmx,'kv'); 
    plt.plot([range(len(v_idx)),range(len(v_idx))],[Vmn,Vmx],'k:')
    plt.plot(b0/vBase,'rx')

    xlm = plt.xlim()
    plt.plot(xlm,[Vmax,Vmax],'r--')
    plt.plot(xlm,[Vmin,Vmin],'r--')
    plt.xlim(xlm)
    plt.grid(True)
    
    plt.title('OpenDSS Solutions')

    plt.show()
    
    if pltSave:
        plt.savefig(sn+'pltBox.png')
    else:
        plt.show()
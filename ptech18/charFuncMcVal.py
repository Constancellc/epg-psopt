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
import scipy.stats

if getpass.getuser()=='chri3793':
    WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"
    sn = r"C:\Users\chri3793\Documents\DPhil\malcolm_updates\wc190204\\charFuncMcVal_"
elif getpass.getuser()=='Matt':
    WD = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18"
    sn = r"C:\Users\Matt\Documents\DPhil\malcolm_updates\wc190204\\charFuncMcVal_"

# Plotting options:
pltGen = True
pltGen = False
pltPdfs = True
pltPdfs = False
pltCdfs = True
pltCdfs = False
pltBox = True
# pltBox = False
pltBoxDss = True
pltBoxDss = False
pltBoxBoth = True
pltBoxBoth = False
pltBoxNorm = True
pltBoxNorm = False
pltLinRst = True
pltLinRst = False

pltSave = True
pltSave = False

ltcModel=True
ltcModel=False

pltCritBus = True
# pltCritBus = False

intgt = 00
intmax = 10
dgn = 1 - (intgt/intmax) # only this percentage of loads are installed.

fdr_i = 5
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7']
feeder = fdrs[fdr_i]
lin_point=0.6
lp_taps='Lpt'

nMc = int(1e2)

ckt = get_ckt(WD,feeder)
fn_ckt = ckt[0]
fn = ckt[1]

Vmax = 1.055
Vmin  = 0.95

ld2mean = 0.5 # ie the mean of those generators which install is 1/2 of their load
ld2mean = 3.0 # ie the mean of those generators which install is 1/2 of their load
# ld2mean = 1.25 # ie the mean of those generators which install is 1/2 of their load

sn0 = sn +  feeder + str(int(lin_point*1e2)) + 'ltc' + str(int(ltcModel)) + 'ld' + str(int(ld2mean*1e2))

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

    b0 = (Ky.dot(xhy0) + Kd.dot(xhd0) + bV)/vBase # in pu

    KyP = Ky[:,:Ky.shape[1]//2]
    KdP = Kd[:,:Kd.shape[1]//2]
    Ktot = np.concatenate((KyP,KdP),axis=1)
elif ltcModel:
    # IF using the LTC model:
    LM = loadLtcModel(feeder,lin_point,WD,'Lpt')
    A=LM['A'];bV=LM['B'];xhy0=LM['xhy0'];xhd0=LM['xhd0']; 
    vBase = LM['Vbase']
    
    x0 = np.concatenate((xhy0,xhd0))
    b0 = (A.dot(x0) + bV)/vBase # in pu
    
    KyP = A[:,0:len(xhy0)//2]
    KdP = A[:,len(xhy0):len(xhy0) + (len(xhd0)//2)]
    
    Ktot = np.concatenate((KyP,KdP),axis=1)
v_idx=LM['v_idx'];
YZp = LM['SyYNodeOrder']
YZd = LM['SdYNodeOrder']

# NB: mean of gamma distribution is k*th; variance is k*(th**2)
rndI = 1e3
xhy0rnd = ld2mean*rndI*np.round(xhy0[:xhy0.shape[0]//2]/rndI  - 1e6*np.finfo(np.float64).eps) # latter required to make sure that this is negative
xhd0rnd = ld2mean*rndI*np.round(xhd0[:xhd0.shape[0]//2]/rndI - 1e6*np.finfo(np.float64).eps)
k = 2.0;  # choose the same for all
# th0 = 

Th = -np.concatenate((xhy0rnd,xhd0rnd))/k # negative so that the pds are positive.
# Sgm = Th/th0
Sgm = Th*(k**0.5) # scale to unity variance
Mns = Th*k

# REDUCE THE LINEAR MODEL to a nice form for multiplication
KtotPu = dsf.vmM(1/vBase,Ktot) # scale to be in pu
KtotPu0 = dsf.mvM(KtotPu,Sgm) # scale the matrices to the input variance
Kmax = np.max(abs(KtotPu0),axis=1)
K1 = dsf.vmM(1/Kmax,KtotPu0) # Finally, scale so that all K are unity


# IDENTIFY WORST BUSES using normal approximation.
Ksgm = np.sqrt(np.sum(abs(K1),axis=1)) # useful for normal approximations
Mm = KtotPu.dot(Mns)
Kk = np.sqrt(np.sum(abs(K1),axis=1))*Kmax

critBuses = dsf.getCritBuses(b0,Vmax,Mm,Kk,Scl=np.arange(0.1,3.10,0.1)/ld2mean)
a = dsf.getBusSens(b0,Vmax,Mm,Kk)
# Scl = np.arange(0.05,3.05,0.05)
# plt.plot(Scl,a[:,critBuses]);
# xlm = plt.xlim()
# plt.plot(xlm,[3.,3.],'r--')
# plt.xlim(xlm); plt.show()

# Choose the scale for x/t
Nmult = np.ceil(10.0 + np.sqrt(Ktot.shape[0]))
Dx = 2*Nmult
dx = 3e-2
x,t = dsf.dy2YzR(dx,Dx)
Nt = len(x)-1

cfTot = np.ones((len(Ktot),Nt//2 + 1),dtype='complex')
pdfV = np.zeros((len(Ktot),int(Nt+1)))
pdfVnorm = np.zeros((len(Ktot),int(Nt+1)))
Vpu = np.zeros((len(Ktot),int(Nt+1)))
print('--- Start DFT Calc ---\n',time.process_time()) # START DFT CALCS HERE ============
for i in range(len(K1)):
    if i%(len(K1)//10)==0:
        print(i,'/',len(K1))
    
    for j in range(K1.shape[1]):
        cfJ = dsf.cf_gm_sh(k,k**-0.5,t*K1[i,j]) # shifted mean; scaled to unit variance
        cfTot[i,:] = cfTot[i,:]*cfJ
    pdfV[i,:] = np.fft.fftshift(np.fft.irfft(cfTot[i,:],n=Nt+1))
    pdfVnorm[i,:] = scipy.stats.norm.pdf(x,scale=Ksgm[i])*dx # unit variance means sum of sqrt(abs(K)).

print('DFT Calc complete.',time.process_time())

# Vpu[i,:] = b0[i] + x*Kmax[i] + KtotPu[i].dot(Mns) # < === per i version here of VVV
Vpu = (b0 + KtotPu.dot(Mns) + dsf.vmM(Kmax,dsf.mvM(np.ones(pdfV.shape),x)).T).T

pdfVsum = (sum(pdfV.T) - 1)*100 # normalised (%)
print('Checksum: max PDF error',max(pdfVsum),'%')
print('Checksum: mean PDF error',np.mean(pdfVsum),'%')

cdfV = np.cumsum(pdfV,axis=1).T
cdfVnorm = np.cumsum(pdfVnorm,axis=1).T
    
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
print('\nNo. Converged:',sum(conv),'/',nMc)

# MC Error analysis:
vOutH0 = vOut[0:nMc//2]
vOutH1 = vOut[nMc//2:]
vOutH0.sort(axis=0)
vOutH1.sort(axis=0)

minVdssH0 = np.min(vOutH0,axis=1)
maxVdssH0 = np.max(vOutH0,axis=1)
minVdssH1 = np.min(vOutH1,axis=1)
maxVdssH1 = np.max(vOutH1,axis=1)

yscale01 = np.linspace(0.0,1.0,nMc//2)

Vp_pct_dss_H0 = 100.0*(1 - yscale01[np.argmin(abs(maxVdssH0 - Vmax))])
Vp_pct_dss_H1 = 100.0*(1 - yscale01[np.argmin(abs(maxVdssH1 - Vmax))])
errH = 100.0*(Vp_pct_dss_H0-Vp_pct_dss_H1)/Vp_pct_dss_H0

print('\n==> HC Ests:',Vp_pct_dss_H0,'%, ',Vp_pct_dss_H1,'%')
print('==> HC Relative Error:',errH,'%')

# NOW: calculate the HC value:
vOut.sort(axis=0)
minVdss = np.min(vOut,axis=1)
maxVdss = np.max(vOut,axis=1)

yscale = np.linspace(0.,1.,len(vOut))

Vp_pct_dss = 100.0*(1 - yscale[np.argmin(abs(maxVdss - Vmax))])

prHcMax = dsf.getHc(Vpu,cdfV,Vmax)
prHcMin = dsf.getHc(Vpu,cdfV,Vmin)

Vp_pct_aly = 100.0*(1 - prHcMax)

print('\n==> HC Analytic value:',Vp_pct_aly,'%')
print('==> HC OpenDSS value:',Vp_pct_dss,'%\n')


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
        plt.savefig(sn0+'pltBoxDss.png')
        plt.close()
    else:
        plt.show()

# ================ PLOTTING FUNCTIONS FROM HERE
if pltGen:
    dP = 10*1e3 # W
    DP = 100000*1e3 # W
    Tpmax = np.pi/(dP) # see WB 22-1-19
    Np = int(DP/dP)
    tp = np.linspace(-Tpmax,Tpmax,int(Np + 1))
    P = dP*np.arange(-Np//2,Np//2 + 1)

    pDnew = np.zeros((Np+1))
    pgTot = np.ones((Np+1),dtype='complex')
    j=0
    for th in Th:
        pgJ = dsf.cf_gm_dgn(k,th,tp,1,dgn);
        pgTot = pgTot*pgJ
        j+=1
    pDnew = abs(np.fft.fftshift(np.fft.ifft(pgTot)))/dP

    plt.plot(P/1e6,pDnew*1e6)
    plt.xlabel('x (Power, MW)')
    plt.ylabel('p(x)')
    plt.xlim((0,5))
    plt.grid(True)
    if pltSave:
        plt.savefig(sn0+'pltGen.png')
    else:
        plt.show()


if pltPdfs:
    for i in range(len(Ktot)):
        plt.plot(Vpu[i,:],pdfV[i,:])
    plt.xlim((0.90,1.1))
    
    # plt.ylim((-5,90))
    ylm = plt.ylim()
    plt.plot(Vmin*np.ones(2),ylm,'r:')
    plt.plot(Vmax*np.ones(2),ylm,'r:')
    plt.ylim(ylm)
    plt.grid(True)
    if pltSave:
        plt.savefig(sn0+'pltPdfs.png')
    else:
        plt.show()

if pltCdfs:
    # Analytic
    plt.subplot(122)
    plt.title('Linear Model')
    plt.plot(Vpu.T,cdfV)
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
    
    plt.subplot(121)
    plt.title('OpenDSS')
    vOutS = vOut.T
    for vout in vOutS:
        plt.plot(vout,yscale)
    plt.plot(minVdss,yscale,'k--',linewidth=2.0)
    plt.plot(maxVdss,yscale,'k--',linewidth=2.0)
    plt.xlim((0.925,1.125))
    ylm = plt.ylim()
    plt.plot([Vmax,Vmax],ylm,'r:')
    plt.plot([Vmin,Vmin],ylm,'r:')
    plt.ylim(ylm)
    plt.xlabel('x (Voltage, pu)')
    plt.ylabel('p(X <= x)')
    plt.grid(True)
    
    if pltSave:
        plt.savefig(sn0+'pltCdfs'+str(int(ld2mean*100))+'.png')
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
        Vmn[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - emn))]
        Vlo[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - elo))]
        Vmd[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - emd))]
        Vhi[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - ehi))]
        Vmx[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - emx))]
        plt.plot(i,Vmn[i],'k^'); 
        plt.plot(i,Vlo[i],'g_'); plt.plot(i,Vmd[i],'b_'); plt.plot(i,Vhi[i],'g_');
        plt.plot(i,Vmx[i],'kv'); 
        plt.plot([i,i],[Vmn[i],Vmx[i]],'k:')
        plt.plot(i,b0[i],'rx')

    plt.xlabel('Bus No.')
    plt.ylabel('Voltage (pu)')
    xlm = plt.xlim()
    plt.plot(xlm,[Vmax,Vmax],'r--')
    plt.plot(xlm,[Vmin,Vmin],'r--')
    plt.xlim(xlm)
    
    if pltCritBus:
        ylm = plt.ylim()
        for critBus in critBuses:
            plt.plot([critBus]*2,ylm,'g',zorder=-1e3)
        plt.ylim(ylm)
    else:
        plt.grid(True)
    
    if pltSave:
        plt.savefig(sn0+'pltBox.png')
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
        Vmn[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - emn))]
        Vlo[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - elo))]
        Vmd[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - emd))]
        Vhi[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - ehi))]
        Vmx[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - emx))]
        plt.plot(i,Vmn[i],'k^'); 
        plt.plot(i,Vlo[i],'g_'); plt.plot(i,Vmd[i],'b_'); plt.plot(i,Vhi[i],'g_');
        plt.plot(i,Vmx[i],'kv'); 
        plt.plot([i,i],[Vmn[i],Vmx[i]],'k:')
        plt.plot(i,b0[i],'rx')

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
    plt.plot(b0,'rx')
    

    xlm = plt.xlim()
    plt.plot(xlm,[Vmax,Vmax],'r--')
    plt.plot(xlm,[Vmin,Vmin],'r--')
    plt.xlim(xlm)
    # plt.grid(True)
    
    plt.title('OpenDSS Solutions')
    
    if pltSave:
        plt.savefig(sn0+'pltBoxBoth.png')
    else:
        plt.show()
        
if pltBoxNorm:
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

    plt.subplot(121)
    for i in range(len(Ktot)):
        Vmn[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - emn))]
        Vlo[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - elo))]
        Vmd[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - emd))]
        Vhi[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - ehi))]
        Vmx[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - emx))]
        plt.plot(i,Vmn[i],'k^'); 
        plt.plot(i,Vlo[i],'g_'); plt.plot(i,Vmd[i],'b_'); plt.plot(i,Vhi[i],'g_');
        plt.plot(i,Vmx[i],'kv'); 
        plt.plot([i,i],[Vmn[i],Vmx[i]],'k:')
        plt.plot(i,b0[i],'rx')

    plt.xlabel('Bus No.')
    plt.ylabel('Voltage (pu)')
    xlm = plt.xlim()
    plt.plot(xlm,[Vmax,Vmax],'r--')
    plt.plot(xlm,[Vmin,Vmin],'r--')
    plt.xlim(xlm)
    plt.grid(True)
    
    plt.title('Linear Model (full)')
    
    plt.subplot(122)
    for i in range(len(Ktot)):
        Vmn[i]=Vpu[i,np.argmin(abs(cdfVnorm[:,i] - emn))]
        Vlo[i]=Vpu[i,np.argmin(abs(cdfVnorm[:,i] - elo))]
        Vmd[i]=Vpu[i,np.argmin(abs(cdfVnorm[:,i] - emd))]
        Vhi[i]=Vpu[i,np.argmin(abs(cdfVnorm[:,i] - ehi))]
        Vmx[i]=Vpu[i,np.argmin(abs(cdfVnorm[:,i] - emx))]
        plt.plot(i,Vmn[i],'k^'); 
        plt.plot(i,Vlo[i],'g_'); plt.plot(i,Vmd[i],'b_'); plt.plot(i,Vhi[i],'g_');
        plt.plot(i,Vmx[i],'kv'); 
        plt.plot([i,i],[Vmn[i],Vmx[i]],'k:')
        plt.plot(i,b0[i],'rx')
    plt.xlabel('Bus No.')
    plt.ylabel('Voltage (pu)')
    xlm = plt.xlim()
    plt.plot(xlm,[Vmax,Vmax],'r--')
    plt.plot(xlm,[Vmin,Vmin],'r--')
    plt.xlim(xlm)
    plt.grid(True)
    
    plt.title('Linear Model (norm approx)')
    
    if pltSave:
        plt.savefig(sn0+'pltBoxNorm.png')
    else:
        plt.show()
        

if pltLinRst:
    Scls = np.arange(0.1,3.01,0.01)
    # Scls = np.arange(0.5,5.5,0.5)
    prScls = np.zeros((len(Scls))); n=0
    prScls2 = np.zeros((len(Scls)))
    for scl in Scls:
        Vpu = (b0 +  scl*(KtotPu.dot(Mns) + dsf.vmM(Kmax,dsf.mvM(np.ones(pdfV.shape),x)).T)).T
        prVmax = np.zeros((len(K1)))
        
        prHc = dsf.getHc(Vpu,cdfV,Vmax)
        prHc2 = dsf.getHc(Vpu,cdfV,Vmin)
        
        prScls[n] = 100*(1-prHc)
        prScls2[n] = 100*prHc2
        # print('Linearly scaled HCs, scale',100*int(scl),'%, HC:',100*(1-prHc),'%')
        n+=1
    
    plt.plot(100*Scls,prScls)
    plt.plot(100*Scls,prScls2)
    plt.xlabel('Scale factor, %')
    plt.ylabel('Probability of an overvoltage, %')
    plt.show()
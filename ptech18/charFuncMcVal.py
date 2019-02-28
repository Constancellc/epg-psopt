# based on the script from 21/1, 'fft_calcs' in matlab.
# Based on script charFuncHcAlys, deleted 30/01

# STEPS:
# Part A: analytic solution
# 1. Load linear model.
# 2. Choose distributions at each bus
# 3. Calculate distribution
# 4. Run MC analysis using linear model
#
# Part B: Run MC analysis using OpenDSS
# 1. load appropriate model
# 2. sample distibution appropriately and run load flow
# 3. Compare results.

import numpy as np
from dss_python_funcs import *

# CHOOSE Network
fdr_i = 9
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']
feeder = fdrs[fdr_i]
# feeder = '213'

netModel=0 # none
netModel=1 # ltc
netModel=2 # fixed

lin_point=0.6
lp_taps='Lpt'

Vmax = 1.055
# Vmax = 1.065 # EPRI ckt 7; 8500 Node
Vmin  = 0.95

ld2mean = 0.5 # ie the mean of those generators which install is 1/2 of their load

nMc = int(1e5)
nMc = int(3e2)
nMc = int(1e2)
# nMc = int(30)

# PDF options
mu_k0 = np.arange(0.5,6.0,0.5) # NB this is as a PERCENTAGE of the chosen nominal powers. 
# mu_k0 = np.arange(0.5,6.0,0.1) # NB this is as a PERCENTAGE of the chosen nominal powers. 
# mu_k0 = 5.5*np.array([1.0]) # NB this is as a PERCENTAGE of the chosen nominal powers. 

# mu_kk = getMu_Kk(feeder,netModel)
mu_kk = 0.4
mu_k = mu_k0*mu_kk
pdfName = 'gamma'
k = np.array([2.0]) # we do not know th, sigma until we know the scaling from mu0.
params = k
pdfData = {'name':pdfName,'prms':params,'mu_k':mu_k,'nP':(len(params),len(mu_k))}

mcLinOn = True
# mcLinOn = False
mcDssOn = True
# mcDssOn = False
useCbs = True
# useCbs = False

# (Active) PLOTTING options:
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
pltBoxNorm = True
pltBoxNorm = False
pltLinRst = True
pltLinRst = False
pltHcBoth = True
# pltHcBoth = False
pltGen = True
pltGen = False

pltCritBus = True
pltCritBus = False

pltSave = True
pltSave = False
# ADMIN =============================================
import numpy.random as rnd
import matplotlib.pyplot as plt
import getpass
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

ckt = get_ckt(WD,feeder)
fn_ckt = ckt[0]
fn = ckt[1]

DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
DSSText = DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution = DSSCircuit.Solution
# sn0 = sn +  feeder + str(int(lin_point*1e2)) + 'mdl' + str(netModel) + 'ld' + str(int(ld2mean*1e2))

# print('Start. Feeder:',feeder,' Linpoint:',lin_point,' Tap Model:',netModel,' Vmax:',Vmax)

# PART A.1 - load model ===========================
if not netModel:
    # IF using the FIXED model:
    LM = loadLinMagModel(feeder,lin_point,WD,'Lpt')
    Ky=LM['Ky'];Kd=LM['Kd'];bV=LM['bV'];xhy0=LM['xhy0'];xhd0=LM['xhd0']
    vBase = LM['vKvbase']

    b0 = (Ky.dot(xhy0) + Kd.dot(xhd0) + bV)/vBase # in pu

    KyP = Ky[:,:Ky.shape[1]//2]
    KdP = Kd[:,:Kd.shape[1]//2]
    Ktot = np.concatenate((KyP,KdP),axis=1)
elif netModel>0: # 1 or 2
    # IF using the LTC model:
    LM = loadNetModel(feeder,lin_point,WD,'Lpt',netModel)
    A=LM['A'];bV=LM['B'];xhy0=LM['xhy0'];xhd0=LM['xhd0']; 
    vBase = LM['Vbase']
    
    x0 = np.concatenate((xhy0,xhd0))
    b0 = (A.dot(x0) + bV)/vBase # in pu
    
    KyP = A[:,0:len(xhy0)//2] # these might be zero if there is no injection (e.g. only Q)
    KdP = A[:,len(xhy0):len(xhy0) + (len(xhd0)//2)]
    
    Ktot = np.concatenate((KyP,KdP),axis=1)

KtotCheck = np.sum(Ktot==0,axis=1)!=Ktot.shape[1]
Ktot = Ktot[KtotCheck]
b0 = b0[KtotCheck]
vBase = vBase[KtotCheck]

v_idx=LM['v_idx'][KtotCheck]
YZp = LM['SyYNodeOrder']
YZd = LM['SdYNodeOrder']

# REDUCE THE LINEAR MODEL to a nice form for multiplication
KtotPu = dsf.vmM(1/vBase,Ktot) # scale to be in pu


# OPENDSS ADMIN =======================================
# B1. load the appropriate model/DSS
DSSText.command='Compile ('+fn+'.dss)'
BB0,SS0 = cpf_get_loads(DSSCircuit)
if lp_taps=='Lpt':
    cpf_set_loads(DSSCircuit,BB0,SS0,lin_point)
    DSSSolution.Solve()

# if not ltcModel:
if not netModel:
    DSSText.command='set controlmode=off'
elif netModel:
# elif ltcModel:
    print(netModel)
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

Vp_pct_aly = np.zeros(pdfData['nP'])
Vp_pct_dss = np.zeros(pdfData['nP'])
Vp_pct_lin = np.zeros(pdfData['nP'])

hc_aly = np.nan*np.zeros(pdfData['nP'])
hc_dss = np.zeros(pdfData['nP'])
hc_lin = np.zeros(pdfData['nP'])
hcGenSet = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],5))
hcGenSetLin = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],5))
hcGen = []; hcGenLin = []

for i in range(pdfData['nP'][0]):
    # PART A.2 - choose distributions and reduce linear model ===========================
    roundI = 1e0
    Mu0_y = -ld2mean*roundI*np.round(xhy0[:xhy0.shape[0]//2]/roundI  - 1e6*np.finfo(np.float64).eps) # latter required to make sure that this is negative
    Mu0_d = -ld2mean*roundI*np.round(xhd0[:xhd0.shape[0]//2]/roundI - 1e6*np.finfo(np.float64).eps)
    Mu0 = np.concatenate((Mu0_y,Mu0_d))
    
    if pdfData['name']=='gamma': # NB: mean of gamma distribution is k*th; variance is k*(th**2)
        k = pdfData['prms'][i]
        sgm = Mu0/np.sqrt(pdfData['prms'][i])
        # Th = Mu0/k # theta as a scale parameter 
    
    KtotPu0 = dsf.mvM(KtotPu,sgm) # scale the matrices to the input variance
    Kmax = np.max(abs(KtotPu0),axis=1)
    K1 = dsf.vmM(1/Kmax,KtotPu0) # Finally, scale so that all K are unity
    
    Kgen = np.ones((1,len(sgm)))
    Kgen0 = dsf.mvM(Kgen,sgm)
    KgenMax = np.max(abs(Kgen0),axis=1)
    Kg = dsf.vmM(1/KgenMax,Kgen0)
    
    # IDENTIFY WORST BUSES using normal approximation.
    Ksgm = np.sqrt(np.sum(abs(K1),axis=1)) # useful for normal approximations
    Mm = KtotPu.dot(Mu0)
    Kk = np.sqrt(np.sum(abs(K1),axis=1))*Kmax
    if useCbs:
        cBs = dsf.getCritBuses(b0,Vmax,Mm,Kk,Scl=np.arange(0.1,3.10,0.1)/ld2mean) # critical buses
    else:
        cBs = np.arange(0,len(b0)) # all buses
    KgSgm = np.sqrt(np.sum(abs(Kg),axis=1)) # useful for normal approximations
    MmGen = Kgen.dot(Mu0)
    KkGen = np.sqrt(np.sum(abs(Kg),axis=1))*KgenMax

    # PART A.3 - Calculate PDF ===========================
    print('---- Start DFT ----',time.process_time())
    params = ['gamma',k,k**-0.5] # parameters chosen to make sure zero mean/unit variance
    # Choose the scale for x/t
    Dx = np.ceil(2*3*2*max(Ksgm));
    dx = 3e-2
    x,t = dsf.dy2YzR(dx,Dx)
    Nt = len(x)-1
    
    pdfV = dsf.calcPdfSum(K1[cBs],x,t,params,verbose=True) # pdf calcs here
    pdfVnorm = dsf.getPdfNormSum(Ksgm,x)
    
    DxG = np.ceil(2*3*2*max(KgSgm));
    dxG = 1e-2
    xG,tG = dsf.dy2YzR(dx,Dx)
    Nt = len(x)-1

    pdfG = dsf.calcPdfSum(Kg,xG,tG,params)[0]
    pdfGnorm = dsf.getPdfNormSum(KgSgm,x)[0]
    
    cdfV = np.cumsum(pdfV,axis=1).T
    cdfVnorm = np.cumsum(pdfVnorm,axis=1).T
    cdfG = np.cumsum(pdfG)
    cdfGnorm = np.cumsum(pdfGnorm)
    
    for jj in range(pdfData['nP'][-1]):
        Mn_k = pdfData['mu_k'][jj]
        # Vpu[i,:] = b0[i] + Mn_k*(x*Kmax[i] + KtotPu[i].dot(Mu0)) # < === per i version here of VVV
        # Vpu = (b0 + Mn_k*(dsf.vmM(Kmax,dsf.mvM(np.ones(pdfV.shape),x)).T + KtotPu.dot(Mu0))).T
        Vpu = (b0[cBs] + Mn_k*(dsf.vmM(Kmax[cBs],dsf.mvM(np.ones(pdfV.shape),x)).T + KtotPu.dot(Mu0)[cBs])).T
        Gpu = Mn_k*((dsf.vmM(KgenMax,dsf.mvM(np.ones((1,pdfG.shape[0])),xG))[0] + Kgen.dot(Mu0)))
        
        prHcMax = dsf.getHc(Vpu,cdfV,Vmax)
        prHcMin = dsf.getHc(Vpu,cdfV,Vmin)
        
        Vp_pct_aly[i,jj] = 100.0*(1 - prHcMax)
        if (1-prHcMax)>1e-7:
            # print('Index:',np.argmin(abs(cdfG - prHcMax))) # for testing
            hc_aly[i,jj] = Gpu[np.argmin(abs(cdfG - prHcMax))]*1e-3 # kW

    print('Complete DFT version.',time.process_time())    

    # PART B FROM HERE ==============================
    if mcLinOn or mcDssOn:
        print('---- Start MC ----',time.process_time())
        # 2a. draw from the correct distributions. (For naming see opendss admin)
        pdfGen = np.zeros((len(genNames),nMc)) # ASK CONSTANCE if this is cheating?
        for j in range(len(genNames)):
            pdfGen[j] = np.random.gamma(k,1e-3*Mu0[j]/k,nMc)
        DvOutLin = (KtotPu.dot(pdfGen).T)*1e3
        
        for jj in range(pdfData['nP'][-1]):
            genTot = np.sum(pdfGen*pdfData['mu_k'][jj],axis=0)
            if mcDssOn:
                # Mns = Mu0*pdfData['mu_k'][jj] # NB: we only scale by the FIRST ONE here
                vOut = np.zeros((nMc,len(v_idx)))
                conv = []
                for j in range(nMc):
                    if j%(nMc//4)==0:
                        print(j,'/',nMc)
                    set_generators( DSSCircuit,genNames,pdfGen[:,j]*pdfData['mu_k'][jj] )
                    DSSSolution.Solve()
                    conv = conv+[DSSSolution.Converged]
                    v00 = abs(tp_2_ar(DSSCircuit.YNodeVarray))
                    vOut[j,:] = v00[3:][v_idx]/vBase
                if sum(conv)!=len(conv):
                    print('\nNo. Converged:',sum(conv),'/',nMc)
                
                # NOW: calculate the HC value:
                
                dsf.mcErrorAnalysis(vOut,Vmax)
                maxV = np.max(vOut,axis=1)
                Vp_pct_dss[i,jj] = 100*(sum(maxV>Vmax)/nMc)
                hcGen = genTot[maxV>Vmax]
                hc_dss[i,jj] = min( np.concatenate((hcGen,np.array([np.inf]))) )
            
            if mcLinOn:
                vOutLin = (DvOutLin*pdfData['mu_k'][jj]) + b0
                maxVlin = np.max(vOutLin,axis=1)
                
                Vp_pct_lin[i,jj] = 100*(sum(maxVlin>Vmax)/nMc)
                hcGenLin = genTot[maxVlin>Vmax]
                hc_lin[i,jj] = min( np.concatenate((hcGenLin,np.array([np.inf]))) )
            

            if len(hcGen)!=0 and mcDssOn:
                hcGen.sort()
                hcGenSet[i,jj,0] = hcGen[np.floor(len(hcGen)*1.0/20.0).astype(int)]
                hcGenSet[i,jj,1] = hcGen[np.floor(len(hcGen)*1.0/4.0).astype(int)]
                hcGenSet[i,jj,2] = hcGen[np.floor(len(hcGen)*1.0/2.0).astype(int)]
                hcGenSet[i,jj,3] = hcGen[np.floor(len(hcGen)*3.0/4.0).astype(int)]
                hcGenSet[i,jj,4] = hcGen[np.floor(len(hcGen)*19.0/20.0).astype(int)]
            if len(hcGenLin)!=0 and mcLinOn:
                hcGenLin.sort()
                hcGenSetLin[i,jj,0] = hcGenLin[np.floor(len(hcGenLin)*1.0/20.0).astype(int)]
                hcGenSetLin[i,jj,1] = hcGenLin[np.floor(len(hcGenLin)*1.0/4.0).astype(int)]
                hcGenSetLin[i,jj,2] = hcGenLin[np.floor(len(hcGenLin)*1.0/2.0).astype(int)]
                hcGenSetLin[i,jj,3] = hcGenLin[np.floor(len(hcGenLin)*3.0/4.0).astype(int)]
                hcGenSetLin[i,jj,4] = hcGenLin[np.floor(len(hcGenLin)*19.0/20.0).astype(int)]
        print('MC complete.',time.process_time())

# alyMinHc = min(Vmax*(KkGen)/(Kk)) # NOT COMPLETELY clear why this isn't working.
# print('Min HC, Pred:',alyMinHc/1e3)
# print('Min HC, Act:',np.nanmin(hc_aly))


# ii = 23
# hist0 = plt.hist(vOutLin[:,ii],bins=100,density=True)[1]
# # plt.hist(vOut[:,ii],bins=100,density=True)
# print(b0[ii])
# plt.plot(Vpu[ii],pdfV[ii]/dsf.get_dx(Vpu[ii]))
# ylm = plt.ylim()
# plt.plot([b0[ii],b0[ii]],ylm,'g--')
# plt.plot([Vmax,Vmax],ylm,'r')
# plt.ylim(ylm)
# plt.show()




# plt.hist(maxVlin,bins=100)
# plt.hist(maxV,bins=100)
# plt.show()

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
    hist = plt.hist(genTot,bins=30,density=True);
    hist = plt.hist(genTot,bins=30,density=True);
    plt.close()

    dhist = dsf.get_dx(hist[1])

    histx = hist[1][:-1] + 0.5*dhist

    pdfY = hist[0]*dhist
    dx0 = dsf.get_dx(Gpu*1e-3)
    ratio = dx0/dhist

    plt.plot(Gpu*1e-3,pdfG);
    plt.plot(histx,pdfY*ratio);

    plt.xlabel('Total Installed Power (kW)')
    plt.ylabel('Probability'); plt.grid(True)
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
        for critBus in cBs:
            plt.plot([critBus]*2,ylm,'g',zorder=-1e3)
        plt.ylim(ylm)
    else:
        plt.grid(True)
    
    if pltSave:
        plt.savefig(sn0+'pltBox.png')
    else:
        plt.show()

if pltBoxBoth:
    Vmn = np.zeros(len(Vpu))
    Vlo = np.zeros(len(Vpu))
    Vmd = np.zeros(len(Vpu))
    Vhi = np.zeros(len(Vpu))
    Vmx = np.zeros(len(Vpu))

    emn = 0.01
    elo = 0.25
    emd = 0.50
    ehi = 0.75
    emx = 0.99
    
    plt.figure(figsize=(9,4))
    
    plt.subplot(122)
    for i in range(len(Vpu)):
        Vmn[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - emn))]
        Vlo[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - elo))]
        Vmd[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - emd))]
        Vhi[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - ehi))]
        Vmx[i]=Vpu[i,np.argmin(abs(cdfV[:,i] - emx))]
        plt.plot(i,Vmn[i],'k^'); 
        plt.plot(i,Vlo[i],'g_'); plt.plot(i,Vmd[i],'b_'); plt.plot(i,Vhi[i],'g_');
        plt.plot(i,Vmx[i],'kv'); 
        plt.plot([i,i],[Vmn[i],Vmx[i]],'k:')
        plt.plot(i,b0[cBs[i]],'rx')

    plt.xlabel('Bus No.')
    plt.ylabel('Voltage (pu)')
    xlm = plt.xlim()
    plt.plot(xlm,[Vmax,Vmax],'r--')
    plt.plot(xlm,[Vmin,Vmin],'r--')
    plt.xlim(xlm)
    if pltCritBus:
        ylm = plt.ylim()
        for critBus in cBs:
            plt.plot([critBus]*2,ylm,'g',zorder=-1e3)
        plt.ylim(ylm)
    else:
        plt.grid(True)

    
    plt.title('Linear Model')
    plt.subplot(121)
    
    Vmn = np.percentile(vOut,1,axis=0)
    Vlo = np.percentile(vOut,25,axis=0)
    Vmd = np.percentile(vOut,50,axis=0)
    Vhi = np.percentile(vOut,75,axis=0)
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
    if pltCritBus:
        ylm = plt.ylim()
        for critBus in cBs:
            plt.plot([critBus]*2,ylm,'g',zorder=-1e3)
        plt.ylim(ylm)
    else:
        plt.grid(True)

    
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

if pltHcBoth:
    plt.subplot(121)
    for i in range(pdfData['nP'][0]):
        plt.plot(pdfData['mu_k'],Vp_pct_dss[i],'ro-')
        plt.plot(pdfData['mu_k'],Vp_pct_aly[i],'bx-')
        plt.plot(pdfData['mu_k'],Vp_pct_lin[i],'g.-')
        

    plt.xlabel('Scale factor');
    plt.title('Prob. of overvoltage');
    plt.grid(True)
    plt.subplot(122)
    for i in range(pdfData['nP'][0]):
        # plt.plot(pdfData['mu_k'],Vp_pct_lin[i]/Vp_pct_aly[i],'g.-')
        # plt.plot(pdfData['mu_k'],hcGenSet[i,:,0],'ro')
        # plt.plot(pdfData['mu_k'],hc_dss[i],'ro-')
        # plt.plot(pdfData['mu_k'],hc_aly[i],'bx-')
        
        plt.plot(pdfData['mu_k'],hcGenSet[i,:,0],'r^'); 
        # plt.plot(pdfData['mu_k'],hcGenSet[i,:,1],'g_'); 
        plt.plot(pdfData['mu_k'],hcGenSet[i,:,2],'r_'); 
        # plt.plot(pdfData['mu_k'],hcGenSet[i,:,3],'g_');
        plt.plot(pdfData['mu_k'],hcGenSet[i,:,4],'rv');        
        
        plt.plot(pdfData['mu_k'],hcGenSetLin[i,:,0],'g^'); 
        # plt.plot(pdfData['mu_k'],hcGenSetLin[i,:,1],'k_'); 
        plt.plot(pdfData['mu_k'],hcGenSetLin[i,:,2],'g_'); 
        # plt.plot(pdfData['mu_k'],hcGenSetLin[i,:,3],'k_');
        plt.plot(pdfData['mu_k'],hcGenSetLin[i,:,4],'gv');
    xlm = plt.xlim()
    plt.xlim((-dsf.get_dx(pdfData['mu_k']),xlm[1]))
    plt.xlabel('Scale factor');
    plt.title('Hosting Capacity (kW)');

    ylm = plt.ylim()
    plt.ylim((0,ylm[1]))
    plt.grid(True)
    plt.show()


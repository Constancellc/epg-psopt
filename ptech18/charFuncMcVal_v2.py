# based on the script from 21/1, 'fft_calcs' in matlab.
# Based on script charFuncHcAlys, deleted 30/01

# STEPS:
# Part A: Linear method
# 1. Load linear model.
# 2. Choose distributions at each bus
# 3. Calculate distribution
# 4. Run MC analysis using linear model
#
# Part B: Run MC analysis using OpenDSS
# 1. load appropriate model
# 2. sample distibution appropriately and run load flow
# 3. Compare results.

import pickle, os, sys, win32com.client, time, scipy.stats
import numpy as np
from dss_python_funcs import *
import numpy.random as rnd
import matplotlib.pyplot as plt
from math import gamma
import dss_stats_funcs as dsf
from sklearn.decomposition import TruncatedSVD

WD = os.path.dirname(sys.argv[0])

# CHOOSE Network
fdr_i = 5
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr','123busCvr']
feeder = fdrs[fdr_i]
# feeder = '213'

netModel=0 # none
netModel=1 # ltc
# netModel=2 # fixed

# lin_point=0.6
lin_point=False
lp_taps='Lpt'

with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','chooseLinPoint.pkl'),'rb') as handle:
    lp0data = pickle.load(handle)

load_point = lp0data['kLo']
ld2mean = load_point - lp0data['k']

Vmax = lp0data['Vp']
Vmin = lp0data['Vm']

# ld2mean = 0.5 # ie the mean of those generators which install is 1/2 of their load

nMc = int(1e4)
nMc = int(3e3)
nMc = int(1e3)
nMc = int(3e2)
# nMc = int(1e2)

# PDF options
mu_k0 = np.arange(0.5,6.0,0.5) # NB this is as a PERCENTAGE of the chosen nominal powers. 
mu_k0 = 0.1*np.arange(0.5,6.0,0.5) # NB this is as a PERCENTAGE of the chosen nominal powers. 
mu_k0 = 0.3*np.arange(0.5,6.0,0.5) # NB this is as a PERCENTAGE of the chosen nominal powers.
mu_k0 = 0.4*np.arange(0.5,6.0,0.5) # NB this is as a PERCENTAGE of the chosen nominal powers.
mu_k0 = 1.0*np.arange(0.05,6.0,0.25) # NB this is as a PERCENTAGE of the chosen nominal powers.
# mu_k0 = 1.0*np.arange(0.05,6.0,0.03) # NB this is as a PERCENTAGE of the chosen nominal powers.
# mu_k0 = 1.2*np.arange(0.05,6.0,0.03) # NB this is as a PERCENTAGE of the chosen nominal powers.
# mu_k0 = 0.6*np.arange(0.05,6.0,0.03) # NB this is as a PERCENTAGE of the chosen nominal powers
# mu_k0 = 0.6*np.arange(0.5,6.0,0.5) # NB this is as a PERCENTAGE of the chosen nominal powers.

# mu_k0 = 5.5*np.array([1.0]) # NB this is as a PERCENTAGE of the chosen nominal powers. 

# mu_kk = getMu_Kk(feeder,netModel)
mu_kk = 1
mu_k = mu_k0*mu_kk
pdfName = 'gamma'
k = np.array([0.5]) # we do not know th, sigma until we know the scaling from mu0.
k = np.array([3.0]) # we do not know th, sigma until we know the scaling from mu0.
# k = np.array([10.0]) # we do not know th, sigma until we know the scaling from mu0.

params = k
pdfData = {'name':pdfName,'prms':params,'mu_k':mu_k,'nP':(len(params),len(mu_k))}

mcLinOn = True
# mcLinOn = False
mcDssOn = True
mcDssOn = False
useCbs = True
useCbs = False

evSvdLim = 0.999
evSvdLim = 0.995
# evSvdLim = 0.98
# evSvdLim = 0.90
nSvdMax = 300
nSvdMax = 100
nSvdMax = 30
nSvdMax = 16


# PLOTTING options:
pltBoxDss = True
pltBoxDss = False
pltHcBoth = True
# pltHcBoth = False
pltHcGen = True
pltHcGen = False
pltPwrCdf = True
pltPwrCdf = False

pltSave = True
pltSave = False
# ADMIN =============================================
if not lin_point:
    lin_point=lp0data['k']

ckt = get_ckt(WD,feeder)
fn_ckt = ckt[0]
fn = ckt[1]

DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
DSSText = DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution = DSSCircuit.Solution

svd = TruncatedSVD(n_components=nSvdMax,algorithm='arpack') # see: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD

print('Start. Feeder:',feeder,' Linpoint:',lin_point,' Load Point:',load_point,' Tap Model:',netModel,' Vmax:',Vmax)

# PART A.1 - load model ===========================
if not netModel:
    # IF using the FIXED model:
    LM = loadLinMagModel(feeder,lin_point,WD,'Lpt')
    Ky=LM['Ky'];Kd=LM['Kd'];bV=LM['bV'];xhy0=LM['xhy0'];xhd0=LM['xhd0']
    vBase = LM['vKvbase']

    xhyN = xhy0/lin_point # needed seperately for lower down
    xhdN = xhd0/lin_point
    # xNom = np.concatenate((xhyN,xhdN))
    
    b0 = (Ky.dot(xhyN*load_point) + Kd.dot(xhdN*load_point) + bV)/vBase # in pu
    # b0 = (Ky.dot(xhy0) + Kd.dot(xhd0) + bV)/vBase # in pu

    KyP = Ky[:,:Ky.shape[1]//2]
    KdP = Kd[:,:Kd.shape[1]//2]
    Ktot = np.concatenate((KyP,KdP),axis=1)
elif netModel>0:
    # IF using the LTC model:
    LM = loadNetModel(feeder,lin_point,WD,'Lpt',netModel)
    A=LM['A'];bV=LM['B'];xhy0=LM['xhy0'];xhd0=LM['xhd0']; 
    vBase = LM['Vbase']
    
    xhyN = xhy0/lin_point # needed seperately for lower down
    xhdN = xhd0/lin_point
    xNom = np.concatenate((xhyN,xhdN))
    b0 = (A.dot(xNom*load_point) + bV)/vBase # in pu
    
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
DSSText.Command='Compile ('+fn+'.dss)'
BB,SS = cpf_get_loads(DSSCircuit)
if lp_taps=='Lpt':
    cpf_set_loads(DSSCircuit,BB,SS,load_point)
    DSSSolution.Solve()

if not netModel:
    DSSText.Command='set controlmode=off'
elif netModel:
    DSSText.Command='set maxcontroliter=300'
DSSText.Command='set maxiterations=100'

YNodeVnom = tp_2_ar(DSSCircuit.YNodeVarray)
YZ = DSSCircuit.YNodeOrder
YZ = vecSlc(YZ[3:],v_idx)

# 2. run MC analysis, going through each generator and setting to a power.
genNamesY = add_generators(DSSObj,YZp,False)
genNamesD = add_generators(DSSObj,YZd,True)
DSSSolution.Solve()

genNames = genNamesY+genNamesD

Vp_pct_dss = np.zeros(pdfData['nP'])
Vp_pct_lin = np.zeros(pdfData['nP'])
Vp_pct_linU = np.zeros(pdfData['nP'])
Vp_pct_linS = np.zeros(pdfData['nP'])

hc_dss = np.zeros(pdfData['nP'])
hc_lin = np.zeros(pdfData['nP'])
hcGenSet = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],5))
hcGenSetLin = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],5))
genTotSet = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],5))
hcGenLinAll = np.array([]); hcGenAll = np.array([])
hcGen = []; hcGenLin=[]
# PART A.2 - choose distributions and reduce linear model ===========================
for i in range(pdfData['nP'][0]):
    roundI = 1e0
    Mu0_y = -ld2mean*roundI*np.round(xhyN[:xhyN.shape[0]//2]/roundI  - 1e6*np.finfo(np.float64).eps) # latter required to make sure that this is negative
    Mu0_d = -ld2mean*roundI*np.round(xhdN[:xhdN.shape[0]//2]/roundI - 1e6*np.finfo(np.float64).eps)
    Mu0 = np.concatenate((Mu0_y,Mu0_d))
    
    if pdfData['name']=='gamma': # NB: mean of gamma distribution is k*th; variance is k*(th**2)
        k = pdfData['prms'][i]
        sgm = Mu0/np.sqrt(pdfData['prms'][i])
    
    KtotPu0 = dsf.mvM(KtotPu,sgm) # scale the matrices to unit input variance
    
    # dVvarPu = Vmax - (b0 + KtotPu.dot(Mu0)) # nb this changes with Mu_k = ... !?
    dVvarPu = Vmax - b0 # nb this changes with Mu_k = ... !?
    KtotU = dsf.vmM(1/dVvarPu,KtotPu0) # scale the matrices to unity output
    
    Us,Ss,Vhs,evS = dsf.trunc_svd(svd,KtotU)
    nSvd = np.argmax(evS>evSvdLim)
    print('Number of Components:',nSvd)
    print('Computational effort saved (%):',100*(1-(nSvd*(KtotU.shape[0] + KtotU.shape[1])/(KtotU.shape[0]*KtotU.shape[1]) )))
    UsSvd = Us[:,:nSvd]
    
    KtotSvd = UsSvd.T.dot(KtotU)
    
    # print('Number of SVD components:',nSvd)    
    
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
        # cBs = dsf.getCritBuses(b0,Vmax,Mm,Kk,Scl=np.arange(0.1,3.10,0.1)/ld2mean) # critical buses
        cBs = np.argmax(abs(Usvd[0:nSvd]),axis=1)
    else:
        cBs = np.arange(0,len(b0)) # all buses
    
    KgSgm = np.sqrt(np.sum(abs(Kg),axis=1)) # useful for normal approximations
    MmGen = Kgen.dot(Mu0)
    KkGen = np.sqrt(np.sum(abs(Kg),axis=1))*KgenMax
    
    # PART B FROM HERE ==============================
    if mcLinOn or mcDssOn:
        print('---- Start MC ----',time.process_time())

        pdfGen0 = (np.random.gamma(k,1/np.sqrt(k),(len(genNames),nMc)))
        pdfGen = dsf.vmM(1e-3*Mu0/np.sqrt(k),pdfGen0) # scale
        pdfGenSh = pdfGen0 - np.sqrt(k) # shift to zero mean
        genTot0 = np.sum(pdfGen,axis=0)
        genTotSort = genTot0.copy()
        genTotSort.sort()
        genTotAll = np.outer(genTot0,pdfData['mu_k'])
        genTotAll = genTotAll.flatten()
        
        DvOutLin = (KtotPu.dot(pdfGen).T)*1e3
        
        DvOutLinK = (KtotU.dot(pdfGenSh).T)
        DvMu0 = KtotU.dot(np.sqrt(k)*np.ones(len(Mu0)))
        
        vOutSvd = UsSvd.dot((KtotSvd.dot(pdfGenSh))).T
        
        for jj in range(pdfData['nP'][-1]):
            # genTot = np.sum(pdfGen*pdfData['mu_k'][jj],axis=0)
            genTot = genTot0*pdfData['mu_k'][jj]
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
                minV = np.min(vOut,axis=1)
                
                Vp_pct_dss[i,jj] = 100*(sum(maxV>Vmax)/nMc)
                hcGen = genTot[maxV>Vmax]
                hc_dss[i,jj] = min( np.concatenate((hcGen,np.array([np.inf]))) )
            
            if mcLinOn:
                vOutLin = (DvOutLin*pdfData['mu_k'][jj]) + b0
                maxVlin = np.max(vOutLin,axis=1)
                minVlin = np.min(vOutLin,axis=1)
                
                vOutULin = (DvOutLinK + DvMu0)*pdfData['mu_k'][jj]
                maxVlinU = np.max(vOutULin,axis=1)
                minVlinU = np.min(vOutULin,axis=1)
                
                vOutSLin = (vOutSvd + DvMu0)*pdfData['mu_k'][jj]
                maxVlinS = np.max(vOutSLin,axis=1)
                minVlinS = np.min(vOutSLin,axis=1)
                
                print('Overvoltage:',100*(sum(maxVlin>Vmax)/nMc))
                print('V. Violation:',100*sum(np.any(np.array([maxVlin>Vmax,minVlin<Vmin]),axis=0))/nMc)
                
                Vp_pct_lin[i,jj] = 100*sum(np.any(np.array([maxVlin>Vmax,minVlin<Vmin]),axis=0))/nMc
                Vp_pct_linU[i,jj] = 100*(sum(maxVlinU>1)/nMc)
                Vp_pct_linS[i,jj] = 100*(sum(maxVlinS>1)/nMc)
                
                # Vp_pct_lin[i,jj] = 100*(sum(maxVlin>Vmax)/nMc)
                # Vp_pct_linU[i,jj] = 100*(sum(maxVlinU>1)/nMc)
                # Vp_pct_linS[i,jj] = 100*(sum(maxVlinS>1)/nMc)
                
                
                hcGenLin = genTot[maxVlin>Vmax]
                hc_lin[i,jj] = min( np.concatenate((hcGenLin,np.array([np.inf]))) )
                

            if len(hcGen)!=0 and mcDssOn:
                hcGenAll = np.concatenate((hcGenAll,hcGen))
                hcGen.sort()
                # hcGenSet[i,jj,0] = hcGen[np.floor(len(hcGen)*1.0/20.0).astype(int)]
                hcGenSet[i,jj,0] = hcGen[0]
                hcGenSet[i,jj,1] = hcGen[np.floor(len(hcGen)*1.0/4.0).astype(int)]
                hcGenSet[i,jj,2] = hcGen[np.floor(len(hcGen)*1.0/2.0).astype(int)]
                hcGenSet[i,jj,3] = hcGen[np.floor(len(hcGen)*3.0/4.0).astype(int)]
                # hcGenSet[i,jj,4] = hcGen[np.floor(len(hcGen)*19.0/20.0).astype(int)]
                hcGenSet[i,jj,4] = hcGen[-1]
            if len(hcGenLin)!=0 and mcLinOn:
                hcGenLinAll = np.concatenate((hcGenLinAll,hcGenLin))
                hcGenLin.sort()
                hcGenSetLin[i,jj,0] = hcGenLin[0]
                # hcGenSetLin[i,jj,0] = hcGenLin[np.floor(len(hcGenLin)*1.0/20.0).astype(int)]
                hcGenSetLin[i,jj,1] = hcGenLin[np.floor(len(hcGenLin)*1.0/4.0).astype(int)]
                hcGenSetLin[i,jj,2] = hcGenLin[np.floor(len(hcGenLin)*1.0/2.0).astype(int)]
                hcGenSetLin[i,jj,3] = hcGenLin[np.floor(len(hcGenLin)*3.0/4.0).astype(int)]
                hcGenSetLin[i,jj,4] = hcGenLin[-1]
                # hcGenSetLin[i,jj,4] = hcGenLin[np.floor(len(hcGenLin)*19.0/20.0).astype(int)]
            
            genTotSet[i,jj,0] = genTotSort[0]*pdfData['mu_k'][jj]
            genTotSet[i,jj,1] = genTotSort[np.floor(len(genTotSort)*1.0/4.0).astype(int)]*pdfData['mu_k'][jj]
            genTotSet[i,jj,2] = genTotSort[np.floor(len(genTotSort)*1.0/2.0).astype(int)]*pdfData['mu_k'][jj]
            genTotSet[i,jj,3] = genTotSort[np.floor(len(genTotSort)*3.0/4.0).astype(int)]*pdfData['mu_k'][jj]
            genTotSet[i,jj,4] = genTotSort[-1]*pdfData['mu_k'][jj]
        print('MC complete.',time.process_time())


if pltPwrCdf:
    hist1 = plt.hist(genTotAll,bins=100,range=(0,max(genTotAll)))
    if mcDssOn:
        hist2 = plt.hist(hcGenAll,bins=100,range=(0,max(genTotAll)))
    hist2lin = plt.hist(hcGenLinAll,bins=100,range=(0,max(genTotAll)))
    plt.close()

    plt.plot(hist1[1][1:],hist2lin[0]/hist1[0])
    if mcDssOn:
        plt.plot(hist1[1][1:],hist2[0]/hist1[0])
    plt.label(('Lin model','OpenDSS'))
    plt.xlabel('Power');
    plt.ylabel('P(.)');
    plt.grid(True)
    plt.show()


# # COMPARE RESULTS ==========
# if pltBoxDss:
    # plt.boxplot(vOut,whis=[1,99])
    # plt.plot(range(1,len(vBase)+1),abs(YNodeVnom[3:])[v_idx]/vBase,'rx')
    # plt.xlabel('Bus no.')
    # plt.ylabel('Voltage (pu)')
    # xlm = plt.xlim()
    # plt.plot(xlm,[Vmax,Vmax],'r--')
    # plt.plot(xlm,[Vmin,Vmin],'r--')
    # plt.xlim(xlm)
    # plt.grid(True)
    # if pltSave:
        # plt.savefig(sn0+'pltBoxDss.png')
        # plt.close()
    # else:
        # plt.show()

# ================ PLOTTING FUNCTIONS FROM HERE
if pltHcBoth:
    plt.subplot(121)
    for i in range(pdfData['nP'][0]):
        # plt.plot(pdfData['mu_k'],Vp_pct_dss[i],'ro-')
        # plt.plot(pdfData['mu_k'],Vp_pct_lin[i],'g.-')
        # plt.ylim((0,20))
        if mcDssOn:
            plt.semilogy(pdfData['mu_k'],Vp_pct_dss[i],'ro-')
        plt.semilogy(pdfData['mu_k'],Vp_pct_linU[i],'rx-')
        plt.semilogy(pdfData['mu_k'],Vp_pct_linS[i],'b.-')
        plt.semilogy(pdfData['mu_k'],Vp_pct_lin[i],'g.-')
        plt.legend(('VpDss','VpNom','VpSvd'))
        # plt.ylim((1e-3,50))
        

    plt.xlabel('Scale factor');
    plt.title('Prob. of overvoltage (logscale)');
    plt.grid(True)
    plt.subplot(122)
    
    if mcDssOn:
        plt.plot(pdfData['mu_k'],Vp_pct_dss[i],'ro-')
    plt.plot(pdfData['mu_k'],Vp_pct_linU[i],'rx-')
    plt.plot(pdfData['mu_k'],Vp_pct_linS[i],'b.-')
    plt.plot(pdfData['mu_k'],Vp_pct_lin[i],'g.-')
    plt.title('Prob. of overvoltage');
    
    plt.grid(True)
    plt.show()
    
if pltHcGen:
    for i in range(pdfData['nP'][0]):
        # plt.plot(pdfData['mu_k'],hcGenSet[i,:,0],'ro')
        # plt.plot(pdfData['mu_k'],hc_dss[i],'ro-')
        
        plt.plot(pdfData['mu_k'],genTotSet[i,:,0],'b-'); 
        plt.plot(pdfData['mu_k'],genTotSet[i,:,2],'b_'); 
        plt.plot(pdfData['mu_k'],genTotSet[i,:,4],'b-');
        if mcDssOn:
            plt.plot(pdfData['mu_k'],hcGenSet[i,:,0],'r-'); 
            plt.plot(pdfData['mu_k'],hcGenSet[i,:,2],'r_'); 
            plt.plot(pdfData['mu_k'],hcGenSet[i,:,4],'r-');        
        if mcLinOn:
            plt.plot(pdfData['mu_k'],hcGenSetLin[i,:,0],'g-'); 
            plt.plot(pdfData['mu_k'],hcGenSetLin[i,:,2],'g_'); 
            plt.plot(pdfData['mu_k'],hcGenSetLin[i,:,4],'g-');

            
    xlm = plt.xlim()
    plt.xlim((-dsf.get_dx(pdfData['mu_k']),xlm[1]))
    plt.xlabel('Scale factor');
    plt.title('Hosting Capacity (kW)');

    # ylm = plt.ylim()
    # plt.ylim((0,ylm[1]))
    plt.grid(True)
    plt.show()


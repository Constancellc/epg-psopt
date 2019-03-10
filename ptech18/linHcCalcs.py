# Based on the previous script charFuncMcVal_v2

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
fdr_i = 18
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']
feeder = fdrs[fdr_i]
# feeder = '213'

netModel=0 # none
# netModel=1 # ltc
# netModel=2 # fixed

lp_taps='Lpt'

with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','chooseLinPoint.pkl'),'rb') as handle:
    lp0data = pickle.load(handle)
load_point = lp0data['kLo']
ld2mean = load_point - lp0data['k']
Vmax = lp0data['Vp']
Vmin = lp0data['Vm']
lin_point=lp0data['k']

nMc = int(1e4)
nMc = int(3e3)
nMc = int(1e3)
nMc = int(3e2)
nMc = int(1e2)
nMc = int(3e1)

if netModel==0:
    circuitK = {'13bus':0.8,'34bus':0.9,'8500node':0.2,'epri5':0.4,'epri7':0.15,'epriJ1':0.2,'epriK1':0.2,'epriM1':0.25,'epri24':0.25}
elif netModel==1:
    circuitK = {'13bus':1.2,'34bus':0.9,'123bus':0.6}
elif netModel==2:
    circuitK = {'8500node':0.2,'epriJ1':0.6,'epriK1':0.2,'epriM1':0.3,'epri24':0.25}
    
# PDF options
dMu = 0.01
# dMu = 0.025
mu_k = circuitK[feeder]*6.0*np.arange(dMu,1.0,dMu) # NB this is as a PERCENTAGE of the chosen nominal powers.

pdfName = 'gamma'
k = np.array([0.5]) # we do not know th, sigma until we know the scaling from mu0.
k = np.array([3.0]) # we do not know th, sigma until we know the scaling from mu0.
# k = np.array([10.0]) # we do not know th, sigma until we know the scaling from mu0.

params = k
pdfData = {'name':pdfName,'prms':params,'mu_k':mu_k,'nP':(len(params),len(mu_k))}

mcLinOn = True
# mcLinOn = False
mcDssOn = True
# mcDssOn = False

# PLOTTING options:
pltHcBoth = True
# pltHcBoth = False
pltHcGen = True
# pltHcGen = False
pltPwrCdf = True
# pltPwrCdf = False

pltBoxDss = True
pltBoxDss = False

pltSave = True
pltSave = False
# ADMIN =============================================
SD = os.path.join(WD,'hcResults',feeder)
SN = os.path.join(SD,'linHcCalcsRslt.pkl')
# sn0 = SD + str(int(lin_point*1e2)) + 'net' + str(int(netModel)) + 'ld' + str(int(ld2mean*1e2))

ckt = get_ckt(WD,feeder)
fn_ckt = ckt[0]
fn = ckt[1]

DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
DSSText = DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution = DSSCircuit.Solution

print('Start. \nFeeder:',feeder,'\nLinpoint:',lin_point,'\nLoad Point:',load_point,'\nTap Model:',netModel)

# PART A.1 - load model ===========================
if not netModel:
    # IF using the FIXED model:
    LM = loadLinMagModel(feeder,lin_point,WD,'Lpt')
    Ky=LM['Ky'];Kd=LM['Kd'];bV=LM['bV'];xhy0=LM['xhy0'];xhd0=LM['xhd0']
    vBase = LM['vKvbase']

    xhyN = xhy0/lin_point # needed seperately for lower down
    xhdN = xhd0/lin_point
    
    b0 = (Ky.dot(xhyN*load_point) + Kd.dot(xhdN*load_point) + bV)/vBase # in pu

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
DSSText.command='Compile ('+fn+'.dss)'
BB0,SS0 = cpf_get_loads(DSSCircuit)

cpf_set_loads(DSSCircuit,BB0,SS0,load_point)
DSSSolution.Solve()

if not netModel:
    DSSText.command='set controlmode=off'
elif netModel:
    DSSText.command='set maxcontroliter=300'
DSSText.command='set maxiterations=100'

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
    
    # PART B FROM HERE ==============================
    print('---- Start MC ----',time.process_time())

    pdfGen0 = (np.random.gamma(k,1/np.sqrt(k),(len(genNames),nMc)))
    
    pdfGen = dsf.vmM(1e-3*Mu0/np.sqrt(k),pdfGen0) # scale
    genTot0 = np.sum(pdfGen,axis=0)
    
    genTotSort = genTot0.copy()
    genTotSort.sort()
    genTotAll = np.outer(genTot0,pdfData['mu_k'])
    genTotAll = genTotAll.flatten()
    
    DvOutLin = (KtotPu.dot(pdfGen).T)*1e3
    
    for jj in range(pdfData['nP'][-1]):
        genTot = genTot0*pdfData['mu_k'][jj]
        
        if mcDssOn:
            vOut = np.zeros((nMc,len(v_idx)))
            conv = []
            for j in range(nMc):
                if j%(nMc//4)==0:
                    print(j,'/',nMc)
                set_generators( DSSCircuit,genNames,pdfGen[:,j]*pdfData['mu_k'][jj] )
                
                DSSText.command='Batchedit load..* vmin=0.33 vmax=3.0 model=1'
                DSSText.command='Batchedit generator..* vmin=0.33 vmax=3.0'
                DSSText.command='Batchedit regcontrol..* band=1.0' # seems to be as low as we can set without bit problems
                
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
            
            inBounds = np.any(np.array([maxV>Vmax,minV<Vmin]),axis=0)
            Vp_pct_dss[i,jj] = 100*sum(inBounds)/nMc
            hcGen = genTot[inBounds]
        
        if mcLinOn:
            vOutLin = (DvOutLin*pdfData['mu_k'][jj]) + b0
            maxVlin = np.max(vOutLin,axis=1)
            minVlin = np.min(vOutLin,axis=1)
            
            # print('Overvoltage:',100*(sum(maxVlin>Vmax)/nMc))
            # print('V. Violation:',100*sum(np.any(np.array([maxVlin>Vmax,minVlin<Vmin]),axis=0))/nMc)
            
            inBoundsLin = np.any(np.array([maxVlin>Vmax,minVlin<Vmin]),axis=0)
            Vp_pct_lin[i,jj] = 100*sum(inBoundsLin)/nMc
            hcGenLin = genTot[inBoundsLin]

        if len(hcGen)!=0 and mcDssOn:
            hcGenAll = np.concatenate((hcGenAll,hcGen))
            hcGen.sort()
            hcGenSet[i,jj,0] = hcGen[0]
            hcGenSet[i,jj,1] = hcGen[np.floor(len(hcGen)*1.0/4.0).astype(int)]
            hcGenSet[i,jj,2] = hcGen[np.floor(len(hcGen)*1.0/2.0).astype(int)]
            hcGenSet[i,jj,3] = hcGen[np.floor(len(hcGen)*3.0/4.0).astype(int)]
            hcGenSet[i,jj,4] = hcGen[-1]
            
        if len(hcGenLin)!=0 and mcLinOn:
            hcGenLinAll = np.concatenate((hcGenLinAll,hcGenLin))
            hcGenLin.sort()
            hcGenSetLin[i,jj,0] = hcGenLin[0]
            hcGenSetLin[i,jj,1] = hcGenLin[np.floor(len(hcGenLin)*1.0/4.0).astype(int)]
            hcGenSetLin[i,jj,2] = hcGenLin[np.floor(len(hcGenLin)*1.0/2.0).astype(int)]
            hcGenSetLin[i,jj,3] = hcGenLin[np.floor(len(hcGenLin)*3.0/4.0).astype(int)]
            hcGenSetLin[i,jj,4] = hcGenLin[-1]
        
        genTotSet[i,jj,0] = genTotSort[0]*pdfData['mu_k'][jj]
        genTotSet[i,jj,1] = genTotSort[np.floor(len(genTotSort)*1.0/4.0).astype(int)]*pdfData['mu_k'][jj]
        genTotSet[i,jj,2] = genTotSort[np.floor(len(genTotSort)*1.0/2.0).astype(int)]*pdfData['mu_k'][jj]
        genTotSet[i,jj,3] = genTotSort[np.floor(len(genTotSort)*3.0/4.0).astype(int)]*pdfData['mu_k'][jj]
        genTotSet[i,jj,4] = genTotSort[-1]*pdfData['mu_k'][jj]
    print('MC complete.',time.process_time())


# NOW: calculate the statistics we want.
    
binNo = int(0.5//dMu)
# binNo = int(1.0//dMu)
hist1 = plt.hist(genTotAll,bins=binNo,range=(0,max(genTotAll)))
if mcDssOn:
    hist2 = plt.hist(hcGenAll,bins=binNo,range=(0,max(genTotAll)))
hist2lin = plt.hist(hcGenLinAll,bins=binNo,range=(0,max(genTotAll)))
plt.close()
    
pp = hist1[1][1:]
ppPdfLin = hist2lin[0]/hist1[0]
    
p0lin = pp[np.argmax(ppPdfLin!=0)]
p10lin = pp[np.argmax(ppPdfLin>=0.1)]
k0lin = pdfData['mu_k'][np.argmax(Vp_pct_lin!=0)]
k10lin = pdfData['mu_k'][np.argmax(Vp_pct_lin>=10.)]

print('\n--- Linear results ---\n\nP0:',p0lin,'\nP10:',p10lin,'\nk0:',k0lin,'\nk10:',k10lin)

if mcDssOn:
    ppPdf = hist2[0]/hist1[0]
    p0 = pp[np.argmax(ppPdf!=0)]
    p10 = pp[np.argmax(ppPdf>=0.1)]
    k0 = pdfData['mu_k'][np.argmax(Vp_pct_dss!=0)]
    k10 = pdfData['mu_k'][np.argmax(Vp_pct_dss>=10.)]
    print('\n--- OpenDSS results ---\n\nP0:',p0,'\nP10:',p10,'\nk0:',k0,'\nk10:',k10)
    
    if not os.path.exists(SD):
        os.makedirs(SD)
    rslt = {'p0lin':p0lin,'p10lin':p10lin,'k0lin':k0lin,'k10lin':k10lin,'p0':p0,'p10':p10,'k0':k0,'k10':k10,'netModel':netModel,'nMc':nMc,'dMu':dMu,'feeder':feeder}
    with open(SN,'wb') as file:
        pickle.dump([rslt],file)
    

# ================ PLOTTING FUNCTIONS FROM HERE
if pltPwrCdf:
    plt.plot(pp,ppPdfLin)
    if mcDssOn:
        plt.plot(pp,ppPdf)        
    plt.legend(('Lin model','OpenDSS'))
    plt.xlabel('Power');
    plt.ylabel('P(.)');
    plt.grid(True)
    plt.show()

if pltHcBoth:
    plt.subplot(121)
    for i in range(pdfData['nP'][0]):
        # plt.plot(pdfData['mu_k'],Vp_pct_dss[i],'ro-')
        # plt.plot(pdfData['mu_k'],Vp_pct_lin[i],'g.-')
        # plt.ylim((0,20))
        if mcDssOn:
            plt.semilogy(pdfData['mu_k'],Vp_pct_dss[i],'ro-')
        plt.semilogy(pdfData['mu_k'],Vp_pct_lin[i],'g.-')
        plt.legend(('VpDss','VpNom','VpSvd'))
        # plt.ylim((1e-3,50))
        

    plt.xlabel('Scale factor');
    plt.title('Prob. of overvoltage (logscale)');
    plt.grid(True)
    plt.subplot(122)
    
    if mcDssOn:
        plt.plot(pdfData['mu_k'],Vp_pct_dss[i],'ro-')
    plt.plot(pdfData['mu_k'],Vp_pct_lin[i],'g.-')
    plt.title('Prob. of overvoltage');
    
    plt.grid(True)
    plt.show()
    
if pltHcGen:
    for i in range(pdfData['nP'][0]):
        # plt.plot(pdfData['mu_k'],hcGenSet[i,:,0],'ro')
        
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


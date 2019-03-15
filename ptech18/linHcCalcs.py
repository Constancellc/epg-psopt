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

WD = os.path.dirname(sys.argv[0])

# CHOOSE Network
fdr_i = 5
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']
feeder = fdrs[fdr_i]
# feeder = '213'

netModel=0 # none
netModel=1 # ltc
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
    circuitK = {'13bus':4.8,'34bus':5.4,'123bus':3.0,'8500node':1.2,'epri5':2.4,'epri7':1.3,'epriJ1':1.2,'epriK1':1.2,'epriM1':1.5,'epri24':1.5}
elif netModel==1:
    circuitK = {'13bus':7.2,'34bus':5.4,'123bus':3.6}
elif netModel==2:
    circuitK = {'8500node':1.2,'epriJ1':3.6,'epriK1':1.5,'epriM1':1.8,'epri24':1.5}

# PDF options
dMu = 0.01
dMu = 0.025
mu_k = circuitK[feeder]*np.arange(dMu,1.0,dMu) # NB this is as a PERCENTAGE of the chosen nominal powers.

pdfName = 'gamma'
k = np.array([0.5]) # we do not know th, sigma until we know the scaling from mu0.
k = np.array([3.0]) # we do not know th, sigma until we know the scaling from mu0.
# k = np.array([20.0]) # we do not know th, sigma until we know the scaling from mu0.
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
pltCns = True
# pltCns = False

pltBoxDss = True
pltBoxDss = False

pltSave = True
pltSave = False

DVmax = 0.06 # percent
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

# PART A.1 - load models ===========================
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
    A=LM['A'];bV=LM['B'];xhy0=LM['xhy0'];xhd0=LM['xhd0']
    vBase = LM['Vbase']
    
    xhyN = xhy0/lin_point # needed seperately for lower down
    xhdN = xhd0/lin_point
    xNom = np.concatenate((xhyN,xhdN))
    b0 = (A.dot(xNom*load_point) + bV)/vBase # in pu
    
    KyP = A[:,0:len(xhy0)//2] # these might be zero if there is no injection (e.g. only Q)
    KdP = A[:,len(xhy0):len(xhy0) + (len(xhd0)//2)]
    
    Ktot = np.concatenate((KyP,KdP),axis=1)

KtotCheck = np.sum(Ktot==0,axis=1)!=Ktot.shape[1] # [can't remember what this is for...]
Ktot = Ktot[KtotCheck]
b0 = b0[KtotCheck]
vBase = vBase[KtotCheck]

v_idx=LM['v_idx'][KtotCheck]
YZp = LM['SyYNodeOrder']
YZd = LM['SdYNodeOrder']

# NOW load the fixed model for calculating voltage deviations
LMfix = loadLinMagModel(feeder,lin_point,WD,'Lpt')
Kyfix=LMfix['Ky'];Kdfix=LMfix['Kd']
dvBase = LMfix['vKvbase'] # NB: this is different to vBase for ltc/regulator models!

KyPfix = Kyfix[:,:Kyfix.shape[1]//2]
KdPfix = Kdfix[:,:Kdfix.shape[1]//2]
Kfix = np.concatenate((KyPfix,KdPfix),axis=1)
Kfix = Kfix[KtotCheck]


# REDUCE THE LINEAR MODEL to a nice form for multiplication
KtotPu = dsf.vmM(1/vBase,Ktot) # scale to be in pu
KfixPu = dsf.vmM(1/dvBase,Kfix)


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
    DSSObj.AllowForms=False
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
Cns_pct_dss = np.zeros(list(pdfData['nP'])+[3])
Vp_pct_lin = np.zeros(pdfData['nP'])
Cns_pct_lin = np.zeros(list(pdfData['nP'])+[3])

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
    
    Mu0[Mu0>(10*Mu0.mean())] = Mu0.mean()
    Mu0[Mu0>(10*Mu0.mean())] = Mu0.mean()
    
    if pdfData['name']=='gamma': # NB: mean of gamma distribution is k*th; variance is k*(th**2)
        k = pdfData['prms'][i]
    
    # PART B FROM HERE ==============================
    print('---- Start MC ----',time.process_time())

    pdfGen0 = (np.random.gamma(k,1/np.sqrt(k),(len(genNames),nMc)))
    
    pdfGen = dsf.vmM( 1e-3*Mu0/np.sqrt(k),pdfGen0 ) # scale into kW
    genTot0 = np.sum(pdfGen,axis=0)
    
    genTotSort = genTot0.copy()
    genTotSort.sort()
    genTotAll = np.outer(genTot0,pdfData['mu_k'])
    genTotAll = genTotAll.flatten()
    
    DelVoutLin = (KtotPu.dot(pdfGen).T)*1e3
    ddVoutLin = abs((KfixPu.dot(pdfGen).T)*1e3) # just get abs change
    
    
    for jj in range(pdfData['nP'][-1]):
        genTot = genTot0*pdfData['mu_k'][jj]
        
        if mcDssOn:
            vOut = np.zeros((nMc,len(v_idx)))
            dvOut = np.zeros((nMc,len(v_idx)))
            conv = []
            DVconv = []
            for j in range(nMc):
                if j%(nMc//4)==0:
                    print(j,'/',nMc)
                set_generators( DSSCircuit,genNames,pdfGen[:,j]*pdfData['mu_k'][jj] )
                
                DSSText.command='Batchedit load..* vmin=0.33 vmax=3.0 model=1'
                DSSText.command='Batchedit generator..* vmin=0.33 vmax=3.0'
                DSSText.command='Batchedit regcontrol..* band=1.0' # seems to be as low as we can set without bit problems
                
                DSSSolution.Solve()
                conv = conv+[DSSSolution.Converged]
                v00 = tp_2_ar(DSSCircuit.YNodeVarray)
                
                DSSText.command='Batchedit generator..* kW=0.001'
                DSSText.command='set controlmode=off'
                DSSSolution.Solve()

                DVconv = DVconv+[DSSSolution.Converged]
                DV00 = tp_2_ar(DSSCircuit.YNodeVarray)
                
                DSSText.command='set controlmode=static'
                
                vOut[j,:] = abs(v00)[3:][v_idx]/vBase
                dvOut[j,:] = abs(abs(v00) - abs(DV00))[3:][v_idx]/vBase
                dvOut[j,:] = abs(abs(v00) - abs(DV00))[3:][v_idx]/vBase
                
                # plt.plot(abs(v00)); plt.plot(abs(DV00)); plt.show()
                # plt.plot(abs(v00)[3:][v_idx]/vBase,'x-'); plt.plot(vOutLin[-1],'x-'); plt.show()
                # YZ
                
            vOut[vOut<0.5] = 1.0
            
            if sum(conv)!=len(conv):
                print('\nNo. Converged:',sum(conv),'/',nMc)
            
            # NOW: calculate the HC value:
            # dsf.mcErrorAnalysis(vOut,Vmax)
            maxV = np.max(vOut,axis=1)
            minV = np.min(vOut,axis=1)
            maxDv = np.max(dvOut,axis=1)
            
            
            Cns_pct_dss[i,jj] = 100*np.array([sum(maxDv>DVmax),sum(maxV>Vmax),sum(minV<Vmin)])/nMc
            # Vhi_pct_dss[i,jj] = 100*/nMc
            # Vlo_pct_dss[i,jj] = 100*/nMc
            
            inBounds = np.any(np.array([maxV>Vmax,minV<Vmin,maxDv>DVmax]),axis=0)

            Vp_pct_dss[i,jj] = 100*sum(inBounds)/nMc
            hcGen = genTot[inBounds]
        
        if mcLinOn:
            vOutLin = (DelVoutLin*pdfData['mu_k'][jj]) + b0
            DvOutLin = ddVoutLin*pdfData['mu_k'][jj]
            vOutLin[vOutLin<0.5] = 1.0
            maxVlin = np.max(vOutLin,axis=1)
            minVlin = np.min(vOutLin,axis=1)
            maxDvLin = np.max(DvOutLin,axis=1)
            
            # print('Overvoltage:',100*(sum(maxVlin>Vmax)/nMc))
            # print('V. Violation:',100*sum(np.any(np.array([maxVlin>Vmax,minVlin<Vmin]),axis=0))/nMc)
            
            Cns_pct_lin[i,jj] = 100*np.array([sum(maxDvLin>DVmax),sum(maxVlin>Vmax),sum(minVlin<Vmin)])/nMc
            inBoundsLin = np.any(np.array([maxVlin>Vmax,minVlin<Vmin,maxDvLin>DVmax]),axis=0)
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

# colors=[#1f77b4,#ff7f0e,#2ca02c]

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
    rslt = {'p0lin':p0lin,'p10lin':p10lin,'k0lin':k0lin,'k10lin':k10lin,'p0':p0,'p10':p10,'k0':k0,'k10':k10,'netModel':netModel,'nMc':nMc,'dMu':dMu,'feeder':feeder,'time2run':time.process_time()}
    with open(SN,'wb') as file:
        pickle.dump([rslt],file)
    

# ================ PLOTTING FUNCTIONS FROM HERE
if pltCns:
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=['red', 'blue', 'green'])
    plt.plot(pdfData['mu_k'],Cns_pct_dss[0]);
    plt.plot(pdfData['mu_k'],Cns_pct_lin[0],'--')
    plt.xlabel('Scale factor');
    plt.ylabel('P(.)');
    plt.title('Constraints')
    plt.legend(('Voltage deviation','Overvoltage','Undervoltage'))
    plt.show()


if pltPwrCdf:
    plt.plot(pp,ppPdfLin)
    if mcDssOn:
        plt.plot(pp,ppPdf)        
    plt.legend(('Lin model','OpenDSS'))
    plt.xlabel('Power');
    plt.ylabel('P(.)');
    plt.grid(True)
    if mcDssOn:
        plt.savefig(os.path.join(SD,'pltPwrCdf.png'))

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
    plt.title('Prob. of violation (logscale)');
    plt.grid(True)
    plt.ylabel('P(.), %')
    plt.subplot(122)
    
    if mcDssOn:
        plt.plot(pdfData['mu_k'],Vp_pct_dss[i],'ro-')
    plt.plot(pdfData['mu_k'],Vp_pct_lin[i],'g.-')
    plt.title('Prob. of violation');
    plt.xlabel('Scale factor');
    plt.ylabel('P(.), %')
    
    plt.grid(True)
    plt.tight_layout()
    if mcDssOn:
        plt.savefig(os.path.join(SD,'pltHcBoth.png'))
    plt.show()
    
if pltHcGen:
    # for i in range(pdfData['nP'][0]):
        # plt.plot(pdfData['mu_k'],hcGenSet[i,:,0],'ro')
    i=0
    plt.plot(pdfData['mu_k'],genTotSet[i,:,0],'b-'); 
    plt.plot(pdfData['mu_k'],genTotSet[i,:,2],'b_'); 
    plt.plot(pdfData['mu_k'],genTotSet[i,:,4],'b-');
    if mcDssOn:
        plt.plot(pdfData['mu_k'],hcGenSet[i,:,0],'r-'); 
        plt.plot(pdfData['mu_k'],hcGenSet[i,:,2],'r_'); 
        plt.plot(pdfData['mu_k'],hcGenSet[i,:,4],'r-');        

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
    if mcDssOn:
        plt.savefig(os.path.join(SD,'pltHcGen.png'))

    plt.show()


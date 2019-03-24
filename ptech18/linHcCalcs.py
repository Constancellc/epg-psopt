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
from linSvdCalcs import hcPdfs, linModel, cnsBdsCalc
from matplotlib import cm

WD = os.path.dirname(sys.argv[0])

mcLinOn = True
# mcLinOn = False
mcDssOn = True
# mcDssOn = False

# PLOTTING options:
pltHcBoth = True
pltHcBoth = False
pltHcGen = True
pltHcGen = False
pltPwrCdf = True
pltPwrCdf = False
pltCns = True
# pltCns = False

pltBoxDss = True
pltBoxDss = False

pltSave = True # for saving both plots and results
# pltSave = False

# CHOOSE Network
fdr_i_set = [5,6,8,9,0,14,17,18,22,19,20,21]
fdr_i_set = [5,6,8,0,14] # fastest few
fdr_i_set = [17,18] # medium length 1
fdr_i_set = [19,20,21] # medium length 2
fdr_i_set = [9] # slow
fdr_i_set = [22] # slowest

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']

# opendss with 'early bindings'
from win32com.client import makepy
sys.argv=["makepy","OpenDSSEngine.DSS"]
makepy.main()
DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")

DSSText = DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution = DSSCircuit.Solution
for fdr_i in fdr_i_set:
    feeder = fdrs[fdr_i]
    # feeder = '213'

    with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','chooseLinPoint.pkl'),'rb') as handle:
        lp0data = pickle.load(handle)
    loadPointLo = lp0data['kLo']
    loadPointHi = lp0data['kHi']

    lin_point=lp0data['k']

    nMc = int(1e4)
    nMc = int(3e3)
    nMc = int(1e3)
    nMc = int(3e2)
    nMc = int(1e2)
    nMc = int(3e1)

    LM = linModel(fdr_i,WD)
    netModel = LM.netModelNom

    pdf = hcPdfs(LM.feeder,netModel=netModel)

    k = pdf.pdf['prms']
    mu_k = pdf.pdf['mu_k']
    pdfData = pdf.pdf
    dMu = pdf.dMu


    DVmax = 0.06 # percent
    # ADMIN =============================================
    SD = os.path.join(WD,'hcResults',feeder)
    SN = os.path.join(SD,'linHcCalcsRslt.pkl')

    ckt = get_ckt(WD,feeder)
    fn = ckt[1]
    
    print('\n\nStart '+'='*30,'\nFeeder:',feeder,'\nLinpoint:',lin_point,'\nLoad Point:',loadPointLo,'\nTap Model:',netModel)

    # # PART A.1 - load models ===========================
    LM.loadNetModel()
    KtotPu = LM.KtotPu
    vBase = LM.vTotBase
    v_idx = LM.v_idx_tot
    YZp = LM.SyYNodeOrderTot
    YZd = LM.SdYNodeOrderTot
    xhyN = LM.xhyNtot
    xhdN = LM.xhdNtot 
    b0ls = LM.b0lo
    b0hs = LM.b0hi

    KfixPu = LM.KfixPu
    dvBase = LM.dvBase

    VpMv = lp0data['VpMv']
    VmMv = lp0data['VmMv']
    VpLv = lp0data['VpLv']
    VmLv = lp0data['VmLv']
    
    mvIdx = np.where(vBase>1000)
    lvIdx = np.where(vBase<=1000)
    vBaseMv = vBase[mvIdx]
    vBaseLv = vBase[lvIdx]
    
    # mvIdxYz = lp0data['mvIdxYz']
    # lvIdxYz = lp0data['lvIdxYz']
    
    
    # OPENDSS ADMIN =======================================
    # B1. load the appropriate model/DSS
    DSSText.Command='Compile ('+fn+'.dss)'
    BB0,SS0 = cpf_get_loads(DSSCircuit)

    cpf_set_loads(DSSCircuit,BB0,SS0,loadPointLo)
    DSSSolution.Solve()

    if not netModel:
        DSSText.Command='set controlmode=off'
    elif netModel:
        DSSText.Command='set maxcontroliter=300'
    DSSObj.AllowForms=False
    DSSText.Command='set maxiterations=100'

    YNodeVnom = tp_2_ar(DSSCircuit.YNodeVarray)
    YZ = DSSCircuit.YNodeOrder
    YZ = vecSlc(YZ[3:],v_idx)

    # 2. run MC analysis, going through each generator and setting to a power.
    genNamesY = add_generators(DSSObj,YZp,False)
    genNamesD = add_generators(DSSObj,YZd,True)
    DSSSolution.Solve()

    genNames = genNamesY+genNamesD
    nCnstr = 9

    Vp_pct_dss = np.zeros(pdfData['nP'])
    Cns_pct_dss = np.zeros(list(pdfData['nP'])+[nCnstr])
    Vp_pct_lin = np.zeros(pdfData['nP'])
    Cns_pct_lin = np.zeros(list(pdfData['nP'])+[nCnstr])

    hcGenSet = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],nCnstr))
    hcGenSetLin = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],nCnstr))
    genTotSet = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],nCnstr))

    hcGenLinAll = np.array([]); hcGenAll = np.array([])
    hcGen = []; hcGenLin=[]
    # PART A.2 - choose distributions and reduce linear model ===========================


    for i in range(pdfData['nP'][0]):
        # PART B FROM HERE ==============================
        print('---- Start MC ----',time.process_time())
        Mu0 = pdf.halfLoadMean(LM.loadScaleNom,xhyN,xhdN)
        pdfGen = pdf.genPdfMcSet(nMc,Mu0,i)[0]
        
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
                # vLsDss = np.ones((nMc,len(v_idx)))
                # vHsDss = np.ones((nMc,len(v_idx)))
                vLsMvDss = np.ones((nMc,len(mvIdx[0])))
                vHsMvDss = np.ones((nMc,len(mvIdx[0])))
                vLsLvDss = np.ones((nMc,len(lvIdx[0])))
                vHsLvDss = np.ones((nMc,len(lvIdx[0])))
                vDvDss = np.ones((nMc,len(v_idx)))
                convLo = []; convDv = []; convHi = []
                print('\nDSS MC Run:',jj,'/',pdfData['nP'][-1])
                
                for j in range(nMc):
                    if j%(nMc//4)==0:
                        print(j,'/',nMc)
                    set_generators( DSSCircuit,genNames,pdfGen[:,j]*pdfData['mu_k'][jj] )
                    
                    if fdr_i != 6:
                        DSSText.Command='Batchedit load..* vmin=0.33 vmax=3.0 model=1'
                    
                    DSSText.Command='Batchedit generator..* vmin=0.33 vmax=3.0'
                    DSSText.Command='Batchedit regcontrol..* band=1.0' # seems to be as low as we can set without bit problems
                    
                    # first solve for the high load point [NB: This order seems best!]
                    cpf_set_loads(DSSCircuit,BB0,SS0,loadPointHi)
                    
                    DSSSolution.Solve()
                    convHi = convHi+[DSSSolution.Converged]
                    vHsDss0 = tp_2_ar(DSSCircuit.YNodeVarray)
                    
                    # then low load point
                    cpf_set_loads(DSSCircuit,BB0,SS0,loadPointLo)
                    
                    DSSSolution.Solve()
                    convLo = convLo+[DSSSolution.Converged]
                    vLsDss0 = tp_2_ar(DSSCircuit.YNodeVarray)
                    
                    # finally solve for voltage deviation. 
                    DSSText.Command='Batchedit generator..* kW=0.001'
                    DSSText.Command='set controlmode=off'
                    DSSSolution.Solve()

                    convDv = convDv+[DSSSolution.Converged]
                    vDv0 = tp_2_ar(DSSCircuit.YNodeVarray)
                    
                    DSSText.Command='set controlmode=static'
                    
                    if convLo and convHi and convDv:
                        # vLsDss[j,:] = abs(vLsDss0)[3:][v_idx]/vBase
                        # vHsDss[j,:] = abs(vHsDss0)[3:][v_idx]/vBase
                        vLsMvDss[j,:] = abs(vLsDss0)[3:][v_idx][mvIdx[0]]/vBaseMv
                        vLsLvDss[j,:] = abs(vLsDss0)[3:][v_idx][lvIdx[0]]/vBaseLv
                        vHsMvDss[j,:] = abs(vHsDss0)[3:][v_idx][mvIdx[0]]/vBaseMv
                        vHsLvDss[j,:] = abs(vHsDss0)[3:][v_idx][lvIdx[0]]/vBaseLv
                        vDvDss[j,:] = abs(abs(vLsDss0) - abs(vDv0))[3:][v_idx]/vBase
                    
                if sum(convLo+convHi+convDv)!=len(convLo+convHi+convDv):
                    print('\nNo. Converged:',sum(convLo+convHi+convDv),'/',nMc*3)
                
                # NOW: calculate the HC value:
                Cns_pct_dss[i,jj], inBoundsDss = cnsBdsCalc(vLsMvDss,vLsLvDss,vHsMvDss,vHsLvDss,vDvDss,lp0data)
                Vp_pct_dss[i,jj] = 100*sum(inBoundsDss)/nMc
                hcGen = genTot[inBoundsDss]
            
            if mcLinOn:
                vLsMvLin = ((DelVoutLin*pdfData['mu_k'][jj]) + b0ls)[:,mvIdx[0]]
                vLsLvLin = ((DelVoutLin*pdfData['mu_k'][jj]) + b0ls)[:,lvIdx[0]]
                vHsMvLin = ((DelVoutLin*pdfData['mu_k'][jj]) + b0hs)[:,mvIdx[0]]
                vHsLvLin = ((DelVoutLin*pdfData['mu_k'][jj]) + b0hs)[:,lvIdx[0]]
                vDvLin = ddVoutLin*pdfData['mu_k'][jj]
                
                Cns_pct_lin[i,jj], inBoundsLin = cnsBdsCalc(vLsMvLin,vLsLvLin,vHsMvLin,vHsLvLin,vDvLin,lp0data)
                
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

    # LM.runLinHc(nMc,pdf.pdf) # equivalent at the moment



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
        if pltSave:
            with open(SN,'wb') as file:
                pickle.dump([rslt],file)
        

    # ================ PLOTTING FUNCTIONS FROM HERE
    if pltCns:
        fig, ax = plt.subplots()
        # clrs = ['#1f77b4','#ff7f0e','red','#2ca02c','green','black','blue']
        clrs = cm.nipy_spectral(np.linspace(0,1,9))
        ax.set_prop_cycle(color=clrs)
        if mcDssOn:
            plt.plot(pdfData['mu_k'],Cns_pct_dss[0])

        plt.plot(pdfData['mu_k'],Cns_pct_lin[0],'--')
        plt.xlabel('Scale factor');
        plt.ylabel('P(.), %');
        plt.title('Constraints, '+feeder)
        plt.legend(('$\Delta V$','$V^{+}_{\mathrm{MV,LS}}$','$V^{-}_{\mathrm{MV,LS}}$','$V^{+}_{\mathrm{LV,LS}}$','$V^{-}_{\mathrm{LV,LS}}$','$V^{+}_{\mathrm{MV,HS}}$','$V^{-}_{\mathrm{MV,HS}}$','$V^{+}_{\mathrm{LV,HS}}$','$V^{-}_{\mathrm{LV,HS}}$'))
        if mcDssOn and pltSave:
            plt.savefig(os.path.join(SD,'pltCns.png'))
        
        plt.show()


    if pltPwrCdf:
        plt.plot(pp,ppPdfLin)
        if mcDssOn:
            plt.plot(pp,ppPdf)

        plt.legend(('Lin model','OpenDSS'))
        plt.xlabel('Power');
        plt.ylabel('P(.)');
        plt.grid(True)
        if mcDssOn and pltSave:
            plt.savefig(os.path.join(SD,'pltPwrCdf.png'))

        plt.show()

    if pltHcBoth:
        plt.subplot(121)
        for i in range(pdfData['nP'][0]):
            # if mcDssOn:
            plt.semilogy(pdfData['mu_k'],Vp_pct_dss[i],'ro-')
            plt.semilogy(pdfData['mu_k'],Vp_pct_lin[i],'g.-')
            plt.legend(('VpDss','VpNom','VpSvd'))

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
        if mcDssOn and pltSave:
            plt.savefig(os.path.join(SD,'pltHcBoth.png'))

        plt.show()
        
    if pltHcGen:
        i=0
        plt.plot(pdfData['mu_k'],genTotSet[i,:,0],'b-'); 
        plt.plot(pdfData['mu_k'],genTotSet[i,:,2],'b_'); 
        plt.plot(pdfData['mu_k'],genTotSet[i,:,4],'b-');
        if mcDssOn:
            plt.plot(pdfData['mu_k'],hcGenSet[i,:,0],'r-'); 
            plt.plot(pdfData['mu_k'],hcGenSet[i,:,2],'r_'); 
            plt.plot(pdfData['mu_k'],hcGenSet[i,:,4],'r-');        

        plt.plot(pdfData['mu_k'],hcGenSetLin[i,:,0],'g-') 
        plt.plot(pdfData['mu_k'],hcGenSetLin[i,:,2],'g_') 
        plt.plot(pdfData['mu_k'],hcGenSetLin[i,:,4],'g-') 
        xlm = plt.xlim()
        plt.xlim((-dsf.get_dx(pdfData['mu_k']),xlm[1]))
        plt.xlabel('Scale factor');
        plt.title('Hosting Capacity (kW)');
        plt.grid(True)
        if mcDssOn and pltSave:
            plt.savefig(os.path.join(SD,'pltHcGen.png'))

        plt.show()


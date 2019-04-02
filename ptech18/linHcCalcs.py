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
from linSvdCalcs import hcPdfs, linModel, cnsBdsCalc, plotCns, plotHcVltn, plotPwrCdf, plotHcGen
from matplotlib import cm

WD = os.path.dirname(sys.argv[0])

mcLinOn = True
# mcLinOn = False
mcDssOn = True
# mcDssOn = False

# PLOTTING options:
pltHcVltn = True
# pltHcVltn = False
pltHcGen = True
# pltHcGen = False
pltPwrCdf = True
# pltPwrCdf = False
pltCns = True
# pltCns = False

pltSave = True # for saving both plots and results
# pltSave = False

# CHOOSE Network
fdr_i_set = [5,6,8,9,0,14,17,18,22,19,20,21]
fdr_i_set = [9,17,18,19,20,21,22]
# fdr_i_set = [5,6,8,0,14] # fastest few
# fdr_i_set = [17,18] # medium length 1
# fdr_i_set = [19,20,21] # medium length 2
fdr_i_set = [17]
nMc = int(3e2)
nMc = 50

pdfName = 'gammaWght'; prms=np.array([3.0])
pdfName = 'gammaFrac'; prms=np.arange(0.02,1.02,0.02)
# pdfName = 'gammaFrac'; prms=np.arange(0.05,1.05,0.05)

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

    LM = linModel(fdr_i,WD)
    netModel = LM.netModelNom
    pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,prms=prms,WD=WD )
    
    mu_k = pdf.pdf['mu_k']
    pdfData = pdf.pdf
    dMu = pdf.dMu

    DVmax = LM.DVmax # percent
    # ADMIN =============================================
    SD = os.path.join(WD,'hcResults',feeder)
    SN = os.path.join(SD,'linHcCalcsRslt_'+pdfName+'.pkl')

    fn = get_ckt(WD,feeder)[1]    
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
    b0ls = LM.b0ls
    b0hs = LM.b0hs

    KfixPu = LM.KfixPu
    dvBase = LM.dvBase

    VpMv = lp0data['VpMv']
    VmMv = lp0data['VmMv']
    VpLv = lp0data['VpLv']
    VmLv = lp0data['VmLv']
    mvIdx = LM.mvIdx
    lvIdx = LM.lvIdx
    vBaseMv = vBase[mvIdx]
    vBaseLv = vBase[lvIdx]
    
    # mvIdxYz = lp0data['mvIdxYz']
    # lvIdxYz = lp0data['lvIdxYz']
    # OPENDSS ADMIN =======================================
    # B1. load the appropriate model/DSS
    DSSText.Command='Compile ('+fn+'.dss)'
    BB0,SS0 = cpf_get_loads(DSSCircuit)
    
    # pp0 = np.array(list(SS0.values())).real
    # pp0 = pp0[pp0<100]
    # print(feeder)
    # print('Power mean:',np.mean(0.5*pp0)) # see notes 31-3-19
    # print('Power std (triangular distribution):',np.sqrt(np.mean((pp0*pp0)/12)))
    
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
    nRcd = 5

    Vp_pct_dss = np.zeros(pdfData['nP'])
    Cns_pct_dss = np.zeros(list(pdfData['nP'])+[nCnstr])
    Vp_pct_lin = np.zeros(pdfData['nP'])
    Cns_pct_lin = np.zeros(list(pdfData['nP'])+[nCnstr])

    hcGenSetDss = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],nRcd))
    hcGenSetLin = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],nRcd))
    genTotSet = np.nan*np.zeros((pdfData['nP'][0],pdfData['nP'][1],nRcd))

    hcGenLinAll = np.array([]); hcGenAll = np.array([])
    hcGen = []; hcGenLin=[]
    genTotAll = np.nan*np.zeros((pdfData['nP'][0],nMc*pdfData['nP'][1])) # NB this gets flattened later
    # PART A.2 - choose distributions and reduce linear model ===========================

    tStart = time.process_time()
    for i in range(pdfData['nP'][0]):
        # PART B FROM HERE ==============================
        if i%(pdfData['nP'][0]//4)==0:
            print('---- Start MC ----',time.process_time(),i+1,'/',pdfData['nP'][0])
        Mu0 = pdf.halfLoadMean(LM.loadScaleNom,xhyN,xhdN)
        pdfGen = pdf.genPdfMcSet(nMc,Mu0,i)[0]
        
        genTot0 = np.sum(pdfGen,axis=0)
        genTotSort = genTot0.copy()
        genTotSort.sort()
        genTotAll0 = np.outer(genTot0,pdfData['mu_k'])
        genTotAll[i] = genTotAll0.flatten()
        
        DelVoutLin = (KtotPu.dot(pdfGen).T)*1e3
        ddVoutLin = abs((KfixPu.dot(pdfGen).T)*1e3) # just get abs change
        
        for jj in range(pdfData['nP'][-1]):
            genTot = genTot0*pdfData['mu_k'][jj]
            
            if mcDssOn:
                vLsMvDss = np.ones((nMc,len(mvIdx)))
                vHsMvDss = np.ones((nMc,len(mvIdx)))
                vLsLvDss = np.ones((nMc,len(lvIdx)))
                vHsLvDss = np.ones((nMc,len(lvIdx)))
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
                        vLsMvDss[j,:] = abs(vLsDss0)[3:][v_idx][mvIdx]/vBaseMv
                        vLsLvDss[j,:] = abs(vLsDss0)[3:][v_idx][lvIdx]/vBaseLv
                        vHsMvDss[j,:] = abs(vHsDss0)[3:][v_idx][mvIdx]/vBaseMv
                        vHsLvDss[j,:] = abs(vHsDss0)[3:][v_idx][lvIdx]/vBaseLv
                        vDvDss[j,:] = abs(abs(vLsDss0) - abs(vDv0))[3:][v_idx]/vBase
                    
                if sum(convLo+convHi+convDv)!=len(convLo+convHi+convDv):
                    print('\nNo. Converged:',sum(convLo+convHi+convDv),'/',nMc*3)
                
                # NOW: calculate the HC value:
                Cns_pct_dss[i,jj], inBoundsDss = cnsBdsCalc(vLsMvDss,vLsLvDss,vHsMvDss,vHsLvDss,vDvDss,lp0data)
                Vp_pct_dss[i,jj] = 100*sum(inBoundsDss)/nMc
                hcGen = genTot[inBoundsDss]
            
            if mcLinOn:
                vLsMvLin = ((DelVoutLin*pdfData['mu_k'][jj]) + b0ls)[:,mvIdx]
                vLsLvLin = ((DelVoutLin*pdfData['mu_k'][jj]) + b0ls)[:,lvIdx]
                vHsMvLin = ((DelVoutLin*pdfData['mu_k'][jj]) + b0hs)[:,mvIdx]
                vHsLvLin = ((DelVoutLin*pdfData['mu_k'][jj]) + b0hs)[:,lvIdx]
                vDvLin = ddVoutLin*pdfData['mu_k'][jj]
                
                Cns_pct_lin[i,jj], inBoundsLin = cnsBdsCalc(vLsMvLin,vLsLvLin,vHsMvLin,vHsLvLin,vDvLin,lp0data)
                
                Vp_pct_lin[i,jj] = 100*sum(inBoundsLin)/nMc
                hcGenLin = genTot[inBoundsLin]

            if len(hcGen)!=0 and mcDssOn:
                hcGenAll = np.concatenate((hcGenAll,hcGen))
                hcGen.sort()
                hcGenSetDss[i,jj,0] = hcGen[0]
                hcGenSetDss[i,jj,1] = hcGen[np.floor(len(hcGen)*1.0/4.0).astype(int)]
                hcGenSetDss[i,jj,2] = hcGen[np.floor(len(hcGen)*1.0/2.0).astype(int)]
                hcGenSetDss[i,jj,3] = hcGen[np.floor(len(hcGen)*3.0/4.0).astype(int)]
                hcGenSetDss[i,jj,4] = hcGen[-1]
                
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
        if i%(pdfData['nP'][0]//4)==0:
            print('MC complete.',time.process_time())
    tEnd = time.process_time()
    # LM.runLinHc(nMc,pdf.pdf) # equivalent at the moment
    
    # NOW: calculate the statistics we want.

    binNo = max(pdf.pdf['nP'])//2
    genTotAll = genTotAll.flatten()
    hist1 = plt.hist(genTotAll,bins=binNo,range=(0,max(genTotAll)))
    if mcDssOn:
        hist2 = plt.hist(hcGenAll,bins=binNo,range=(0,max(genTotAll)))
    hist2lin = plt.hist(hcGenLinAll,bins=binNo,range=(0,max(genTotAll)))
    plt.close()

    pp = hist1[1][1:]
    ppPdfLin = hist2lin[0]/hist1[0]
        
    p0lin = pp[np.argmax(ppPdfLin!=0)]
    p10lin = pp[np.argmax(ppPdfLin>=0.1)]

    if pdf.pdf['name']=='gammaWght':
        k0lin = pdfData['mu_k'][np.argmax(Vp_pct_lin!=0)]
        k10lin = pdfData['mu_k'][np.argmax(Vp_pct_lin>=10.)]
    elif pdf.pdf['name']=='gammaFrac':
        k0lin = pdfData['prms'][np.argmax(Vp_pct_lin!=0)]
        k10lin = pdfData['prms'][np.argmax(Vp_pct_lin>=10.)]

    print('\n--- Linear results ---\n\nP0:',p0lin,'\nP10:',p10lin,'\nk0:',k0lin,'\nk10:',k10lin)

    if mcDssOn:
        ppPdfDss = hist2[0]/hist1[0]
        p0 = pp[np.argmax(ppPdfDss!=0)]
        p10 = pp[np.argmax(ppPdfDss>=0.1)]        
        if pdf.pdf['name']=='gammaWght':        
            k0 = pdfData['mu_k'][np.argmax(Vp_pct_dss!=0)]
            k10 = pdfData['mu_k'][np.argmax(Vp_pct_dss>=10.)]
        elif pdf.pdf['name']=='gammaFrac':
            k0 = pdfData['prms'][np.argmax(Vp_pct_dss!=0)]
            k10 = pdfData['prms'][np.argmax(Vp_pct_dss>=10.)]

        print('\n--- OpenDSS results ---\n\nP0:',p0,'\nP10:',p10,'\nk0:',k0,'\nk10:',k10)
        
        if not os.path.exists(SD):
            os.makedirs(SD)
        rslt = {'p0lin':p0lin,'p10lin':p10lin,'k0lin':k0lin,'k10lin':k10lin,'p0':p0,'p10':p10,'k0':k0,'k10':k10,'netModel':netModel,'nMc':nMc,'dMu':dMu,'feeder':feeder,'time2run':tStart-tEnd}
        if pltSave:
            with open(SN,'wb') as file:
                pickle.dump([rslt],file)
    
    # ================ PLOTTING FUNCTIONS FROM HERE
    if pltCns:
        fig, ax = plt.subplots()
        if mcDssOn:
            ax = plotCns(pdfData['mu_k'],pdfData['prms'],Cns_pct_dss,feeder=feeder,lineStyle='-',ax=ax,pltShow=False)
        
        ax = plotCns(pdfData['mu_k'],pdfData['prms'],Cns_pct_lin,feeder=feeder,lineStyle='--',ax=ax,pltShow=False)
        if mcDssOn and pltSave:
            plt.savefig(os.path.join(SD,'pltCns_'+pdfName+'.png'))
        
        plt.show()


    if pltPwrCdf:
        ax = plotPwrCdf(pp,ppPdfLin,lineStyle='--',pltShow=False)
        if mcDssOn:
            ax = plotPwrCdf(pp,ppPdfDss,ax=ax,lineStyle='-',pltShow=False)
            ax.legend(('Lin model','OpenDSS'))
        plt.xlabel('Power');
        plt.ylabel('P(.)');
        ax.grid(True)
        if mcDssOn and pltSave:
            plt.savefig(os.path.join(SD,'pltPwrCdf_'+pdfName+'.png'))
        
        plt.show()

    if pltHcVltn:
        fig = plt.subplot()
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        if mcDssOn:
            plotHcVltn(pdfData['mu_k'],pdfData['prms'],Vp_pct_dss,ax=ax1,pltShow=False,feeder=feeder,logScale=True)
            plotHcVltn(pdfData['mu_k'],pdfData['prms'],Vp_pct_dss,ax=ax2,pltShow=False,feeder=feeder,logScale=False)
        
        plotHcVltn(pdfData['mu_k'],pdfData['prms'],Vp_pct_lin,ax=ax1,pltShow=False,feeder=feeder,logScale=True)
        plotHcVltn(pdfData['mu_k'],pdfData['prms'],Vp_pct_lin,ax=ax2,pltShow=False,feeder=feeder,logScale=False)
        if mcDssOn:
            ax1.legend(('Vltn., DSS','Vltn., Nom'))
        
        plt.tight_layout()
        if mcDssOn and pltSave:
            plt.savefig(os.path.join(SD,'pltHcVltn_'+pdfName+'.png'))
        
        plt.show()
        
    if pltHcGen:
        ax = plotHcGen(mu_k,prms,genTotSet[:,:,0::2],'k')
        ax = plotHcGen(mu_k,prms,hcGenSetLin[:,:,0::2],'r',ax=ax)
        if mcDssOn:
            ax = plotHcGen(mu_k,prms,hcGenSetLin[:,:,0::2],'g',ax=ax)
        xlm = ax.get_xlim()
        ax.set_xlim((0,xlm[1]))
        plt.xlabel('Scale factor');
        plt.title('Hosting Capacity (kW)');
        plt.grid(True)
        if mcDssOn and pltSave:
            plt.savefig(os.path.join(SD,'pltHcGen_'+pdfName+'.png'))
        plt.show()
# Based on the previous script charFuncMcVal_v2
# >>>>> Part A: Linear method
# 1. Load linear model.
# 2. Choose distributions at each bus
# 3. Calculate distribution
# 4. Run MC analysis using linear model
# >>>>> Part B: Run MC analysis using OpenDSS
# 1. load appropriate model
# 2. sample distibution appropriately and run load flow
# 3. Compare results.

import pickle, os, sys, win32com.client, time, scipy.stats
import numpy as np
from dss_python_funcs import *
import matplotlib.pyplot as plt
import dss_stats_funcs as dsf
from linSvdCalcs import hcPdfs, linModel, cnsBdsCalc, plotCns, plotHcVltn, plotPwrCdf, plotHcGen, plotBoxWhisk
from dss_voltage_funcs import getRegBwVrto,getRegIcVr,getRx


mcLinOn = True # most basic MC run.
mcLinOn = False
mcLinVal = True
mcLinVal = False
mcLinSns = True
mcLinSns = False
mcDssOn = True
mcDssOn = False

# mcDssBw = 1
# mcFullSet = 1
# mcLinUpg = 1
# mcLinLds = 1
# mcLinPrg = 1
# mcTapSet = 1
# mcTapMultSet = 1



# # PLOTTING options:
# pltHcVltn = 1
# pltHcGen = 1
# pltPwrCdf = 1
# pltCns = 1
plotShow = 1

nMc = 100 # nominal value of 100
# nMc = 10 # nominal value of 100

pltSave = True # for saving both plots and results
# pltSave = False

regBand=0 # opendss options
# regBand=1.0 # opendss options
setCapsOpt = 'linModel' # opendss options. 'linModels' is the 'right' option, cf True/False

# CHOOSE Network
fdr_i_set = [5,6,8,9,0,14,17,18,22,19,20,21]
# fdr_i_set = [5,6,8,0,14,17,18,20,21]
fdr_i_set = [6,8,9,17,18,19,20,21,22]
fdr_i_set = [8,20,17,18,21,19,22,9]
fdr_i_set = [6,8,20,17,18,21]

pdfName = 'gammaWght'
pdfName = 'gammaFrac'; prms=np.array([]) 
# pdfName = 'gammaFrac'; prms=np.arange(0.05,1.05,0.05) # to make it faster
# pdfName = 'gammaFrac'; prms=np.arange(0.1,1.10,0.10) # to make it faster

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr','123busCvr']

tmax = 0.1 # for LP HC runs

# # PF is -0.95
# optMultJ1 = 0.995**np.array([5,4,0,1,2,4,0,5.5,4]) 
# optMult8500 = 0.995**np.array([3,1,2,2,1,0,1,0,0,3,3,1])
# upgReg = {'8500node':optMult8500,'epri24':0.981,'epriJ1':optMultJ1,'epriK1':0.984,'epriM1':0.980} #NB epri M1 is slightly lower than the actual optimal lambda of 0.984
# upgPf = -0.95

# # PF is 1.00:
# optMultJ1 = 0.995**np.array([5,4,0,-1,2,4,0,5.5,4]) 
# optMult8500 = 0.995**np.array([3,1,2,2,1,0,1,0,0,3,3,1])
# upgReg = {'8500node':optMult8500,'epri24':0.980,'epriJ1':optMultJ1,'epriK1':0.99,'epriM1':0.983} #NB epri M1 is slightly lower than the actual optimal lambda of 0.984
# upgPf = 1.00

# PF is -0.95, nominal taps
optMultJ1 = np.ones((0.995**np.array([5,4,0,1,2,4,0,5.5,4]) ).shape)
optMult8500 = np.ones((0.995**np.array([3,1,2,2,1,0,1,0,0,3,3,1])).shape)
upgReg = {'8500node':optMult8500,'epri24':1.0,'epriJ1':optMultJ1,'epriK1':1.0,'epriM1':1.0} #NB epri M1 is slightly lower than the actual optimal lambda of 0.984
upgPf = -0.95

# opendss with 'early bindings'
WD = os.path.dirname(sys.argv[0])
from win32com.client import makepy
sys.argv=["makepy","OpenDSSEngine.DSS"]
makepy.main()
DSSObj = win32com.client.Dispatch("OpenDSSEngine.DSS")

DSSText = DSSObj.Text
DSSCircuit = DSSObj.ActiveCircuit
DSSSolution = DSSCircuit.Solution


def getResultSensitivity(linResult):
    dBnds = 0.05
    HcA = np.sum(linResult['Lp_pct']<(1+dBnds),axis=2)
    HcB = np.sum(linResult['Lp_pct']<(1-dBnds),axis=2)
    return np.sum(np.abs(HcA - HcB))/(2*dBnds*HcA.shape[0])
    

for fdr_i in fdr_i_set:
    feeder = fdrs[fdr_i]
    # feeder = '213'

    with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','chooseLinPoint.pkl'),'rb') as handle:
        lp0data = pickle.load(handle)

    # # PART A.1 - load models ===========================
    LM = linModel(fdr_i,WD,setCapsModel=setCapsOpt)
    YZp = LM.SyYNodeOrderTot
    YZd = LM.SdYNodeOrderTot
    pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,WD=WD,nMc=nMc,rndSeed=0,prms=prms ) # use
    
    if mcLinVal or ('mcFullSet' in locals()) or ('mcTapSet' in locals()):
        LMval = linModel(fdr_i,WD,setCapsModel=setCapsOpt)
        pdfVal = hcPdfs(LMval.feeder,netModel=LMval.netModelNom,pdfName=pdfName,WD=WD,nMc=nMc,rndSeed=2**31,prms=prms )
    
    if mcLinSns:
        LMsns = linModel(fdr_i,WD,setCapsModel=setCapsOpt)
        
    if ('mcLinUpg' in locals()) or ('mcLinPrg' in locals()):
        # LMupg = linModel(fdr_i,WD,QgenPf=-0.95)
        LMupg = linModel(fdr_i,WD,QgenPf=upgPf)
        LMupg.updateDcpleModel(LMupg.regVreg0*upgReg[feeder])
    
    # ADMIN =============================================
    SD = os.path.join(WD,'hcResults',feeder)

    fn = get_ckt(WD,feeder)[1]    
    print('\n\nStart '+'='*30,'\nFeeder:',feeder,'\nLinpoint:',lp0data['k'],'\nLoad Point:',lp0data['kLo'],'\nTap Model:',LM.netModelNom)
    
    # OPENDSS ADMIN =======================================
    # B1. load the appropriate model/DSS, and find nominal taps as required.
    DSSText.Command='Compile ('+fn+'.dss)'
    BB0,SS0 = cpf_get_loads(DSSCircuit)
    
    lin_point=lp0data['k']
    cpf_set_loads(DSSCircuit,BB0,SS0,lin_point,setCaps=setCapsOpt,capPos=lp0data['capPosOut'])
    DSSText.Command='Batchedit load..* vminpu=0.33 vmaxpu=3' # to match linearise_manc stuff exactly.
    DSSSolution.Solve()
    LM.TC_No0 = find_tap_pos(DSSCircuit)
    
    DSSText.Command='Compile ('+fn+'.dss)'
    cpf_set_loads(DSSCircuit,BB0,SS0,lp0data['kLo'],setCaps=setCapsOpt,capPos=lp0data['capPosOut'])
    DSSSolution.Solve()

    if not LM.netModelNom:
        DSSText.Command='set controlmode=off'
    elif LM.netModelNom:
        DSSText.Command='set maxcontroliter=300'
    DSSObj.AllowForms=False
    DSSText.Command='set maxiterations=100'

    # 2. run MC analysis, going through each generator and setting to a power.
    genNamesY = add_generators(DSSObj,YZp,False)
    genNamesD = add_generators(DSSObj,YZd,True)
    DSSSolution.Solve()
    genNames = genNamesY+genNamesD
    
    # PART A.2 - choose distributions and reduce linear model ===========================
    if 'mcFullSet' in locals():
        # 'nominal' linear
        LM.runLinHc(pdf,fast=True) # equivalent at the moment
        linHcRslNom = LM.linHcRsl
        
        # precondition linear model + MC run 2
        Mu,Sgm = pdf.getMuStd(LM,0)
        LM.busViolationVar(Sgm)
        LM.makeCorrModel()
        LM.runLinHc(pdfVal,model='cor',fast=True) # equivalent at the moment. PDFVAL is what changes this here cf 'pdf'.
        linHcRslNmc = LM.linHcRsl
        preCndLeft = len(LM.NSetCor[0])/len(LM.varKfullU)*100
        print('Pre conditioning left:',preCndLeft)
        
        # Finally run linear model to which everything is compared
        LM.runLinHc(pdf,model='cor',fast=True)
        linHcRsl = LM.linHcRsl
        print('Linear Run Time:',linHcRsl['runTime'])
        print('Sampling Time:',linHcRsl['runTimeSample'])
        
        # Nominal DSS model:
        LM.runDssHc(pdf,DSSObj,genNames,BB0,SS0,regBand=regBand,setCapsModel=setCapsOpt)
        dssHcRslNom = LM.dssHcRsl
        
        # Low BW DSS model:
        LM.runDssHc(pdf,DSSObj,genNames,BB0,SS0,regBand=1.0,setCapsModel=setCapsOpt)
        dssHcRslTgt = LM.dssHcRsl
        
        dssMae = LM.calcLinPdfError(dssHcRslNom,model='dss')
        dssReg = LM.calcLinPdfError(dssHcRslNom,type='reg',model='dss')
        
        # calculate error:
        maeVals = {'nomMae':LM.calcLinPdfError(linHcRslNom),'nmcMae':LM.calcLinPdfError(linHcRslNmc), 'dssNomMae':LM.calcLinPdfError(dssHcRslNom),'dssTgtMae':LM.calcLinPdfError(dssHcRslTgt),'dssMae':dssMae}
        rgeVals = {'nomRge':LM.calcLinPdfError(linHcRslNom,type='reg'),'nmcRge':LM.calcLinPdfError(linHcRslNmc,type='reg'), 'dssNomRge':LM.calcLinPdfError(dssHcRslNom,type='reg'),'dssTgtRge':LM.calcLinPdfError(dssHcRslTgt,type='reg'),'dssReg':dssReg}
        
        rslt = {'linHcRsl':linHcRsl,'linHcRslNom':linHcRslNom,'linHcRslNmc':linHcRslNmc,'dssHcRslNom':dssHcRslNom,'dssHcRslTgt':dssHcRslTgt,'pdfData':pdf.pdf,'pdfDataNmc':pdfVal.pdf,'feeder':feeder,'maeVals':maeVals,'rgeVals':rgeVals,'preCndLeft':preCndLeft}
        if pltSave:
            SN = os.path.join(SD,'linHcCalcsRslt_'+pdfName+'_finale_dpndnt.pkl')
            with open(SN,'wb') as file:
                pickle.dump(rslt,file)
        
    if mcLinOn or mcLinSns or mcDssOn:
        LM.runLinHc(pdf) # equivalent at the moment
        
    if mcDssOn:
        LM.runDssHc(pdf,DSSObj,genNames,BB0,SS0,regBand=regBand,setCapsModel=setCapsOpt)
        dssRegMae = LM.calcLinPdfError(LM.dssHcRsl)
        print('\n -------- Complete -------- ')
        
        if not os.path.exists(SD):
            os.makedirs(SD)
        rslt = {'dssHcRsl':LM.dssHcRsl,'linHcRsl':LM.linHcRsl,'pdfData':pdf.pdf,'feeder':feeder,'regMae':dssRegMae}
        if pltSave:
            SN = os.path.join(SD,'linHcCalcsRslt_'+pdfName+'_reg'+str(regBand)+'_new.pkl')
            with open(SN,'wb') as file:
                pickle.dump(rslt,file)
    
    if 'mcDssBw' in locals():
        LM.runDssHc(pdf,DSSObj,genNames,BB0,SS0,regBand=1.0,setCapsModel=setCapsOpt)
        dssRegMae = LM.calcLinPdfError(LM.dssHcRsl)
        print('\n -------- Complete -------- ')
        
        rslt = {'dssHcRsl':LM.dssHcRsl,'linHcRsl':LM.linHcRsl,'pdfData':pdf.pdf,'feeder':feeder,'regMae':dssRegMae}
        if pltSave:
            SN = os.path.join(SD,'linHcCalcsRslt_'+pdfName+'_reg'+str(regBand)+'_bw.pkl')
            with open(SN,'wb') as file:
                pickle.dump(rslt,file)
    
    if 'mcTapSet' in locals():
        # 'nominal' linear
        LM.runLinHc(pdf,fast=False) # equivalent at the moment
        linHcRslNom = LM.linHcRsl
        
        # precondition linear model + MC run 2
        Mu,Sgm = pdf.getMuStd(LM,0)
        LM.busViolationVar(Sgm)
        LM.makeCorrModel()
        LM.runLinHc(pdfVal,model='cor',fast=True) # equivalent at the moment. PDFVAL is what changes this here cf 'pdf'.
        linHcRslNmc = LM.linHcRsl
        preCndLeft = len(LM.NSetCor[0])/len(LM.varKfullU)*100
        print('Pre conditioning left:',preCndLeft)
        
        # Finally run linear model to which everything is compared
        LM.runLinHc(pdf,model='cor',fast=True)
        linHcRsl = LM.linHcRsl
        print('Linear Run Time:',linHcRsl['runTime'])
        print('Sampling Time:',linHcRsl['runTimeSample'])
        
        # Nominal DSS model:
        LM.runDssHc(pdf,DSSObj,genNames,BB0,SS0,regBand=regBand,setCapsModel=setCapsOpt,runType='tapSet',tapPosStart=linHcRslNom['tapPosSeq'])
        dssHcRslTapSet = LM.dssHcRsl
        
        if fdr_i==17 or fdr_i==18:
            dssHcRslTapLck = dssHcRslTapSet
            dssHcRslTapTgt = dssHcRslTapSet
        else:
            LM.runDssHc(pdf,DSSObj,genNames,BB0,SS0,regBand=regBand,setCapsModel=setCapsOpt,runType='tapSetLock',tapPosStart=linHcRslNom['tapPosSeq'])
            dssHcRslTapLck = LM.dssHcRsl
            
            LM.runDssHc(pdf,DSSObj,genNames,BB0,SS0,regBand=1.0,setCapsModel=setCapsOpt,runType='tapSet',tapPosStart=linHcRslNom['tapPosSeq'])
            dssHcRslTapTgt = LM.dssHcRsl
        
        dssMae = LM.calcLinPdfError(dssHcRslTapTgt,model='dss')
        dssRge = LM.calcLinPdfError(dssHcRslTapTgt,type='reg',model='dss')
        
        # calculate error:
        maeVals = {'nomMae':LM.calcLinPdfError(linHcRslNom),'nmcMae':LM.calcLinPdfError(linHcRslNmc), 'dssSetMae':LM.calcLinPdfError(dssHcRslTapSet),'dssLckMae':LM.calcLinPdfError(dssHcRslTapLck),'dssTgtMae':LM.calcLinPdfError(dssHcRslTapTgt),'dssMae':dssMae}
        rgeVals = {'nomRge':LM.calcLinPdfError(linHcRslNom,type='reg'),'nmcRge':LM.calcLinPdfError(linHcRslNmc,type='reg'),'dssSetRge':LM.calcLinPdfError(dssHcRslTapSet,type='reg'),'dssLckRge':LM.calcLinPdfError(dssHcRslTapLck,type='reg'),'dssTgtRge':LM.calcLinPdfError(dssHcRslTapTgt,type='reg'),'dssRge':dssRge}
        
        rslt = {'linHcRsl':linHcRsl,'linHcRslNom':linHcRslNom,'linHcRslNmc':linHcRslNmc,'dssHcRslTapSet':dssHcRslTapSet,'dssHcRslTapLck':dssHcRslTapLck,'dssHcRslTapTgt':dssHcRslTapTgt,'pdfData':pdf.pdf,'pdfDataNmc':pdfVal.pdf,'feeder':feeder,'maeVals':maeVals,'rgeVals':rgeVals,'preCndLeft':preCndLeft,'TC_No0':LM.TC_No0}
        if pltSave:
            SN = os.path.join(SD,'linHcCalcsRslt_'+pdfName+'_tapSet.pkl')
            with open(SN,'wb') as file:
                pickle.dump(rslt,file)
    
    if 'mcTapMultSet' in locals():
        Mu,Sgm = pdf.getMuStd(LM,0)
        LM.busViolationVar(Sgm)
        LM.makeCorrModel()
        
        nMult = 5 # This version should work up to 40 or so
        
        multResults = {}
        ibResults = {}
        svtyResults = np.array([])
        maeSet = np.array([])
        for ii in range(nMult):
            pdfMult = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,WD=WD,nMc=nMc,rndSeed=int(ii*1e8),prms=prms )
            LM.runLinHc(pdfMult,model='cor',fast=False)
            linHcRsl = LM.linHcRsl
            print('Linear Run Time:',linHcRsl['runTime'])
            print('Sampling Time:',linHcRsl['runTimeSample'])
            
            # NB: TIGHT running is required! (regBand=1.0)
            LM.runDssHc(pdf,DSSObj,genNames,BB0,SS0,regBand=1.0,setCapsModel=setCapsOpt,runType='tapSet',tapPosStart=linHcRsl['tapPosSeq'])
            dssHcRslTapSet = LM.dssHcRsl
            multResults[ii] = {'dss':dssHcRslTapSet,'lin':linHcRsl}
            ibResults[ii] = {'dss':np.sum(dssHcRslTapSet['inBds'],axis=2).flatten()/nMc,
                             'lin':np.sum(linHcRsl['inBds'],axis=2).flatten()/nMc}
            
            svtyResults = np.r_[svtyResults,getResultSensitivity(linHcRsl)]
            maeSet = np.r_[ maeSet, np.sum( np.abs(ibResults[ii]['lin'] - ibResults[ii]['dss']) )/pdfMult.pdf['nP'][0] ]
        
        for ii in range(nMult):
            print( 'MAE ',ii,':',maeSet[ii] )
            print( 'svtyResults ',ii,':',svtyResults[ii] )
            # plt.plot(pdf.pdf['prms'],ibResults[ii]['dss'],label='dss')
            # plt.plot(pdf.pdf['prms'],ibResults[ii]['lin'],label='lin');
            # plt.legend(); plt.show()

        rslt = {'ibResults':ibResults,'multResults':multResults,'maeSet':maeSet,'svtyResults':svtyResults}
        if pltSave:
            SN = os.path.join(os.path.dirname(SD),'TapMultSet',LM.feeder+'linHcCalcsRslt_'+pdfName+'_tapMultSet.pkl')
            with open(SN,'wb') as file:
                pickle.dump(rslt,file)
        
    
    if mcLinSns:
        regs0 = LMsns.regVreg0
        LMsns.updateDcpleModel(regs0*1.00625)
        LMsns.runLinHc(pdf)
        rsl0 = LMsns.linHcRsl
        
        LMsns.updateDcpleModel(regs0/1.00625)
        LMsns.runLinHc(pdf)
        rsl1 = LMsns.linHcRsl
        
        snsRegMaes = [LM.calcLinPdfError(rsl0)]
        snsRegMaes.append(LM.calcLinPdfError(rsl1))
        print('\n -------- Complete -------- ')
        
        rslt = {'linHcRsl':LM.linHcRsl,'linHcSns0':rsl0,'linHcSns1':rsl1,'pdf':pdf.pdf,'feeder':feeder,'regMaes':snsRegMaes}
        if pltSave:
            SN = os.path.join(SD,'linHcCalcsSns_'+pdfName+'_new.pkl')
            with open(SN,'wb') as file:
                pickle.dump(rslt,file)
        
    if mcLinVal:
        LMval.runLinHc(pdfVal) # equivalent at the moment
        valRegErrors = LM.calcLinPdfError(LMval.linHcRsl)
        print('\n -------- Complete -------- ')
        
        rslt = {'linHcRsl':LM.linHcRsl,'linHcVal':LMval.linHcRsl,'pdfLin':pdf.pdf,'pdfVal':pdfVal.pdf,'feeder':feeder,'regError':valRegErrors}
        if pltSave:
            SN = os.path.join(SD,'linHcCalcsVal_'+pdfName+str(nMc)+'_new.pkl')
            with open(SN,'wb') as file:
                pickle.dump(rslt,file)
        
    if 'mcLinUpg' in locals():
        LM.runLinHc(pdf)
        linHcRslBef = LM.linHcRsl
        
        LMupg.runLinHc(pdf)
        linHcRslAft = LMupg.linHcRsl
        
        rslt = {'linHcRslBef':linHcRslBef,'linHcRslAft':linHcRslAft,'pdfData':pdf.pdf,'feeder':feeder,'mae':LM.calcLinPdfError(linHcRslBef),'rge':LM.calcLinPdfError(linHcRslBef,type='reg')}
        if pltSave:
            # SN = os.path.join(SD,'linHcCalcsUpg.pkl')
            SN = os.path.join(SD,'linHcCalcsUpgNom.pkl')
            with open(SN,'wb') as file:
                pickle.dump(rslt,file)
        
    if 'mcLinPrg' in locals():
        LM.runLinHc(pdf)
        linHcRsl = LM.linHcRsl
        
        LMupg.runLinHc(pdf)
        linHcUpg = LMupg.linHcRsl
        
        
        LMupg.runLinLp(pdf,tmax=tmax,qmax=pf2kq(abs(upgPf)))
        linLpRslTQ = LMupg.linLpRsl
        LMupg.runLinLp(pdf,tmax=tmax,qmax=0)
        linLpRslT0 = LMupg.linLpRsl 
        LMupg.runLinLp(pdf,tmax=0,qmax=0)
        linLpRsl00 = LMupg.linLpRsl
        
        rslt = {'linHcRsl':linHcRsl,'linHcUpg':linHcUpg,'linLpRslTQ':linLpRslTQ,'linLpRslT0':linLpRslT0,'linLpRsl00':linLpRsl00,'pdfData':pdf.pdf,'feeder':feeder}
        if pltSave:
            SN = os.path.join(SD,'linHcPrg.pkl')
            with open(SN,'wb') as file:
                pickle.dump(rslt,file)
        
    
    
    # ================ PLOTTING FUNCTIONS FROM HERE
    if 'pltCns' in locals():
        fig, ax = plt.subplots()
        if mcDssOn:
            ax = plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.dssHcRsl['Cns_pct'],feeder=feeder,lineStyle='-',ax=ax,pltShow=False)
        if mcLinVal:
            ax = plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LMval.linHcRsl['Cns_pct'],feeder=feeder,lineStyle=':',ax=ax,pltShow=False)
        if mcLinSns:
            ax = plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LMsns.linHcRsl['Cns_pct'],feeder=feeder,lineStyle=':',ax=ax,pltShow=False)
        if 'mcDssPar' in locals():
            ax = plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],rsltSeq['Cns_pct'],feeder=feeder,lineStyle=':',ax=ax,pltShow=False)
            ax = plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],rsltPar['Cns_pct'],feeder=feeder,lineStyle='-.',ax=ax,pltShow=False)

        ax = plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=feeder,lineStyle='--',ax=ax,pltShow=False)

        if 'plotShow' in locals():
            plt.show()
    if 'pltPwrCdf' in locals():
        ax = plotPwrCdf(LM.linHcRsl['pp'],LM.linHcRsl['ppPdf'],lineStyle='--',pltShow=False)
        if mcDssOn:
            ax = plotPwrCdf(LM.linHcRsl['pp'],LM.dssHcRsl['ppPdf'],ax=ax,lineStyle='-',pltShow=False)
            ax.legend(('Lin model','OpenDSS'))
        plt.xlabel('Power');
        plt.ylabel('P(.)');
        ax.grid(True)
        
        if 'plotShow' in locals():
            plt.show()
    if 'pltHcVltn' in locals():
        fig = plt.subplot()
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        # if mcDssOn:
        plotHcVltn(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.dssHcRsl['Vp_pct'],ax=ax1,pltShow=False,feeder=feeder,logScale=True)
        plotHcVltn(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.dssHcRsl['Vp_pct'],ax=ax2,pltShow=False,feeder=feeder,logScale=False)
        
        plotHcVltn(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Vp_pct'],ax=ax1,pltShow=False,feeder=feeder,logScale=True)
        plotHcVltn(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Vp_pct'],ax=ax2,pltShow=False,feeder=feeder,logScale=False)
        if mcDssOn:
            ax1.legend(('Vltn., DSS','Vltn., Nom'))
        plt.tight_layout()
        if 'plotShow' in locals():
            plt.show()
    if 'pltHcGen' in locals():
        ax = plotHcGen(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['genTotSet'][:,:,0::2],'k')
        ax = plotHcGen(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['hcGenSet'][:,:,0::2],'r',ax=ax)
        if mcDssOn:
            ax = plotHcGen(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.dssHcRsl['hcGenSet'][:,:,0::2],'g',ax=ax)
        xlm = ax.get_xlim()
        ax.set_xlim((0,xlm[1]))
        plt.xlabel('Scale factor');
        plt.title('Hosting Capacity (kW)');
        plt.grid(True)
        if 'plotShow' in locals():
            plt.show()


# # 1. seeing how the tap positions change for the feeder
# plt.scatter((linHcRslNom['tapPosSeq'] + LM.TC_No0).flatten(),dssHcRslTapSet['tapPosSet'].flatten())
# plt.plot((-10,10),(-10,10),'k')
# plt.show()

# # # 2. Seeing how out of bandwidth the regulators are at the specified nominal tap positions
# plt.scatter(dssHcRslTapLck['regVI'][:,:,0].flatten(),dssHcRslTapLck['regVI'][:,:,1].flatten())
# plt.scatter(dssHcRslTapSet['regVI'][:,:,0].flatten(),dssHcRslTapSet['regVI'][:,:,1].flatten())
# plt.plot((-1,1,1,-1,-1),(-1,-1,1,1,-1),'k')
# plt.grid(True)
# plt.show()

# # 3. Plotting the sensitivity to tap position
# tapPosSns = linHcRslNom['tapPosSns']
# tapMinLo = np.min(tapPosSns[:,:,1,:],axis=2)
# prms = np.linspace(100/tapMinLo.shape[0],100,tapMinLo.shape[0])
# fig,ax = plt.subplots()
# jj = 0
# for tapMin in tapMinLo[::2]:
    # pctls = np.percentile(tapMin,[5,25,50,75,95])
    # rngs = np.percentile(tapMin,[0,100])
    # plotBoxWhisk(ax,prms[jj],1,pctls,bds=rngs)
    # jj+=2

# xlm = ax.get_xlim()
# ax.plot(xlm,(2,2),'k--')
# ax.set_xlim(xlm)
# ax.set_xlabel('Fraction of Loads with PV')
# ax.set_ylabel('No. taps to upper voltage constraint')
# ax.set_ylim((-3.5,4.5))
# plt.show()
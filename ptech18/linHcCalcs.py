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
import numpy.random as rnd
import matplotlib.pyplot as plt
from math import gamma
import dss_stats_funcs as dsf
from linSvdCalcs import hcPdfs, linModel, cnsBdsCalc, plotCns, plotHcVltn, plotPwrCdf, plotHcGen
from matplotlib import cm

mcLinOn = True
# mcLinOn = False
mcLinVal = True
mcLinVal = False
mcLinSns = True
mcLinSns = False
mcDssOn = True
# mcDssOn = False

# PLOTTING options:
pltHcVltn = True
pltHcVltn = False
pltHcGen = True
pltHcGen = False
pltPwrCdf = True
pltPwrCdf = False
pltCns = True
pltCns = False

nMc = 50 # nominal value
# nMc = 300

pltSave = True # for saving both plots and results
# pltSave = False

plotShow=True
plotShow=False

# CHOOSE Network
fdr_i_set = [5,6,8,9,0,14,17,18,22,19,20,21]
# fdr_i_set = [5,6,8,0] # fast
# fdr_i_set = [14,17,18] # medium length 1
# fdr_i_set = [20,21] # medium length 2
# fdr_i_set = [9] # slow 1
# fdr_i_set = [22] # slow 2
fdr_i_set = [19] # slow 3
# fdr_i_set = [18,9]

pdfName = 'gammaWght'
pdfName = 'gammaFrac'
# pdfName = 'gammaFrac'; prms=np.arange(0.05,1.05,0.05) # to make it a bit faster

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24']

# opendss with 'early bindings'
WD = os.path.dirname(sys.argv[0])
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

    # # PART A.1 - load models ===========================
    LM = linModel(fdr_i,WD)
    LM.loadNetModel()
    YZp = LM.SyYNodeOrderTot
    YZd = LM.SdYNodeOrderTot
    pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,WD=WD,nMc=nMc,rndSeed=0 ) # use
    
    if mcLinVal:
        LMval = linModel(fdr_i,WD)
        LMval.loadNetModel()
        pdfVal = hcPdfs(LMval.feeder,netModel=LMval.netModelNom,pdfName=pdfName,WD=WD,nMc=nMc,rndSeed=2**31 )
    
    if mcLinSns:
        LMsns = linModel(fdr_i,WD)
        LMsns.loadNetModel()
        newRegs = LMsns.regVreg0*(1+(np.array(LMsns.regBandwidth)*0.5/120.0))
        LMsns.updateDcpleModel(newRegs)
        
    
    # ADMIN =============================================
    SD = os.path.join(WD,'hcResults',feeder)

    fn = get_ckt(WD,feeder)[1]    
    print('\n\nStart '+'='*30,'\nFeeder:',feeder,'\nLinpoint:',lp0data['k'],'\nLoad Point:',lp0data['kLo'],'\nTap Model:',LM.netModelNom)
    
    # OPENDSS ADMIN =======================================
    # B1. load the appropriate model/DSS
    DSSText.Command='Compile ('+fn+'.dss)'
    BB0,SS0 = cpf_get_loads(DSSCircuit)
    
    cpf_set_loads(DSSCircuit,BB0,SS0,lp0data['kLo'])
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
    if mcLinOn:
        LM.runLinHc(pdf) # equivalent at the moment
        
    if mcLinSns:
        LMsns.runLinHc(pdf)
        print('\n--- Linear results 1 ---\n\nP0:',LM.linHcRsl['ppCdf'][0],'\nP10:',LM.linHcRsl['ppCdf'][2],'\nk0:',LM.linHcRsl['kCdf'][0],'\nk10:',LM.linHcRsl['kCdf'][2])
        print('\n--- Linear results 2 ---\n\nP0:',LMsns.linHcRsl['ppCdf'][0],'\nP10:',LMsns.linHcRsl['ppCdf'][2],'\nk0:',LMsns.linHcRsl['kCdf'][0],'\nk10:',LMsns.linHcRsl['kCdf'][2])
        if not os.path.exists(SD):
            os.makedirs(SD)
        rslt = {'linHcRsl':LM.linHcRsl,'linHcSns':LMsns.linHcRsl,'reg0':LM.regVreg0,'newReg':newRegs,'pdf':pdf.pdf,'feeder':feeder}
        if pltSave:
            SN = os.path.join(SD,'linHcCalcsSns_'+pdfName+'.pkl')
            with open(SN,'wb') as file:
                pickle.dump(rslt,file)
        
    if mcLinVal:
        LMval.runLinHc(pdfVal) # equivalent at the moment
        print('\n--- Linear results 1 ---\n\nP0:',LM.linHcRsl['ppCdf'][0],'\nP10:',LM.linHcRsl['ppCdf'][2],'\nk0:',LM.linHcRsl['kCdf'][0],'\nk10:',LM.linHcRsl['kCdf'][2])
        print('\n--- Linear results 2 ---\n\nP0:',LMval.linHcRsl['ppCdf'][0],'\nP10:',LMval.linHcRsl['ppCdf'][2],'\nk0:',LMval.linHcRsl['kCdf'][0],'\nk10:',LMval.linHcRsl['kCdf'][2])
        if not os.path.exists(SD):
            os.makedirs(SD)
        rslt = {'linHcRsl':LM.linHcRsl,'linHcVal':LMval.linHcRsl,'pdfLin':pdf.pdf,'pdfVal':pdfVal.pdf,'feeder':feeder}
        if pltSave:
            SN = os.path.join(SD,'linHcCalcsVal_'+pdfName+str(nMc)+'.pkl')
            with open(SN,'wb') as file:
                pickle.dump(rslt,file)
        
        
    if mcDssOn:
        regBand=0
        # LM.runDssHc(pdf,DSSObj,genNames,BB0,SS0,regBand=1) # old results
        LM.runDssHc(pdf,DSSObj,genNames,BB0,SS0,regBand=regBand)
        
        print('\n--- Linear results ---\n\nP0:',LM.linHcRsl['ppCdf'][0],'\nP10:',LM.linHcRsl['ppCdf'][2],'\nk0:',LM.linHcRsl['kCdf'][0],'\nk10:',LM.linHcRsl['kCdf'][2])
        print('\n--- OpenDSS results ---\n\nP0:',LM.dssHcRsl['ppCdf'][0],'\nP10:',LM.dssHcRsl['ppCdf'][2],'\nk0:',LM.dssHcRsl['kCdf'][0],'\nk10:',LM.dssHcRsl['kCdf'][2])
        
        if not os.path.exists(SD):
            os.makedirs(SD)
        rslt = {'dssHcRsl':LM.dssHcRsl,'linHcRsl':LM.linHcRsl,'pdfData':pdf.pdf,'feeder':feeder}
        if pltSave:
            SN = os.path.join(SD,'linHcCalcsRslt_'+pdfName+'_reg'+str(regBand)+'.pkl')
            with open(SN,'wb') as file:
                pickle.dump(rslt,file)
    
    
    
    # ================ PLOTTING FUNCTIONS FROM HERE
    if pltCns:
        fig, ax = plt.subplots()
        if mcDssOn:
            ax = plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.dssHcRsl['Cns_pct'],feeder=feeder,lineStyle='-',ax=ax,pltShow=False)
        if mcLinVal:
            ax = plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LMval.linHcRsl['Cns_pct'],feeder=feeder,lineStyle=':',ax=ax,pltShow=False)
        if mcLinSns:
            ax = plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LMsns.linHcRsl['Cns_pct'],feeder=feeder,lineStyle=':',ax=ax,pltShow=False)
        
        ax = plotCns(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Cns_pct'],feeder=feeder,lineStyle='--',ax=ax,pltShow=False)
        if mcDssOn and pltSave:
            plt.savefig(os.path.join(SD,'pltCns_'+pdfName+'.png'))
        
        if plotShow:
            plt.show()
    if pltPwrCdf:
        ax = plotPwrCdf(LM.linHcRsl['pp'],LM.linHcRsl['ppPdf'],lineStyle='--',pltShow=False)
        if mcDssOn:
            ax = plotPwrCdf(LM.linHcRsl['pp'],LM.dssHcRsl['ppPdf'],ax=ax,lineStyle='-',pltShow=False)
            ax.legend(('Lin model','OpenDSS'))
        plt.xlabel('Power');
        plt.ylabel('P(.)');
        ax.grid(True)
        if mcDssOn and pltSave:
            plt.savefig(os.path.join(SD,'pltPwrCdf_'+pdfName+'.png'))
        
        if plotShow:
            plt.show()
    if pltHcVltn:
        fig = plt.subplot()
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        if mcDssOn:
            plotHcVltn(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.dssHcRsl['Vp_pct'],ax=ax1,pltShow=False,feeder=feeder,logScale=True)
            plotHcVltn(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.dssHcRsl['Vp_pct'],ax=ax2,pltShow=False,feeder=feeder,logScale=False)
        
        plotHcVltn(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Vp_pct'],ax=ax1,pltShow=False,feeder=feeder,logScale=True)
        plotHcVltn(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['Vp_pct'],ax=ax2,pltShow=False,feeder=feeder,logScale=False)
        if mcDssOn:
            ax1.legend(('Vltn., DSS','Vltn., Nom'))
        
        plt.tight_layout()
        if mcDssOn and pltSave:
            plt.savefig(os.path.join(SD,'pltHcVltn_'+pdfName+'.png'))
        
        if plotShow:
            plt.show()
    if pltHcGen:
        ax = plotHcGen(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['genTotSet'][:,:,0::2],'k')
        ax = plotHcGen(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.linHcRsl['hcGenSet'][:,:,0::2],'r',ax=ax)
        if mcDssOn:
            ax = plotHcGen(pdf.pdf['mu_k'],pdf.pdf['prms'],LM.dssHcRsl['hcGenSet'][:,:,0::2],'g',ax=ax)
        xlm = ax.get_xlim()
        ax.set_xlim((0,xlm[1]))
        plt.xlabel('Scale factor');
        plt.title('Hosting Capacity (kW)');
        plt.grid(True)
        if mcDssOn and pltSave:
            plt.savefig(os.path.join(SD,'pltHcGen_'+pdfName+'.png'))
        if plotShow:
            plt.show()
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

plotShow=True
plotShow=False

# CHOOSE Network
# fdr_i_set = [5,6,8,9,0,14,17,18,22,19,20,21]
# fdr_i_set = [9,17,18,19,20,21,22]
fdr_i_set = [5,6,8,0] # fastest few
# fdr_i_set = [14,17,18] # medium length 1
# fdr_i_set = [20,21] # medium length 2
# fdr_i_set = [9]
# fdr_i_set = [22]

pdfName = 'gammaWght'
# pdfName = 'gammaFrac'
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

    LM = linModel(fdr_i,WD)
    netModel = LM.netModelNom
    pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName=pdfName,WD=WD ) # use
    
    pdfData = pdf.pdf
    # ADMIN =============================================
    SD = os.path.join(WD,'hcResults',feeder)
    SN = os.path.join(SD,'linHcCalcsRslt_'+pdfName+'.pkl')

    fn = get_ckt(WD,feeder)[1]    
    print('\n\nStart '+'='*30,'\nFeeder:',feeder,'\nLinpoint:',lp0data['k'],'\nLoad Point:',lp0data['kLo'],'\nTap Model:',netModel)

    # # PART A.1 - load models ===========================
    LM.loadNetModel()
    v_idx = LM.v_idx_tot
    YZp = LM.SyYNodeOrderTot
    YZd = LM.SdYNodeOrderTot

    # OPENDSS ADMIN =======================================
    # B1. load the appropriate model/DSS
    DSSText.Command='Compile ('+fn+'.dss)'
    BB0,SS0 = cpf_get_loads(DSSCircuit)
    
    cpf_set_loads(DSSCircuit,BB0,SS0,lp0data['kLo'])
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
    # PART A.2 - choose distributions and reduce linear model ===========================
    if mcLinOn:
        LM.runLinHc(pdf) # equivalent at the moment
        print('\n--- Linear results ---\n\nP0:',LM.linHcRsl['ppCdf'][0],'\nP10:',LM.linHcRsl['ppCdf'][2],'\nk0:',LM.linHcRsl['kCdf'][0],'\nk10:',LM.linHcRsl['kCdf'][2])
        
    if mcDssOn:
        LM.runDssHc(pdf,DSSObj,genNames,BB0,SS0)
        
        print('\n--- Linear results ---\n\nP0:',LM.linHcRsl['ppCdf'][0],'\nP10:',LM.linHcRsl['ppCdf'][2],'\nk0:',LM.linHcRsl['kCdf'][0],'\nk10:',LM.linHcRsl['kCdf'][2])
        print('\n--- OpenDSS results ---\n\nP0:',LM.dssHcRsl['ppCdf'][0],'\nP10:',LM.dssHcRsl['ppCdf'][2],'\nk0:',LM.dssHcRsl['kCdf'][0],'\nk10:',LM.dssHcRsl['kCdf'][2])
        
        if not os.path.exists(SD):
            os.makedirs(SD)
        rslt = {'dssHcRsl':LM.dssHcRsl,'linHcRsl':LM.linHcRsl,'pdfData':pdfData,'feeder':feeder}
        if pltSave:
            with open(SN,'wb') as file:
                pickle.dump(rslt,file)
    
    
    
    # ================ PLOTTING FUNCTIONS FROM HERE
    if pltCns:
        fig, ax = plt.subplots()
        if mcDssOn:
            ax = plotCns(pdfData['mu_k'],pdfData['prms'],LM.dssHcRsl['Cns_pct'],feeder=feeder,lineStyle='-',ax=ax,pltShow=False)
        
        ax = plotCns(pdfData['mu_k'],pdfData['prms'],LM.linHcRsl['Cns_pct'],feeder=feeder,lineStyle='--',ax=ax,pltShow=False)
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
            plotHcVltn(pdfData['mu_k'],pdfData['prms'],LM.dssHcRsl['Vp_pct'],ax=ax1,pltShow=False,feeder=feeder,logScale=True)
            plotHcVltn(pdfData['mu_k'],pdfData['prms'],LM.dssHcRsl['Vp_pct'],ax=ax2,pltShow=False,feeder=feeder,logScale=False)
        
        plotHcVltn(pdfData['mu_k'],pdfData['prms'],LM.linHcRsl['Vp_pct'],ax=ax1,pltShow=False,feeder=feeder,logScale=True)
        plotHcVltn(pdfData['mu_k'],pdfData['prms'],LM.linHcRsl['Vp_pct'],ax=ax2,pltShow=False,feeder=feeder,logScale=False)
        if mcDssOn:
            ax1.legend(('Vltn., DSS','Vltn., Nom'))
        
        plt.tight_layout()
        if mcDssOn and pltSave:
            plt.savefig(os.path.join(SD,'pltHcVltn_'+pdfName+'.png'))
        
        if plotShow:
            plt.show()
    if pltHcGen:
        ax = plotHcGen(pdfData['mu_k'],pdfData['prms'],LM.linHcRsl['genTotSet'][:,:,0::2],'k')
        ax = plotHcGen(pdfData['mu_k'],pdfData['prms'],LM.linHcRsl['hcGenSet'][:,:,0::2],'r',ax=ax)
        if mcDssOn:
            ax = plotHcGen(pdfData['mu_k'],pdfData['prms'],LM.dssHcRsl['hcGenSet'][:,:,0::2],'g',ax=ax)
        xlm = ax.get_xlim()
        ax.set_xlim((0,xlm[1]))
        plt.xlabel('Scale factor');
        plt.title('Hosting Capacity (kW)');
        plt.grid(True)
        if mcDssOn and pltSave:
            plt.savefig(os.path.join(SD,'pltHcGen_'+pdfName+'.png'))
        if plotShow:
            plt.show()
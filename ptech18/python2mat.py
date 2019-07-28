# This function will (i) read in the appropriate python files then save as a .mat file for 
# direct import into matlab for further analysis.
import numpy as np
import scipy.io as sio
import os, sys
import matplotlib.pyplot as plt
from linSvdCalcs import hcPdfs, linModel, cnsBdsCalc, plotCns, plotHcVltn, plotPwrCdf, plotHcGen, plotBoxWhisk

setCapsOpt = 'linModel' # opendss options. 'linModels' is the 'right' option, cf True/False
WD = os.path.dirname(sys.argv[0])

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr','123busCvr']
fdr_i = 6
fdr_iSet = [0]

for fdr_i in fdr_iSet:
    LM = linModel(fdr_i,WD,setCapsModel=setCapsOpt)
    print(fdr_i)
    pdf = hcPdfs(LM.feeder,netModel=LM.netModelNom,pdfName='gammaFrac',WD=WD ) # use
    Mu,Sgm = pdf.getMuStd(LM,0)
    LM.busViolationVar(np.ones(LM.nS))

    if fdr_i==0:
        LM.makeCorrModel(stdLim=0.9999,corrLim=[0.9999])
    else:
        LM.makeCorrModel(stdLim=0.99,corrLim=[0.99])

    # copied from runLinHc
    nV = LM.KtotPu.shape[0]
    vars = LM.varKfullU.copy()
    varSortN = vars.argsort()[::-1]
    NSet = np.array(varSortN[LM.NSetCor[0]])
    NSetTot = NSet[NSet<nV]
    NSetFix = NSet[NSet>=nV] - nV
    NSetAll = np.arange(nV)

    NSet = np.unique(np.r_[NSetFix,NSetTot])

    My = np.c_[LM.KtotPu,np.zeros(LM.KtotPu.shape)][NSet] # in pu per (k?)W
    V0 = LM.b0ls[0:3]
    a = (LM.b0ls[NSet]).reshape(My.shape[0],1)
    xhy0 = (1e-16*(LM.xNomTot + 1e-15)).reshape(My.shape[1],1)
    xhy0nom = LM.xNomTot
    lin_point = LM.linPoint

    mdict = {'My':My,'V0':V0,'a':a,'lin_point':lin_point,'xhy0':xhy0,'xhy0nom':xhy0nom}

    fileName = os.path.join(WD,'lin_models',fdrs[fdr_i]+'.mat')
    sio.savemat(fileName,mdict)
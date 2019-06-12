import lineariseDssModels, sys, os, pickle, random
from importlib import reload
import numpy as np
from dss_python_funcs import vecSlc, getBusCoords, getBusCoordsAug, tp_2_ar
import matplotlib.pyplot as plt

from scipy.linalg import toeplitz

FD = sys.argv[0]

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr']


def main(fdr_i=5,linPoint=1.0,pCvr=0.8,method='fpl',saveModel=False,modelType=None,pltSave=False):
    reload(lineariseDssModels)
    
    SD = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=fdr_i,linPoints=[linPoint],pCvr=pCvr,
                                                modelType=modelType,method=method,saveModel=saveModel,SD=SD,pltSave=pltSave)
    return blm

# self = main(5,modelType='plotOnly',pltSave=False)#
self = main(0)
self.showQpSln(self.slnX,self.slnF)
self.showQpSln(self.sln2X,self.sln2F)
self.showQpSln(self.sln3X,self.sln3F)

# linPoints = [1.0,0.2]
# aCvrs = [0.8,0.25]
# fdr_i_set = [5,6,8,24,0]
# for fdr_i in fdr_i_set:
    # for linPoint in linPoints:
        # for aCvr in aCvrs:
            # self = main(fdr_i,linPoint=linPoint,pCvr=aCvr)
            # self.saveLinModel()

# # self.setupConstraints(qlim=1e4)
# # self.runCvrQp('full')
# # self.plotNetBuses('qSln')
# # self.plotNetBuses('v0')
# # self.showQpSln(self.slnX,self.slnF)
# # self.snapQpComparison()

# # fdr_i = 24
# # loadPoint = '020'
# # aCvr = '025'
# # SN = os.path.join(os.path.dirname(FD),'lin_models','cvr_models',fdrs[fdr_i],fdrs[fdr_i]+'P'+loadPoint+'A'+aCvr+'.pkl')
# # with open(SN,'rb') as inFile:
    # # self = pickle.load(inFile)

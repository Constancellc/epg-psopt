import lineariseDssModels, sys, os, pickle, random
from importlib import reload
import numpy as np
from dss_python_funcs import vecSlc, getBusCoords, getBusCoordsAug, tp_2_ar
import matplotlib.pyplot as plt
from lineariseDssModels import dirPrint

FD = sys.argv[0]

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr']


def main(fdr_i=5,linPoint=1.0,pCvr=0.8,method='fpl',saveModel=False,modelType=None,pltSave=False):
    reload(lineariseDssModels)
    
    SD = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=fdr_i,linPoints=[linPoint],pCvr=pCvr,
                                                modelType=modelType,method=method,saveModel=saveModel,SD=SD,pltSave=pltSave)
    return blm

# # NETWORKS:
# self = main(8,modelType='loadModel')
# self = main(8,modelType='loadOnly')
self = main(8)
self.plotNetBuses('qSln')

self.printQpSln(self.slnX,self.slnF)
self.printQpSln(self.slnX,self.slnFdss)
self.printQpSln(np.zeros(self.nCtrl),self.slnF0)
self.printQpSln(np.zeros(self.nCtrl),self.slnF0dss)

# self.showQpSln(self.slnX,self.slnF)
# self.snapQpComparison()


# # # fdr_i = 24
# # # loadPoint = '020'
# # # aCvr = '025'
# # # SN = os.path.join(os.path.dirname(FD),'lin_models','cvr_models',fdrs[fdr_i],fdrs[fdr_i]+'P'+loadPoint+'A'+aCvr+'.pkl')
# # # with open(SN,'rb') as inFile:
    # # # self = pickle.load(inFile)


# cns = self.cns0
# # for feeder 'n1'
# cns['mvLo'] = 0.99
# cns['plim'] = 10000
# # for feeder 123bus
# cns['plim'] = 100000
# # for feeder epri24
# cns['mvHi'] = 1.10
# cns['mvLo'] = 0.92
# cns['lvHi'] = 1.10
# cns['lvLo'] = 0.92
# cns['iScale'] = 4.0

# # Case:
# # EU LV 
# # 1. Load HC
# self = main(8,modelType='loadModel')
# self.setupConstraints(cns)

# self.runCvrQp('genHostingCap',optType=['cvxopt'])
# self.plotNetBuses('qSln')
# self.printQpSln(self.slnX,self.slnF)
# self.showQpSln(self.slnX,self.slnF)

# self.runCvrQp('loadHostingCap',optType=['cvxopt'])
# self.runCvrQp('loadHostingCap',optType=['cvxMosek'])
# self.plotNetBuses('qSln')
# self.showQpSln(self.slnX,self.slnF)

# self.runCvrQp('full',optType=['cvxopt'])
# self.runCvrQp('full',optType=['mosekInt'])
# self.plotNetBuses('qSln')
# self.showQpSln(self.slnX,self.slnF)

# # 2. Gen HC


# # 3. Operating costs


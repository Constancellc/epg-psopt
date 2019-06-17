import lineariseDssModels, sys, os, pickle, random
from importlib import reload
import numpy as np
from dss_python_funcs import vecSlc, getBusCoords, getBusCoordsAug, tp_2_ar
import matplotlib.pyplot as plt

FD = sys.argv[0]

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr']


# f_bulkBuildModels = 1


def main(fdr_i=5,linPoint=1.0,pCvr=0.8,method='fpl',saveModel=False,modelType=None,pltSave=False):
    reload(lineariseDssModels)
    
    SD = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=fdr_i,linPoints=[linPoint],pCvr=pCvr,
                                                modelType=modelType,method=method,saveModel=saveModel,SD=SD,pltSave=pltSave)
    return blm

self = main('n10',modelType='plotOnly',pltSave=False)

cns = self.cns0
cns['mvLo']=0.92
cns['mvLo']=0.98
self.setupConstraints(cns)
self.runCvrQp('full','mosekInt')

# STEP 1: saving the models.
if f_bulkBuildModels in locals():
    feederSet = [5,6,8,24,18,'n4','n1','n10','n27',17]
    linPoints = [0.2,0.6,1.0]
    pCvrSet = [0.2,0.8]
    for feeder in feederSet:
        for linPoint in linPoints:
            for pCvr in pCvrSet:
                main(feeder,pCvr=pCvr,modelType='buildSave',linPoint=linPoint)

self = main(5,pCvr=0.8,modelType='buildTestSave',linPoint=1.0)
# STEP 2: Running the models, obtaining the optimization results.
# feederSet = [5,6,8,24,18,'n4','n1','n10','n27',17]
# feederSet = [5,6]
feederSet = [5]

# self = main(0,modelType='loadAndRun')
# feederSet = [5,6,8,24,18,'n4','n1','n10','n27',17]
feederSet = [5,6,8,24,18,17]
# linPoints = [0.2,1.0]
# pCvrSet = [0.2,0.8]
linPoints = [0.6]
pCvrSet = [0.2,0.8]

for feeder in feederSet:
    print('============================================= Feeder:',feeder)
    for linPoint in linPoints:
        for pCvr in pCvrSet:
            self = main(feeder,pCvr=pCvr,modelType='loadAndRun',linPoint=linPoint)

feederSet = [5,6,8,24,18,'n4','n1','n10','n27',17]
for feeder in feederSet:
    for linPoint in linPoints:
        for pCvr in pCvrSet:
            self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint)
            self.loadQpSet()
            for key in self.qpSolutions:
                print(self.feeder + '; sln type ' + key+', linPoint,' +str(linPoint)+' : ',self.qpSolutions[key][2])
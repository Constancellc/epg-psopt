import lineariseDssModels, sys, os, pickle, random, time
from importlib import reload
import numpy as np
from dss_python_funcs import vecSlc, getBusCoords, getBusCoordsAug, tp_2_ar
import matplotlib.pyplot as plt

FD = sys.argv[0]

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr']


# f_bulkBuildModels = 1
# f_bulkRunModels = 1
# f_checkFeasibility = 1
# f_checkError = 1
# f_valueComparison = 1

def main(fdr_i=5,modelType=None,linPoint=1.0,pCvr=0.8,method='fpl',saveModel=False,pltSave=False):
    reload(lineariseDssModels)
    
    SD = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=fdr_i,linPoints=[linPoint],pCvr=pCvr,
                                                modelType=modelType,method=method,saveModel=saveModel,SD=SD,pltSave=pltSave)
    return blm


self = main(0,'loadOnly',linPoint=1.0); self.loadQpSet(); self.loadQpSln('full','opCst'); self.plotNetBuses('qSlnPh',minMax=[-1.0,1.0])
self = main(0,'loadOnly',linPoint=0.1); self.loadQpSet(); self.loadQpSln('full','hcGen'); self.plotNetBuses('qSlnPh',minMax=[-1.0,1.0])
self = main(8,'loadOnly',linPoint=1.0); self.loadQpSet(); self.loadQpSln('full','opCst'); self.plotNetBuses('qSlnPh',minMax=[-1.0,1.0])
self = main(8,'loadOnly',linPoint=0.1); self.loadQpSet(); self.loadQpSln('full','hcGen'); self.plotNetBuses('qSlnPh',minMax=[-1.0,1.0])


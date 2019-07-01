import lineariseDssModels, sys, os, pickle, random, time
from importlib import reload
import numpy as np
from dss_python_funcs import vecSlc, getBusCoords, getBusCoordsAug, tp_2_ar, runCircuit
import matplotlib.pyplot as plt
from lineariseDssModels import dirPrint
import dss_stats_funcs as dsf


from scipy import sparse

FD = sys.argv[0]

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr']


def main(fdr_i=5,modelType=None,linPoint=1.0,pCvr=0.8,method='fpl',saveModel=False,pltSave=False):
    reload(lineariseDssModels)
    
    SD = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=fdr_i,linPoints=[linPoint],pCvr=pCvr,
                                                modelType=modelType,method=method,saveModel=saveModel,SD=SD,pltSave=pltSave)
    return blm


# self = main(8,'buildSave')

# self = main(0,modelType='loadOnly'); self.runCvrQp('full','opCst'); self.showQpSln()

# self = main(5,modelType='loadOnly');
# self.loadQpSet(); self.loadQpSln(strategy,obj); self.showQpSln()

# self.setQlossOfs(lossFact=0.05,kQlossQ=0.5,kQlossL=0.0,kQlossC=0.5,qlossRegC=0.0); self.runCvrQp('full','opCst'); self.showQpSln()

# print(sum(self.runQp(self.slnX,True)[:4]))
# print(sum(self.runQp(self.slnX,False)[:4]))

# self = main('n1',modelType='loadOnly'); self.loadQpSet(); self.loadQpSln('full','opCst'); self.showQpSln()
# self.setQlossOfs(0.05,0.5,0.0);self.runCvrQp('full','opCst'); self.showQpSln()

# self = main(0,modelType='loadOnly'); self.runCvrQp('full','opCst'); self.showQpSln()

# self = main(0,modelType='loadOnly'); self.runCvrQp('full','opCst'); self.showQpSln()
# self = main(0,modelType='loadOnly'); self.setQlossOfs(lossFact=0.02,kQlossL=0.0); self.runCvrQp('full','opCst'); self.showQpSln()

# self = main(17,'loadOnly'); self.initialiseOpenDss(); self.testQpVcpf(); plt.show()

feeder = 6
obj = 'hcLds'
strategy = 'full'
pCvr = 0.6
linPoint = 0.6
self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint); # self.initialiseOpenDss();
self.loadQpSet(); self.loadQpSln(strategy,obj); self.showQpSln(); 

self.plotNetBuses('qSlnPh')

# self.testGenSetting(k=np.arange(-10,11,2),dPlim=0.10,dQlim=0.10); plt.show()
# self.slnD = self.qpDssValidation(method='relaxT')
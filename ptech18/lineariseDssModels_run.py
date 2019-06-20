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


self = main(5)


# feeder = 17
# obj = 'hcLds'
# strategy = 'part'
# pCvr = 0.8
# self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPointsDict[feeder][obj][-1])
# self.Mc2i = dsf.mvM( np.concatenate( (self.WyXfmr[:,:self.nPy],self.WdXfmr[:,:self.nPd],
                                    # self.WyXfmr[:,self.nPy::],self.WdXfmr[:,self.nPd::],
                                    # self.WtXfmr),axis=1), 1/self.xScale ) # limits for these are in self.iXfmrLims.
# self.initialiseOpenDss();
# self.testGenSetting(k=np.arange(-10,11,2),dPlim=0.10,dQlim=0.10); plt.show()

# self.loadQpSet()
# self.loadQpSln(strategy,obj)
# self.showQpSln()
# self.plotNetBuses('qSln')

# self.showQpSln()
# self.slnD = self.qpDssValidation(method='relaxT')
# self.showQpSln()



# feeder = 'n10'
# obj = 'hcLds'
# strategy = 'full'
# self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPointsDict[feeder][2])
# self.loadQpSet()
# self.loadQpSln(strategy,obj)

# self.showQpSln()


# # for feeder epri24
# cns['mvHi'] = 1.10
# cns['mvLo'] = 0.92
# cns['lvHi'] = 1.10
# cns['lvLo'] = 0.92
# cns['iScale'] = 4.0
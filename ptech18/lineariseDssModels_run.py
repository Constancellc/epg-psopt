import lineariseDssModels, sys, os, pickle, random
from importlib import reload
import numpy as np
from dss_python_funcs import vecSlc, getBusCoords, getBusCoordsAug, tp_2_ar
import matplotlib.pyplot as plt
from lineariseDssModels import dirPrint

FD = sys.argv[0]

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr']


def main(fdr_i=5,modelType=None,linPoint=1.0,pCvr=0.8,method='fpl',saveModel=False,pltSave=False):
    reload(lineariseDssModels)
    
    SD = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=fdr_i,linPoints=[linPoint],pCvr=pCvr,
                                                modelType=modelType,method=method,saveModel=saveModel,SD=SD,pltSave=pltSave)
    return blm


# self = main(5,modelType='loadModel')
# self = main('n10',modelType='loadModel',linPoint=1.8)
# self = main('n10',modelType='loadModel')
# self = main(5,modelType='loadModel')
# self = main(0,modelType='loadModel',linPoint=0.1)

# mc2iIi = self.Mc2i[ii].dot(oneHat)
# plt.plot(mc2iIi.real); plt.plot(mc2iIi.imag); plt.show()
# np.r_[ [mc2iIi.real],[mc2iIi.imag]].dot(oneHat)

# self = main(0,'buildSave',linPoint=0.1)
# self = main(0,linPoint=0.1); self.printQpSln()
self = main(0,linPoint=0.1)

# self = main(5,'buildSave',linPoint=1.0)
# self = main(5); self.printQpSln()

# self = main(5,'buildSave',linPoint=0.1)

# self.initialiseOpenDss(); self.testCvrQp()


# self = main(0,linPoint=0.1)

# feeder = 0
# obj = 'opCst'
# strategy = 'full'
# pCvr = 0.8
# linPoint = 0.1
# self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint); # self.initialiseOpenDss();
# self.loadQpSet(); self.loadQpSln(strategy,obj); self.showQpSln()

# self.loadQpSet(); self.loadQpSln(strategy,'opCst')
# self.showQpSln()

# self = main(8,linPoint=0.6,modelType='loadModel')
# self = main(8,linPoint=1.0,modelType='loadModel')


# self.testGenSetting(k=np.arange(-10,11,2),dPlim=0.10,dQlim=0.10); plt.show() # <--- here the current limits are still shite!

# self.showQpSln()
# self.slnD = self.qpDssValidation(method='relaxT')
# self.showQpSln()


# self.testGenSetting(k=np.arange(-10,11,2),dPlim=0.10,dQlim=0.10); plt.show()

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
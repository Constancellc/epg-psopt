import lineariseDssModels, sys, os, pickle, random
from importlib import reload
import numpy as np
from dss_python_funcs import vecSlc, getBusCoords, getBusCoordsAug, tp_2_ar
import matplotlib.pyplot as plt

FD = sys.argv[0]

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr']


# f_bulkBuildModels = 1
f_bulkRunModels = 1
# f_checkFeasibility = 1
f_checkError = 1

def main(fdr_i=5,linPoint=1.0,pCvr=0.8,method='fpl',saveModel=False,modelType=None,pltSave=False):
    reload(lineariseDssModels)
    
    SD = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=fdr_i,linPoints=[linPoint],pCvr=pCvr,
                                                modelType=modelType,method=method,saveModel=saveModel,SD=SD,pltSave=pltSave)
    return blm

# self = main('n10',modelType='plotOnly',pltSave=False)

feederSet = [5,6,8,24,18,'n4','n1','n10','n27',17]
linPointsA = [0.1,0.6,1.0]
linPointsB = [0.1,0.3,0.6] # no. 6 + 18
linPointsDict = {5:linPointsA,6:linPointsB,8:linPointsA,24:linPointsA,18:linPointsB,'n4':linPointsA,
                                'n1':linPointsA,'n10':linPointsA,'n27':linPointsA,17:linPointsA}
# pCvrSet = [0.2,0.8]
pCvrSet = [0.8]

# STEP 1: building and saving the models. =========================
if 'f_bulkBuildModels' in locals():
    for feeder in feederSet:
        linPoints = linPointsDict[feeder]
        for linPoint in linPoints:
            for pCvr in pCvrSet:
                main(feeder,pCvr=pCvr,modelType='buildSave',linPoint=linPoint)

# STEP 2: Running the models, obtaining the optimization results.
if 'f_bulkRunModels' in locals():
    for feeder in feederSet:
        linPoints = linPointsDict[feeder]
        print('============================================= Feeder:',feeder)
        for linPoint in linPoints:  
            for pCvr in pCvrSet:
                self = main(feeder,pCvr=pCvr,modelType='loadAndRun',linPoint=linPoint)

# STEP 3: check the feasibility of all solutions
if 'f_checkFeasibility' in locals():
    for feeder in feederSet:
        linPoints = linPointsDict[feeder]
        for linPoint in linPoints:
            for pCvr in pCvrSet:
                self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint)
                self.loadQpSet()
                for key in self.qpSolutions:
                    print(self.feeder + '; sln type ' + key+', linPoint,' +str(linPoint)+' : ',self.qpSolutions[key][2])

# STEP 4: consider the accuracy of the models
if 'f_checkError' in locals():
    pCvr = 0.8
    strategy='part'
    objSet = ['opCst','opCst','opCst','hcGen','hcLds']
    resultTable = [['Feeder','opCstA','opCstB','opCstC','hcGen','hcLds']]
    successTable = [['Feeder','A','B','C','G','L']]
    i = 1
    for feeder in feederSet:
        linPoints = linPointsDict[feeder]
        resultTable.append([feeder])
        successTable.append([feeder])
        linPointSet = vecSlc(linPoints,[0,1,2,0,2])
        for j in range(5):
            self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPointSet[j])
            resultTable[i].append( str(self.qpSolutionDssError(strategy,objSet[j])*100)[:7] )
            successTable[i].append( self.qpSolutionDssError(strategy,objSet[j])*100<0.5 )
        
        i+=1
    print(*resultTable,sep='\n')
    print(*successTable,sep='\n')

# STEP 5: consider the value of the different control schemes ----> do here!
import lineariseDssModels, sys, os, pickle, random
from importlib import reload
import numpy as np
from dss_python_funcs import vecSlc, getBusCoords, getBusCoordsAug, tp_2_ar
import matplotlib.pyplot as plt

FD = sys.argv[0]

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr']


f_bulkBuildModels = 1
f_bulkRunModels = 1
# f_checkFeasibility = 1
# f_checkError = 1
# f_valueComparison = 1

def main(fdr_i=5,linPoint=1.0,pCvr=0.8,method='fpl',saveModel=False,modelType=None,pltSave=False):
    reload(lineariseDssModels)
    
    SD = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=fdr_i,linPoints=[linPoint],pCvr=pCvr,
                                                modelType=modelType,method=method,saveModel=saveModel,SD=SD,pltSave=pltSave)
    return blm

# self = main('n10',modelType='plotOnly',pltSave=False)

feederSet = [5,6,8,24,18,'n4','n1','n10','n27',17,0]
linPointsA = [0.1,0.6,1.0]
linPointsB = [0.1,0.3,0.6]
linPointsC = [1.0]
# linPointsA = [1.0]
# linPointsB = [0.6]
linPointsDict = {5:linPointsA,6:linPointsB,8:linPointsA,24:linPointsA,18:linPointsB,'n4':linPointsA,
                                'n1':linPointsA,'n10':linPointsA,'n27':linPointsA,17:linPointsA,0:linPointsC}
pCvrSet = [0.2,0.8]
# pCvrSet = [0.8]

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
    # objSet = ['opCst','hcGen','hcLds']
    resultTable = [['Feeder','opCstA','opCstB','opCstC','hcGen','hcLds']]
    successTable = [['Feeder','A','B','C','G','L']]
    i = 1
    for feeder in feederSet:
        linPoints = linPointsDict[feeder]
        resultTable.append([feeder])
        successTable.append([feeder])
        linPointSet = vecSlc(linPoints,[0,1,2,0,2])
        # linPointSet = vecSlc(linPoints,[0,0,0])
        for j in range(len(linPointSet)):
            self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPointSet[j])
            resultTable[i].append( str(self.qpSolutionDssError(strategy,objSet[j])*100)[:7] )
            successTable[i].append( self.qpSolutionDssError(strategy,objSet[j])*100<0.5 )
        
        i+=1
    print(*resultTable,sep='\n')
    print(*successTable,sep='\n')

# STEP 5: consider the value of the different control schemes ----> do here!
if 'f_valueComparison' in locals():
    pCvr = 0.8
    
    objSet = ['opCst','opCst','opCst','hcGen','hcLds']
    objSet = ['opCst','hcGen','hcLds']
    strategies = ['full','part','minTap','maxTap']
    # strategies = ['part']
    
    opCstTable = [['Operating cost (kW)'],['Feeder',*strategies]]
    opCstTableA = [['Operating cost (kW)'],[*strategies]]
    opCstTableB = [['Operating cost (kW)'],[*strategies]]
    opCstTableC = [['Operating cost (kW)'],[*strategies]]
    hcGenTable = [['Generation (kW)'],['Feeder',*strategies]]
    hcLdsTable = [['Load (kW)'],['Feeder',*strategies]]
    
    i = 2
    for feeder in feederSet:
        print(feeder)
        linPoints = linPointsDict[feeder]
        opCstTableA.append([])
        opCstTableB.append([])
        opCstTableC.append([])
        opCstTable.append([feeder])
        hcGenTable.append([feeder])
        hcLdsTable.append([feeder])
        linPointSet = vecSlc(linPoints,[0,1,2,0,2])
        # linPointSet = vecSlc(linPoints,[0,0,0])
        for strategy in strategies:
            for j in range(len(linPointSet)):
                obj = objSet[j]
                self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPointSet[j])
                val = self.qpVarValue(strategy,obj)
                
                if obj=='opCst' and j==0: opCstTableA[i].append( val )
                if obj=='opCst' and j==1: opCstTableB[i].append( val )
                if obj=='opCst' and j==2: opCstTableC[i].append( val )
                if obj=='hcGen': hcGenTable[i].append( str(val)[:7] )
                if obj=='hcLds': hcLdsTable[i].append( str(val)[:7] )
        i+=1
    
    w = [0.3333,0.3333,0.3334]
    opCstTable_ = (w[0]*np.array(opCstTableA[2:]) + w[1]*np.array(opCstTableB[2:]) + w[2]*np.array(opCstTableC[2:])).tolist()
    # opCstTable_ = opCstTableA[2:]
    
    for i in range(2,len(feederSet)+2):
        for j in range(len(strategies)):
            opCstTable[i].append( str(opCstTable_[i-2][j])[:7] )
        
    print(*opCstTable,sep='\n')
    print(*hcGenTable,sep='\n')
    print(*hcLdsTable,sep='\n')

# feeder = 0
# obj = 'hcLds'
# strategy = 'part'
# self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPointsDict[feeder][0])
# self.loadQpSet()
# self.loadQpSln(strategy,obj)
# self.showQpSln()
# self.plotNetBuses('qSln')


# feeder = 'n10'
# obj = 'hcLds'
# strategy = 'full'
# self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPointsDict[feeder][2])
# self.loadQpSet()
# self.loadQpSln(strategy,obj)

# self.showQpSln()
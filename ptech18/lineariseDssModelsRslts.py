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
f_psccResults = 1

def main(fdr_i=5,modelType=None,linPoint=1.0,pCvr=0.8,method='fpl',saveModel=False,pltSave=False):
    reload(lineariseDssModels)
    
    SD = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=fdr_i,linPoints=[linPoint],pCvr=pCvr,
                                                modelType=modelType,method=method,saveModel=saveModel,SD=SD,pltSave=pltSave)
    return blm

# self = main('n10',modelType='plotOnly',pltSave=False)
feederSet = [5,6,8,24,0,18,17,'n4','n1','n10','n27']
feederSet = [0,17,'n1',8,'n27',24] # results set

lpA = [0.1,0.6,1.0];        lpB = [0.1,0.3,0.6];       lpC = [1.0]
linPointsA = {'all':lpA,'opCst':lpA,'hcGen':[lpA[0]],'hcLds':[lpA[-1]]}
linPointsB = {'all':lpB,'opCst':lpB,'hcGen':[lpB[0]],'hcLds':[lpB[-1]]}
linPointsC = {'all':lpC,'opCst':lpC,'hcGen':lpC,'hcLds':lpC}
objSet = ['opCst','hcGen','hcLds']

# NB remember to update n10!
linPointsDict = {5:linPointsA,6:linPointsB,8:linPointsA,24:linPointsA,18:linPointsB,'n4':linPointsA,
                                'n1':linPointsA,'n10':linPointsA,'n27':linPointsA,17:linPointsA,0:linPointsA,25:linPointsC}
pCvrSet = [0.2,0.8]
pCvrSet = [0.4,0.8]

# STEP 1: building and saving the models. =========================
tBuild = []
if 'f_bulkBuildModels' in locals():
    for feeder in feederSet:
        t0 = time.time()
        linPoints = linPointsDict[feeder]['all']
        print('============================================= Feeder:',feeder)
        for linPoint in linPoints:
            for pCvr in pCvrSet:
                self = main(feeder,pCvr=pCvr,modelType='buildSave',linPoint=linPoint)
        tBuild.append(time.time()-t0)

# STEP 2: Running the models, obtaining the optimization results.
tSet = []
if 'f_bulkRunModels' in locals():
    for feeder in feederSet:
        t0 = time.time()
        linPoints = linPointsDict[feeder]['all']
        print('============================================= Feeder:',feeder)
        for linPoint in linPoints:  
            for pCvr in pCvrSet:
                self = main(feeder,pCvr=pCvr,modelType='loadAndRun',linPoint=linPoint) # see "runQpSet"
        tSet.append(time.time()-t0)

# STEP 3: check the feasibility of all solutions
if 'f_checkFeasibility' in locals():
    for feeder in feederSet:
        linPoints = linPointsDict[feeder]['all']
        for linPoint in linPoints:
            for pCvr in pCvrSet:
                self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint)
                self.loadQpSet()
                for key in self.qpSolutions:
                    print(self.feeder + '; sln type ' + key+', linPoint,' +str(linPoint)+' : ',self.qpSolutions[key][2])

# STEP 4: consider the accuracy of the models
if 'f_checkError' in locals():
    pCvr = 0.8
    strategy='full'
    resultTableV = [['V error (%)'],['Feeder','opCstA','opCstB','opCstC','hcGen','hcLds']]
    resultTableI = [['I error (%)'],['Feeder','opCstA','opCstB','opCstC','hcGen','hcLds']]
    
    successTable = [['Success Table'],['Feeder','A','B','C','G','L']]
    i = len(successTable)
    for feeder in feederSet:
        print('Feeder ',feeder)
        resultTableV.append([feeder])
        resultTableI.append([feeder])
        successTable.append([feeder])
        for obj in objSet:
            linPoints = linPointsDict[feeder][obj]
            for linPoint in linPoints:
                self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint)
                resultTableV[i].append( "%.6f" % (self.qpSolutionDssError(strategy,obj,err='V')*100))
                resultTableI[i].append( "%.6f" % (self.qpSolutionDssError(strategy,obj,err='I')*100))
                successTable[i].append( self.qpSolutionDssError(strategy,obj)*100<0.5 and
                                        self.qpSolutionDssError(strategy,obj,err='I')<0.05 )
        i+=1
    print(*resultTableV,sep='\n')
    print(*resultTableI,sep='\n')
    print(*successTable,sep='\n')

# STEP 5: consider the value of the different control schemes ----> do here!
if 'f_valueComparison' in locals():
    pCvr = 0.8
    
    # strategies = ['full','part','phase','minTap','maxTap','nomTap']
    strategies = ['full','phase','minTap','maxTap']
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
        opCstTableA.append([]); opCstTableB.append([]); opCstTableC.append([])
        opCstTable.append([feeder]); hcGenTable.append([feeder]); hcLdsTable.append([feeder])
        for strategy in strategies:
            for obj in objSet:
                linPoints = linPointsDict[feeder][obj]
                j=0
                for linPoint in linPoints:
                    self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint)
                    val = self.qpVarValue(strategy,obj,'power')
                    
                    if obj=='opCst' and j==0: opCstTableA[i].append( val )
                    if obj=='opCst' and j==1: opCstTableB[i].append( val )
                    if obj=='opCst' and j==2: opCstTableC[i].append( val )
                    if obj=='hcGen': hcGenTable[i].append( str(val)[:7] )
                    if obj=='hcLds': hcLdsTable[i].append( str(val)[:7] )
                    j+=1
                
        i+=1
    
    w = [0.3333,0.3333,0.3334]
    opCstTable_ = (w[0]*np.array(opCstTableA[2:]) + w[1]*np.array(opCstTableB[2:]) + w[2]*np.array(opCstTableC[2:])).tolist()
    # opCstTable_ = opCstTableA[2:]
    
    for i in range(2,len(feederSet)+2):
        for j in range(len(strategies)):
            opCstTable[i].append( "%.6f" % opCstTable_[i-2][j] )
        
    print(*opCstTable,sep='\n')
    print(*hcGenTable,sep='\n')
    print(*hcLdsTable,sep='\n')

if 'f_psccResults' in locals():
    feederSet = ['n1','n27']
    
    sRated = 2
    # lossFracS0s = [1.45,0.72,0.88] # from paper by Notton et al
    # lossFracSmaxs = [4.37/(sRated**2),3.45/(sRated**2),11.49/(sRated**2)] # from paper by Notton et al
    lossSettings = {'Low':[  ],'Med':[ ], 'Hi':[ ] }
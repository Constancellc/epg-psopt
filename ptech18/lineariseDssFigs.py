import lineariseDssModels, sys, os, pickle, random, time
from importlib import reload
import numpy as np
from dss_python_funcs import vecSlc, getBusCoords, getBusCoordsAug, tp_2_ar
import matplotlib.pyplot as plt
from matplotlib import cm, rc
plt.style.use('tidySettings')

FD = sys.argv[0]

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr']

# f_valueComparisonChart = 1
# f_plotOnly = 1
# f_plotInvLoss = 1

feederSet = [0,17,'n1',8,24,'n27']

strategies = ['full','phase','minTap','maxTap']
objSet = ['opCst','hcGen']

lpA = [0.1,0.6,1.0];        lpB = [0.1,0.3,0.6];       lpC = [1.0]
linPointsA = {'all':lpA,'opCst':lpA,'hcGen':[lpA[0]],'hcLds':[lpA[-1]]}
linPointsB = {'all':lpB,'opCst':lpB,'hcGen':[lpB[0]],'hcLds':[lpB[-1]]}
linPointsC = {'all':lpC,'opCst':lpC,'hcGen':lpC,'hcLds':lpC}
linPointsDict = {5:linPointsA,6:linPointsB,8:linPointsA,24:linPointsA,18:linPointsB,'n4':linPointsA,
                                'n1':linPointsA,'n10':linPointsA,'n27':linPointsA,17:linPointsA,0:linPointsA,25:linPointsC}

def main(fdr_i=5,modelType=None,linPoint=1.0,pCvr=0.8,method='fpl',saveModel=False,pltSave=False):
    reload(lineariseDssModels)
    
    SD = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=fdr_i,linPoints=[linPoint],pCvr=pCvr,
                                                modelType=modelType,method=method,saveModel=saveModel,SD=SD,pltSave=pltSave)
    return blm

feederTidy=[]
for feeder in feederSet:
    try:
        feederTidy.append(fdrs[feeder])
    except:
        feederTidy.append(feeder)

# Networks considered
if 'f_plotOnly' in locals():
    for feeder in feederSet:
        self = main(feeder,'plotOnly')

if 'f_plotInvLoss' in locals():
    # MAYBE this is a 'nicer looking' graph of losses?
    x = np.linspace(-1,1,1001)
    # ct1 = 0.25; ct2 = 1-ct1
    # c0 = 0.5*ct1; c2 = ct2 + c0
    c0 = 0.1; c2 = 1-c0
    ct2 = c2-c0; ct1=2*c0
    y = 5*( ct1*np.abs(x) + ct2*(x**2) )
    z = 5*( c0 + c2*(x**2))
    z[len(z)//2]=0
    plt.plot(100*x,y); plt.xlabel('Q, %'); plt.ylabel('Losses - fraction of peak Q (%)')
    plt.plot(100*x,z); plt.xlabel('Q, %'); plt.ylabel('Losses - fraction of peak Q (%)')
    plt.ylim((-0.0,6.1))
    plt.grid()
    plt.show()
    



if 'f_valueComparisonChart' in locals():
    pCvr = 0.8
    opCstTable = [['Operating cost (kW)'],['Feeder',*strategies]]
    opCstTableA = [['Operating cost (kW)'],[*strategies]];    opCstTableB = [['Operating cost (kW)'],[*strategies]]
    opCstTableC = [['Operating cost (kW)'],[*strategies]];    hcGenTable = [['Generation (kW)'],['Feeder',*strategies]]
    i = 2
    for feeder in feederSet:
        print(feeder)
        linPoints = linPointsDict[feeder]
        opCstTableA.append([]); opCstTableB.append([]); opCstTableC.append([])
        opCstTable.append([feeder]); hcGenTable.append([feeder]); # hcLdsTable.append([feeder])
        for strategy in strategies:
            for obj in objSet:
                linPoints = linPointsDict[feeder][obj]
                j=0
                for linPoint in linPoints:
                    self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint)
                    val = self.qpVarValue(strategy,obj,'norm')
                    
                    if obj=='opCst' and j==0: opCstTableA[i].append( val )
                    if obj=='opCst' and j==1: opCstTableB[i].append( val )
                    if obj=='opCst' and j==2: opCstTableC[i].append( val )
                    if obj=='hcGen': hcGenTable[i].append( str(val)[:7] )
                    j+=1
        i+=1
    
    # w = [0.3333,0.3333,0.3334]
    w = [0.0,0.0,1.0]
    opCstTable_ = (w[0]*np.array(opCstTableA[2:]) + w[1]*np.array(opCstTableB[2:]) + w[2]*np.array(opCstTableC[2:])).tolist()
    
    for i in range(2,len(feederSet)+2):
        for j in range(len(strategies)):
            opCstTable[i].append( "%.6f" % opCstTable_[i-2][j] )
        
    print(*opCstTable,sep='\n')
    print(*hcGenTable,sep='\n')
    
    tables={'opCst':opCstTable,'hcGen':hcGenTable}
    ylabels={'opCst':'Total P in / Total P in Ref., %','hcGen':'Hosting capacity (% of nominal load)'}
    ylims={'opCst':(82.5,117.5),'hcGen':(-10,220)}

    for obj in objSet:
        fig,ax = plt.subplots()
        table = tables[obj]
        for i in range(len(feederSet)):
            for j in range(len(strategies)):
                ax.bar(i-0.25+ (0.25*2*np.arange(len(strategies))/(len(strategies)-1)),100*np.array(table[i+2][1:]).astype('float'),
                                                                        width=0.08,color=cm.matlab(range(len(strategies))) )
        
        for i in range(len(strategies)): #for the legend
            ax.plot(0,0,label=strategies[i])
        
        ax.set_ylim(ylims[obj])
        ax.set_xticks(np.arange(len(feederSet)))
        ax.set_xticklabels(feederTidy,rotation=90)
        ax.legend(fontsize='small')
        ax.set_ylabel(ylabels[obj])
        ax.set_title(obj)
        plt.tight_layout()
        plt.show()


# self = main(0,'loadOnly',linPoint=1.0); self.loadQpSet(); self.loadQpSln('full','opCst'); self.plotNetBuses('qSlnPh',minMax=[-1.0,1.0])
# self = main(0,'loadOnly',linPoint=0.1); self.loadQpSet(); self.loadQpSln('full','hcGen'); self.plotNetBuses('qSlnPh',minMax=[-1.0,1.0])
# self = main(8,'loadOnly',linPoint=1.0); self.loadQpSet(); self.loadQpSln('full','opCst'); self.plotNetBuses('qSlnPh',minMax=[-1.0,1.0])
# self = main(8,'loadOnly',linPoint=0.1); self.loadQpSet(); self.loadQpSln('full','hcGen'); self.plotNetBuses('qSlnPh',minMax=[-1.0,1.0])


# self = main(8,'loadOnly',linPoint=0.1); self.loadQpSet(); self.loadQpSln('full','hcGen'); self.showQpSln()
# self = main('n1','loadOnly',linPoint=0.1); self.loadQpSet(); self.loadQpSln('full','hcGen'); self.showQpSln()

# self = main('n1','loadOnly',linPoint=0.1); self.runCvrQp('full','hcGen')
# self = main('n1','loadOnly',linPoint=0.1); self.runCvrQp('full','hcGen')

# self = main('n1','plotOnly')
# self = main('n1','plotOnly')

# self = main('n27','loadOnly',linPoint=0.1); self.loadQpSet(); self.loadQpSln('full','hcGen'); self.showQpSln()
# self.plotNetBuses('qSlnPh',minMax=[-1.0,1.0])

# self = main('n27','loadOnly',linPoint=1.0); self.initialiseOpenDss(); self.testCvrQp()
# self = main(17,'loadOnly',linPoint=1.0); self.initialiseOpenDss(); self.testCvrQp()
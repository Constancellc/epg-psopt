import lineariseDssModels, sys, os, pickle, random, time
from importlib import reload
import numpy as np
from dss_python_funcs import vecSlc, getBusCoords, getBusCoordsAug, tp_2_ar, basicTable, sdt
import matplotlib.pyplot as plt

FD = sys.argv[0]

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr','123busCvr']


# f_bulkBuildModels = 1
# f_bulkRunModels = 1
# f_checkFeasibility = 1
# f_checkError = 1
# f_valueComparison = 1
# f_psccAbstract = 1
# t_costFuncStudy = 1


SD0 = os.path.join(os.path.join(os.path.expanduser('~')), 'Documents','DPhil','papers','psjul19')

def main(fdr_i=5,modelType=None,linPoint=1.0,pCvr=0.6,method='fpl',saveModel=False,pltSave=False):
    reload(lineariseDssModels)
    SD = SD0
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=fdr_i,linPoints=[linPoint],pCvr=pCvr,
                                                modelType=modelType,method=method,saveModel=saveModel,SD=SD,pltSave=pltSave)
    return blm

# self = main('n10',modelType='plotOnly',pltSave=False)
feederSet = [5,6,8,26,24,0,18,17,'n4','n1','n10','n27']
feederSet = [6,17,'n1',26,'n27',24,'n10',18] # results set
# feederSet = [6,26,24,17,18,'n1'] # introductory set
feederSet = [26]
# feederSet = [26,0,17,24,18,'n1','n10','n27','n4']

methodChosen = 'fot'

feederIdxTidy = {5:'13 Bus',6:'34 Bus',8:'123 Bus',9:'8500 Node',19:'Ckt. J1',20:'Ckt. K1',21:'Ckt. M1',17:'Ckt. 5',18:'Ckt. 7',22:'Ckt. 24',26:'123 Bus',24:'Ckt. K1','n1':'EULVa','n27':'EULVa-r',0:'EULV','n4':'Nwk. 4','n10':'Nwk. 10'}


lpA = [0.1,0.6,1.0];        lpB = [0.1,0.3,0.6];       lpC = [1.0]
linPointsA = {'all':lpA,'opCst':lpA,'hcGen':[lpA[0]],'hcLds':[lpA[-1]]}
linPointsB = {'all':lpB,'opCst':lpB,'hcGen':[lpB[0]],'hcLds':[lpB[-1]]}
linPointsC = {'all':lpC,'opCst':lpC,'hcGen':lpC,'hcLds':lpC}

objSet = ['opCst','hcGen','hcLds']
objSet = ['opCst']
strategySet = { 'opCst':['full','phase','nomTap','load','loss'],'hcGen':['full','phase','nomTap','maxTap'],'hcLds':['full','phase','nomTap','minTap'] }

# NB remember to update n10!
linPointsDict = {5:linPointsA,6:linPointsB,26:linPointsA,24:linPointsA,18:linPointsB,'n4':linPointsA,
                                'n1':linPointsA,'n10':linPointsA,'n27':linPointsA,17:linPointsA,0:linPointsA,25:linPointsC}
pCvrSet = [0.0,0.3,0.6,0.9]
pCvrSet = [0.0,0.3,0.9]
pCvrSet = [0.6]

# STEP 1: building and saving the models. =========================
tBuild = []
if 'f_bulkBuildModels' in locals():
    for feeder in feederSet:
        t0 = time.time()
        linPoints = linPointsDict[feeder]['all']
        linPoints = [linPointsDict[feeder]['all'][-1]]
        print('============================================= Feeder:',feeder)
        for linPoint in linPoints:
            for pCvr in pCvrSet:
                self = main(feeder,pCvr=pCvr,modelType='buildSave',linPoint=linPoint,method=methodChosen)
        tBuild.append(time.time()-t0)

# STEP 2: Running the models, obtaining the optimization results.
tSet = []
if 'f_bulkRunModels' in locals():
    for feeder in feederSet:
        t0 = time.time()
        linPoints = linPointsDict[feeder]['all']
        linPoints = [linPointsDict[feeder]['all'][-1]]
        print('============================================= Feeder:',feeder)
        for linPoint in linPoints:  
            for pCvr in pCvrSet:
                self = main(feeder,pCvr=pCvr,modelType='loadAndRun',linPoint=linPoint,method=methodChosen) # see "runQpSet"
        tSet.append(time.time()-t0)

# STEP 3: check the feasibility of all solutions
if 'f_checkFeasibility' in locals():
    for feeder in feederSet:
        linPoints = linPointsDict[feeder]['all']
        for linPoint in linPoints:
            for pCvr in pCvrSet:
                self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint,method=methodChosen)
                self.loadQpSet()
                for key in self.qpSolutions:
                    print(self.feeder + '; sln type ' + key+', linPoint,' +str(linPoint)+' : ',self.qpSolutions[key][2])

# STEP 4: consider the accuracy of the models
if 'f_checkError' in locals():
    pCvr = 0.6
    strategy='full'
    heading = ['Feeder','opCstA','opCstB','opCstC','hcGen','hcLds']
    resultTableV = [['Voltage error, $\|V_{\mathrm{Apx}} - V_{\mathrm{DSS}}\|_{2}/\|V_{\mathrm{DSS}}\|_{2}$, \%'],heading]
    resultTableI = [['Current error, $\|I_{\mathrm{Apx}} - I_{\mathrm{DSS}}\|_{2}/\|I_{\mathrm{Xfmr}}\|_{2}$, \%'],heading]
    resultTableP = [['Power error, $(P_{\mathrm{feeder}}^{\mathrm{Apx.}} - P_{\mathrm{feeder}}^{\mathrm{DSS.}})/P_{\mathrm{feeder}}$, \%'],heading]
    
    successTable = [['Success Table'],['Feeder','A','B','C','G','L']]
    i = len(successTable)
    for feeder in feederSet:
        feederTidy = feederIdxTidy[feeder]
        print('Feeder ',feeder)
        resultTableV.append([feederTidy])
        resultTableI.append([feederTidy])
        resultTableP.append([feederTidy])
        successTable.append([feederTidy])
        for obj in objSet:
            linPoints = linPointsDict[feeder][obj]
            linPoints = [linPointsDict[feeder]['all'][-1]]
            for linPoint in linPoints:
                self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint,method=methodChosen)
                resultTableV[i].append( "%.6f" % (self.qpSolutionDssError(strategy,obj,err='V')*100))
                resultTableI[i].append( "%.6f" % (self.qpSolutionDssError(strategy,obj,err='I')*100))
                resultTableP[i].append( "%.6f" % (self.qpSolutionDssError(strategy,obj,err='P')*100))
                successTable[i].append( self.qpSolutionDssError(strategy,obj)*100<0.5 and
                                        self.qpSolutionDssError(strategy,obj,err='I')<0.05 and
                                        self.qpSolutionDssError(strategy,obj,err='P')<0.05 )
        i+=1
    print(*resultTableV,sep='\n')
    print(*resultTableI,sep='\n')
    print(*resultTableP,sep='\n')
    print(*successTable,sep='\n')
    TD = os.path.join(SD0,'tables\\')
    basicTable(resultTableV[0][0],'checkErrorV',heading,resultTableV[2:],TD)
    basicTable(resultTableI[0][0],'checkErrorI',heading,resultTableI[2:],TD)
    basicTable(resultTableP[0][0],'checkErrorP',heading,resultTableP[2:],TD)
    
# STEP 5: consider the value of the different control schemes ----> do here!
if 'f_valueComparison' in locals():
    pCvr = 0.6
    
    opCstTable = [['Operating cost (kW)'],['Feeder',*strategySet['opCst']]]
    opCstTableA = [['Operating cost (kW)'],[*strategySet['opCst']]]
    opCstTableB = [['Operating cost (kW)'],[*strategySet['opCst']]]
    opCstTableC = [['Operating cost (kW)'],[*strategySet['opCst']]]
    hcGenTable = [['Generation (kW)'],['Feeder',*strategySet['hcGen']]]
    hcLdsTable = [['Load (kW)'],['Feeder',*strategySet['hcLds']]]
    
    i = 2
    for feeder in feederSet:
        print(feeder)
        linPoints = linPointsDict[feeder]
        opCstTableA.append([]); opCstTableB.append([]); opCstTableC.append([])
        opCstTable.append([feeder]); hcGenTable.append([feeder]); hcLdsTable.append([feeder])
        for obj in objSet:
            for strategy in strategySet[obj]:
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
        for j in range(len(strategySet['opCst'])):
            opCstTable[i].append( "%.6f" % opCstTable_[i-2][j] )
    
    
    
    print(*opCstTable,sep='\n')
    print(*hcGenTable,sep='\n')
    print(*hcLdsTable,sep='\n')

if 'f_psccAbstract' in locals():
    # feederSet = ['n1','n27']
    # feederSet = 8
    feeder = 24
    # feeder = 'n27'
    # feeder = 5
    linPoint = 1.0
    # linPoint = 0.6
    pCvr=0.6
    
    strategy='full'; obj='opCst'
    
    sRated = 2 #kVA
    lossFracS0s = 1e-2*np.array([1.45,0.72,0.88])*(sRated**1) # from paper by Notton et al
    lossFracSmaxs = 1e-2*np.array([4.37,3.45,11.49])*(sRated**-1) # from paper by Notton et al
    
    lossSettings = {'Low':[lossFracS0s[1],lossFracSmaxs[1]],'Med':[lossFracS0s[2],lossFracSmaxs[2]], 'Hi':[lossFracS0s[0],lossFracSmaxs[0]] }
    lossSetting = 'Hi'
    
    loss = lossSettings[lossSetting]
    slnSet = [['','Base','Ignore losses','Incl. losses'],['$f^{*}$, kW'],['$\Delta f^{*}$, kW'],['$\|Q^{*}\|_{1}$, kVAr']]
    
    
    # RESULTS TABLE ================
    self = main(feeder,linPoint=linPoint,pCvr=pCvr,modelType='loadOnly')
    
    self.setQlossOfs(kQlossQ=loss[1],kQlossC=0)
    self.qLim = 0
    self.runCvrQp(); # self.showQpSln()
    
    fStar0 = sum(self.runQp(self.slnX)[0:4])
    fStar = sum(self.runQp(self.slnX)[0:4])
    slnSet[1].append( '%.1f' % np.round(fStar,1) )
    slnSet[2].append( '%.2f' % np.round(fStar-fStar0,2) )
    slnSet[3].append( '%.1f' % np.linalg.norm(self.slnX[self.nPctrl:self.nSctrl],ord=1))
    
    # first solve with no inverter losses (but get the solution with losses)
    self = main(feeder,linPoint=linPoint,pCvr=pCvr,modelType='loadOnly')
    self.setQlossOfs(kQlossQ=0.0,kQlossC=0.0)
    self.runCvrQp()
    
    self.setQlossOfs(kQlossQ=loss[1],kQlossC=0)
    
    fStar = sum(self.runQp(self.slnX)[0:4])
    slnSet[1].append( '%.1f' % np.round(fStar,1) )
    slnSet[2].append( '%.2f' % np.round(fStar-fStar0,2) )
    slnSet[3].append( '%.1f' % np.linalg.norm(self.slnX[self.nPctrl:self.nSctrl],ord=1))
    
    self.runCvrQp()
    fStar = sum(self.runQp(self.slnX)[0:4])
    slnSet[1].append( '%.1f' % np.round(fStar,1) )
    slnSet[2].append( '%.2f' % np.round(fStar-fStar0,2) )
    slnSet[3].append( '%.1f' % np.linalg.norm(self.slnX[self.nPctrl:self.nSctrl],ord=1))
    
    caption='EPRI Circuit K1 Optimization Results'
    heading = slnSet[0]
    data = slnSet[1:]
    TD = r"C:\Users\Matt\Documents\DPhil\papers\pscc20\tables\\"
    label='results'
    basicTable(caption,label,heading,data,TD)
    print(*slnSet,sep='\n')
    
    # RESULTS FIGURE ================
    if feeder==5: qlossRegs = np.linspace(0,0.14,15)
    if feeder==24: qlossRegs = np.linspace(0.0,0.6,30)
    if feeder=='n27': qlossRegs = np.linspace(0.00,0.14,15)
    # qlossRegs = np.linspace(0,0.45,45) # 123bus (?)
    # qlossRegs = np.linspace(0,1.0,4)
    
    self = main(feeder,linPoint=linPoint,pCvr=pCvr,modelType='loadOnly')
    self.setupConstraints()
    
    costA = []; costB = []
    nnz = []; taps = []
    i=0
    for qlossRegC in qlossRegs:
        print(i); i+=1
        self.setQlossOfs(kQlossQ=loss[1],kQlossC=loss[0],qlossRegC=qlossRegC)
        self.runCvrQp()
        costA.append(sum(self.runQp(self.slnX,True)[:4]))
        costB.append(sum(self.runQp(self.slnX,False)[:4]))
        nnz.append(np.sum( (np.abs(self.slnX)>self.qlossCzero)[self.nPctrl:self.nSctrl] ))
        taps.append(self.slnX[self.nSctrl:])

    fig,ax = plt.subplots(figsize=(4.7,2.2))

    fig,ax = plt.subplots(figsize=(4.7,1.8))
    ax.plot(qlossRegs,costA,'k',label='With $c_{\mathrm{Turn\,on}}$');
    ax.plot(qlossRegs,costB,'k--',label='No $c_{\mathrm{Turn\,on}}$');
    ax.legend(loc='lower right')
    ax.set_ylabel('Cost $f$ (kW)')
    ax.set_xlabel('Regularization parameter $\eta$')
    ax.annotate("Optimal point",
                xy=(0.05, 12425), xycoords='data',
                xytext=(0.045, 12440), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"),
                )
    ax.annotate("$||Q^{*}||_{1}=0$",
                xy=(0.45, 12463), xycoords='data',
                xytext=(0.43, 12450), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"),
                )            
    plt.grid(); plt.xlim((0.0,0.6))
    plt.tight_layout(); 
    # plt.savefig(r"C:\Users\Matt\Documents\DPhil\papers\pscc20\figures\rglrz.png",bbox_inches='tight', pad_inches=0)
    # plt.savefig(r"C:\Users\Matt\Documents\DPhil\papers\pscc20\figures\rglrz.pdf",bbox_inches='tight', pad_inches=0)
    plt.show()
        
    # taps= np.array(taps);
    fig,[ax0,ax1,ax2] = plt.subplots(3,sharex=True)
    ax0.plot(qlossRegs,costB,label='Incl turn on');
    ax0.plot(qlossRegs,costA,label='No turn on');
    ax0.set_ylabel('Cost (kW)'); 
    ax0.legend(); 
    ax1.plot(qlossRegs,nnz)
    ax1.set_ylabel('No. zero elements')
    ax2.step(qlossRegs,taps)
    ax2.set_ylabel('Tap Positions')
    ax2.set_xlabel('Regularization parameter')
    plt.tight_layout()
    plt.show()


if 't_costFuncStudy' in locals():
    from numpy.linalg import norm, solve, lstsq
    #things to do.
    # 1. load linTot, qpQlss and qpQtot Find the units of them.
    # 2. Find Q^-1 L; then, solve using the 'fast' method, and compare.
    epsXoff = np.zeros((len(feederSet),3))
    x1sets = {}
    feederSetOrdered = [17,18,'n1','n10',26,24,'n27'] # ordered results set
    for feeder,ii in zip(feederSetOrdered,range(len(feederSetOrdered))):
        print(feeder)
        linPoints = linPointsDict[feeder]['all']
        linPoint = linPoints[-1]
        x1sets[feeder] = {}
        self = main(feeder,modelType='loadOnly',linPoint=linPoint,method=methodChosen)
        self.solveQpUnc()
        qpQlss = 1e3*self.qQpUnc[self.nPctrl:self.nSctrl,self.nPctrl:self.nSctrl]
        
        types = ['Low','Hi']
        sRateds = [1.0,4.0,16.0]
        invCoeffs = [[],[]]
        for type,i in zip(types,range(2)):
            for sRated in sRateds:
                kQlossC,kQlossQ = self.getInvLossCoeffs(sRated,type)
                self.setQlossOfs(kQlossQ=kQlossQ,kQlossC=0)
                invCoeffs[i].append(1e3*np.linalg.norm(self.qlossQdiag,ord=np.inf))
        
        pLoad = self.ploadL[self.nPctrl:self.nSctrl]
        pLoss = self.qpLlss[self.nPctrl:self.nSctrl]
        p = 1e3*(pLoad + pLoss)
        
        xOffValue = 1.05
        sln0 = lstsq(qpQlss,-0.5*p,rcond=None)
        cQ = sln0[0]
        bQ = lstsq(-qpQlss,cQ,rcond=None)[0]

        c0 = cQ.dot(cQ)*(xOffValue**2 - 1) # 0.19
        c1 = 2*cQ.dot(bQ)
        c2 = bQ.dot(bQ)
        epsXoff[ii,2] = np.min(np.roots([c2,c1,c0]))
        # x1sets[feeder] = [x1Set,cQ]

        aQ = lstsq(qpQlss,bQ,rcond=None)

        cQ = -0.5*p
        bQ = -qpQlss.dot(cQ)
        c0 = cQ.dot(cQ)*(1 - xOffValue**2)
        c1 = 2*cQ.dot(bQ)
        c2 = bQ.dot(bQ)
        # epsXoff[ii,1] = np.max(1/np.roots([c2,c1,c0])) # not used as too often outside of the right point.
        
        sln0s = sln0[3]
        m_eps = np.finfo(np.float64).eps
        sln0s = sln0s[sln0s>( m_eps*len(sln0s)*sln0s[0] )]
        
        epsXoff[ii,0:2] = [sln0s[0],sln0s[-1]]
        
    epsXoffTbl = []
    for row,feeder in zip(epsXoff,feederSet):
        epsXoffTbl.append([])
        epsXoffTbl[-1].append(feederIdxTidy[feeder])
        epsXoffTbl[-1].append( '%.2e' % row[2])
        epsXoffTbl[-1].append( '%.2e' % row[1])
        epsXoffTbl[-1].append( '%.2e' % row[0])

    TD = sdt('t3','t')
        
    label = 'costFuncStudy'
    heading = ['Feeder','$\delta _{5\%}$','Min sing. value','Max sing. value']
    caption = 'Minimum/Maximum singular values of network loss quadratic matrices, and the estimate of the 5\% solution reduction value $\delta _{5\%}$.'
    data = epsXoffTbl
    basicTable(caption,label,heading,data,TD)

    
    

# self = main(feeder,linPoint=linPoint,pCvr=pCvr,modelType='loadOnly')
# self.setQlossOfs(kQlossQ=0.5,kQlossC=0.0)
# self.runCvrQp();sum(self.slnF[:4])

# self = main(feeder,linPoint=linPoint,pCvr=pCvr,modelType='loadOnly')
# self.setQlossOfs(kQlossQ=0.5,kQlossC=0.0)
# self.runCvrQp(); sum(self.slnF[:4])


# self.setQlossOfs(kQlossQ=0.5,kQlossC=0.0,qlossRegC=1.0)
# self.runCvrQp()
# sum(self.slnF[:4])

# self = main(feeder,linPoint=linPoint,pCvr=pCvr,modelType='loadOnly')
# self.setupConstraints()
# self.setQlossOfs(kQlossQ=loss[1])
# self.runCvrQp()
# self.showQpSln()
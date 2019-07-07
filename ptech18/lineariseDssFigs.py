import lineariseDssModels, sys, os, pickle, random, time
from importlib import reload
import numpy as np
from dss_python_funcs import vecSlc, getBusCoords, getBusCoordsAug, tp_2_ar, plotSaveFig, basicTable, np2lsStr
import matplotlib.pyplot as plt
from matplotlib import cm, rc
plt.style.use('tidySettings')

FD = sys.argv[0]

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr','123busCvr']

feederAllTidy = {'13bus':'13 Bus','34bus':'34 Bus','123bus':'123 Bus','8500node':'8500 Node','epriJ1':'Ckt. J1','epriK1':'Ckt. K1','epriM1':'Ckt. M1','epri5':'Ckt. 5','epri7':'Ckt. 7','epri24':'Ckt. 24','123busCvr':'123 Bus','epriK1cvr':'Ckt. K1','n1':'EULV-A','n27':'EULV-AR','eulv':'EULV','n4':'Ntwk. 4','n10':'Ntwk. 10'}

feederIdxTidy = {5:'13 Bus',6:'34 Bus',8:'123 Bus',9:'8500 Node',19:'Ckt. J1',20:'Ckt. K1',21:'Ckt. M1',17:'Ckt. 5',18:'Ckt. 7',22:'Ckt. 24',26:'123 Bus',24:'Ckt. K1','n1':'EULV-A','n27':'EULV-AR',0:'EULV','n4':'Ntwk. 4','n10':'Ntwk. 10'}


# f_valueComparisonChart = 1
# f_plotOnly = 1
# f_plotInvLoss = 1
# f_batchTest = 1
# f_modelValidation = 1
# f_daisy = 1
# f_solutionError = 1
# f_caseStudyChart = 1
# f_sensitivities_base = 1
# f_sensitivities_invLoss = 1
# f_sensitivities_efficacy = 1
# f_sensitivities_aCvr = 1
# f_sensitivities_loadPoint = 1
# f_checkErrorSummary = 1

pltSave=1

SD0 = os.path.join(os.path.join(os.path.expanduser('~')), 'Documents','DPhil','papers','psjul19')
SDfig = os.path.join(SD0,'figures')
TD = os.path.join(SD0,'tables\\')


# feederSet = [0,17,'n1',26,24,'n27']
# feederSet = [0,17,'n1',26,24,'n27']
feederSet = ['n1','n10',17,18,26,24,'n27']
feederSet = ['n1','n10',17,18,26,24,'n27']
# feederSet = [0,5] # fast

strategySet = { 'opCst':['full','phase','nomTap','load','loss'],'hcGen':['full','phase','nomTap','minTap'],'hcLds':['full','phase','nomTap','maxTap'] }
objSet = ['opCst','hcGen','hcLds']

lpA = [0.1,0.6,1.0];        lpB = [0.1,0.3,0.6];       lpC = [1.0]
linPointsA = {'all':lpA,'opCst':lpA,'hcGen':[lpA[0]],'hcLds':[lpA[-1]]}
linPointsB = {'all':lpB,'opCst':lpB,'hcGen':[lpB[0]],'hcLds':[lpB[-1]]}
linPointsC = {'all':lpC,'opCst':lpC,'hcGen':lpC,'hcLds':lpC}
linPointsDict = {5:linPointsA,6:linPointsB,26:linPointsA,24:linPointsA,18:linPointsB,'n4':linPointsA,
                                'n1':linPointsA,'n10':linPointsA,'n27':linPointsA,17:linPointsA,0:linPointsA,25:linPointsC}

def main(fdr_i=5,modelType=None,linPoint=1.0,pCvr=0.6,method='fpl',saveModel=False,pltSave=False):
    reload(lineariseDssModels)
    
    SD = os.path.join(os.path.join(os.path.expanduser('~')), 'Documents','DPhil','papers','psjul19','figures')
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=fdr_i,linPoints=[linPoint],pCvr=pCvr,
                                                modelType=modelType,method=method,saveModel=saveModel,SD=SD,pltSave=pltSave)
    return blm

feederTidy=[]
for feeder in feederSet:
    if type(feeder) is int:
        feederTidy.append(feederAllTidy[fdrs[feeder]])
    elif feeder[0]=='n':
        feederTidy.append(feederAllTidy[feeder])
    else:
        print('error in feederTidy!!!!')
        feederTidy.append(feeder)

# Networks considered
if 'f_plotOnly' in locals():
    self = main('n1','plotOnly',pltSave=True)
    for feeder in feederSet:
        self = main(feeder,'plotOnly')

if 'f_plotInvLoss' in locals():
    sRated = 2
    x = np.linspace(0,sRated,100)
    # lossFracSmaxs = [4.37/(sRated**2),3.45/(sRated**2),11.49/(sRated**2)] # from paper by Notton et al
    # lossFracS0s = [1.45,0.72,0.88] # from paper by Notton et al
    lossFracSmaxs = [3.45/(sRated**2),11.49/(sRated**2)] # from paper by Notton et al
    lossFracS0s = [0.72,0.88] # from paper by Notton et al

    fig,[ax0,ax1] = plt.subplots(ncols=2,figsize=(5.5,2.6))
    for i in range(len(lossFracSmaxs)):
        lossFracSmax = lossFracSmaxs[i]
        lossFracS0 = lossFracS0s[i]
        Plss = 0.01*sRated*( lossFracS0 + lossFracSmax*(x**2) )
        
        ax0.plot(100*x/sRated,100*Plss/sRated);
        ax0.set_ylabel('Inverter Losses, % of $S_{rated}$')
        ax0.set_ylim((0.0,15.0)); ax0.set_xlim((0.0,100))
        ax0.set_xlabel('Apparent Power $S_{\mathrm{Inv}}$, % of $S_{rated}$')
        
        ax1.plot(100*x/sRated, 100*x/(Plss + x)); 
        ax1.set_ylim((86,100));   ax1.set_xlim((0.0,100))
        ax1.set_xlabel('Apparent Power $S_{\mathrm{Inv}}$, % of $S_{rated}$')
        ax1.set_ylabel('Efficiency, %')
    ax0.legend(('Low loss','High loss'))
    plt.tight_layout()
    if 'pltSave' in locals():
        SN = os.path.join(SDfig,'invLoss')
        plt.savefig(SN+'.png',bbox_inches='tight', pad_inches=0.01)
        plt.savefig(SN+'.pdf',bbox_inches='tight', pad_inches=0.01)
    
    plt.show()
    

    for i in range(len(lossFracSmaxs)):
        lossFracSmax = lossFracSmaxs[i]
        lossFracS0 = lossFracS0s[i]
        
        lossFrac = ( lossFracS0 + lossFracSmax*(1**2) )*sRated # losses per Q
        c0 = lossFracS0*sRated/lossFrac # losses per Q
        c2 = 1-c0
        x = np.linspace(-1,1,1001)
        ct2 = c2-c0; ct1=2*c0
        y = lossFrac*( ct1*np.abs(x) + ct2*(x**2) )
        z = lossFrac*( c0 + c2*(x**2))
        print('lossFrac',lossFrac); print('c0',c0); print('c2',c2); print('ct1',ct1); print('ct2',ct2)
        z[len(z)//2]=0
        plt.plot(100*x,y); plt.xlabel('Q, %'); plt.ylabel('Losses - fraction of peak Q (%)')
        plt.plot(100*x,z); plt.xlabel('Q, %'); plt.ylabel('Losses - fraction of peak Q (%)')
        
    plt.grid()
    plt.show()
        


if 'f_valueComparisonChart' in locals():
    # pCvr = 0.6
    # pCvr = 0.0
    opCstTable = [['Operating cost (kW)'],['Feeder',*strategySet['opCst']]]
    opCstTableA = [['Operating cost (kW)'],[*strategySet['opCst']]];    opCstTableB = [['Operating cost (kW)'],[*strategySet['opCst']]]
    opCstTableC = [['Operating cost (kW)'],[*strategySet['opCst']]];    
    hcGenTable = [['Generation (kW)'],['Feeder',*strategySet['hcGen']]]; hcLdsTable = [['Load (kW)'],['Feeder',*strategySet['hcLds']]]
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
                    val = self.qpVarValue(strategy,obj,'norm')
                    
                    if obj=='opCst' and j==0: opCstTableA[i].append( val )
                    if obj=='opCst' and j==1: opCstTableB[i].append( val )
                    if obj=='opCst' and j==2: opCstTableC[i].append( val )
                    if obj=='hcGen': hcGenTable[i].append( str(val)[:7] )
                    if obj=='hcLds': hcLdsTable[i].append( str(val)[:7] )
                    j+=1
        i+=1
    
    # w = [0.3333,0.3333,0.3334]
    w = [0.0,1.0,0.0]
    w = [0.0,0.0,1.0]
    opCstTable_ = (w[0]*np.array(opCstTableA[2:]) + w[1]*np.array(opCstTableB[2:]) + w[2]*np.array(opCstTableC[2:])).tolist()
    
    for i in range(2,len(feederSet)+2):
        for j in range(len(strategySet['opCst'])):
            opCstTable[i].append( "%.6f" % opCstTable_[i-2][j] )
        
    print(*opCstTable,sep='\n')
    print(*hcGenTable,sep='\n')
    print(*hcLdsTable,sep='\n')
    
    tables={'opCst':opCstTable,'hcGen':hcGenTable,'hcLds':hcLdsTable}
    ylabels={'opCst':'Total P in / Total P in Ref., %','hcGen':'Generation (% of nominal load)','hcLds':'Load (% of nominal load)'}
    ylims={'opCst':(92.0,106.0),'hcGen':(-0,220),'hcLds':(-0,100)}
    
    colorSet = {'opCst':cm.matlab([0,1,2,3,4]),'hcGen':cm.matlab([0,1,2,5]),'hcLds':cm.matlab([0,1,2,6])}
    
    objSet = ['opCst']
    for obj in objSet:
        fig,ax = plt.subplots(figsize=(5.5,3.0))
        table = tables[obj]
        nS = len(strategySet[obj])
        for i in range(len(feederSet)):
            for j in range(nS):
                ax.bar(i-0.25+ (0.25*2*np.arange(nS)/(nS-1)),100*np.array(table[i+2][1:]).astype('float'),
                                                                        width=0.08,color=colorSet[obj] )
                                                                        # width=0.08,color=cm.matlab(range(nS)) )
        
        for i in range(nS): #for the legend
            ax.plot(0,0,label=strategySet[obj][i],color=colorSet[obj][i])
        
        ax.set_ylim(ylims[obj])
        ax.set_xticks(np.arange(len(feederSet)))
        ax.set_xticklabels(feederTidy,rotation=90)
        # ax.legend(fontsize='small')
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),fontsize='small')
        ax.set_ylabel(ylabels[obj])
        ax.set_title(obj)
        if 'pltSave' in locals():
            plotSaveFig(os.path.join(SDfig,'valueComparisonChart_'+obj))
        plt.tight_layout()
        plt.show()

if 'f_caseStudyChart' in locals():
    ylabels={'opCst':'Total P in / Total P in Ref., %','hcGen':'Generation (% of nominal load)','hcLds':'Load (% of nominal load)'}
    ylims={'opCst':(82.5,117.5),'hcGen':(-10,220),'hcLds':(-10,100)}
    
    pCvr = 0.6
    feeder = 26
    obj = 'opCst'
    nS = len(strategySet[obj])
    
    
    # first do the cost functions
    opCstTable = np.zeros( (3,len(strategySet[obj])) )
    i=0
    for strategy in strategySet[obj]:
        linPoints = linPointsDict[feeder][obj]
        j=0
        for linPoint in linPoints:
            self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint)
            opCstTable[j,i] = self.qpVarValue(strategy,obj,'norm')
            j+=1
        i+=1
    
    figFrac = 0.6
    fig,ax = plt.subplots(figsize=(5.5*figFrac,3.0))
    for i in range(3):
        for j in range(nS):
            ax.bar(i-0.25+ (0.25*2*np.arange(nS)/(nS-1)),100*opCstTable[i],
                                                                    width=0.08,color=cm.matlab(range(nS)) )
    for i in range(nS): #for the legend
        ax.plot(0,0,label=strategySet[obj][i])
    
    ax.set_ylim((95,107.5))
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(('10% load','60% load','100% load'),rotation=90)
    ax.legend(fontsize='small')
    ax.set_ylabel(ylabels[obj])
    ax.set_title(obj)
    plt.tight_layout()
    if 'pltSave' in locals():
        plotSaveFig(os.path.join(SDfig,'caseStudyChartOpCst'))
    plt.show()

    obj = 'hcGen'
    nSg = len(strategySet[obj])
    hcGenTable = np.zeros( 0 )
    linPoint = linPointsDict[feeder][obj][0]
    self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint)
    for strategy in strategySet[obj]:
        hcGenTable = np.r_[(hcGenTable,self.qpVarValue(strategy,obj,'norm'))]

    obj = 'hcLds'
    nSl = len(strategySet[obj])

    hcLdsTable = np.zeros( 0 )
    linPoint = linPointsDict[feeder][obj][0]
    self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint)
    for strategy in strategySet[obj]:
        hcLdsTable = np.r_[(hcLdsTable,self.qpVarValue(strategy,obj,'norm'))]

    fig,ax = plt.subplots(figsize=(5.5*(1-figFrac),3.0))
    ax.bar(0-0.25+ (0.25*2*np.arange(nSg)/(nSg-1)),100*hcGenTable,
                                            width=0.08,color=cm.matlab([0,1,2,5]) )
    
    ax.bar(1-0.25+ (0.25*2*np.arange(nSl)/(nSl-1)),100*hcLdsTable,
                                            width=0.08,color=cm.matlab([0,1,2,6]) )
    
    matlabClrs = [0,1,2,5,6]
    strategySetAug = strategySet['hcGen']+[strategySet['hcLds'][-1]]
    for i in range(len(strategySetAug)): #for the legend
        ax.plot(0,0,label=strategySetAug[i],color=cm.matlab(matlabClrs[i]))
    
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(('Gen.','Load'),rotation=90)
    ax.legend(fontsize='small')
    ax.set_ylabel('Power (kW)')
    plt.tight_layout()
    if 'pltSave' in locals():
        plotSaveFig(os.path.join(SDfig,'caseStudyChartHc'))
    plt.show()


if 'f_batchTest' in locals():
    feederTest = [8,24,'n27',0,'n1',17]
    for feeder in feederTest:
        self = main(feeder,'loadOnly',linPoint=0.6); self.initialiseOpenDss(); self.testQpVcpf(); plt.show()
        self.testQpScpf(); plt.show()
        self.testQpTcpf(); plt.show()

if 'f_modelValidation' in locals():
    feeder = 24
    self = main(feeder,'loadOnly',linPoint=1.0,pCvr=0.6); self.initialiseOpenDss(); 
    # V0,K0 = self.testQpVcpf()[1:]; plt.show()
    TLboth,TLestBoth,PLboth,PLestBoth,vErrBoth,Sset = self.testQpScpf()

    ii = 0
    fig,[ax0,ax1,ax2] = plt.subplots(ncols=3,figsize=(6.2,2.8))
    ax0.plot(Sset[ii],100*vErrBoth[ii])
    ax0.set_xlabel('Reactive power (kVAr)')
    ax0.set_ylabel('Voltage error, $\\dfrac{||V_{\mathrm{DSS}} - V_{\mathrm{Lin}}||_{2}}{||V_{\mathrm{DSS}}||_{2}}$, %')
    ax1.plot(Sset[ii],TLboth[ii],label='DSS.')
    ax1.plot(Sset[ii],TLestBoth[ii],label='Apx.')
    ax1.set_xlabel('Reactive power (kVAr)')
    ax1.set_ylabel('Losses (kW)')
    ax1.legend(fontsize='small')
    ax2.plot(Sset[ii],PLboth[ii],label='DSS.')
    ax2.plot(Sset[ii],PLestBoth[ii],label='Apx.')
    ax2.set_xlabel('Reactive power (kVAr)')
    ax2.set_ylabel('Load (kW)')
    ax2.legend(fontsize='small')
    plt.tight_layout(); 
    if 'pltSave' in locals():
        SN = os.path.join(SDfig,'modelValidationQ')
        plt.savefig(SN+'.png',bbox_inches='tight',pad_inches=0.01)
        plt.savefig(SN+'.pdf',bbox_inches='tight',pad_inches=0.01)

    plt.show()

    TL,TLest,PL,PLest,vErr,dxScale = self.testQpTcpf()
    ii = 0
    fig,[ax0,ax1,ax2] = plt.subplots(ncols=3,figsize=(6.2,2.8))
    ax0.plot(dxScale,100*vErr,'x-')
    ax0.set_xlabel('Tap')
    ax0.set_ylabel('Voltage error, $\\dfrac{||V_{\mathrm{DSS}} - V_{\mathrm{Lin}}||_{2}}{||V_{\mathrm{DSS}}||_{2}}$, %')
    ax1.plot(dxScale,TL,'x-',label='DSS.')
    ax1.plot(dxScale,TLest,'x-',label='Apx.')
    ax1.set_xlabel('Tap')
    ax1.set_ylabel('Losses (kW)')
    ax1.legend(fontsize='small')
    # ax1.set_ylim((0,10000))
    ax2.plot(dxScale,PL,'x-',label='DSS.')
    ax2.plot(dxScale,PLest,'x-',label='Apx.')
    ax2.set_xlabel('Tap')
    ax2.set_ylabel('Load (kW)')
    ax2.legend(fontsize='small')
    plt.tight_layout(); 
    
    if 'pltSave' in locals():
        plotSaveFig(os.path.join(SDfig,'modelValidationT'))
    plt.show()
    
if 'f_daisy' in locals():
    feeder = 8
    self = main(feeder,'loadOnly',linPoint=1.0); self.loadQpSet(); self.loadQpSln('full','opCst')
    self.plotNetBuses('qSlnPh',pltShow=False)
    SN = os.path.join(SDfig,'daisy')
    if 'pltSave' in locals():
        plotSaveFig(os.path.join(SDfig,'daisy'),pltClose=True)

if 'f_checkErrorSummary' in locals():
    pCvr = 0.6
    strategy='full'
    # heading = ['Feeder','Voltage error, $\|V_{\mathrm{Apx}} - V_{\mathrm{DSS}}\|_{2}/\|V_{\mathrm{DSS}}\|_{2}$, \%','Current error, $\|I_{\mathrm{Apx}} - I_{\mathrm{DSS}}\|_{2}/\|I_{\mathrm{Xfmr}}\|_{2}$, \%','Power error, $(P_{\mathrm{feeder}}^{\mathrm{Apx.}} - P_{\mathrm{feeder}}^{\mathrm{DSS.}})/P_{\mathrm{feeder}}$, \%']
    # heading = ['Feeder','Voltage error, \\ $\|\Delta V\|_{2}/\|V_{\mathrm{DSS}}\|_{2}$, \%','Current error, $\|\Delta I\|_{2}/\|I_{\mathrm{Xfmr}}\|_{2}$, \%','Power error, $(\Delta P)/P$, \%']
    heading = ['Feeder','Voltage error, $\delta _{V}$','Current error, $\delta _{I}$','Power error, $\delta _{P}$']
    
    resultTable = [heading]
    i = 1
    for feeder in feederSet:
        feederTidy = feederIdxTidy[feeder]
        resultTable.append([feederTidy])
        for obj in objSet:
            linPoints = linPointsDict[feeder][obj]
            voltages = []; currents = []; powers = []
            for linPoint in linPoints:
                self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint)
                voltages.append(self.qpSolutionDssError(strategy,obj,err='V')*100)
                currents.append(self.qpSolutionDssError(strategy,obj,err='I')*100)
                powers.append(self.qpSolutionDssError(strategy,obj,err='P')*100)
        resultTable[i].append( "%.3f" % (np.max(voltages)))
        resultTable[i].append( "%.3f" % (np.max(currents)))
        resultTable[i].append( "%.3f" % (np.max(powers)))
        i+=1
    print(*resultTable,sep='\n')
    TD = os.path.join(SD0,'tables\\')
    if 'pltSave' in locals(): basicTable('Maximum error, \%','checkErrorSummary',heading,resultTable[1:],TD)
    


if 'f_sensitivities_base' in locals():
    # sensitivity_base: just the results of smart inverter control
    # go through and calculate the benefits of full (SI) control versus nominal control
    strategies = ['full','phase','nomTap']
    opCst = np.zeros((3,len(feederSet)))
    pCvr = 0.6;     obj='opCst'
    i = 0
    heading = ['Control']
    for feeder in feederSet:
        heading.append(feederIdxTidy[feeder])
        j=0
        for strategy in strategies:
            self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPointsDict[feeder][obj][-1])
            opCst[j,i] = self.qpVarValue(strategy,obj,'norm',invType='Low')
            j+=1
        i+=1
    benefits = -100*(opCst[0:2] - opCst[2])
    benefits = np2lsStr(benefits,3)
    data = [ ['Full']+benefits[0],['Phase']+benefits[1] ]
    caption='Smart inverter benefits, \% of load'
    label='sensitivities_base'
    if 'pltSave' in locals(): basicTable(caption,label,heading,data,TD)
    print(heading); print(*data,sep='\n')
    
if 'f_sensitivities_invLoss' in locals() or 'f_sensitivities_efficacy' in locals():
    # sensitivities_invLoss, sensitivities_efficacy: the results considering smaller and larger losses
    pCvr = 0.6;     obj='opCst'
    strategies = ['full','nomTap']
    invTypes = ['None','Low','Hi']
    opCst = np.zeros((2,len(feederSet),3))
    wCst = np.zeros((2,len(feederSet),3))
    qCst = np.zeros((2,len(feederSet),3))
    i=0
    heading = ['Inv. Losses']
    for feeder in feederSet:
        heading.append(feederIdxTidy[feeder])
        j=0
        for strategy in strategies:
            k=0
            self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPointsDict[feeder][obj][-1])
            for invType in invTypes:
                opCst[j,i,k] = self.qpVarValue(strategy,obj,'norm',invType=invType)
                wCst[j,i,k] = self.qpVarValue(strategy,obj,'power',invType=invType)
                qCst[j,i,k] = np.linalg.norm(self.slnX[self.nPctrl:self.nSctrl],ord=1)
                k+=1
            j+=1
        i+=1

    benefits = -100*(opCst[0] - opCst[1])
    benefits = np2lsStr(benefits.T,3)
    wBenefit = -(wCst[0] - wCst[1])
    qCost = qCst[0]
    efficacy = np2lsStr( (1e3*wBenefit/qCost).T,2)

    data = [ ['None']+benefits[0],['Low (base)']+benefits[1],['High']+benefits[2] ]
    eData = [ ['None']+efficacy[0],['Low (base)']+efficacy[1],['High']+efficacy[2] ]
    
    label='sensitivities_invLoss'
    if 'pltSave' in locals(): basicTable(caption,label,heading,data,TD)
    print(heading); print(*data,sep='\n')

    # heading[0]='Inv. Losses'
    label='sensitivities_efficacy'
    caption='Smart inverter efficacy ($P/||Q||_{1}$), W/kVAr'
    if 'pltSave' in locals(): basicTable(caption,label,heading,eData,TD)
    print(heading); print(*eData,sep='\n')

if 'f_sensitivities_aCvr' in locals():
    # sensitivities_aCvr
    caption='Smart inverter benefits, \% of load'
    pCvrSet = [0.0,0.3,0.6,0.9]
    obj='opCst'
    strategies = ['full','nomTap']
    opCst = np.zeros((2,len(feederSet),len(pCvrSet)))
    i=0
    heading = ['$\\alpha_{\mathrm{CVR}}$']
    for feeder in feederSet:
        heading.append(feederIdxTidy[feeder])
        j=0
        for strategy in strategies:
            k=0
            for pCvr in pCvrSet:
                self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPointsDict[feeder][obj][-1])
                opCst[j,i,k] = self.qpVarValue(strategy,obj,'norm')
                k+=1
            j+=1
        i+=1

    benefits = -100*(opCst[0] - opCst[1])
    benefits = np2lsStr(benefits.T,3)
    data = [ ['0.0']+benefits[0],['0.3']+benefits[1],['0.6']+benefits[2],['0.9']+benefits[3] ]
    
    label='sensitivities_aCvr'
    if 'pltSave' in locals(): basicTable(caption,label,heading,data,TD)
    print(heading); print(*data,sep='\n')
    
if 'f_sensitivities_loadPoint' in locals():
    opCst = np.zeros((2,len(feederSet),3))
    strategies = ['full','nomTap']; obj='opCst'; pCvr=0.6
    i=0
    heading=['Loading']
    for feeder in feederSet:
        heading.append(feederIdxTidy[feeder])
        j=0
        for strategy in strategies:
            k=0
            for linPoint in linPointsDict[feeder][obj]:
                self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint)
                opCst[j,i,k] = self.qpVarValue(strategy,obj,'norm')
                k+=1
            j+=1
        i+=1

    benefits = -100*(opCst[0] - opCst[1])
    benefits = np2lsStr(benefits.T,3)
    data = [ ['10\\%']+benefits[0],['60\\%']+benefits[1],['100\\%']+benefits[2] ]

    label='sensitivities_loadPoint'
    if 'pltSave' in locals(): basicTable(caption,label,heading,data,TD)
    print(heading); print(*data,sep='\n')
    
    
if 'f_solutionError' in locals():
    feeder = 26
    # self = main(feeder,'loadOnly',linPoint=1.0); self.loadQpSet(); self.loadQpSln('full','opCst'); self.plotArcy()
    # self = main(feeder,'loadOnly',linPoint=1.0); self.loadQpSet(); self.loadQpSln('full','hcLds'); self.plotArcy()
    # self = main(feeder,'loadOnly',linPoint=1.0); self.loadQpSet(); self.loadQpSln('full','hcGen'); self.plotArcy()

    self = main(feeder,'loadOnly',linPoint=1.0); self.loadQpSet(); 
    self.loadQpSln('full','opCst')
    self.plotArcy(pltShow=False)
    if 'pltSave' in locals():
        plotSaveFig(os.path.join(SDfig,'solutionErrorOpCst'),pltClose=True)
        
    self.loadQpSln('full','hcLds'); 
    self.plotArcy(pltShow=False)
    if 'pltSave' in locals():
        plotSaveFig(os.path.join(SDfig,'solutionErrorHcLds'),pltClose=True)
    
    self = main(feeder,'loadOnly',linPoint=0.1); self.loadQpSet(); 
    self.loadQpSln('full','hcGen'); 
    self.plotArcy(pltShow=False)
    if 'pltSave' in locals():
        plotSaveFig(os.path.join(SDfig,'solutionErrorHcGen'),pltClose=True)
    
    # feeder = 18
    # self = main(feeder,'loadOnly',linPoint=0.6); self.loadQpSet(); self.loadQpSln('full','opCst'); self.plotArcy()
    # self = main(feeder,'loadOnly',linPoint=0.6); self.loadQpSet(); self.loadQpSln('full','hcLds'); self.plotArcy()
    # self = main(feeder,'loadOnly',linPoint=0.1); self.loadQpSet(); self.loadQpSln('full','hcGen'); self.plotArcy()

# if 'f_caseStudy' in locals():
    # self = 

# self = main(8,'loadOnly',linPoint=0.1); self.loadQpSet(); self.loadQpSln('full','hcGen'); self.showQpSln()
# self = main('n1','loadOnly',linPoint=0.1); self.runCvrQp('full','hcGen')
# self = main('n1','plotOnly')
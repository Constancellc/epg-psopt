import lineariseDssModels, sys, os, pickle, random, time
from importlib import reload
import numpy as np
from dss_python_funcs import vecSlc, getBusCoords, getBusCoordsAug, tp_2_ar, plotSaveFig, basicTable, np2lsStr, sdt, createYbus,find_tap_pos, getBranchNames, makeYprim, getBranchYprims
from dss_voltage_funcs import getCapPos
import matplotlib.pyplot as plt
from matplotlib import cm, rc
plt.style.use('tidySettings')
from lineariseDssModels import dirPrint

FD = sys.argv[0]

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr','123busCvr']

feederAllTidy = {'13bus':'13 Bus','34bus':'34 Bus','123bus':'123 Bus','8500node':'8500 Node','epriJ1':'Ckt. J1','epriK1':'Ckt. K1','epriM1':'Ckt. M1','epri5':'Ckt. 5','epri7':'Ckt. 7','epri24':'Ckt. 24','123busCvr':'123 Bus','epriK1cvr':'Ckt. K1','n1':'EULVa','n27':'EULVa-r','eulv':'EULV','n4':'Nwk. 4','n10':'Nwk. 10'}

feederIdxTidy = {5:'13 Bus',6:'34 Bus',8:'123 Bus',9:'8500 Node',19:'Ckt. J1',20:'Ckt. K1',21:'Ckt. M1',17:'Ckt. 5',18:'Ckt. 7',22:'Ckt. 24',26:'123 Bus',24:'Ckt. K1','n1':'EULVa','n27':'EULVa-r',0:'EULV','n4':'Nwk. 4','n10':'Nwk. 10'}



# # THREE EXAMPLES from transmission papers?
# x = [-100,0,100]
# y= [3.25,2.9,3.82]
# xL = np.linspace(-120,120,1000)

# x = [0,100,200]
# y = [0,100,200]

# # paper 1
# c = np.polyfit(x,y,2)
# plt.plot(xL,c[0]*(xL**2) + c[1]*xL + x[0]);plt.show()
# plt.plot(x,y,'x');
# plt.plot(xL,2*c[0]*xL + c[1]);plt.show()
# plt.plot(xL,1000*(2*c[0]*xL + c[1]));plt.show()

# plt.plot(xL,c[0]*(xL**2) + c[1]*xL + c[2]);plt.show()

# f_valueComparisonChart = 1
# f_plotOnly = 1
# f_plotInvLoss = 1
# f_batchTest = 1
# f_currentErrors = 1
# f_modelValidation = 1
# f_modelValidationSet = 1
# f_modelValidationSetPscc = 1
# f_daisy = 1
# f_123busVlts = 1
# f_123busVltsPscc = 1
# f_solutionError = 1
# f_solutionErrorPscc = 1
# f_caseStudyChart = 1
# t_sensitivities_base_pscc = 1
# t_sensitivities_base = 1
# f_sensitivities_all = 1
# f_sensitivities_aCvr = 1
# f_sensitivities_loadPoint = 1
# t_checkErrorSummary = 1
# t_networkSummary = 1
# t_networkSummaryPscc = 1
# t_networkMetrics = 1
# f_epriK1detail = 1
# f_costFunc = 1
# f_37busVal = 1
# t_thssSizes = 1
# f_thssSparsity = 1
# f_runTsAnalysis = 1
# f_plotTsAnalysis = 1
# f_convCapability = 1
# f_scaling = 1
# f_costFuncXmpl = 1
# f_costFuncAsym = 1
# f_psdFigs = 1
# t_loadModelVecs = 1
# f_uncSolution = 1
# t_uncBoundViolations = 1
# f_hcParamsFig = 1

# pltSave=1

SD0 = os.path.join(os.path.join(os.path.expanduser('~')), 'Documents','DPhil','papers','psjul19')
SDT = sdt()
SDfig = os.path.join(SD0,'figures')
TD = os.path.join(SD0,'tables\\')


# feederSet = [0,17,'n1',26,24,'n27']
# feederSet = [0,17,'n1',26,24,'n27']
feederSet = [17,18,'n1','n10',26,24,'n27']
# feederSet = [24]
# feederSet = [8]
# feederSet = [0,5] # fast

strategySet = { 'opCst':['full','phase','nomTap','load','loss'],'hcGen':['full','phase','nomTap','maxTap'],'hcLds':['full','phase','nomTap','minTap'] }
# objSet = ['opCst','hcGen','hcLds']
objSet = ['opCst']

ssSmart = {'full':'Full','phase':'Phase','nomTap':'Nominal','load':'Load','loss':'Loss'}

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
    
if 'f_epriK1detail' in locals():
    self = main('epriK1','plotOnly')
    self.plotNetwork(pltShow=False)
    plt.annotate('Substation', (2637962.769000+4e2, 623062.072000),xycoords='data')
    plt.scatter(2642631.410000, 628706.594000,zorder=+20,color=cm.matlab(3),edgecolor='k')
    plt.annotate('Capacitor\nbank', (2642631.410000-39e2, 628706.594000-9e2),xycoords='data',zorder=20)
    plt.scatter(2644756.213000, 631315.808000,marker='p',zorder=+20,color=cm.matlab(4),s=60,edgecolor='k')
    plt.annotate('Generator', (2644756.213000+6e2, 631315.808000-3e2),xycoords='data')
    if 'pltSave' in locals(): plotSaveFig(os.path.join(SDT,'c2litreview','c2figures','epriK1detail'),pltClose=True)
    plt.show()
    

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
        ax0.set_xlabel('Apparent Power $S_{\mathrm{invr}}$, % of $S_{rated}$')
        
        ax1.plot(100*x/sRated, 100*x/(Plss + x)); 
        ax1.set_ylim((86,100));   ax1.set_xlim((0.0,100))
        ax1.set_xlabel('Apparent Power $S_{\mathrm{invr}}$, % of $S_{rated}$')
        ax1.set_ylabel('Efficiency, %')
    ax0.legend(('Low loss','High loss'))
    plt.tight_layout()
    if 'pltSave' in locals():
        SN = os.path.join(SDfig,'invLoss')
        plt.savefig(SN+'.png',bbox_inches='tight', pad_inches=0.01)
        plt.savefig(SN+'.pdf',bbox_inches='tight', pad_inches=0.01)
    
    plt.show()
    
    fig,ax0 = plt.subplots(ncols=1,figsize=(2.8,2.6))
    for i in range(len(lossFracSmaxs)):
        lossFracSmax = lossFracSmaxs[i]
        lossFracS0 = lossFracS0s[i]
        Plss = 0.01*sRated*( lossFracS0 + lossFracSmax*(x**2) )
        
        ax0.plot(100*x/sRated,100*Plss/sRated);
        ax0.set_ylabel('Inverter Losses, % of $S_{rated}$')
        ax0.set_ylim((0.0,15.0)); ax0.set_xlim((0.0,100))
        ax0.set_xlabel('Apparent Power $S_{\mathrm{invr}}$, % of $S_{rated}$')
    ax0.legend(('Low loss','High loss'))
    plt.tight_layout()
    if 'pltSave' in locals(): plotSaveFig(os.path.join(sdt('t3','f'),'invLossLss'),pltClose=True)
    
    fig,ax1 = plt.subplots(ncols=1,figsize=(2.8,2.6))
    for i in range(len(lossFracSmaxs)):
        lossFracSmax = lossFracSmaxs[i]
        lossFracS0 = lossFracS0s[i]
        Plss = 0.01*sRated*( lossFracS0 + lossFracSmax*(x**2) )
    
        ax1.plot(100*x/sRated, 100*x/(Plss + x)); 
        ax1.set_ylim((86,100));   ax1.set_xlim((0.0,100))
        ax1.set_xlabel('Apparent Power $S_{\mathrm{invr}}$, % of $S_{rated}$')
        ax1.set_ylabel('Throughput, %')
    plt.tight_layout()
    if 'pltSave' in locals(): plotSaveFig(os.path.join(sdt('t3','f'),'invLossThr'),pltClose=True)

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
    # psccVersion = 1 # TOGGLE ME HERE to change what the annotation says
    
    pCvr = 0.6
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
                # linPoints = [linPointsDict[feeder][obj][-1]]
                j=0
                for linPoint in linPoints:
                    self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint,method='fot')
                    val = self.qpVarValue(strategy,obj,'norm')
                    
                    if obj=='opCst' and j==0: opCstTableA[i].append( val )
                    if obj=='opCst' and j==1: opCstTableB[i].append( val )
                    if obj=='opCst' and j==2: opCstTableC[i].append( val )
                    if obj=='hcGen': hcGenTable[i].append( str(val)[:7] )
                    if obj=='hcLds': hcLdsTable[i].append( str(val)[:7] )
                    j+=1
        i+=1
    
    w = [0.0,0.0,1.0]
    opCstTable_ = (w[0]*np.array(opCstTableA[2:]) + w[1]*np.array(opCstTableB[2:]) + w[2]*np.array(opCstTableC[2:])).tolist()
    # opCstTable_ = (np.array(opCstTableA[2:])).tolist()
    
    for i in range(2,len(feederSet)+2):
        for j in range(len(strategySet['opCst'])):
            opCstTable[i].append( "%.6f" % opCstTable_[i-2][j] )
        
    print(*opCstTable,sep='\n')
    # print(*hcGenTable,sep='\n')
    # print(*hcLdsTable,sep='\n')
    
    # tables={'opCst':opCstTable,'hcGen':hcGenTable,'hcLds':hcLdsTable}
    # ylabels={'opCst':'Total P in / Total P in Ref., %','hcGen':'Generation (% of nominal load)','hcLds':'Load (% of nominal load)'}
    ylims={'opCst':(92.0,106.0),'hcGen':(-0,220),'hcLds':(-0,100)}
    
    tables={'opCst':opCstTable}
    colorSet = {'opCst':cm.matlab([0,1,2,3,4]),'hcGen':cm.matlab([0,1,2,5]),'hcLds':cm.matlab([0,1,2,6])}
    
    # objSet = ['opCst','hcGen','hcLds']
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
            ax.plot(0,0,label=ssSmart[strategySet[obj][i]],color=colorSet[obj][i])

    ax.plot((3.5,3.5),ylims[obj],'k--')
    if 'psccVersion' in locals():
        ax.text(3.6,ylims[obj][-1]-2.8,'With tap\ncontrol')
        ax.text(-0.5,ylims[obj][-1]-2.8,'Without\ntap control')    
    else:
        ax.text(3.6,ylims[obj][-1]-2.8,'With\nregs.')
        ax.text(-0.5,ylims[obj][-1]-2.8,'No\nregs.')
    ax.set_ylim(ylims[obj])
    ax.set_xticks(np.arange(len(feederSet)))
    ax.set_xticklabels(feederTidy,rotation=90)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),fontsize='small',title='Control')
    # ax.set_ylabel(ylabels[obj])
    ax.set_ylabel('Power ratio $\dfrac{P_{\mathrm{feeder}}( x^{\\dagger} )}{ P_{\mathrm{feeder}}(0)}$, %')
    # ax.set_title(obj)
    plt.tight_layout()
    # if 'pltSave' in locals(): plotSaveFig(os.path.join(SDfig,'valueComparisonChart_'+obj))
    if 'psccVersion' in locals():
        FD = r'C:\Users\Matt\Documents\DPhil\papers\pscc20\full\figures'
        if 'pltSave' in locals(): plotSaveFig(os.path.join(FD,'valueComparisonChart_'+obj),pltClose=True)
    else:
        if 'pltSave' in locals(): plotSaveFig(os.path.join(sdt('t3','f'),'valueComparisonChart_'+obj),pltClose=True)
    
    
    asdasd = np.array(opCstTable_)
            
    plt.show()



if 'f_caseStudyChart' in locals():
    ylabels={'opCst':'Power ratio $\dfrac{P_{\mathrm{feeder}}( x^{\\dagger} )}{ P_{\mathrm{feeder}}(0)}$, %','hcGen':'Generation (% of nominal load)','hcLds':'Load (% of nominal load)'}
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
            self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint,method='fot')
            opCstTable[j,i] = self.qpVarValue(strategy,obj,'norm')
            j+=1
        i+=1
    
    figFrac = 1.0
    fig,ax = plt.subplots(figsize=(5.5*figFrac,2.5))
    for i in range(3):
        for j in range(nS):
            ax.bar(i-0.25+ (0.25*2*np.arange(nS)/(nS-1)),100*opCstTable[i],
                                                                    width=0.08,color=cm.matlab(range(nS)) )
    for i in range(nS): #for the legend
        ax.plot(0,0,label=ssSmart[strategySet[obj][i]])
    
    ax.set_ylim((95,100.5))
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(('10% load','60% load','100% load'))
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),title='Control')
    ax.set_ylabel(ylabels[obj])
    # ax.set_title(obj)
    plt.tight_layout()
    if 'pltSave' in locals():
        plotSaveFig(os.path.join(SDfig,'caseStudyChartOpCst'))
        plotSaveFig(os.path.join(sdt('t3','f'),'caseStudyChartOpCst'))
    plt.show()

    # obj = 'hcGen'
    # nSg = len(strategySet[obj])
    # hcGenTable = np.zeros( 0 )
    # linPoint = linPointsDict[feeder][obj][0]
    # self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint)
    # for strategy in strategySet[obj]:
        # hcGenTable = np.r_[(hcGenTable,self.qpVarValue(strategy,obj,'norm'))]

    # obj = 'hcLds'
    # nSl = len(strategySet[obj])

    # hcLdsTable = np.zeros( 0 )
    # linPoint = linPointsDict[feeder][obj][0]
    # self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint)
    # for strategy in strategySet[obj]:
        # hcLdsTable = np.r_[(hcLdsTable,self.qpVarValue(strategy,obj,'norm'))]

    # fig,ax = plt.subplots(figsize=(5.5*(1-figFrac),3.0))
    # ax.bar(0-0.25+ (0.25*2*np.arange(nSg)/(nSg-1)),100*hcGenTable,
                                            # width=0.08,color=cm.matlab([0,1,2,5]) )
    
    # ax.bar(1-0.25+ (0.25*2*np.arange(nSl)/(nSl-1)),100*hcLdsTable,
                                            # width=0.08,color=cm.matlab([0,1,2,6]) )
    
    # matlabClrs = [0,1,2,5,6]
    # strategySetAug = strategySet['hcGen']+[strategySet['hcLds'][-1]]
    # for i in range(len(strategySetAug)): #for the legend
        # ax.plot(0,0,label=strategySetAug[i],color=cm.matlab(matlabClrs[i]))
    
    # ax.set_xticks(np.arange(2))
    # ax.set_xticklabels(('Gen.','Load'),rotation=90)
    # ax.legend(fontsize='small')
    # ax.set_ylabel('Power (kW)')
    # plt.tight_layout()
    # if 'pltSave' in locals():
        # plotSaveFig(os.path.join(SDfig,'caseStudyChartHc'))
    # plt.show()


if 'f_batchTest' in locals():
    feederTest = [26,24,'n27',0,'n1',17]
    feederTest = [26]
    for feeder in feederTest:
        self = main(feeder,'loadOnly',linPoint=0.6); self.initialiseOpenDss(); 
        self.testQpVcpf(); plt.show()
        self.testQpScpf(); plt.show()
        self.testQpTcpf(); plt.show()

if 'f_currentErrors' in locals():
    feederTest = [26,17]
    # feederTest = [17]
    # feederTest = [26]
    for feeder in feederTest:
        self = main(feeder,'loadOnly',linPoint=0.6); 
        self.recreateKc2i = 1
        self.recreateMc2v = 1
        self.initialiseOpenDss()
        self.makeCvrQp()
        k,ice,iae = self.testQpVcpf()[2:]; plt.close()
        fig,ax = plt.subplots(figsize=(3.8,2.8))
        ax.plot(k,ice,label='Complex')
        ax.plot(k,iae,label='Real')
        ax.set_xlabel('Load scale factor')
        ax.set_ylabel('Error, $\\frac{1}{\\sqrt{N}}|\!| (I_{\mathrm{Lin}} - I_{\mathrm{DSS}})/I_{\mathrm{Xfmr}} |\!|_{2}$, %')
        ylm = ax.get_ylim()
        ax.set_ylim((0,ylm[1]))
        ax.legend(title='Domain of $I$')
        plt.tight_layout()
        if 'pltSave' in locals(): plotSaveFig(os.path.join(sdt('t3','f'),'currentErrors_'+self.feeder),pltClose=True)
        plt.show()


if 'f_scaling' in locals():
    # feederTest = [18,26]
    # feederTest = [26]
    feederTest = [18]
    for feeder in feederTest:
        linPoint = linPointsDict[feeder]['all'][-1]
        self = main(feeder,'loadOnly',linPoint=linPoint,pCvr=0.6,method='fot'); self.initialiseOpenDss(); 
        vae06,k06 = self.testQpVcpf(k=np.arange(0.6,1.425,0.025)*linPoint)[1:3]; plt.close()
        
        self = main(feeder,'loadOnly',linPoint=linPoint,pCvr=0.0,method='fot'); self.initialiseOpenDss(); 
        vae00,k00 = self.testQpVcpf(k=np.arange(0.6,1.425,0.025)*linPoint)[1:3]; plt.close()
        
        fig,ax = plt.subplots(figsize=(3.8,2.8))
        ax.plot(k06/linPoint,100*vae06,label='$\\alpha_{\mathrm{CVR}}=0.6$');
        ax.plot(k00/linPoint,100*vae00,label='$\\alpha_{\mathrm{CVR}}=0.0$');
        ax.set_xlabel('Load scaling factor')
        ax.set_ylabel('Error, $ |\!| V_{\mathrm{Lin}} - V_{\mathrm{DSS}} |\!| / |\!|V_{\mathrm{DSS}} |\!|$, %')
        ax.legend(title='CVR factor')
        ylm = ax.get_ylim()
        ax.set_ylim((0,ylm[1]))
        plt.tight_layout()
        if 'pltSave' in locals(): plotSaveFig(os.path.join(sdt('t3','f'),'scalingCvr_'+self.feeder),pltClose=True)
        plt.show()
        # self.testQpScpf(); plt.show()
        # self.testQpTcpf(); plt.show()

if 'f_convCapability' in locals():
    sMax = 100
    qMax = 60
    pqXff = np.sqrt(sMax**2 - qMax**2)
    fig,ax = plt.subplots(figsize=(3.6,3.0))
    ax.plot(0,0,'.',color=cm.matlab(0))
    ax.plot([0,0],[-qMax,qMax],'--')
    ax.plot([0,pqXff],[-qMax,-qMax],color=cm.matlab(0))
    ax.plot([0,pqXff],[qMax,qMax],color=cm.matlab(0))
    thta = np.linspace(-np.arcsin(qMax/sMax),np.arcsin(qMax/sMax),1000)
    ax.plot(sMax*np.cos(thta),sMax*np.sin(thta),color=cm.matlab(0))
    ax.axis('equal')
    ax.set_xlabel('$P_{\mathrm{invr}}$, % of $S_{\mathrm{Rated}}$')
    ax.set_ylabel('$Q_{\mathrm{invr}}$, % of $S_{\mathrm{Rated}}$')
    ax.set_xlim((-40,140));
    ax.set_ylim((-90,90));
    plt.tight_layout()
    if 'pltSave' in locals(): plotSaveFig(os.path.join(sdt('t3','f'),'convCapability'),pltClose=True)
    plt.show()

if 'f_modelValidation' in locals():
    feederSet = [18,24]
    feederSet = [24]
    feederSet = [18]
    for feeder in feederSet:
        linPoints = linPointsDict[feeder]['all']
        self = main(feeder,'loadOnly',linPoint=linPoints[-1],pCvr=0.6); self.initialiseOpenDss(); 
        TLboth,TLestBoth,PLboth,PLestBoth,vErrBoth,iErrBoth,Sset = self.testQpScpf(); 
        
        plt.close(); plt.close()
        
        figSze = (4.5,5.0)
        if feeder==18:
            figSze = (5.5,5.0)
        
        ii = 0
        # fig,[ax0,ax3,ax1,ax2] = plt.subplots(nrows=2,ncols=2,sharex=True,figsize=figSze)
        fig,[[ax0,ax3],[ax1,ax2]] = plt.subplots(nrows=2,ncols=2,sharex=True,figsize=figSze)
        ax0.plot(Sset[ii],100*vErrBoth[ii])
        # ax0.set_xlabel('Rctv. power, kVAr')
        ax0.set_ylabel('$V$ error, $\\dfrac{||V_{\mathrm{DSS}} - V_{\mathrm{Lin}}||_{2}}{||V_{\mathrm{DSS}}||_{2}}$, %')
        ax3.plot(Sset[ii],100*iErrBoth[ii])
        # ax3.set_xlabel('Rctv. power, kVAr')
        ax3.set_ylabel('$I$ error, $\\dfrac{1}{\\sqrt{N_{\mathrm{I}}}}||\\dfrac{I_{\mathrm{DSS}} - I_{\mathrm{Lin}}}{I_{\mathrm{Xfmr}}}||_{2}$, %')
        ax1.plot(Sset[ii],TLboth[ii],label='DSS.')
        ax1.plot(Sset[ii],TLestBoth[ii],label='Apx.')
        ax1.set_xlabel('$Q_{\mathrm{\,invt}}$ (per gen.), kVAr')
        ax1.set_ylabel('Losses, kW')
        ax1.legend(fontsize='small')
        ax2.plot(Sset[ii],PLboth[ii]*1e-3,label='DSS.')
        ax2.plot(Sset[ii],PLestBoth[ii]*1e-3,label='Apx.')
        ax2.set_xlabel('$Q_{\mathrm{\,invt}}$ (per gen.), kVAr')
        ax2.set_ylabel('Load, MW')
        ax2.legend(fontsize='small')
        plt.tight_layout(); 
        if 'pltSave' in locals():
            # SN = os.path.join(SDfig,'modelValidationQ')
            # plt.savefig(SN+'.png',bbox_inches='tight',pad_inches=0.01)
            # plt.savefig(SN+'.pdf',bbox_inches='tight',pad_inches=0.01)
            plotSaveFig(os.path.join(sdt('t3','f'),'modelValidationQ'+self.feeder),pltClose=True)

        plt.show()
        if self.nT>0:
            TL,TLest,PL,PLest,vErr,iErr,dxScale = self.testQpTcpf()
            plt.close(); 
            # plt.show()
            ii = 0
            # fig,[ax0,ax3,ax1,ax2] = plt.subplots(ncols=4,figsize=figSze)
            fig,[[ax0,ax3],[ax1,ax2]] = plt.subplots(nrows=2,ncols=2,sharex=True,figsize=figSze)
            ax0.plot(dxScale,100*vErr,'x-')
            # ax0.set_xlabel('Tap')
            ax0.set_ylabel('$V$ error, $\\dfrac{||V_{\mathrm{DSS}} - V_{\mathrm{Lin}}||_{2}}{||V_{\mathrm{DSS}}||_{2}}$, %')
            ax3.plot(dxScale,100*iErr)
            # ax3.set_xlabel('Tap')
            ax3.set_ylabel('$I$ error, $\\dfrac{1}{\\sqrt{N_{\mathrm{I}}}}||\\dfrac{I_{\mathrm{DSS}} - I_{\mathrm{Lin}}}{I_{\mathrm{Xfmr}}}||_{2}$, %')
            ax1.plot(dxScale,TL,'x-',label='DSS.')
            ax1.plot(dxScale,TLest,'x-',label='Apx.')
            ax1.set_xlabel('Tap')
            ax1.set_ylabel('Losses (kW)')
            ax1.legend(fontsize='small')
            # ax1.set_ylim((0,10000))
            ax2.plot(dxScale,PL*1e-3,'x-',label='DSS.')
            ax2.plot(dxScale,PLest*1e-3,'x-',label='Apx.')
            ax2.set_xlabel('Tap')
            ax2.set_ylabel('Load, MW')
            ax2.legend(fontsize='small')
            plt.tight_layout(); 
            
            if 'pltSave' in locals():
                # plotSaveFig(os.path.join(SDfig,'modelValidationT'+self.feeder))
                plotSaveFig(os.path.join(sdt('t3','f'),'modelValidationT'+self.feeder),pltClose=True)
            plt.show()

if 'f_modelValidationSetPscc' in locals():
    feederSet = [18,24]
    feederSet = [26,17,18,24]
    # feederSet = [26]
    for feeder in feederSet:
        linPoints = linPointsDict[feeder]['all']
        self = main(feeder,'loadOnly',linPoint=linPoints[-1],pCvr=0.6,method='fot'); self.initialiseOpenDss()
        self.recreateKc2i = 1
        self.recreateMc2v = 1
        self.initialiseOpenDss()
        self.makeCvrQp()
        
        self.qLim = 4 # increase slightly
        
        print('Solve 1...')
        TLboth,TLestBoth,PLboth,PLestBoth,vErrBoth,iErrBoth,Sset,iN,icN,iaN,vcErrBoth = self.testQpScpf(); 
        # if self.nT>0:
            # print('Solve 2...')
            # TL,TLest,PL,PLest,vErr,iErr,dxScale,iNt,icNt,iaNt = self.testQpTcpf()
        
        self = main(feeder,'loadOnly',linPoint=linPoints[-1],pCvr=0.0,method='fot'); self.initialiseOpenDss()
        self.recreateKc2i = 1
        self.recreateMc2v = 1
        self.initialiseOpenDss()
        self.makeCvrQp()
        
        self.qLim = 4 # increase slightly
        print('Solve 1...')
        TLboth,TLestBoth,PLboth,PLestBoth,vErrBoth,iErrBoth,Sset2,iN,icN,iaN,vcErrBoth2 = self.testQpScpf(); 
        plt.close('all')
        ii=0
        
        figSze = (3.0,2.8)
        
        figname='modelValidationSetVpscc'
        sd = sdt('t3','f')
        fig,ax=plt.subplots(figsize=figSze)
        ax.plot(Sset[ii],100*vcErrBoth[ii],label='$\\alpha_{\mathrm{CVR}} = 0.6$')
        ax.plot(Sset2[ii],100*vcErrBoth2[ii],label='$\\alpha_{\mathrm{CVR}} = 0.0$')
        # ax.plot(Sset[ii+1],100*vcErrBoth[ii+1],label='P')
        ax.set_xlabel('$Q_{\mathrm{\,invr}}$ (per invr.), kVAr')
        ax.set_ylabel('Voltage error, $\dfrac{|\!|V_{\mathrm{Lin}} - V_{\mathrm{DSS}}|\!|_{2}}{|\!|V_{\mathrm{DSS}}|\!|_{2}}$ %')
        # ax.set_ylabel('Voltage error, $|\!|V_{\mathrm{Lin}} - V_{\mathrm{DSS}}|\!|/|\!|V_{\mathrm{DSS}}|\!| %')
        ax.legend(fontsize='small',title='CVR factor')
        plt.tight_layout()
        # psccFD = r'C:\Users\Matt\Documents\DPhil\papers\pscc20\full\figures'
        psccFD = r'D:\DocumentsD_l\DPhil\papers\pscc20\full\figures'
        if 'pltSave' in locals(): plotSaveFig(os.path.join(psccFD,figname+self.feeder),pltClose=True)
        plt.show()


if 'f_modelValidationSet' in locals():
    feederSet = [18,24]
    feederSet = [26]
    pltSet = ['lds','lss','curr','vlt']
    pltSet = ['vlt']
    for feeder in feederSet:
        linPoints = linPointsDict[feeder]['all']
        self = main(feeder,'loadOnly',linPoint=linPoints[-1],pCvr=0.6,method='fot'); self.initialiseOpenDss()
        if 'curr' in pltSet or 'vlt' in pltSet:
            self.recreateKc2i = 1
            self.recreateMc2v = 1
            self.initialiseOpenDss()
            self.makeCvrQp()
    
        print('Solve 1...')
        TLboth,TLestBoth,PLboth,PLestBoth,vErrBoth,iErrBoth,Sset,iN,icN,iaN,vcErrBoth = self.testQpScpf(); 
        if self.nT>0:
            print('Solve 2...')
            TL,TLest,PL,PLest,vErr,iErr,dxScale,iNt,icNt,iaNt = self.testQpTcpf()
        
        plt.close('all')
        ii=0
        
        figSze = (2.4,2.8)
        
        if 'lds' in pltSet:
            figname='modelValidationSetLoad'
            sd = sdt('t3','f')
            fig,ax=plt.subplots(figsize=figSze)
            ax.plot(Sset[ii],PLboth[ii]*1e-3,label='DSS.')
            ax.plot(Sset[ii],PLestBoth[ii]*1e-3,label='Lin.')
            ax.set_xlabel('$Q_{\mathrm{\,invt}}$ (per gen.), kVAr')
            ax.set_ylabel('Load, MW')
            plt.tight_layout()
            if 'pltSave' in locals(): plotSaveFig(os.path.join(sd,figname+self.feeder),pltClose=True)
            plt.show()
            if self.nT>0:
                figname='modelValidationSetLoadTaps'
                fig,ax=plt.subplots(figsize=figSze)
                ax.plot(dxScale,PL*1e-3,'x-',label='DSS.')
                ax.plot(dxScale,PLest*1e-3,'x-',label='Lin.')
                ax.set_xlabel('Tap')
                ax.set_ylabel('Load, MW')
                ax.legend()
                plt.tight_layout()
                if 'pltSave' in locals(): plotSaveFig(os.path.join(sd,figname+self.feeder),pltClose=True)
                plt.show()
        
        if 'lss' in pltSet:
            figname='modelValidationSetLss'
            sd = sdt('t3','f')
            fig,ax=plt.subplots(figsize=figSze)
            ax.plot(Sset[ii],TLboth[ii],label='DSS.')
            ax.plot(Sset[ii],TLestBoth[ii],label='Lin.')
            ax.set_xlabel('$Q_{\mathrm{\,invt}}$ (per gen.), kVAr')
            ax.set_ylabel('Losses, kW')
            plt.tight_layout()
            if 'pltSave' in locals(): plotSaveFig(os.path.join(sd,figname+self.feeder),pltClose=True)
            plt.show()
            if self.nT>0:
                figname='modelValidationSetLssTaps'
                fig,ax=plt.subplots(figsize=figSze)
                ax.plot(dxScale,TL,'x-',label='DSS.')
                ax.plot(dxScale,TLest,'x-',label='Lin.')
                ax.set_xlabel('Tap')
                ax.set_ylabel('Losses, kW')
                ax.legend()
                plt.tight_layout()
                if 'pltSave' in locals(): plotSaveFig(os.path.join(sd,figname+self.feeder),pltClose=True)
                plt.show()
                
        if 'curr' in pltSet:
            figname='modelValidationSetI'
            sd = sdt('t3','f')
            fig,ax=plt.subplots(figsize=figSze)
            ax.plot(Sset[ii],100*iN[ii],label='DSS.')
            ax.plot(Sset[ii],100*icN[ii],label='Lin. clx.')
            ax.plot(Sset[ii],100*iaN[ii],'--',label='Lin. mag.',color=cm.matlab(1))
            ax.set_xlabel('$Q_{\mathrm{\,invt}}$ (per gen.), kVAr')
            ax.set_ylabel('Current, % of $I_{\mathrm{rated}}$')
            ax.legend(fontsize='small')
            plt.tight_layout()
            if 'pltSave' in locals(): plotSaveFig(os.path.join(sd,figname+self.feeder),pltClose=True)
            plt.show()
            if self.nT>0:
                figname='modelValidationSetITaps'
                fig,ax=plt.subplots(figsize=figSze)
                ax.plot(dxScale,100*iNt,'x-',label='DSS.')
                ax.plot(dxScale,100*icNt,'x-',label='Lin. clx.')
                ax.plot(dxScale,100*iaNt,'x--',label='Lin. mag.',color=cm.matlab(1))
                ax.set_xlabel('Tap')
                ax.set_ylabel('Current, % of $I_{\mathrm{rated}}$')
                ax.legend(fontsize='small')
                plt.tight_layout()
                if 'pltSave' in locals(): plotSaveFig(os.path.join(sd,figname+self.feeder),pltClose=True)
                plt.show()
        if 'vlt' in pltSet:
            figname='modelValidationSetV'
            sd = sdt('t3','f')
            fig,ax=plt.subplots(figsize=figSze)
            ax.plot(Sset[ii],100*vcErrBoth[ii],label='DSS.')
            ax.plot(Sset[ii+1],100*vcErrBoth[ii+1],label='DSS.')
            ax.set_xlabel('$Q_{\mathrm{\,invt}}$ (per gen.), kVAr')
            ax.set_ylabel('Votlage error, %')
            ax.legend(fontsize='small')
            plt.tight_layout()
            # if 'pltSave' in locals(): plotSaveFig(os.path.join(sd,figname+self.feeder),pltClose=True)
            # plt.show()
        
        # ax0.plot(Sset[ii],100*vErrBoth[ii])
        # # ax0.set_xlabel('$Q_{\mathrm{\,invt}}$ (per gen.), kVAr')
        # ax0.set_ylabel('$V$ error, $\\dfrac{||V_{\mathrm{DSS}} - V_{\mathrm{Lin}}||_{2}}{||V_{\mathrm{DSS}}||_{2}}$, %')
        # ax3.plot(Sset[ii],100*iErrBoth[ii])
        # # ax3.set_xlabel('$Q_{\mathrm{\,invt}}$ (per gen.), kVAr')
        # ax3.set_ylabel('$I$ error, $\\dfrac{1}{\\sqrt{N_{\mathrm{I}}}}||\\dfrac{I_{\mathrm{DSS}} - I_{\mathrm{Lin}}}{I_{\mathrm{Xfmr}}}||_{2}$, %')

        # if self.nT>0:
            # TL,TLest,PL,PLest,vErr,iErr,dxScale = self.testQpTcpf()
            # ax0.plot(dxScale,100*vErr,'x-')
            # # ax0.set_xlabel('Tap')
            # ax0.set_ylabel('$V$ error, $\\dfrac{||V_{\mathrm{DSS}} - V_{\mathrm{Lin}}||_{2}}{||V_{\mathrm{DSS}}||_{2}}$, %')
            # ax3.plot(dxScale,100*iErr)
            # # ax3.set_xlabel('Tap')
            # ax3.set_ylabel('$I$ error, $\\dfrac{1}{\\sqrt{N_{\mathrm{I}}}}||\\dfrac{I_{\mathrm{DSS}} - I_{\mathrm{Lin}}}{I_{\mathrm{Xfmr}}}||_{2}$, %')



if 'f_daisy' in locals():
    # feeder = 8
    feeder = 26
    self = main(feeder,'loadOnly',linPoint=1.0,method='fot'); self.loadQpSet(); self.loadQpSln('full','opCst')
    self.plotNetBuses('qSlnPh',pltShow=False)
    SN = os.path.join(SDfig,'daisy')
    if 'pltSave' in locals(): plotSaveFig(os.path.join(SDfig,'daisy'),pltClose=False)
    if 'pltSave' in locals(): plotSaveFig(os.path.join(sdt('t3','f'),'daisy'),pltClose=True)
    plt.show()

if 'f_123busVlts' in locals():
    feeder = 26
    self = main(feeder,'loadOnly',linPoint=1.0,method='fot'); self.loadQpSet(); self.loadQpSln('full','opCst')
    
    figname='123busVlts'
    sd = sdt('t3','f')
    # nicked from showQpSln
    TL,PL,TC,CL,V,I,Vc,Ic = self.slnF
    TL0,PL0,TC0,CL0,V0,I0,Vc0,Ic0 = self.slnF0
    TLd,PLd,TCd,CLd,Vd,Id,Vcd,Icd = self.slnD

    fig,ax0=plt.subplots(figsize=(4.4,3.0))
    # ax0.plot((V/self.vKvbase)[self.vIn],'x',markeredgewidth=0.7,label='QP Sln.');
    ax0.plot((V/self.vKvbase)[self.vIn],'o',markerfacecolor='None',markersize=3.0,label='QP Sln.');
    ax0.plot((V0/self.vKvbase)[self.vIn],'o',markerfacecolor='None',markersize=3.0,label='Nom. Sln.');
    ax0.plot((Vd/self.vKvbase)[self.vIn],'o',markerfacecolor='None',markersize=2.0,markeredgewidth=0.7,label='DSS. Sln.');
    ax0.plot((self.vHi/self.vKvbase)[self.vIn],'k_');
    ax0.plot((self.vLo/self.vKvbase)[self.vIn],'k_');
    ax0.set_xlabel('Bus Index')
    ax0.set_ylabel('Voltage, pu')
    ax0.grid(True)
    ax0.set_ylim((0.89,1.06))
    ax0.legend()
    # ax0.show()
    plt.tight_layout()
    if 'pltSave' in locals(): plotSaveFig(os.path.join(sd,figname),pltClose=True)
    plt.show()
    
    self.printQpSln()
    self.printQpSln(np.zeros(self.nCtrl),self.slnD0)
    
    
    figname='123busVltsCtrl'
    fig,ax2=plt.subplots(figsize=(4.4,3.0))
    self.getLdsPhsIdx()
    ax2.plot(range(0,self.nPh1),
                        100*(self.slnX[self.nPctrl:self.nPctrl*2][self.Ph1])/self.qLim,'x-',label='Q, phs. A')
    ax2.plot(range(self.nPh1,self.nPh1+self.nPh2),
                        100*self.slnX[self.nPctrl:self.nPctrl*2][self.Ph2]/self.qLim,'x-',label='Q, phs. B')
    ax2.plot(range(self.nPh1+self.nPh2,self.nPctrl),
                        100*self.slnX[self.nPctrl:self.nPctrl*2][self.Ph3]/self.qLim,'x-',label='Q, phs. C')
    ax2.plot(range(self.nPctrl,self.nPctrl + self.nT),100*self.slnX[self.nPctrl*2:]/self.tLim,'x-',label='Tap')
    ax2.set_xlabel('Control Index')
    ax2.set_ylabel('Control effort, %')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim((-110,110))
    plt.tight_layout()
    if 'pltSave' in locals(): plotSaveFig(os.path.join(sd,figname),pltClose=True)
    plt.show()

if 'f_123busVltsPscc' in locals():
    feeder = 26
    self = main(feeder,'loadOnly',linPoint=1.0,method='fot'); self.loadQpSet(); self.loadQpSln('full','opCst')
    
    figname='123busVlts'
    # sd = sdt('t3','f')
    # sd = r'C:\Users\Matt\Documents\DPhil\papers\pscc20\full\figures'
    sd = r'D:\DocumentsD_l\DPhil\papers\pscc20\full\figures'
    # nicked from showQpSln
    TL,PL,TC,CL,V,I,Vc,Ic = self.slnF
    TL0,PL0,TC0,CL0,V0,I0,Vc0,Ic0 = self.slnF0
    TLd,PLd,TCd,CLd,Vd,Id,Vcd,Icd = self.slnD

    fig,ax0=plt.subplots(figsize=(4.4,3.0))
    # ax0.plot((V/self.vKvbase)[self.vIn],'x',markeredgewidth=0.7,label='QP Sln.');
    ax0.plot((V0/self.vKvbase)[self.vIn],'o',markerfacecolor='None',markersize=3.0,label='Base Sln.',color=cm.matlab(2));
    ax0.plot((V/self.vKvbase)[self.vIn],'o',markerfacecolor='None',markersize=3.0,label='QP Sln.',color=cm.matlab(0));
    ax0.plot((Vd/self.vKvbase)[self.vIn],'o',markerfacecolor='None',markersize=2.0,markeredgewidth=0.7,label='DSS. Sln.',color=cm.matlab(1));
    ax0.plot((self.vHi/self.vKvbase)[self.vIn],'k_');
    ax0.plot((self.vLo/self.vKvbase)[self.vIn],'k_');
    ax0.set_xlabel('Bus Index')
    ax0.set_ylabel('Voltage, pu')
    ax0.grid(True)
    ax0.set_ylim((0.89,1.06))
    ax0.legend()
    # ax0.show()
    plt.tight_layout()
    if 'pltSave' in locals(): plotSaveFig(os.path.join(sd,figname),pltClose=True)
    plt.show()
    
    self.printQpSln()
    self.printQpSln(np.zeros(self.nCtrl),self.slnD0)
    
    
    figname='123busVltsCtrl'
    fig,ax2=plt.subplots(figsize=(4.4,3.0))
    self.getLdsPhsIdx()
    ax2.plot(range(0,self.nPh1),
                    100*(self.slnX[self.nPctrl:self.nPctrl*2][self.Ph1])/self.qLim,'x-',label='$Q_{\mathrm{invr}}$, phs. A')
    ax2.plot(range(self.nPh1,self.nPh1+self.nPh2),
                        100*self.slnX[self.nPctrl:self.nPctrl*2][self.Ph2]/self.qLim,'x-',label='$Q_{\mathrm{invr}}$, phs. B')
    ax2.plot(range(self.nPh1+self.nPh2,self.nPctrl),
                        100*self.slnX[self.nPctrl:self.nPctrl*2][self.Ph3]/self.qLim,'x-',label='$Q_{\mathrm{invr}}$, phs. C')
    ax2.plot(range(self.nPctrl,self.nPctrl + self.nT),100*self.slnX[self.nPctrl*2:]/self.tLim,'x-',label='Tap pos., $T$')
    ax2.set_xlabel('Control Index')
    ax2.set_ylabel('Control effort, %')
    ax2.legend(fontsize='small')
    ax2.grid(True)
    ax2.set_ylim((-110,110))
    plt.tight_layout()
    if 'pltSave' in locals(): plotSaveFig(os.path.join(sd,figname),pltClose=True)
    plt.show()


    

if 't_checkErrorSummary' in locals():
    pCvr = 0.6
    strategy='full'
    heading = ['Feeder','Voltage error, \%','Current error, \%','Power error, \%']
    
    resultTable = [heading]
    i = 1
    for feeder in feederSet:
        feederTidy = feederIdxTidy[feeder]
        resultTable.append([feederTidy])
        for obj in objSet:
            linPoints = linPointsDict[feeder][obj]
            voltages = []; currents = []; powers = []
            for linPoint in linPoints:
                self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint,method='fot')
                voltages.append(self.qpSolutionDssError(strategy,obj,err='V')*100)
                currents.append(self.qpSolutionDssError(strategy,obj,err='I')*100)
                powers.append(self.qpSolutionDssError(strategy,obj,err='P')*100)
        resultTable[i].append( "%.3f" % (np.max(voltages)))
        resultTable[i].append( "%.3f" % (np.max(currents)))
        resultTable[i].append( "%.3f" % (np.max(powers)))
        i+=1
    print(*resultTable,sep='\n')
    TD = os.path.join(SD0,'tables\\')
    caption = 'Maximum error across all three loading points (10\%, 60\%, 100\%), when the optimal solution $x^{\dagger}$ is applied to both OpenDSS and the MI-QCQP model. Voltage, current and power errors are as calculated in \\eqref{e:vltErr}, \\eqref{e:crrErr} and \\eqref{e:pwrErr}.'
    label = 'checkErrorSummary'
    # if 'pltSave' in locals(): basicTable(caption,label,heading,resultTable[1:],TD)
    if 'pltSave' in locals(): basicTable(caption,label,heading,resultTable[1:],sdt('t3','t'))

# feeder = 'n1'
# strategy='full'; obj='opCst'; optType = ['mosekFull']
# self = main(feeder,pCvr=0.6,modelType='loadOnly',linPoint=1.0,method='fot')
# # self.initialiseOpenDss()

# # self = main(feeder,pCvr=0.6,modelType='buildOnly',linPoint=1.0,method='fot')
# self.setupConstraints() # this reloads the constraints in.
# self.runCvrQp(strategy=strategy,obj=obj,optType=optType)

# dVset = np.abs(self.slnF[4][self.vIn] - self.slnF0[4][self.vIn])/self.vKvbase[self.vIn]
# dVsetTrue = np.abs(self.slnD[4][self.vIn] - self.slnD0[4][self.vIn])/self.vKvbase[self.vIn]

# # plt.plot(np.log10(abs(dVset))); plt.plot(np.log10(abs(dVsetTrue))); plt.show()
# # plt.plot(np.log10(abs((dVset - dVsetTrue)))); plt.show()
# # plt.plot(np.log10(abs((dVset - dVsetTrue)/dVsetTrue ))); plt.show()


if 't_sensitivities_base' in locals():
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
            self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPointsDict[feeder][obj][-1],method='fot')
            opCst[j,i] = self.qpVarValue(strategy,obj,'norm',invType='Low')
            j+=1
        i+=1
    benefits = -100*(opCst[0:2] - opCst[2])
    benefitsNos = -100*(opCst[0:2] - opCst[2]) # use to find Network 10 results
    benefitsRatio = benefits[1]/benefits[0]
    benefits = np2lsStr(benefits,3)
    benefitsRatio = np2lsStr(benefitsRatio*100,3)
    # data = [ ['Full']+benefits[0],['Phase']+benefits[1] ]
    # data = [ ['Full, \%:']+benefits[0],['Phase, \%:']+benefits[1],['Ratio (\%):']+benefitsRatio ]
    data = [ ['Full, \%:']+benefits[0],['Phase, \%:']+benefits[1]]
    caption='Smart inverter benefits, as a \% of load, compared to the Nominal control case (which only has tap controls activated, if there are any), calculated using the MI-QCQP model. With the exception of Network 10, the minimum ratio of phase to full load is 73\%, and the maximum ratio of benefits is 98\%.'
    label='sensitivities_base'
    if 'pltSave' in locals(): basicTable(caption,label,heading,data,TD)
    if 'pltSave' in locals(): basicTable(caption,label,heading,data,sdt('t3','t'))
    print(heading); print(*data,sep='\n')
    print('Benefits ratio:\n',benefitsRatio)
    
    
if 't_sensitivities_base_pscc' in locals():
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
            self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPointsDict[feeder][obj][-1],method='fot')
            opCst[j,i] = self.qpVarValue(strategy,obj,'norm',invType='Low')
            j+=1
        i+=1
    benefits = -100*(opCst[0:2] - opCst[2])
    benefitsNos = -100*(opCst[0:2] - opCst[2])
    benefitsRatio = benefits[1]/benefits[0]
    benefits = np2lsStr(benefits,3)
    benefitsRatio = np2lsStr(benefitsRatio*100,3)
    # data = [ ['Full']+benefits[0],['Phase']+benefits[1] ]
    # data = [ ['Full, \%:']+benefits[0],['Phase, \%:']+benefits[1],['Ratio (\%):']+benefitsRatio ]
    
    TD = r'C:\Users\Matt\Documents\DPhil\papers\pscc20\full\tables\\'
    
    data = [ ['Full, \%:']+benefits[0],['Phase, \%:']+benefits[1] ,['Ratio, \%:']+benefitsRatio ]
    data = list(map(list, zip(*data)))
    # hdng = data[0]
    # dta = data[1:]
    newList = []
    for hdg,dta in zip(heading,data):
        newList.append([hdg]+dta)
    
    heading = newList[0]
    data = newList[1:]
    caption='Smart inverter benefits, as a \% of load, compared to the Nominal control case (which only has tap controls activated, if there are any).'
    label='sensitivities_base_pscc'
    
    if 'pltSave' in locals(): basicTable(caption,label,heading,data,TD)
    print(*newList,sep='\n')
    
if 'f_sensitivities_all' in locals():
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
            self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPointsDict[feeder][obj][-1],method='fot')
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
    efficacy = np2lsStr( (1e3*wBenefit/qCost).T,1)

    data = [ ['None']+benefits[0],['Low']+benefits[1],['High']+benefits[2] ]
    eData = [ ['None']+efficacy[0],['Low']+efficacy[1],['High']+efficacy[2] ]
    
    label='sensitivities_invLoss'
    # if 'pltSave' in locals(): basicTable(caption,label,heading,data,TD)
    print(heading); print(*data,sep='\n')

    label='sensitivities_efficacy'
    caption='Smart inverter efficacy ($P/||Q||_{1}$), W/kVAr'
    # if 'pltSave' in locals(): basicTable(caption,label,heading,eData,TD)
    print(heading); print(*eData,sep='\n')
    
    heading[0]='Network'
    newData = list(map(list, zip(*(  [heading]+data+eData  ))))
    caption='Smart inverter load benefit (\%) and efficacy ($P/||Q||_{1}$, W/kVAr)'
    label='sensitivities_all'
    if 'pltSave' in locals(): basicTable(caption,label,newData[0],newData[1:],TD)
    if 'pltSave' in locals(): basicTable(caption,label,newData[0],newData[1:],sdt('t3','t'))
    

if 'f_sensitivities_aCvr' in locals():
    # sensitivities_aCvr
    caption='Smart inverter benefits, \% of load'
    pCvrSet = [0.0,0.3,0.6,0.9]
    obj='opCst'
    strategies = ['full','nomTap']
    opCst = np.zeros((2,len(feederSet),len(pCvrSet)))
    wCst = np.zeros((2,len(feederSet),len(pCvrSet)))
    qCst = np.zeros((2,len(feederSet),len(pCvrSet)))
    i=0
    heading = ['$\\alpha_{\mathrm{CVR}}$']
    nonLinearity = ['Non-lin']
    for feeder in feederSet:
        heading.append(feederIdxTidy[feeder])
        j=0
        for strategy in strategies:
            k=0
            for pCvr in pCvrSet:
                self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPointsDict[feeder][obj][-1],method='fot')
                opCst[j,i,k] = self.qpVarValue(strategy,obj,'norm')
                wCst[j,i,k] = self.qpVarValue(strategy,obj,'power')
                qCst[j,i,k] = np.linalg.norm(self.slnX[self.nPctrl:self.nSctrl],ord=1)                
                k+=1
            j+=1
        i+=1
    
    nLinCvr = ['Non-lin meas']
    benefits = -100*(opCst[0] - opCst[1])
    # for benefit in benefits:
        # cfs = np.polyfit(pCvrSet,benefit,2)
        # nLinCvr.append('%.3f' % (cfs[0]) )
        # print(cfs)
    
    wBenefit = -(wCst[0] - wCst[1])
    qCost = qCst[0]
    efficacy = np2lsStr( (1e3*wBenefit/qCost).T,1)
    benefits = np2lsStr(benefits.T,2)
    data = [ ['0.0']+benefits[0],['0.3']+benefits[1],['0.6']+benefits[2],['0.9']+benefits[3] ]
    eData = [ ['0.0']+efficacy[0],['0.3']+efficacy[1],['0.6']+efficacy[2],['0.9']+efficacy[3] ]

    label='sensitivities_aCvr'
    if 'pltSave' in locals(): basicTable(caption,label,heading,data,TD)
    print(heading); print(*data,sep='\n')
    print(heading); print(*eData,sep='\n')
    
    heading[0]='Network'
    newData = list(map(list, zip(*(  [heading]+data+eData  ))))
    caption='Smart inverter load benefit (\%) and efficacy ($P/||Q||_{1}$, W/kVAr) against $\\alpha_{\mathrm{CVR}}$.'
    label='sensitivities_aCvr'
    if 'pltSave' in locals(): basicTable(caption,label,newData[0],newData[1:],TD)
    if 'pltSave' in locals(): basicTable(caption,label,newData[0],newData[1:],sdt('t3','t'))
    
    
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
    
if 't_networkSummary' in locals():
    # NB: annoyingly the regulator is at a higher tap nominally so the values are a bit different
    # self2 = main('n27','loadOnly',linPoint=linPointsDict[feeder]['opCst'][0])
    # self2.TC_No0
    
    heading = ['Network', 'No. Lds.', 'No. Buses', 'Q-sensitivity $\eta_{\mathrm{Sns}}$','Unc. cost $\delta^{*}$']
    data = []; i=0
    for feeder in feederSet:
        self = main(feeder,'loadOnly',linPoint=linPointsDict[feeder]['opCst'][-1])
        data.append([feederIdxTidy[feeder]])
        data[i].append(str(self.nPctrl))
        data[i].append(str(self.Kc2v.shape[0] + 3))
        slnXunc = self.solveQpUnc()[0]
        data[i].append( '%.1f' % self.solveQpUnc()[2] )
        slnFunc = self.solveQpUnc()[1]
        data[i].append( '%.3f' % ( -100*( np.sum(slnFunc[0:4]) - np.sum(self.slnF0[0:4]))/np.sum(self.slnF0[0:4]))  )
        i+=1
    
    label='t_networkSummary'
    caption='Summary of Network'
    if 'pltSave' in locals(): basicTable(caption,label,heading,data,TD)
    print(heading); print(*data,sep='\n')

feederIdxTidy = {5:'13 Bus',6:'34 Bus',8:'123 Bus',9:'8500 Node',19:'Ckt. J1',20:'Ckt. K1',21:'Ckt. M1',17:'Ckt. 5',18:'Ckt. 7',22:'Ckt. 24',26:'123 Bus',24:'Ckt. K1','n1':'EULVa','n27':'EULVa-r',0:'EULV','n4':'Nwk. 4','n10':'Nwk. 10'}


if 't_networkSummaryPscc' in locals():
    heading = ['Network','No. Buses', 'No. Lds.', 'No. Taps']
    data = []; i=0
    feederSet = [17,18,'n1','n10',26,24,'n27']
    for feeder in feederSet:
        self = main(feeder,'loadOnly',linPoint=linPointsDict[feeder]['opCst'][-1])
        data.append([feederIdxTidy[feeder]])
        data[i].append(str(self.Kc2v.shape[0] + 3))
        data[i].append(str(self.nPctrl))
        data[i].append(str(self.nCtrl - self.nSctrl))
        i+=1
    
    TD = r'C:\Users\Matt\Documents\DPhil\papers\pscc20\full\tables\\'
    label='t_networkSummaryNew'
    caption='Summary of Network'
    # if 'pltSave' in locals(): basicTable(caption,label,heading,data,TD)
    basicTable(caption,label,heading,data,TD)
    print(heading); print(*data,sep='\n')





if 't_networkMetrics' in locals():
    heading = ['Network','Average sense $\hat{\lambda}_{\mathrm{Ave}}$, kVAr','Average cost saving, $\hat{f}_{\mathrm{Ave}}$, W','Unc. cost $f_{\mathrm{Q\,Unc}}^{*}$, \%','Control efficacy, $f_{\mathrm{Q\,Unc}}^{*}/\|x_{\mathrm{Q\,Unc}}^{*}\|$, W/kVAr']
    data = []; i=0
    for feeder in feederSet:
        self = main(feeder,'loadOnly',linPoint=linPointsDict[feeder]['opCst'][-1])
        
        data.append([feederIdxTidy[feeder]])
        
        slnXunc, slnFunc, gradP = self.solveQpUnc()[0:3]
        
        data[i].append( '%.1f' % gradP)
        data[i].append( '%.1f' % ( 1e3*(np.sum(slnFunc[0:4]) - np.sum(self.slnF0[0:4]))/self.nPctrl ) )
        data[i].append( '%.3f' % ( -100*(np.sum(slnFunc[0:4]) - np.sum(self.slnF0[0:4]) )/np.sum(self.slnF0[0:4])) )
        data[i].append( '%.3f' % ( -1e3*(np.sum(slnFunc[0:4]) - np.sum(self.slnF0[0:4]))/np.linalg.norm(slnXunc) ) )
        i+=1
    
    label='t_networkMetrics'
    caption='Unconstrained Network Analysis'
    if 'pltSave' in locals(): basicTable(caption,label,heading,data,TD)
    print(heading); print(*data,sep='\n')
    
if 'f_costFunc' in locals():
    feederSet = [24,'n27']
    for feeder in feederSet:
        self = main(feeder,modelType='loadOnly',linPoint=1.0,method='fot')
        fig,ax1 = plt.subplots(figsize=(3.3,2.8))

        ldsQ = 1e3*self.ploadL[self.nPctrl:self.nSctrl]
        lssQ = 1e3*self.qpLlss[self.nPctrl:self.nSctrl]
        tot = lssQ + ldsQ
        ax1.plot(tot,'k',label='$\\psi_{\mathrm{Q}}$ (tot)');
        ax1.plot(ldsQ,'--',label='$\\psi_{\mathrm{Q,\,lds}}$');
        ax1.plot(lssQ,'--',label='$\\psi_{\mathrm{Q,\,lss}}$');
        ax1.legend(fontsize='small')
        ax1.set_xlabel('Bus index $i$')
        ax1.set_ylabel('QP Sensitivity  $\\psi_{\mathrm{Q}}^{[i]}$, W/kVAr')

        # H = self.getHmat()
        # qpQlss = 1e3*H.dot(H.T)
        # qpQtot = qpQlss + np.diag(1e3*self.qlossQdiag)
        # # plt.plot(np.diag(qpQlss)[self.nPctrl:self.nSctrl],label='qpQtot')
        # ax2.plot(np.sum(np.abs(qpQtot),axis=1)[self.nPctrl:self.nSctrl],'k',label='$\Lambda_{\mathrm{QQ}}$ (tot)')
        # ax2.plot(np.sum(np.abs(qpQlss),axis=1)[self.nPctrl:self.nSctrl],'--',label='$\Lambda_{\mathrm{QQ}}^{\mathrm{Loss}}$')
        # ax2.plot(1e3*self.qlossQdiag[self.nPctrl:self.nSctrl],'--',label='$\Lambda_{\mathrm{QQ}}^{\mathrm{Inverter}}$')
        # ax2.set_xlabel('Bus index $i$')
        # ax2.set_ylabel('QP Hessian  $\|\| \Lambda_{\mathrm{QQ}}^{[i,:]} \|\|_{1}$, W/kVAr$^{2}$ ')
        # ax2.legend(fontsize='small',loc='upper left')
        # ax2.set_ylim((0,100))
        
        plt.tight_layout()
        # if 'pltSave' in locals(): plotSaveFig(os.path.join(SDfig,'costFunc_'+self.feeder),pltClose=True)
        if 'pltSave' in locals(): plotSaveFig(os.path.join(sdt('t3','f'),'costFunc_'+self.feeder),pltClose=True)
        plt.show()


if 'f_psdFigs' in locals():
    feeder = 6
    self = main(feeder,modelType='loadOnly',linPoint=0.6,method='fot')
    self.solveQpUnc()
    qpQlss = 1e3*self.qQpUnc[self.nPctrl:self.nSctrl,self.nPctrl:self.nSctrl]
    
    pLoad = self.ploadL[self.nPctrl:self.nSctrl]
    pLoss = self.qpLlss[self.nPctrl:self.nSctrl]
    p = 1e3*(pLoad + pLoss)
        
    from numpy.linalg import svd
    [UU,ss,VVh] = svd(qpQlss)

    figname = 'psdFigsSs'
    sd = sdt('t3','f')
    fig,ax = plt.subplots(figsize=(2.8,2.8))
    ax.plot(ss,'x')
    ax.set_yscale('log')
    ax.set_xlabel('Singular value no.')
    ax.set_ylabel('Singular value, W/kVA$^{2}$')
    plt.tight_layout()
    if 'pltSave' in locals(): plotSaveFig(os.path.join(sd,figname),pltClose=True)

    plt.show()

    figname = 'psdFigsUu'
    sd = sdt('t3','f')
    fig,ax = plt.subplots(figsize=(2.8,2.8))
    ax.plot(VVh[-1],'x',label='N-1')
    ax.plot(VVh[-2],'+',label='N-2')
    ax.set_xlabel('Control index, $i$')
    ax.set_ylabel('Basis vector value $U^{[j,i]}$')
    ylm = ax.get_ylim()
    ylm = (-0.48,0.38)
    ax.plot((39.5,39.5),ylm,'k--')
    ax.set_ylim(ylm)
    ax.text(1,-0.45,'Wye\nloads')
    ax.text(41,-0.45,'Delta\nloads')
    plt.tight_layout()
    if 'pltSave' in locals(): plotSaveFig(os.path.join(sd,figname),pltClose=True)

    plt.show()
    # print(vecSlc(self.YZ,self.pyIdx[0][20:26]))

    figname = 'psdFigsUlmb'
    sd = sdt('t3','f')
    fig,ax = plt.subplots(figsize=(2.8,2.8))
    ax.plot(1e3*np.abs(VVh.dot(pLoad)),'+',label='Load')
    ax.plot(1e3*np.abs(VVh.dot(pLoss)),'x',label='Ntwk. Loss')
    ax.set_xlabel('Singular value no.')
    ax.set_ylabel('Response $U^{\intercal}\\psi_{(\cdot)}$, W/kVAr')
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    if 'pltSave' in locals(): plotSaveFig(os.path.join(sd,figname),pltClose=True); plt.show()

    plt.show()

if 'f_costFuncXmpl' in locals():
    from numpy.linalg import norm, solve, lstsq, svd
    feederSet = ['n1',26]
    # feederSet = ['n1']
    for feeder in feederSet:
        self = main(feeder,modelType='loadOnly',linPoint=1.0,method='fot')
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
        
        # based on solveQpUnc 
        xIvt = np.logspace(-2.0,3.0,20) # W/kVAr^2
        x1Set = []
        for xivt in xIvt:
            qpQivt = xivt*np.eye(len(qpQlss))
            np.linalg.norm(qpQlss,ord=2)
            Q = qpQivt + qpQlss
            xStar = np.r_[np.zeros(self.nPctrl),lstsq(Q,-0.5*p,rcond=None)[0],np.zeros(self.nT)]
            fStar = self.runQp(xStar)
            # x1Set.append(norm(xStar))
            x1Set.append(norm(xStar,ord=1))
            c0star = -(1/xivt)*0.5*p
        # ylm = (-5,105)
        # ylm = (-0.0,2.5)
        ylm = (0.0,5)
        sln0x = lstsq(qpQlss,-0.5*p,rcond=None)[0]
        fig,ax = plt.subplots(figsize=(3.6,2.6))
        # ax.plot(xIvt,100*np.array(x1Set)/norm(sln0x))
        # ax.plot(xIvt,np.array(x1Set)/len(sln0x))
        ax.plot(xIvt,np.array(x1Set)/len(sln0x),label='$\\dfrac{||x_{\mathrm{Q,\,Unc}}^{\\dagger}||_{1}}{N_{\mathrm{lds}}}$')
        ax.set_xscale('log'); 
        ax.set_xlabel('Inverter loss coefficient $c_{R}$, W/kVAr$^{2}$')
        # ax.set_ylabel('$Q$ per gen., $\\dfrac{||x_{\mathrm{Q,\,Unc}}^{\\dagger}||_{1}}{N_{\mathrm{lds}}}$, kVAr')
        ax.set_ylabel('Mean $Q$ per invt., kVAr')
        ax.plot([invCoeffs[0][0]]*2,ylm,'--',color=cm.matlab(1))
        ax.plot([invCoeffs[1][0]]*2,ylm,'--',color=cm.matlab(1))
        ax.text(invCoeffs[0][0]*1.6,0.75*ylm[1],'1 kVA inverter',color=cm.matlab(1),rotation=90)
        ax.plot([invCoeffs[0][1]]*2,ylm,':',color=cm.matlab(2))
        ax.plot([invCoeffs[1][1]]*2,ylm,':',color=cm.matlab(2))
        ax.text(invCoeffs[0][1]*1.6,0.75*ylm[1],'4 kVA inverter',color=cm.matlab(2),rotation=90)
        ax.plot([invCoeffs[0][2]]*2,ylm,'-.',color=cm.matlab(3))
        ax.plot([invCoeffs[1][2]]*2,ylm,'-.',color=cm.matlab(3))
        ax.text(invCoeffs[0][2]/1.6,0.75*ylm[1],'16 kVA inverter',color=cm.matlab(3),rotation=90)
        ax.set_ylim(ylm)
        ax.set_xlim((xIvt[0],xIvt[-1]))
        ax.legend(fontsize='small')
        plt.tight_layout()
        if 'pltSave' in locals(): plotSaveFig(os.path.join(sdt('t3','f'),'costFuncXmpl_'+self.feeder),pltClose=True)
        plt.show()
        
if 'f_costFuncAsym' in locals():
    from numpy.linalg import norm, solve, lstsq
    feeder = 'n1'
    feeder = 17
    # xIvt = np.logspace(-2.0,3.0,10) # W/kVAr^2
    xIvt = np.logspace(-2.0,3.0,60) # W/kVAr^2
    # self = main(feeder,modelType='loadOnly',linPoint=1.0)
    self = main(feeder,modelType='loadOnly',linPoint=1.0,method='fot')
    self.solveQpUnc()
    qpQlss = 1e3*self.qQpUnc[self.nPctrl:self.nSctrl,self.nPctrl:self.nSctrl]
    types = ['Low','Hi']
    pLoad = self.ploadL[self.nPctrl:self.nSctrl]
    pLoss = self.qpLlss[self.nPctrl:self.nSctrl]
    p = 1e3*(pLoad + pLoss)
    
    # based on solveQpUnc 
    x1Set = []; fSet = []; xInvSet = []; xLinSet = []
    for xivt in xIvt:
        print(xivt)
        qpQivt = xivt*np.eye(len(qpQlss))
        np.linalg.norm(qpQlss,ord=2)
        Q = qpQivt + qpQlss
        try:
            xStar = np.r_[np.zeros(self.nPctrl),lstsq(Q,-0.5*p,rcond=None)[0],np.zeros(self.nT)]
        except:
            pass
        fStar = self.runQp(xStar)
        x1Set.append(norm(xStar))
        fSet.append(np.sum(fStar[0:4]))
        
        c0star = -(1/xivt)*0.5*p
        xInvStar = (np.eye(len(qpQlss)) - ((1/xivt)*qpQlss))*c0star
        xInvSet.append(norm(xInvStar))
        
        b0star = lstsq(qpQlss,-0.5*p,rcond=None)[0]
        xLinStar = b0star + ( xivt*lstsq(-qpQlss,b0star,rcond=None)[0] )
        xLinSet.append(norm(xLinStar))
        
    
    sln0 = lstsq(qpQlss,-0.5*p,rcond=None)
    sln0x = sln0[0]
    sln0s = sln0[3]
    m_eps = np.finfo(np.float64).eps
    sln0s = sln0s[sln0s>( m_eps*len(sln0s)*sln0s[0] )]
    sMin = sln0s
    
    xOffValue = 1.05
    sln0 = lstsq(qpQlss,-0.5*p,rcond=None)

    cQ = sln0[0]
    bQ = lstsq(-qpQlss,cQ,rcond=None)[0]
    c0 = cQ.dot(cQ)*(xOffValue**2 - 1) # 0.19
    c1 = 2*cQ.dot(bQ)
    c2 = bQ.dot(bQ)
    epsXoffLo = np.min(np.roots([c2,c1,c0]))

    aQ = lstsq(qpQlss,bQ,rcond=None)

    xOffValue = 0.95
    cQ = -0.5*p
    bQ = -qpQlss.dot(cQ)
    c0 = cQ.dot(cQ)*(1 - xOffValue**2)
    c1 = 2*cQ.dot(bQ)
    c2 = bQ.dot(bQ)
    epsXoffHi = np.max(1/np.roots([c2,c1,c0]))
    epsXoff = [epsXoffLo,epsXoffHi,sln0s[-1],sln0s[0]]
    print(epsXoff)

    xRcpSet = 0.5*norm(p)/xIvt

    ylm = (-5,105)

    fig,ax = plt.subplots(figsize=(5.7,2.6))
    ax.plot(xIvt,100*np.array(x1Set)/norm(sln0x),label='True')
    ax.plot(xIvt,100*np.array(xInvSet)/norm(sln0x),'--',color=cm.matlab(1),label='Asym. apx,\nhigh $c_{R}$')
    ax.plot(xIvt,100*np.array(xLinSet)/norm(sln0x),'-.',color=cm.matlab(3),label='Asym. apx,\nlow $c_{R}$')
    ax.set_xscale('log'); 
    ax.set_xlabel('Inverter loss coefficient $c_{R}$, W/kVAr$^{2}$')
    ax.set_ylabel('Solution fraction $\\dfrac{||x^{\\dagger}(c_{R})||_{2}}{||x^{\\dagger}(0)||_{2}}$, %')
    ax.plot([epsXoff[2]]*2,ylm,'k-.')
    ax.plot([epsXoff[3]]*2,ylm,'k-.')
    ax.text(epsXoff[2]/1.6,60,'Min sing. val.',color='k',rotation=90)
    ax.text(epsXoff[3]*1.5,95,'Max sing. val.',color='k',rotation=90)
    ax.plot([epsXoff[0]]*2,ylm,'k-.')
    # ax.text(epsXoff[0]/1.6,65,'95% cutoff',color='k',rotation=90)
    ax.text(epsXoff[0]/1.7,70,'95% cutoff, lo. $c_{R}$',color='k',rotation=90)
    ax.plot([epsXoff[1]]*2,ylm,'k-.')
    ax.text(epsXoff[1]/1.7,80,'95% cutoff, hi. $c_{R}$',color='k',rotation=90)
    ax.set_ylim(ylm)
    ax.set_xlim((xIvt[0],xIvt[-1]))
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),fontsize='small',title='Solution',labelspacing=0.8)
    plt.tight_layout()
    # plt.show()
    if 'pltSave' in locals(): plotSaveFig(os.path.join(sdt('t3','f'),'costFuncAsym_'+self.feeder),pltClose=True)
    plt.show()



if 'f_solutionError' in locals():
    feeder = 26
    self = main(feeder,'loadOnly',linPoint=1.0,method='fot'); self.loadQpSet(); 
    self.loadQpSln('full','opCst')
    self.plotArcy(pltShow=False)
    if 'pltSave' in locals():
        plotSaveFig(os.path.join(SDfig,'solutionErrorOpCst'),pltClose=False)
        plotSaveFig(os.path.join(sdt('t3','f'),'solutionErrorOpCst'),pltClose=True)
    
    self.printQpSln(self.slnX,self.slnF)
    self.printQpSln(self.slnX,self.slnD)
    
    
    dPa = sum(self.slnF[0:4]) - sum(self.slnD0[0:4])
    dPb = sum(self.slnD[0:4]) - sum(self.slnD0[0:4])
    errP = (dPa - dPb)/dPb
    print(dPa)
    print(dPb)
    print(errP)
    
    # self.loadQpSln('full','hcLds'); 
    # self.plotArcy(pltShow=False)
    # if 'pltSave' in locals(): plotSaveFig(os.path.join(SDfig,'solutionErrorHcLds'),pltClose=True)
    # plt.show()
    
    # self = main(feeder,'loadOnly',linPoint=0.1); self.loadQpSet(); 
    # self.loadQpSln('full','hcGen'); 
    # self.plotArcy(pltShow=False)
    # if 'pltSave' in locals(): plotSaveFig(os.path.join(SDfig,'solutionErrorHcGen'),pltClose=True)
    plt.show()
    
if 'f_solutionErrorPscc' in locals():
    feeder = 26
    self = main(feeder,'loadOnly',linPoint=1.0,method='fot'); self.loadQpSet(); 
    self.loadQpSln('full','opCst')
    self.plotArcy(pltShow=False,psccVrs=True)
    # if 'pltSave' in locals():
        # plotSaveFig(os.path.join(SDfig,'solutionErrorOpCst'),pltClose=False)
        # plotSaveFig(os.path.join(sdt('t3','f'),'solutionErrorOpCst'),pltClose=True)
    
    self.printQpSln(self.slnX,self.slnF)
    self.printQpSln(self.slnX,self.slnD)
    
    
    dPa = sum(self.slnF[0:4]) - sum(self.slnD0[0:4])
    dPb = sum(self.slnD[0:4]) - sum(self.slnD0[0:4])
    errP = (dPa - dPb)/dPb
    print(dPa)
    print(dPb)
    print(errP)
    
    FD = r'C:\Users\Matt\Documents\DPhil\papers\pscc20\full\figures'
    if 'pltSave' in locals(): plotSaveFig(os.path.join(FD,'solutionErrorOpCstPscc'),pltClose=True)
    # self.loadQpSln('full','hcLds'); 
    # self.plotArcy(pltShow=False)
    # if 'pltSave' in locals(): plotSaveFig(os.path.join(SDfig,'solutionErrorHcLds'),pltClose=True)
    # plt.show()
    
    # self = main(feeder,'loadOnly',linPoint=0.1); self.loadQpSet(); 
    # self.loadQpSln('full','hcGen'); 
    # self.plotArcy(pltShow=False)
    # if 'pltSave' in locals(): plotSaveFig(os.path.join(SDfig,'solutionErrorHcGen'),pltClose=True)
    plt.show()
    

if 'f_37busVal' in locals():
    self = main(10,'linOnly')
    [vce,vae,k] = self.testVoltageModel(k=np.arange(-1.5,1.51,0.01))
    fig,ax = plt.subplots(figsize=(4.4,2.8))
    plt.plot(k,100*abs(vce),label='Cplx. Model');
    plt.plot(k,100*vae,label='Mag. Model');
    plt.xlim((-1.5,1.5))
    plt.ylim((0,0.45))
    plt.xlabel('Load scaling factor')
    plt.ylabel('Error, $||V_{\mathrm{DSS}} - V_{\mathrm{Lin}}||_{2}/||V_{\mathrm{DSS}}||_{2}$, %')
    plt.legend()
    plt.tight_layout()
    if 'pltSave' in locals():
        plotSaveFig(os.path.join(sdt('t2'),'37busBal'),pltClose=False)
        
if 't_thssSizes' in locals():
    feederSet = [6,8,20,19,21,17,18,9,22]
    # feederSet = [6,8]
    # heading = ['Feeder','Size, $(N_{Y_{\mathrm{bus}}})^{2}$','$\mathrm{nnz}(Y_{\mathrm{bus}})$', '$\dfrac{\mathrm{nnz}(Y_{\mathrm{bus}})}{N_{Y_{\mathrm{bus}}}^{2}}$, \%','Inverse calc. time, s']
    # heading = ['Feeder','$\mathrm{numel}(Y_{\mathrm{bus}})$','$\mathrm{nnz}(Y_{\mathrm{bus}})$','$Y_{\mathrm{bus}}$ Inverse calc. time, s']
    heading = ['Feeder','$\mathrm{nnz}(Y_{\mathrm{bus}})$','\dfrac{\mathrm{nnz}(Y_{\mathrm{bus}})}{N_{V}^{2}}$, \%', '$t_{\mathrm{Lin}}$, s','$\dfrac{t_{\mathrm{Lin}}\times 10^{7}}{\mathrm{nnz}(Y_{\mathrm{bus}})N_{V} }$']
    data = []; i=0
    for feeder in feederSet:
        data.append([feederIdxTidy[feeder]])
        
        self = main(feeder,modelType='linOnly')
        
        # data[i].append( '%d' % self.YbusN2)
        data[i].append( '%d' % self.YbusNnz )
        data[i].append( '%.2f' % (100*self.YbusFrac) )
        data[i].append( '%.2f' % self.Mtime )
        data[i].append( '%.2f' % (1e7*self.Mtime/(np.sqrt(self.YbusNnz*self.YbusN2)**1.5)) )
        i+=1
    # modifying the table - this could be tided up.
    for i in range(9):
        nnz = float(data[i][1])
        numel = nnz/(0.01*float(data[i][2]))
        tms = float(data[i][3])
        n0 = np.sqrt(nnz)
        n1 = np.sqrt(numel)
        k = 2
        print(tms*1e7/((n0**k)*(n1**(3-k))))
        data[i][4] = ( '%.2f' % (tms*1e7/((n0**k)*(n1**(3-k)))) )
        
    TD = sdt('c4','t') + '\\'
    label='t_thssSizes'
    caption='Sparsity properties of nine admittance matrices, and the corresponding time required to build the linear model, $t_{\mathrm{Lin}}$.'
    if 'pltSave' in locals(): basicTable(caption,label,heading,data,TD)

    print(heading); print(*data,sep='\n')

if 'f_thssSparsity' in locals():
    # feederSet = [6,8]
    feederSet = [5,6,8]
    markerSzs = {5:0.8,6:0.3,8:0.1}
    for feeder in feederSet:
        self = main(feeder,modelType='linOnly')
        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff

        [DSSObj,DSSText,DSSCircuit,DSSSolution] = self.dssStuff
        # >>> 1. Run the DSS; fix loads and capacitors at their linearization points, then load the Y-bus matrix at those points.
        DSSText.Command='Compile ('+self.fn+'.dss)'
        DSSText.Command='Batchedit load..* vminpu=0.02 vmaxpu=50 model=1 status=variable'
        DSSSolution.Tolerance=1e-10
        DSSSolution.LoadMult = 1.0
        DSSSolution.Solve()
        print('\nNominally converged:',DSSSolution.Converged)

        self.TC_No0 = find_tap_pos(DSSCircuit) # NB TC_bus is nominally fixed
        self.Cap_No0 = getCapPos(DSSCircuit)
        Ybus, YNodeOrder = createYbus( DSSObj,self.TC_No0,self.capPosLin )
        
        plt.spy(Ybus.toarray(),markersize=markerSzs[feeder])
        plt.grid()
        if 'pltSave' in locals():
            plotSaveFig(os.path.join(sdt('t2','f'),'thssSparsity'+fdrs[feeder]),pltClose=False)
        plt.show()
        
if 'f_runTsAnalysis' in locals():
    FNlds = r'C:\Users\Matt\Documents\MATLAB\DPhil\rpc_loss_mtlb\data'
    import pandas as pd
    TR = pd.read_csv(os.path.join(FNlds,'extract_profiles','aggregated_n.csv'))
    TR = np.array(TR); 
    TR = TR.flatten()
    TR = np.round(TR,10) #if this isn't here then EPRI K1 complains(!)

    TG = pd.read_csv(os.path.join(FNlds,'pvwatts_hourly.csv'))
    TG = TG.iloc[17:-1,10] # 1/2 bugs - also, DST needs accounting for
    TG = np.array(TG).astype('float')
    TGi = np.interp(np.linspace(0,len(TG)-1,len(TG)*2-1),np.arange(len(TG)),TG);
    TGi = TGi/max(TGi)
    TGi = np.round(TGi,10) #if this isn't here then EPRI K1 complains(!)

    genKW = 3.81
    genKVA = 4 # implies 4 kVA inverter here.
    qLimMax = 2.4 # kVAr

    dayNoA = 364-31-30-16 #15th Oct
    dayNoB = 31+28+31+30+31+30+19 #20th July
    dsto = 2 # dstOffset
    genSet = [TGi[dayNoA*48 - dsto:(dayNoA+1)*48 - dsto],TGi[dayNoB*48 - dsto:(dayNoB+1)*48 - dsto]]
    ldsSet = [TR[dayNoA*48 - dsto:(dayNoA+1)*48 - dsto],TR[dayNoB*48 - dsto:(dayNoB+1)*48 - dsto]]
    
    # # for faster debugging:
    # ldsTs = np.array([3,2,3,3,4,5,6,6,5,4,6,8,9,8,9,10,9,8,7,5,4,3,2,2])/10; ldsSet = [ldsTs,ldsTs];
    # genTs = np.array([0,0,0,0,0,0.5,1,2,3,6,8,9,10,10,10,9,8,6,3,2,1,0.5,0,0])/10; genSet = [genTs,genTs];
    # ldsTs = np.array([0.5,1]); ldsSet = [ldsTs,ldsTs];
    # genTs = np.array([0,1]); genSet = [genTs,genTs]; 

    dayNms = ['wtr','smr']
    nTS = len(genSet[0])
    
    for feeder in feederSet:
        for genTs,ldsTs,dayType in zip(genSet,ldsSet,dayNms):
            # THINGS TO RECORD.
            ts0s = np.zeros((3,nTS))
            slnMid = {'vMin':ts0s.copy(),'vMinLv':ts0s.copy(),'vMinMv':ts0s.copy(),'vMax':ts0s.copy(),'tPwr':ts0s.copy(),'tLss':ts0s.copy(),'qPhs':ts0s.copy(),'tSet':ts0s.copy()}

            # initialise a few bits and pieces
            t0 = time.time()
            self = main(feeder,modelType='loadOnly');
            dx0 = genKW*np.r_[np.ones(self.nPctrl),np.zeros(self.nPctrl+self.nT)]
            cns = self.getConstraints()
            qLimSet = 1e3*np.minimum(qLimMax*np.ones(nTS),np.sqrt(genKVA**2 - (genKW*genTs)**2))
            qLimSet[genTs==0]=0

            for lp,gen,i in zip(ldsTs,genTs,range(nTS)):
                print('\n---------------> i = '+str(i))
                print('t = '+str(time.time()-t0))
                # first, linearise at the chosen point
                self = main(feeder,modelType='buildOnly',linPoint=lp)
                cns['plim']=1e3*genKW*gen
                
                # just use the solution with genTs on
                # then, find the solutions with the different operating modes using 3x Q points
                slnF0 = self.runQp(gen*dx0)
                slnMid = self.tsRecordSnap(slnMid,(0,i),gen*dx0,slnF0)
                
                # No reactive power (approx):
                cns['qlim']=1
                self.setupConstraints(cns)
                self.runCvrQp('phase','tsCst')
                slnMid = self.tsRecordSnap(slnMid,(1,i))

                # Per phase control (also record phase info)
                cns['qlim']=qLimSet[i]
                self.setupConstraints(cns)
                self.runCvrQp('phase','tsCst')
                slnMid = self.tsRecordSnap(slnMid,(2,i))
                
                self.getLdsPhsIdx()

                slnMid['qPhs'][0,i] = self.slnX[self.nPctrl:self.nPctrl*2][self.Ph1][0]
                slnMid['qPhs'][1,i] = self.slnX[self.nPctrl:self.nPctrl*2][self.Ph2][0]
                slnMid['qPhs'][2,i] = self.slnX[self.nPctrl:self.nPctrl*2][self.Ph3][0]
            print('\n === COMPLETE ===\nTime = '+str(time.time()-t0))
            
            slnTs = {'ldsTs':ldsTs,'genTs':genTs,'cns':cns,'genKW':genKW,'genKVA':genKVA,'qLimMax':qLimMax,'qLimSet':qLimSet,'nTS':nTS,'nPctrl':self.nPctrl,**slnMid}

            SD = os.path.join( os.path.dirname(self.getSaveDirectory()),'results',self.feeder+'_ts_out')
            SN = os.path.join(SD,self.feeder+'ts_i'+self.invLossType+'_'+dayType+'_sln.pkl')
            if not os.path.exists(SD):
                os.mkdir(SD)
            
            if 'pltSave' in locals():
                with open(SN,'wb') as outFile:
                    print('Results saved to '+ SN)
                    pickle.dump(slnTs,outFile)


if 'f_plotTsAnalysis' in locals():
    feederSet=[17,24]
    # feederSet=[24]
    figNameSet = ['tPwr','vMinMax','qPhs','dPwr','efcy','tapSet']
    figNameSet = ['vMinMax']
    dayTypes = ['wtr','smr']
    # dayTypes = ['smr']
    # pltSave=1
    for feeder in feederSet:
        self = main(feeder,modelType='loadOnly');
        self.getLdsPhsIdx()
        SD = os.path.join( os.path.dirname(self.getSaveDirectory()),'results',self.feeder+'_ts_out')
        tsFigSze = (4.0,2.5)
        SDT5 = sdt('t3','f')
        
        for dayType in dayTypes:
            SN = os.path.join(SD,self.feeder+'ts_i'+self.invLossType+'_'+dayType+'_sln.pkl')
            with open(SN,'rb') as inFile:
                tsRslt = pickle.load(inFile)

            times = np.linspace(0,24 - (24/tsRslt['nTS']),tsRslt['nTS'])
            xlm = [0,24]
            xtcks = [0,4,8,12,16,20,24]
            
            figName = 'tPwr'
            if figName in figNameSet:
                fig,ax = plt.subplots(figsize=tsFigSze)
                ax.plot(times,tsRslt['genTs']*tsRslt['nPctrl']*tsRslt['genKW'],'k-.',label='Generation')
                ax.plot(times,tsRslt['tPwr'][0],'k:',label='Feeder net load')
                ax.set_xlabel('Time, hour')
                ax.set_ylabel('Power, kW')
                ax.set_xlim(xlm); ax.set_xticks(xtcks)
                plt.legend()
                plt.tight_layout()
                if 'pltSave' in locals(): plotSaveFig(os.path.join(SDT5,figName+self.feeder+dayType+self.invLossType),pltClose=True)
                plt.show()


            figName = 'vMinMax'
            if figName in figNameSet:
                fig,ax = plt.subplots(figsize=tsFigSze)
                # ax.set_prop_cycle(color=cm.matlab([0,1,2]))
                ax.set_prop_cycle(color=cm.matlab([0,1]))
                # ax.plot(times, tsRslt['vMinMv'][1:3].T,':',labe);
                # ax.plot(times, tsRslt['vMinLv'][1:3].T);
                # ax.plot(times, tsRslt['vMax'][1:3].T);
                ax.plot(times, tsRslt['vMinMv'][1].T,':',label='$V_{\mathrm{MV,\,min}}^{\mathrm{Nom.}}$',color=cm.matlab(0));
                ax.plot(times, tsRslt['vMinLv'][1].T,label='$V_{\mathrm{LV,\,min}}^{\mathrm{Nom.}}$',color=cm.matlab(0));
                ax.plot(times, tsRslt['vMax'][1].T,'-.',label='$V_{\mathrm{max}}^{\mathrm{Nom.}}$',color=cm.matlab(0));
                ax.plot(times, tsRslt['vMinMv'][2].T,':',label='$V_{\mathrm{MV,\,min}}^{\mathrm{Phase}}$',color=cm.matlab(1));
                ax.plot(times, tsRslt['vMinLv'][2].T,label='$V_{\mathrm{LV,\,min}}^{\mathrm{Phase.}}$',color=cm.matlab(1));
                ax.plot(times, tsRslt['vMax'][2].T,'-.',label='$V_{\mathrm{max}}^{\mathrm{Phase.}}$',color=cm.matlab(1));
                
                ax.plot(xlm,[tsRslt['cns']['mvHi']]*2,'k--')
                ax.plot(xlm,[tsRslt['cns']['mvLo']]*2,'k:')
                ax.plot(xlm,[tsRslt['cns']['lvLo']]*2,'k--')
                if dayType=='wtr' and feeder==17:
                    ax.text(0.3,tsRslt['cns']['mvLo']+0.005,'$V_{-,\,\\mathrm{MV}}$')
                    ax.text(0.3,tsRslt['cns']['lvLo']+0.006,'$V_{-,\,\\mathrm{LV}}$')
                if dayType=='wtr' and feeder==24:
                    ax.text(0.3,tsRslt['cns']['mvLo']+0.007,'$V_{-,\,\\mathrm{MV}}$')
                    ax.text(0.3,tsRslt['cns']['lvLo']+0.006,'$V_{-,\,\\mathrm{LV}}$')
                
                if dayType=='smr' and feeder==17:
                    ax.legend(fontsize='small',loc='lower left',ncol=2)
                    # ax.legend(loc='lower left',labelspacing=0.1,ncol=2)
                if dayType=='smr' and feeder==24:
                    ax.legend(fontsize='small',loc='center right',ncol=2)
                
                ax.set_xlim(xlm); ax.set_xticks(xtcks)
                ax.set_xlabel('Time, hour')
                ax.set_ylabel('Voltage, pu')
                plt.tight_layout()
                if 'pltSave' in locals(): plotSaveFig(os.path.join(SDT5,figName+self.feeder+dayType+self.invLossType),pltClose=True)
                plt.show()

            figName = 'qPhs'
            if figName in figNameSet:
                fig,ax = plt.subplots(figsize=tsFigSze)
                ax.set_prop_cycle(linestyle=['-','--','-.'])
                # plts=ax.plot(times, tsRslt['nPctrl']*tsRslt['qPhs'].T,color=cm.matlab(1))
                # ax.plot(times,1e-3*tsRslt['nPctrl']*tsRslt['qLimSet'],'k--')
                # ax.plot(times,-1e-3*tsRslt['nPctrl']*tsRslt['qLimSet'],'k--')
                plts=ax.plot(times, tsRslt['qPhs'].T,color=cm.matlab(1))
                ax.plot(times,1e-3*tsRslt['qLimSet'],'k--')
                # ax.step(times,1e-3*tsRslt['qLimSet'],'k',linestyle='--')
                ax.plot(times,-1e-3*tsRslt['qLimSet'],'k--')
                # ax.step(times,-1e-3*tsRslt['qLimSet'],'k',linestyle='--')
                ax.set_xlabel('Time, hour')
                ax.set_ylabel('Reactive power\n per generator, kVAr')
                ax.legend(plts,['a','b','c'])
                ax.set_xlim(xlm); ax.set_xticks(xtcks)
                if dayType=='wtr':
                    ax.text(6.8,2.04,'$Q_{+,\,\mathrm{invr}}$')
                    ax.text(6.8,-2.2,'$Q_{-,\,\mathrm{invr}}$')
                plt.tight_layout()
                if 'pltSave' in locals(): plotSaveFig(os.path.join(SDT5,figName+self.feeder+dayType+self.invLossType),pltClose=True)
                plt.show()
            
            tPwr = tsRslt['tPwr']
            tLss = tsRslt['tLss']
            figName = 'dPwr'
            if figName in figNameSet:
                fig,ax = plt.subplots(figsize=tsFigSze)
                # ax.set_prop_cycle(color=cm.matlab([1,2]))
                ax.set_prop_cycle(color=cm.matlab([1]))
                # ax.plot(times,tPwr[0]-tPwr[1])
                # ax.plot(times,tPwr[0]-tPwr[2],label='Tot. power')
                ax.plot(times,tPwr[1]-tPwr[2],label='Tot. power')
                # ax.plot(times,tLss[0]-tLss[1],'-.')
                # ax.plot(times,tLss[0]-tLss[2],'-.',label='Losses')
                ax.plot(times,tLss[1]-tLss[2],'-.',label='Losses')
                ax.set_xlim(xlm); ax.set_xticks(xtcks)
                ax.set_xlabel('Time, hour')
                ax.set_ylabel('Power, kW')
                ax.legend()
                plt.tight_layout()
                if 'pltSave' in locals(): plotSaveFig(os.path.join(SDT5,figName+self.feeder+dayType+self.invLossType),pltClose=True)
                plt.show()
            
            print('Power saved:',sum(tPwr[1]-tPwr[2])*0.5,'kWh over the day')
            print('Power saved:',sum(tPwr[0])*0.5,'kWh over the day (load)')
            print('Power saved:',sum(tPwr[1]-tPwr[2])*0.5/tsRslt['nPctrl'],'kWh over the day per customer')
            print('Power saved:',sum(tPwr[1]-tPwr[2])*0.5*365/tsRslt['nPctrl'],'kWh over the year per customer')
            # print('Power saved:',sum(tPwr[1]-tPwr[2])*0.5/24,'average kW')
            # print('Power saved:',sum(tPwr[1]-tPwr[2])*0.5/(24*tsRslt['nPctrl']),'average kW per customer')
                    
            figName = 'efcy'
            if figName in figNameSet:
                oldSettings = np.seterr()
                np.seterr(all='ignore') # we want to ignore divide by zero.
                phsVec = np.array([self.nPh1,self.nPh2,self.nPh3])
                efcy = (1e3*(tPwr[1]-tPwr[2]))/(phsVec.dot(np.abs(tsRslt['qPhs'])))
                np.seterr(**oldSettings)
                efcy[np.isinf(efcy)] = 0
                
                fig,ax = plt.subplots(figsize=tsFigSze)
                ax.plot(times,efcy,color=cm.matlab(1))
                ax.set_xlabel('Time, hour')
                ax.set_ylabel('Efficacy, W/kVAr')
                ax.set_xlim(xlm); ax.set_xticks(xtcks)
                plt.tight_layout()
                if 'pltSave' in locals(): plotSaveFig(os.path.join(SDT5,figName+self.feeder+dayType+self.invLossType),pltClose=True)
                plt.show()

            figName = 'tapSet'
            if figName in figNameSet:
                fig,ax = plt.subplots(figsize=tsFigSze)
                ax.plot(times,tsRslt['tSet'][1:3].T)
                ax.plot(xlm,[16,16],'k--')
                ax.plot(xlm,[-16,-16],'k--')
                ax.set_xlim(xlm); ax.set_xticks(xtcks)
                ax.set_xlabel('Time, hour')
                ax.set_ylabel('Tap position')
                plt.tight_layout()
                if 'pltSave' in locals(): plotSaveFig(os.path.join(SDT5,figName+self.feeder+dayType+self.invLossType),pltClose=True)
                plt.show()

if 't_loadModelVecs' in locals():
    from numpy.linalg import norm
    tableSet = []
    # feederSet = [26]
    for feeder in feederSet:
        tableSet.append([])
        linPoints = linPointsDict[feeder]['all']
        self = main(feeder,modelType='loadOnly',linPoint=linPoints[-1],method='fot')
        
        pLoad = 1e3*self.ploadL[self.nPctrl:self.nSctrl]
        pLoss = 1e3*self.qpLlss[self.nPctrl:self.nSctrl]
        p = (pLoad + pLoss)
        
        tableSet[-1].append(feederIdxTidy[feeder])
        tableSet[-1].append( '%.2f' % (norm(pLoad,ord=1)/len(pLoad)))
        tableSet[-1].append( '%.2f' % (norm(pLoss,ord=1)/len(pLoad)))
        tableSet[-1].append( '%.2f' % (norm(p,ord=1)/len(pLoad)))
        tableSet[-1].append( '%.1f' % (np.arccos( (pLoad.dot(pLoss))/(norm(pLoad)*norm(pLoss)) )*180/np.pi))
        
    head0 = ['Feeder','Plds','Plss','P','Theta']
    heading = ['Feeder','$\\dfrac{\\| \\psi_{\mathrm{Q, lds}}\\|}{\\sqrt{N_{\mathrm{Q}}}}$, W/kVAr',
                        '$\\dfrac{\\| \\psi_{\mathrm{Q, lss}}\\|}{\\sqrt{N_{\mathrm{Q}}}}$, W/kVAr',
                        '$\\dfrac{\\| \\psi_{\mathrm{Q}} \\|}{\\sqrt{N_{\mathrm{Q}}}}$, W/kVAr',
                        '$ \\theta_{\\psi},\,^{\\circ}$']
    caption = 'Summary of sensitivities to reactive power injections in load, loss, and the total for both of these combined. The angle $\\theta_{\\psi}$ is calculated between the load and loss sensitivity vectors; $N_{\mathrm{Q}}$ is the number of elements in the sensitivity vector $\\psi_{\mathrm{Q}}$.'
    print(head0)
    print(*tableSet,sep='\n')
    label = 'loadModelVecs'
    
    if 'pltSave' in locals(): basicTable( caption, label, heading, tableSet, sdt('t3','t') )
    
    
    
if 'f_uncSolution' in locals():
    feeder = 17
    linPoints = linPointsDict[feeder]['all']
    self = main(feeder,modelType='loadOnly',linPoint=linPoints[-1],method='fot')
    slnX,slnF = self.solveQpUnc()[:2]
    # self.showQpSln(xStar,fStar)
    pOut = slnX[:self.nPctrl]
    qOut = slnX[self.nPctrl:self.nPctrl*2]
    tOut = slnX[self.nPctrl*2:]

    TL,PL,TC,CL,V,I,Vc,Ic = slnF
    TL0,PL0,TC0,CL0,V0,I0,Vc0,Ic0 = self.slnF0
    iDrn = (-1)**( np.abs(np.angle(Ic/Ic0))>np.pi/2 )

    figName='uncSolutionI_'
    fig,ax1 = plt.subplots(figsize=(4.4,3.0))
    ax1.plot(100*iDrn*abs(Ic/(self.iScale*self.iXfmrLims)),'o',markerfacecolor='None',markersize=1.4,markeredgewidth=0.6,label='Unc. QP Sln')
    ax1.plot(100*abs(Ic0)/(self.iScale*self.iXfmrLims),'o',markerfacecolor='None',markersize=1.4,markeredgewidth=0.6,label='Nom. Sln')
    ax1.plot(100*np.ones(len(self.iXfmrLims)),'k_',markeredgewidth=0.5)
    ax1.set_xlabel('Xfmr Index')
    ax1.set_ylabel('Current, % of $I_{\\mathrm{rated}}$')
    ax1.legend(fontsize='small')
    ax1.set_ylim( (-70,130) )
    ax1.set_xlim((-3,len(Ic0)+3))
    ax1.grid(True)
    plt.tight_layout()
    if 'pltSave' in locals(): plotSaveFig(os.path.join(sdt('t3','f'),figName+self.feeder),pltClose=True)

    plt.show()
    
    
    vltgs = (V/self.vKvbase)[self.vIn]
    vltgsHi = vltgs>((self.vHi/self.vKvbase)[self.vIn])
    vltgsLo = vltgs<((self.vLo/self.vKvbase)[self.vIn])
    sum(vltgsHi)
    sum(vltgsLo)/len(vltgs) # used to calculate the number of loads out of bounds.
    
    figName='uncSolutionV_'
    fig,ax0 = plt.subplots(figsize=(4.4,3.0))
    # plot voltages versus voltage limits
    ax0.plot((V/self.vKvbase)[self.vIn],'o',markerfacecolor='None',markersize=0.8,markeredgewidth=0.6);
    ax0.plot((V0/self.vKvbase)[self.vIn],'o',markerfacecolor='None',markersize=0.8,markeredgewidth=0.6);
    ax0.plot([0,0],[0,0],'o',markerfacecolor='None',markersize=1.2,markeredgewidth=1.0,label='Unc. QP Sln',color=cm.matlab(0))
    ax0.plot([0,0],[0,0],'o',markerfacecolor='None',markersize=1.2,markeredgewidth=1.0,label='Nom. Sln',color=cm.matlab(1))
    ax0.plot((self.vHi/self.vKvbase)[self.vIn],'k_',markeredgewidth=0.5);
    ax0.plot((self.vLo/self.vKvbase)[self.vIn],'k_',markeredgewidth=0.5);
    ax0.set_xlabel('Bus Index')
    ax0.set_ylabel('Voltage, pu')
    ax0.grid(True)
    ax0.legend(fontsize='small')
    ax0.set_ylim((0.89,1.06))
    plt.tight_layout()
    if 'pltSave' in locals(): plotSaveFig(os.path.join(sdt('t3','f'),figName+self.feeder),pltClose=True)
    
    print(max(abs(slnX[self.nPctrl:self.nPctrl*2])))
    plt.show()


if 't_uncBoundViolations' in locals():
    feederSet = [17,18,'n1','n10',26,24,'n27']
    # feederSet = [17,26]
    resultTable = []
    for feeder in feederSet:
        resultTable.append([])
        linPoints = linPointsDict[feeder]['all']
        self = main(feeder,modelType='loadOnly',linPoint=linPoints[-1],method='fot')
        slnX,slnF = self.solveQpUnc()[:2]
        TL,PL,TC,CL,V,I,Vc,Ic = slnF
        vMaxOut = max(abs(V/self.vKvbase)[self.vIn])
        vMinOut = min(abs(V/self.vKvbase)[self.vIn])
        iMaxOut = max(100*abs(Ic)/(self.iScale*self.iXfmrLims))
        qMax = max(abs(slnX[self.nPctrl:self.nPctrl*2]))
        
        resultTable[-1].append(feederAllTidy[self.feeder])
        resultTable[-1].append( "%.3f" % (vMaxOut))
        resultTable[-1].append( "%.3f" % (vMinOut))
        resultTable[-1].append( "%.1f" % (iMaxOut))
        resultTable[-1].append( "%.2f" % (qMax))
        
    caption = 'Constraint extrema at the unconstrained optimization point $x_{\\mathrm{Q,\,Unc}}^{\dagger}$ (i.e. solved in reactive power only).'
    heading = ['Feeder','max$(V)$, pu','min$(V)$, pu','max$(I_{\\mathrm{xfmr}})$, \\%','max$(Q_{\\mathrm{invr}})$, kVAr']
    label = 't_uncBoundViolations'
    if 'pltSave' in locals(): basicTable(caption,label,heading,resultTable,sdt('t3','t'))




# self = main('n27',modelType='loadOnly',linPoint=1.0)
# self = main(26,modelType='loadOnly',linPoint=1.0)
# self = main(24,modelType='loadOnly',linPoint=1.0)
# self.runQpSet()

if 'f_hcParamsFig' in locals():
    x = np.linspace(0,100)
    y = 100*np.exp( 0.20*(x - 60) )/(1 + np.exp( 0.20*(x - 60) ))
    fig,ax0 = plt.subplots(figsize=(4.4,2.5))
    plt.xlim((0,100))
    plt.ylim((-2,102))
    plt.plot(x,y)
    # plt.grid()
    plt.xlabel('Penetration $p$, %')
    plt.ylabel('Likelihood of constraint\nviolation, $F_{p}(p)$ %')
    plt.tight_layout()
    # if 'pltSave' in locals(): plotSaveFig(os.path.join(sdt('t2','f'),'hcParamsFig'),pltClose=True)
    if 'pltSave' in locals():
        SN = os.path.join(sdt('t2','f'),'hcParamsFig')
        plt.savefig(SN+'.png',bbox_inches='tight',pad_inches=0.03)
        plt.savefig(SN+'.pdf',bbox_inches='tight',pad_inches=0.03)
        plt.close()
    plt.show()
    
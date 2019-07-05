import pickle, sys, os, getpass
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy.stats import pearsonr
plt.style.use('tidySettings')

WD = os.path.dirname(sys.argv[0])
sys.path.append(os.path.dirname(WD))

from dss_python_funcs import basicTable, vecSlc, set_ax_size
from linSvdCalcs import plotBoxWhisk, getKcdf, plotCns
from matplotlib.patches import Ellipse

# feeders = ['13bus','34bus','123bus','8500node','eulv','usLv','epriJ1','epriK1','epriM1','epri5','epri7','epri24']
feeders = ['34bus','123bus','8500node','epriJ1','epriK1','epriM1','epri5','epri7','epri24']
# feeders = ['34bus','123bus','8500node','epriJ1','epriK1','epriM1','epri24']
# feeders = ['34bus','123bus','epriJ1','epriK1','epriM1','epri5','epri7']

feedersTidy = {'34bus':'34 Bus','123bus':'123 Bus','8500node':'8500 Node','epriJ1':'Ckt. J1','epriK1':'Ckt. K1','epriM1':'Ckt. M1','epri5':'Ckt. 5','epri7':'Ckt. 7','epri24':'Ckt. 24'}

feeders_mult = ['34bus','123bus','epriK1','epri5','epri7','epriM1','epriJ1','epri24']

# feeders_dcp = ['8500node','epriJ1','epriK1','epriM1','epri24']
feeders_dcp = ['8500node','epriJ1','epriM1','epri24']
feeders_lp = feeders_dcp

# t_timeTable = 1 # timeTable # <--- now not in use!
# t_rsltSvty = 1 # sensitivity table # <--- now not in use!
# t_results = 1
# f_dssVlinWght = 1 # gammaFrac boxplot results
# f_dssVlinWghtConservative = 1
# f_mcLinUpg = 1
# f_mcLinCmp = 1
# f_linMcSns = 1
# f_plotCns = 1 # <--- also useful for debugging.
# f_plotCns_paramUpdate = 1
# f_plotLpUpg = 1
# f_plotLp = 1
f_errorCorr = 1
# f_maeRerun = 1

consFactor = 1.10 # for f_dssVlinWghtConservative

pltSave=True
pltShow=True

figSze0 = (5.2,3.4)
figSze1 = (5.2,2.5)
figSze2 = (5.2,3.0)
figSze3 = (5.2,2.2)
figSze4 = (5.2,1.8)
TD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\tables\\"
FD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\figures\\"

rsltsFrac = {}; rsltsSns = {}; rsltsUpg = {}; rsltsLp = {}; rsltsUnom = {}; rsltsTap = {}; rsltsMult = {}
for feeder in feeders:
    # RD = os.path.join(WD,feeder,'linHcCalcsRslt_gammaFrac_finale.pkl')
    RD = os.path.join(WD,feeder,'linHcCalcsRslt_gammaFrac_finale_dpndnt.pkl')
    with open(RD,'rb') as handle:
        rsltsFrac[feeder] = pickle.load(handle)
    RD = os.path.join(WD,feeder,'linHcCalcsRslt_gammaFrac_tapSet.pkl')
    with open(RD,'rb') as handle:
        rsltsTap[feeder] = pickle.load(handle)

for feeder in feeders_dcp:
    RDsns = os.path.join(WD,feeder,'linHcCalcsSns_gammaFrac_new.pkl')
    RDupg = os.path.join(WD,feeder,'linHcCalcsUpg.pkl')
    
    with open(RDsns,'rb') as handle:
        rsltsSns[feeder] = pickle.load(handle)
    with open(RDupg,'rb') as handle:
        rsltsUpg[feeder] = pickle.load(handle)
for feeder in feeders_lp:
    RD = os.path.join(WD,feeder,'linHcPrg.pkl')
    RDupgNom = os.path.join(WD,feeder,'linHcCalcsUpgNom.pkl')
    
    with open(RD,'rb') as handle:
        rsltsLp[feeder] = pickle.load(handle)
    with open(RDupgNom,'rb') as handle:
        rsltsUnom[feeder] = pickle.load(handle)
# SDerrors = os.path.join(WD,'feederErrors.pkl')
SDerrors = os.path.join(WD,'feederErrors_corr.pkl')
with open(SDerrors,'rb') as saveFile:
    feederErrors = pickle.load(saveFile) # NB these are actually sensitivities
feederDict = {'34bus':6,'123bus':8,'8500node':9,'epri5':17,'epri7':18,'epriJ1':19,'epriK1':20,'epriM1':21,'epri24':22}

for feeder in feeders_mult:
    RDmult = os.path.join(WD,'tapMultSet',feeder+'linHcCalcsRslt_gammaFrac_tapMultSet.pkl')
    with open(RDmult,'rb') as handle:
        rsltsMult[feeder] = pickle.load(handle)


# kCdfLin = [];    kCdfDss = []; kCdfNom = []; 
# LmeanNorm = []; feederTidySet = []
# timeTableData = []
# rsltSvtyData = []
# resultsData = []
# dssComparisonData = []
# runTimeSample = []
# runTimeFull = []
# rslt34 = rsltsFrac['34bus'] # useful for debugging

# idxChosen = [0,1,5,10,15,19,20]
# # idxChosen = [0,2,5,10,15,18,20]

# for rslt in rsltsFrac.values():
    # dataSet = []
    # dataSet.append(feedersTidy[rslt['feeder']])
    # dataSet.append('%.2f' % rslt['dssHcRslNom']['runTime'])
    # dataSet.append('%.2f' % rslt['linHcRsl']['runTime'])
    # dataSet.append('%.2f' %  rslt['maeVals']['dssTgtMae'])
    # dataSet.append('%.2f' %  rslt['maeVals']['dssNomMae'])
    # timeTableData.append(dataSet)

    # dataSet = []
    # dataSet.append(feedersTidy[rslt['feeder']])
    # dataSet.append('%.1f' % (100-rslt['preCndLeft']))
    # dataSet.append('%.2f' % rslt['maeVals']['nomMae'])
    # dataSet.append('%.2f' % (rslt['maeVals']['nmcMae'])) # <--- to do! [?]
    # rsltSvtyData.append(dataSet)
    
    # dataSet = []
    # dataSet.append(feedersTidy[rslt['feeder']])
    # dataSet.append('%.2f' %  rslt['maeVals']['dssTgtMae'])
    # dataSet.append('%.2f' % (rslt['maeVals']['nmcMae'])) # <--- to do! [?]
    # dataSet.append('%.2f' % rslt['dssHcRslNom']['runTime'])
    # dataSet.append('%.2f' % rslt['linHcRsl']['runTime'])
    # resultsData.append(dataSet)
    
    # dataSet = []
    # dataSet.append(feedersTidy[rslt['feeder']])
    # dataSet.append('%.2f' % (rslt['maeVals']['dssMae']))
    # dssComparisonData.append(dataSet)
    
    # dataSet = []
    # dataSet.append('%.2f' % (rslt['linHcRsl']['runTimeSample'])) # <--- to do! [?]
    # runTimeSample.append(dataSet)
    
    # dataSet = []
    # dataSet.append('%.2f' % (rslt['linHcRslNom']['runTime'])) # <--- to do! [?]
    # runTimeFull.append(dataSet)
    
    # kCdfLin.append(rslt['linHcRsl']['kCdf'][idxChosen])
    # kCdfDss.append(rslt['dssHcRslTgt']['kCdf'][idxChosen])
    # kCdfNom.append(rslt['dssHcRslNom']['kCdf'][[0,1,5,10,15,19,20]])
    # LmeanNorm.append( np.mean(np.abs(rslt['dssHcRslTgt']['Vp_pct']-rslt['linHcRsl']['Vp_pct']))*0.01 )
    # feederTidySet.append(feedersTidy[rslt['feeder']])
    
kCdfLin = [];    kCdfDss = []; kCdfNom = []; kCdfLinConservative=[]
LmeanNorm = []; feederTidySet = []; corrPlot = []
timeTableData = []
rsltSvtyData = []
resultsData = []
dssComparisonData = []
runTimeSample = []
runTimeFull = []
rslt34 = rsltsTap['34bus'] # useful for debugging

idxChosen = [0,1,5,10,15,19,20]
idxChosenConservative = (np.array(idxChosen)*5).tolist()
# idxChosen = [0,2,5,10,15,18,20]

for rslt in rsltsTap.values():
    dataSet = []
    dataSet.append(feedersTidy[rslt['feeder']])
    dataSet.append('%.2f' % rslt['dssHcRslTapSet']['runTime'])
    dataSet.append('%.2f' % rslt['linHcRsl']['runTime'])
    dataSet.append('%.2f' %  rslt['maeVals']['dssSetMae'])
    dataSet.append('%.2f' %  rslt['maeVals']['dssLckMae'])
    timeTableData.append(dataSet)

    dataSet = []
    dataSet.append(feedersTidy[rslt['feeder']])
    dataSet.append('%.1f' % (100-rslt['preCndLeft']))
    dataSet.append('%.2f' % rslt['maeVals']['nomMae'])
    dataSet.append('%.2f' % (rslt['maeVals']['nmcMae'])) # <--- to do! [?]
    rsltSvtyData.append(dataSet)
    
    dataSet = []
    dataSet.append(feedersTidy[rslt['feeder']])
    # dataSet.append('%.2f' %  rslt['maeVals']['dssLckMae'])
    dataSet.append('%.2f' %  rslt['maeVals']['dssTgtMae'])
    dataSet.append('%.2f' % rslt['dssHcRslTapSet']['runTime'])
    dataSet.append('%.2f' % rslt['linHcRsl']['runTime'])
    dataSet.append('%.2f' % (rslt['maeVals']['nmcMae'])) # <--- to do! [?]
    resultsData.append(dataSet)
    
    dataSet = []
    dataSet.append(feedersTidy[rslt['feeder']])
    dataSet.append('%.2f' % (rslt['maeVals']['dssMae']))
    dssComparisonData.append(dataSet)
    
    dataSet = []
    dataSet.append('%.2f' % (rslt['linHcRsl']['runTimeSample'])) # <--- to do! [?]
    runTimeSample.append(dataSet)
    
    dataSet = []
    dataSet.append('%.2f' % (rslt['linHcRsl']['runTime'])) # <--- to do! [?]
    runTimeFull.append(dataSet)
    
    kCdfLin.append(rslt['linHcRsl']['kCdf'][idxChosen])
    # kCdfLin.append(rslt['linHcRslNom']['kCdf'][idxChosen])
    
    param = rslt['pdfData']['prms']
    Vp_pct_conservative = np.sum( rslt['linHcRslNom']['Lp_pct'][:,0,:]<consFactor,axis=1 )
    kCdfLinConservative.append(  (100*getKcdf(param,Vp_pct_conservative)[0])[idxChosen])
    
    # kCdfDss.append(rslt['dssHcRslTapLck']['kCdf'][idxChosen])
    # kCdfDss.append(rslt['dssHcRslTapSet']['kCdf'][idxChosen])
    kCdfDss.append(rslt['dssHcRslTapTgt']['kCdf'][idxChosen])
    # kCdfNom.append(rslt['dssHcRslTapLck']['kCdf'][idxChosen])
    
    # LmeanNorm.append( np.mean(np.abs(rslt['dssHcRslTgt']['Vp_pct']-rslt['linHcRsl']['Vp_pct']))*0.01 )
    feederTidySet.append(feedersTidy[rslt['feeder']])
    
    corrPlot.append( [ feederErrors[feederDict[rslt['feeder']]], rslt['maeVals']['dssTgtMae'] ] )
    
        
kCdfSns0 = []; kCdfSns1 = []; kCdfLinSns = [] # new lin needed coz this is only some models
feederSnsSmart = []; LsnsNorm = []
for rslt in rsltsSns.values():
    kCdfLinSns.append(rslt['linHcRsl']['kCdf'][[0,1,5,10,15,19,20]])
    kCdfSns0.append(rslt['linHcSns0']['kCdf'][[0,1,5,10,15,19,20]])
    kCdfSns1.append(rslt['linHcSns1']['kCdf'][[0,1,5,10,15,19,20]])
    
    LsnsNorm.append(rslt['regMaes'])
    feederSnsSmart.append(feedersTidy[rslt['feeder']])
i=0
linHcRsl = rslt['linHcRsl'] # for debugging

kCdfUpgBef = []; kCdfUpgAft = []
for rslt in rsltsUpg.values():
    kCdfUpgBef.append(rslt['linHcRslBef']['kCdf'][[0,1,5,10,15,19,20]])
    kCdfUpgAft.append(rslt['linHcRslAft']['kCdf'][[0,1,5,10,15,19,20]])

kCdfLpNom = []; kCdfLpUpg = []; kCdfLpTQ = []; kCdfLpT0 = []; kCdfLp00 = []
feederLpSmart = []
for rslt in rsltsLp.values():
    kCdfLpNom.append(rslt['linHcRsl']['kCdf'][[0,1,5,10,15,19,20]])
    # lpLpNom.append(rslt['linHcRsl']['Lp_pct'])
    kCdfLpUpg.append(rslt['linHcUpg']['kCdf'][[0,1,5,10,15,19,20]])
    kCdfLpTQ.append(rslt['linLpRslTQ']['kCdf'][[0,1,5,10,15,19,20]])
    kCdfLpT0.append(rslt['linLpRslT0']['kCdf'][[0,1,5,10,15,19,20]])
    kCdfLp00.append(rslt['linLpRsl00']['kCdf'][[0,1,5,10,15,19,20]])
    feederLpSmart.append(feedersTidy[rslt['feeder']])
Xlp = np.arange(len(kCdfLpNom),0,-1)

kCdfUpgNomBef = []; kCdfUpgNomAft = []
for rslt in rsltsUnom.values():
    kCdfUpgNomBef.append(rslt['linHcRslBef']['kCdf'][[0,1,5,10,15,19,20]])
    kCdfUpgNomAft.append(rslt['linHcRslAft']['kCdf'][[0,1,5,10,15,19,20]])


# print('Sensitivity errors:')
# print(*LsnsNorm,sep='\n')

print('DSS static versus HC:')
print(*dssComparisonData,sep='\n')

print('\nTime to generate MC runs:')
print(*runTimeSample,sep='\n')

print('\nFull linear model runtime:')
print(*runTimeFull,sep='\n')


# TABLE 1 - timings + MAE ======================= 
if 't_timeTable' in locals():
    caption='OpenDSS and Linear models result comparison'
    label='timeTable'
    heading = ['Model','OpenDSS time','Linear time','MAE (tight), \%','MAE (nominal), \%']
    data = timeTableData
    if 'pltShow' in locals():
        print('\n',heading)
        print(*data,sep='\n')
    if 'pltSave' in locals():
        basicTable(caption,label,heading,data,TD)
# ===============================================

# TABLE 2 - sensitivity MAE ===================== 
heading = ['Model','Precond. calc effort, \%','Precond. MAE','Nmc MAE']
data = rsltSvtyData
# if 'pltShow' in locals():
print('\nPreconditioning effort saved:')
print(heading)
print(*data,sep='\n')
if 't_rsltSvty' in locals():
    caption='Sensitivity to simulation parameters'
    label='rsltSvty'
    if 'pltSave' in locals():
        basicTable(caption,label,heading,data,TD)
# ===============================================

# TABLE 3 - results table (single table) ===================== 
if 't_results' in locals():
    caption='Sensitivity to simulation parameters'
    label='results'
    heading = ['Model','MAE (DSS.), \%','OpenDSS time','Linear time','MAE (MC rerun), \%']
    data = resultsData
    if 'pltShow' in locals():
        print('\n',heading)
        print(*data,sep='\n')
    if 'pltSave' in locals():
        basicTable(caption,label,heading,data,TD)
# ===============================================


dx = 0.175; ddx=dx/1.5
X = np.arange(len(kCdfLin),0,-1)
i=0
# clrA,clrB,clrC,clrD,clrE = cm.tab10(np.arange(5))
clrA,clrB,clrC,clrD,clrE,clrF,clrG = cm.matlab(np.arange(7))
clrH = '#777777'

# # RESULTS 1 - opendss vs linear model, k =====================
if 'f_dssVlinWght' in locals():
    fig = plt.figure(figsize=figSze0)
    ax = fig.add_subplot(111)
    i=0
    for x in X:
        ax = plotBoxWhisk(ax,x+dx,ddx,kCdfLin[i][1:-1],clrA,bds=kCdfLin[i][[0,-1]],transpose=True)
        ax = plotBoxWhisk(ax,x-dx,ddx,kCdfDss[i][1:-1],clrB,bds=kCdfDss[i][[0,-1]],transpose=True)
        i+=1
    ax.plot(0,0,'-',color=clrA,label='Linear, $\hat{f}$')
    ax.plot(0,0,'-',color=clrB,label='OpenDSS, $f$')
    plt.legend(fontsize='small')
    
    plt.xlim((-3,103))
    plt.ylim((0.4,9.6))
    plt.grid(True)
    plt.yticks(X,feederTidySet)
    plt.xlabel('Loads with PV installed, %')
    plt.tight_layout()
    if 'pltSave' in locals():
        plt.savefig(FD+'dssVlinWght.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'dssVlinWght.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()

# # RESULTS 1 - opendss vs linear model, k =====================
if 'f_dssVlinWghtConservative' in locals():
    fig = plt.figure(figsize=figSze2)
    ax = fig.add_subplot(111)
    i=0
    for x in X:
        ax = plotBoxWhisk(ax,x+1.5*dx,ddx,kCdfLin[i][1:-1],clrA,bds=kCdfLin[i][[0,-1]],transpose=True)
        ax = plotBoxWhisk(ax,x+0*dx,ddx,kCdfDss[i][1:-1],clrB,bds=kCdfDss[i][[0,-1]],transpose=True)
        ax = plotBoxWhisk(ax,x-1.5*dx,ddx,kCdfLinConservative[i][1:-1],clrC,bds=kCdfLinConservative[i][[0,-1]],transpose=True)
        i+=1
    ax.plot(0,0,'-',color=clrA,label='Linear, $\hat{f}$')
    ax.plot(0,0,'-',color=clrB,label='OpenDSS, $f$')
    ax.plot(0,0,'-',color=clrC,label='Linear, $\hat{f}^{\:110\%}$ ')
    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),fontsize='small')
    
    plt.xlim((-3,103))
    plt.ylim((0.4,9.6))
    plt.grid(True)
    plt.yticks(X,feederTidySet)
    plt.xlabel('Fraction of Loads with PV, %')
    plt.tight_layout()
    if 'pltSave' in locals():
        plt.savefig(FD+'dssVlinWghtConservative.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'dssVlinWghtConservative.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()

# RESULTS 5 =====================
Xsns = np.arange(len(kCdfLinSns),0,-1)
if 'f_linMcSns' in locals():
    fig = plt.figure(figsize=figSze1)
    ax = fig.add_subplot(111)
    i=0
    for x in Xsns:
        ax = plotBoxWhisk(ax,x-1.3*dx   ,0.66*ddx,kCdfSns0[i][1:-1],clrD,bds=kCdfSns0[i][[0,-1]],transpose=True)
        ax = plotBoxWhisk(ax,x,0.66*ddx,kCdfLinSns[i][1:-1],clrC,bds=kCdfLinSns[i][[0,-1]],transpose=True)
        ax = plotBoxWhisk(ax,x+1.3*dx,0.66*ddx,kCdfSns1[i][1:-1],clrE,bds=kCdfSns1[i][[0,-1]],transpose=True)
        i+=1
    ax.plot(0,0,'-',color=clrD,label='t = +1')
    ax.plot(0,0,'-',color=clrC,label='t = 0')
    ax.plot(0,0,'-',color=clrE,label='t = -1')
    plt.xlim((-3,103))
    plt.ylim((0.4,5.6))
    plt.legend()
    plt.grid(True)
    plt.yticks(Xsns,feederSnsSmart)
    plt.xlabel('Loads with PV installed, %')
    plt.tight_layout()
    if 'pltSave' in locals():
        plt.savefig(FD+'linMcSns.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'linMcSns.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()

if 'f_mcLinUpg' in locals():
    fig, ax = plt.subplots(figsize=figSze1)
    i=0
    for x in Xsns:
        ax = plotBoxWhisk(ax,x+dx,ddx,kCdfUpgBef[i][1:-1],clrA,bds=kCdfUpgBef[i][[0,-1]],transpose=True)
        ax = plotBoxWhisk(ax,x-dx,ddx,kCdfUpgAft[i][1:-1],clrE,bds=kCdfUpgAft[i][[0,-1]],transpose=True)
        i+=1
    ax.plot(0,0,'-',color=clrA,label='Before')
    ax.plot(0,0,'-',color=clrE,label='After')
    plt.xlim((-3,103))
    plt.ylim((0.4,5.6))
    plt.legend()
    plt.grid(True)
    plt.yticks(Xsns,feederSnsSmart)
    plt.xlabel('Loads with PV installed, %')
    plt.tight_layout()
    if 'pltSave' in locals():
        plt.savefig(FD+'mcLinUpg.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'mcLinUpg.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()

# RESULTS 7 - OpenDSS vs Linear pltCons
if 'f_plotCns' in locals():
    # feederPlot='8500node'
    feederPlot='epriJ1'
    # rsltM1 = rsltsFrac[feederPlot]
    rsltM1 = rsltsTap[feederPlot]
    pdf = rsltM1['pdfData']
    linRsl = rsltM1['linHcRslNom'] # remark: it has to be this so that the second set of constraints shows up.
    # dssRsl = rsltM1['dssHcRslTgt']
    dssRsl = rsltM1['dssHcRslTapTgt']
    
    fig, (ax1,ax0) = plt.subplots(figsize=(5.9,4.0),sharex=True,nrows=2,ncols=1)
    
    clrs = cm.matlab(np.arange(4)+2)
    ax0.set_prop_cycle(color=clrs)
    Cns_dss = dssRsl['Cns_pct']
    Cns_lin = linRsl['Cns_pct']
    x_vals = 100*pdf['prms']
    y_dss = Cns_dss[:,0,:][:,[7,1,3,0]]
    y_lin = Cns_lin[:,0,:][:,[7,1,3,0]]

    ax0.plot(x_vals,y_lin,'-')
    ax0.plot(x_vals,y_dss,'--')
    ax0.set_xlabel('Fraction of loads with PV, %');
    
    ax0.legend(['$V_{+}^{\mathrm{LV}}$, Hi $S_{\mathrm{Load}}$','$V_{+}^{\mathrm{MV}}$, Lo $S_{\mathrm{Load}}$','$V_{+}^{\mathrm{LV}}$, Lo $S_{\mathrm{Load}}$','$\Delta V$ (Vlt. Dev.)'],loc='center left', bbox_to_anchor=(1, 0.5),fontsize='small',title='Constraint Type')
    
    if feederPlot=='8500node':
        ax0.annotate('OpenDSS',xytext=(53,52),xy=(69,28),arrowprops={'arrowstyle':'->','linewidth':1.0})
        ax0.annotate('Linear',xytext=(75,7),xy=(75,33),arrowprops={'arrowstyle':'->','linewidth':1.0})
    if feederPlot=='epriJ1':
        ax0.annotate('OpenDSS',xytext=(19,30),xy=(23,61),arrowprops={'arrowstyle':'->','linewidth':1.0},fontsize=10)
        ax0.annotate('Linear',xytext=(5,72),xy=(20,55),arrowprops={'arrowstyle':'->','linewidth':1.0},fontsize=10)
    
    ax0.set_ylabel('Constraint violations, %');
    ax0.set_xlim((0,100))
    ax0.set_ylim((-3,103))
    
    ax1.plot(x_vals,linRsl['Vp_pct'])
    ax1.plot(x_vals,dssRsl['Vp_pct'],'--')
    
    ax1.legend(['Linear, $\hat{f}$','OpenDSS, $f$'],loc='center left', bbox_to_anchor=(1.01, 0.5),fontsize='small',title='Model')
    
    ax1.set_ylabel('Constraint violations, %');
    ax1.set_xlim((0,100))
    ax1.set_ylim((-3,103))
    ax1.grid(True)
    plt.tight_layout()
    
    if 'pltSave' in locals():
        plt.savefig(FD+'plotCns.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'plotCns.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()
        
# RESULTS 8 - OpenDSS vs Linear pltCons
if 'f_plotCns_paramUpdate' in locals():
    LD = os.path.join(WD,'hcParamSlctnCaseStudy.pkl')
    with open(LD,'rb') as file:
        rslts = pickle.load(file)
    rsltBef = rslts['rsltBef']
    rsltAft = rslts['rsltAft']
    pdf = rslts['pdf'].pdf
    
    fig, (ax0,ax1) = plt.subplots(figsize=(5.9,4.0),sharex=True,nrows=2,ncols=1)
    clrs = cm.matlab(np.arange(4)+2)
    ax1.set_prop_cycle(color=clrs)
    Cns_bef = rsltBef['Cns_pct']
    Cns_aft = rsltAft['Cns_pct']
    x_vals = 100*pdf['prms']
    y_aft = Cns_aft[:,0,:][:,[7,1,3,0]]
    y_bef = Cns_bef[:,0,:][:,[7,1,3,0]]

    ax1.plot(x_vals,y_bef,'-')
    ax1.plot(x_vals,y_aft,'-.')
    
    ax1.legend(['$V_{+}^{\mathrm{LV}}$, Hi $S_{\mathrm{Load}}$','$V_{+}^{\mathrm{MV}}$, Lo $S_{\mathrm{Load}}$','$V_{+}^{\mathrm{LV}}$, Lo $S_{\mathrm{Load}}$','$\Delta V$ (Vlt. Dev.)'],loc='center left', bbox_to_anchor=(1, 0.5),fontsize='small',title='Constraint Type')
    
    ax1.annotate('PF: 1.00',xytext=(5,80),xy=(19,54),arrowprops={'arrowstyle':'->','linewidth':1.0})
    ax1.annotate('PF: 0.95',xytext=(53,50),xy=(65,24),arrowprops={'arrowstyle':'->','linewidth':1.0})
    
    ax1.set_ylabel('Constraint violations, %');
    ax1.set_xlabel('Fraction of loads with PV, %');
    ax1.set_xlim((0,100))
    ax1.set_ylim((-3,103))
    ax1.grid(True)
    
    ax0.plot(x_vals,rsltBef['Vp_pct'],'-')
    ax0.plot(x_vals,rsltAft['Vp_pct'],'-.',color=clrA)
    
    ax0.legend(['1.00','0.95 (lag.)'],loc='center left', bbox_to_anchor=(1.04, 0.5),fontsize='small',title='Power Factor')
    
    ax0.set_ylabel('Constraint violations, %');
    # ax0.set_xlabel('Fraction of loads with PV, %');
    ax0.set_xlim((0,100))
    ax0.set_ylim((-3,103))
    ax0.grid(True)
    plt.tight_layout()
    if 'pltSave' in locals():
        plt.savefig(FD+'plotCns_paramUpdate.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'plotCns_paramUpdate.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()

if 'f_mcLinCmp' in locals():
    fig = plt.figure(figsize=figSze1)
    ax = fig.add_subplot(111)
    # Xused = np.array([9,8,7,6,5,4,3,2,1])
    # iUsed = [8,7,6,5,4,3,2,1,0]
    Xused = np.array([7,6,5,4,1])
    iUsed = [8,5,4,3,2]
    i=0
    ii=0
    for x in X:
        if x in Xused:
            ax = plotBoxWhisk(ax,len(iUsed)-1-ii+dx,ddx,kCdfNom[i][1:-1],clrH,bds=kCdfNom[i][[0,-1]],transpose=True)
            ax = plotBoxWhisk(ax,len(iUsed)-1-ii-dx,ddx,kCdfDss[i][1:-1],clrB,bds=kCdfDss[i][[0,-1]],transpose=True)
            ii += 1
        i+=1
    ax.plot(0,0,'-',color=clrH,label='Tight (HC) bandwidth')
    ax.plot(0,0,'-',color=clrB,label='Full (QSTS) bandwidth)')
    plt.legend(fontsize='small')
    
    plt.xlim((-3,103))
    plt.ylim((-0.6,len(iUsed)-1+0.6))
    
    plt.grid(True)
    feederTidyUsed = vecSlc(feederTidySet,iUsed)
    plt.yticks(range(len(iUsed)),feederTidyUsed)
    plt.xlabel('Loads with PV installed, %')
    plt.tight_layout()
    if 'pltSave' in locals():
        plt.savefig(FD+'mcLinCmp.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'mcLinCmp.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()
        
if 'f_plotLpUpg' in locals():
    # i=0
    i = 1 # Circuit J1
    # for x in Xlp:
    fig,ax = plt.subplots(figsize=figSze4)
    ax = plotBoxWhisk(ax,5,0.2,kCdfUpgNomBef[i][1:-1],clrA,bds=kCdfUpgNomBef[i][[0,-1]],transpose=True) # NB not checked!
    ax = plotBoxWhisk(ax,4,0.2,kCdfUpgNomAft[i][1:-1],clrA,bds=kCdfUpgNomAft[i][[0,-1]],transpose=True)
    ax = plotBoxWhisk(ax,3,0.2,kCdfLp00[i][1:-1],clrG,bds=kCdfLp00[i][[0,-1]],transpose=True)
    ax = plotBoxWhisk(ax,2,0.2,kCdfLpT0[i][1:-1],clrG,bds=kCdfLpT0[i][[0,-1]],transpose=True)
    ax = plotBoxWhisk(ax,1,0.2,kCdfLpTQ[i][1:-1],clrG,bds=kCdfLpTQ[i][[0,-1]],transpose=True)
    plt.yticks([5,4,3,2,1],['PF = 1.00 (RLF)','PF = 0.95 (RLF)','No T or Q','T, no Q (DERMS)','T and Q (DERMS)'])
    plt.xlim((-2,102))
    plt.grid(True)
    plt.xlabel('Fraction of Loads with PV, %')
    plt.title(feederLpSmart[i])
    plt.tight_layout()
    if 'pltSave' in locals():
        plt.savefig(FD+'mcLinCmp'+feeders_lp[i]+'.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'mcLinCmp'+feeders_lp[i]+'.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()
    # i+=1

# if 'f_plotLp' in locals(): # OLD version
    # fig = plt.figure(figsize=figSze3)
    # ax = fig.add_subplot(111)
    # i=0
    # for x in Xlp:
        # # ax = plotBoxWhisk(ax,x+1.3*dx,ddx*0.5,kCdfLpNom[i][1:-1],clrA,bds=kCdfLpNom[i][[0,-1]],transpose=True)
        # # ax = plotBoxWhisk(ax,x,ddx*0.5,kCdfLpUpg[i][1:-1],clrE,bds=kCdfLpUpg[i][[0,-1]],transpose=True)
        # ax = plotBoxWhisk(ax,x+1.3*dx,ddx*0.5,kCdfUpgNomBef[i][1:-1],clrA,bds=kCdfUpgNomBef[i][[0,-1]],transpose=True)
        # ax = plotBoxWhisk(ax,x,ddx*0.5,kCdfUpgNomAft[i][1:-1],clrE,bds=kCdfUpgNomAft[i][[0,-1]],transpose=True)
        # ax = plotBoxWhisk(ax,x-1.3*dx,ddx*0.5,kCdfLpTQ[i][1:-1],clrG,bds=kCdfLpTQ[i][[0,-1]],transpose=True)
        # i+=1
    # ax.plot(0,0,'-',color=clrA,label='Dcpld. (Nom)')
    # ax.plot(0,0,'-',color=clrE,label='Dcpld. (Upg)')
    # ax.plot(0,0,'-',color=clrG,label='DERMS')
    # plt.legend(title='Linear Model:',fontsize='small',loc='center left', bbox_to_anchor=(1.01, 0.5))
    # plt.yticks(Xlp,feederLpSmart)
    # plt.xlim((-2,102))
    # plt.ylim((0.6,5.4))
    # plt.grid(True)
    # plt.xlabel('Loads with PV installed, %')
    # plt.tight_layout()
    # if 'pltSave' in locals():
        # plt.savefig(FD+'plotLp.png',pad_inches=0.02,bbox_inches='tight')
        # plt.savefig(FD+'plotLp.pdf',pad_inches=0.02,bbox_inches='tight')
    # if 'pltShow' in locals():
        # plt.show()
        
if 'f_plotLp' in locals():
    fig,[ax0,ax1] = plt.subplots(2,sharex=True,figsize=figSze1)
    i=0
    for x in Xlp:
        ax0 = plotBoxWhisk(ax0,x+dx,ddx*1.0,kCdfUpgNomBef[i][1:-1],clrA,bds=kCdfUpgNomBef[i][[0,-1]],transpose=True)
        ax0 = plotBoxWhisk(ax0,x-dx,ddx*1.0,kCdfLpT0[i][1:-1],clrG,bds=kCdfLpT0[i][[0,-1]],transpose=True)
        ax1 = plotBoxWhisk(ax1,x+dx,ddx*1.0,kCdfUpgNomAft[i][1:-1],clrA,bds=kCdfUpgNomAft[i][[0,-1]],transpose=True)
        ax1 = plotBoxWhisk(ax1,x-dx,ddx*1.0,kCdfLpTQ[i][1:-1],clrG,bds=kCdfLpTQ[i][[0,-1]],transpose=True)
        i+=1
    ax0.plot(0,0,'-',color=clrA,label='RLF')
    ax0.plot(0,0,'-',color=clrG,label='DERMS')
    ax1.plot(0,0,'-',color=clrA,label='RLF')
    ax1.plot(0,0,'-',color=clrG,label='DERMS')
    ax0.legend(title='Unity PF',fontsize='small',loc='center left', bbox_to_anchor=(1.01, 0.5))
    ax1.legend(title='0.95 lag. PF',fontsize='small',loc='center left', bbox_to_anchor=(1.01, 0.5))
    
    ax0.set_yticks(Xlp)
    ax0.set_yticklabels(feederLpSmart)
    ax0.set_xlim((-2,102));
    ax0.set_ylim((0.6,4.4));
    ax1.set_yticks(Xlp)
    ax1.set_yticklabels(feederLpSmart)
    ax1.set_xlim((-2,102));
    ax1.set_ylim((0.6,4.4));
    
    # fig.subplots_adjust(wspace=300.0)
    # ax0.set_ymargin(4.3, 3)
    # ax1.set_ymargin(6.3, 3)
    
    plt.grid(True)
    plt.xlabel('Fraction of Loads with PV, %')
    plt.tight_layout()
    if 'pltSave' in locals():
        plt.savefig(FD+'plotLp.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'plotLp.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()




if 'f_errorCorr' in locals():
    # corrPlot = np.array(corrPlot).T
    corrPlotAve = np.zeros((len(rsltsMult),2))
    corrElps = {}
    fig,ax = plt.subplots()
    ii = 0
    for feeder,rslt in rsltsMult.items():
        ax.scatter(rslt['svtyResults'],rslt['maeSet'],color=cm.Dark2(ii),marker='.')
        corrPlotAve[ii] = [np.mean(rslt['svtyResults']),np.mean(rslt['maeSet'])]
        ax.plot(corrPlotAve[ii,0],corrPlotAve[ii,1],'X',color=cm.Dark2(ii),label=feedersTidy[feeder],markeredgecolor='k')
        corrElps[ii] = np.cov(rslt['svtyResults'],rslt['maeSet'])
        lambda_, v = np.linalg.eig(corrElps[ii])
        lambda_ = np.sqrt(lambda_)
        # for j in range(1, 4): # This makes the variance look quite big, it seems.
            # ell = Ellipse(xy=corrPlotAve[ii],
                    # width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                    # angle=np.rad2deg(np.arccos(v[0, 0])))
            # ell.set_facecolor(cm.Dark2(ii))
            # ell.set_alpha(0.15)
            # ell.set_edgecolor('k')
            # ax.add_artist(ell)
        ii+=1

    # ax.set_xlim((-5,160))
    # ax.set_ylim((-0.01,0.18))
    ax.set_ylabel('Linear MAE, %')
    ax.set_xlabel('Hosting capacity sensitivity $f_{\mathrm{S}}$, %')
    plt.legend(title='Feeder',fontsize='small',loc='center left', bbox_to_anchor=(1.01, 0.5)); 
    set_ax_size(3.0,2.2,ax)
    plt.tight_layout()
    
    print('Pearson correlation coefficient:',pearsonr(corrPlotAve.T[0],corrPlotAve.T[1]) )
    if 'pltSave' in locals():
        plt.savefig(FD+'errorCorr.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'errorCorr.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():    
        plt.show()

# # 1. seeing how the tap positions change for the feeder
# for feeder in feeders:
    # plt.scatter((rsltsTap[feeder]['linHcRslNom']['tapPosSeq'] + rsltsTap[feeder]['TC_No0']).flatten(),rsltsTap[feeder]['dssHcRslTapSet']['tapPosSet'].flatten())
    # plt.plot((-10,10),(-10,10),'k')
    # plt.show()

# # 2. Seeing how out of bandwidth the regulators are at the specified nominal tap positions
# for feeder in feeders:
    # dssHcRslTapLck = rsltsTap[feeder]['dssHcRslTapLck']
    # dssHcRslTapSet = rsltsTap[feeder]['dssHcRslTapSet']
    # plt.scatter(dssHcRslTapLck['regVI'][:,:,0].flatten(),dssHcRslTapLck['regVI'][:,:,1].flatten())
    # plt.scatter(dssHcRslTapSet['regVI'][:,:,0].flatten(),dssHcRslTapSet['regVI'][:,:,1].flatten())
    # plt.plot((-1,1,1,-1,-1),(-1,-1,1,1,-1),'k')
    # plt.gca().set_aspect('equal')
    # plt.grid(True)
    # plt.title(feeder)
    # plt.tight_layout()
    # plt.show()

# # 2 plotting the nice tables
# feeder = 'epriM1'
# rsltX = rsltsTap[feeder]
# params = rsltX['pdfData']['prms']
# nMc = rsltX['pdfData']['nMc']
# lpSns = rsltX['linHcRslNom']['Lp_pct'][:,0,:] # this is the only one which actually records the sensitivity

# # 2b plotting the stuff again
# I = [9,19,29,39,49] # 40%

# mults = np.linspace(0.7,1.3)
# for i in I:
    # frac = []
    # for mult in mults:
        # fracOut = 100*np.sum((lpSns[i]/mult)<1.0)/nMc
        # frac.append(fracOut)
    # plt.plot(100*mults,frac,label=str(100*params[i])+' %')

# plt.title(feeder)
# plt.legend()
# plt.xlabel('Scaled generation (%)')
# plt.ylabel('Constraint violations (%)')
# plt.show()

# stvy = np.sum(lpSns<0.95,axis=1)
# sntvyVals = [0.95,1.00,1.05]

# plt.plot(stvy); plt.plot(rsltX['linHcRslNom']['Vp_pct']); plt.show()

# fig,ax = plt.subplots()
# jj = 0
# for sns in lpSns[::1]:
    # pctls = np.percentile(sns,[5,25,50,75,95])
    # rngs = np.percentile(sns,[0,100])
    # plotBoxWhisk(ax,params[jj],0.01,pctls,bds=rngs)
    # jj+=1

# xlm = ax.get_xlim()
# ax.plot(xlm,(1.0,1.0),'k--')
# ax.set_xlim(xlm)
# ax.set_ylim((0,4.5))
# ax.set_xlabel('Fraction of Loads with PV')
# ax.set_ylabel('Gen scale factor to violation')
# plt.show()

# # 3. Plotting the sensitivity to tap position
# tapPosSns = rsltX['linHcRslNom']['tapPosSns'] # this is the only one which actually records the sensitivity
# Bwdth = 0.666 #taps
# linM1 = rsltX['linHcRslNom']

# tapPsnMin = tapPosSns[:,:,1,:]
# tapPsnOutBnds = (tapPsnMin<Bwdth)
# # tapPsnOut = 100*np.sum(tapPsnOutBnds,axis=1)/nMc
# tapPsnOut = 100*np.sum(tapPsnOutBnds,axis=1)/nMc

# kCdf = 100*getKcdf(params,tapPsnOut)[0][idxChosen]
# # kCdf2 = 100*getKcdf(params,linM1['Vp_pct'])[0][idxChosen]

# i=5
# x = X[i]
# fig,ax = plt.subplots(figsize=figSze0)
# plt.plot(100*params,tapPsnOut,label='Lin \'conservative\'')
# plt.plot(100*params,linM1['Vp_pct'],label='Lin \'nominal\'')
# plt.plot(100*params,rsltX['dssHcRslTapTgt']['Vp_pct'],label='Dss \'Unlocked\'')
# plt.plot(100*params,rsltX['dssHcRslTapLck']['Vp_pct'],label='Dss \'Locked tap\'')
# plt.legend()
# plt.xlabel('Fraction of loads with PV')
# plt.ylabel('% constraint violations')
# plt.show()

# # # plt.plot(params,rsltX['dssHcRslTapSet']['Vp_pct'],label='Dss Set')


# fig,ax = plt.subplots(figsize=figSze0)
# ax = plotBoxWhisk(ax,x,ddx,kCdf[1:-1],clrC,bds=kCdf[[0,-1]],transpose=True)
# ax = plotBoxWhisk(ax,x+2*dx,ddx,kCdfLin[i][1:-1],clrA,bds=kCdfLin[i][[0,-1]],transpose=True)
# ax = plotBoxWhisk(ax,x-2*dx,ddx,kCdfDss[i][1:-1],clrB,bds=kCdfDss[i][[0,-1]],transpose=True)
# ax.plot(0,0,'-',color=clrA,label='Linear, $\hat{f}$')
# ax.plot(0,0,'-',color=clrB,label='OpenDSS, $f$')
# ax.plot(0,0,'-',color=clrC,label='Linear conservative')
# plt.legend(fontsize='small')
# plt.xlim((-3,103))
# plt.ylim((0.4,9.6))
# plt.grid(True)
# # plt.yticks(X,feederTidySet)
# plt.xlabel('Loads with PV installed, %')
# plt.tight_layout()
# plt.show()

# # # tapMinLo = np.min(tapPosSns[:,:,1,:],axis=2)
# # # prms = np.linspace(100/tapMinLo.shape[0],100,tapMinLo.shape[0])
# # # fig,ax = plt.subplots()
# # # jj = 0
# # # for tapMin in tapMinLo[::1]:
    # # # pctls = np.percentile(tapMin,[5,25,50,75,95])
    # # # rngs = np.percentile(tapMin,[0,100])
    # # # plotBoxWhisk(ax,prms[jj],1,pctls,bds=rngs)
    # # # jj+=1

# # # xlm = ax.get_xlim()
# # # ax.plot(xlm,(2,2),'k:')
# # # ax.plot(xlm,(-2,-2),'k:')
# # # ax.plot(xlm,(0.416,0.416),'k--')
# # # ax.plot(xlm,(-0.416,-0.416),'k--')
# # # ax.set_xlim(xlm)
# # # ax.set_xlabel('Fraction of Loads with PV')
# # # ax.set_ylabel('No. taps to upper voltage constraint')
# # # ax.set_ylim((-3.5,4.5))
# # # plt.show()

# for rslt in rsltsTap.values():
    # linHc = rslt['linHcRslNom']
    # Lp_pct = (linHc['Lp_pct'][:,0,:]<1)
    # Lp50 = np.sum(Lp_pct[:,:50],axis=1)/50
    # mae = []
    # for i in range(1,Lp_pct.shape[1]):
        # mae.append( (1/50)*np.sum(np.abs( np.sum(Lp_pct[:,:i],axis=1)/i - Lp50)) )
    
    # plt.plot(mae); plt.title(rslt['feeder']); plt.show()

# dssHc = rslt34['dssHcRslTapTgt']
import pickle, sys, os, getpass
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
plt.style.use('tidySettings')

WD = os.path.dirname(sys.argv[0])
sys.path.append(os.path.dirname(WD))

from dss_python_funcs import basicTable
from linSvdCalcs import plotBoxWhisk, getKcdf, plotCns

# feeders = ['13bus','34bus','123bus','8500node','eulv','usLv','epriJ1','epriK1','epriM1','epri5','epri7','epri24']
feeders = ['34bus','123bus','8500node','epriJ1','epriK1','epriM1','epri5','epri7','epri24']
# feeders = ['34bus','123bus','epriJ1','epriK1','epriM1','epri5','epri7']

feedersTidy = {'34bus':'34 Bus','123bus':'123 Bus','8500node':'8500 Node','epriJ1':'Ckt. J1','epriK1':'Ckt. K1','epriM1':'Ckt. M1','epri5':'Ckt. 5','epri7':'Ckt. 7','epri24':'Ckt. 24'}

feeders_dcp = ['8500node','epriJ1','epriK1','epriM1','epri24']

# t_timeTable = 1 # timeTable
# t_rsltSvty = 1 # sensitivity table
# f_dssVlinWght = 1 # gammaFrac boxplot results
f_mcLinUpg = 1
# f_linMcSns = 1
# f_plotCns = 1 # <--- also useful for debugging.
# f_plotCns_paramUpdate = 1

# pltSave=True
pltShow=True

figSze0 = (5.2,3.4)
figSze1 = (5.2,2.5)
figSze2 = (5.2,3.0)
figSze3 = (5.2,2.2)
TD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\tables\\"
FD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\figures\\"

rsltsFrac = {}; rsltsSns = {}; rsltsUpg = {}
for feeder in feeders:
    RD = os.path.join(WD,feeder,'linHcCalcsRslt_gammaFrac_finale.pkl')
    with open(RD,'rb') as handle:
        rsltsFrac[feeder] = pickle.load(handle)
for feeder in feeders_dcp:
    RDsns = os.path.join(WD,feeder,'linHcCalcsSns_gammaFrac_new.pkl')
    RDupg = os.path.join(WD,feeder,'linHcCalcsUpg.pkl')
    with open(RDsns,'rb') as handle:
        rsltsSns[feeder] = pickle.load(handle)
    with open(RDupg,'rb') as handle:
        rsltsUpg[feeder] = pickle.load(handle)
    

kCdfLin = [];    kCdfDss = []; 
LmeanNorm = []; feederTidySet = []
timeTableData = []
rsltSvtyData = []
rslt34 = rsltsFrac['34bus'] # useful for debugging

for rslt in rsltsFrac.values():
    dataSet = []
    dataSet.append(feedersTidy[rslt['feeder']])
    dataSet.append('%.2f' % rslt['dssHcRslNom']['runTime'])
    dataSet.append('%.2f' % rslt['linHcRsl']['runTime'])
    dataSet.append('%.2f' %  rslt['maeVals']['dssTgtMae'])
    dataSet.append('%.2f' %  rslt['maeVals']['dssNomMae'])
    timeTableData.append(dataSet)
    
    dataSet = []
    dataSet.append(feedersTidy[rslt['feeder']])
    dataSet.append('%.1f' % (100-rslt['preCndLeft']))
    dataSet.append('%.2f' % rslt['maeVals']['nomMae'])
    dataSet.append('%.2f' % (rslt['maeVals']['nmcMae'])) # <--- to do!
    rsltSvtyData.append(dataSet)
    
    kCdfLin.append(rslt['linHcRsl']['kCdf'][[0,1,5,10,15,19,20]])
    kCdfDss.append(rslt['dssHcRslTgt']['kCdf'][[0,1,5,10,15,19,20]])
    # kCdfDss.append(rslt['dssHcRslNom']['kCdf'][[0,1,5,10,15,19,20]])
    LmeanNorm.append( np.mean(np.abs(rslt['dssHcRslTgt']['Vp_pct']-rslt['linHcRsl']['Vp_pct']))*0.01 )
    feederTidySet.append(feedersTidy[rslt['feeder']])
        
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

print('Sensitivity errors:')
print(*LsnsNorm,sep='\n')


# TABLE 1 - timings + MAE ======================= 
if 't_timeTable' in locals():
    caption='OpenDSS and Linear models result comparison'
    label='timeTable'
    heading = ['Model','OpenDSS time','Linear time','MAE (tight), \%','MAE (nominal), \%']
    data = timeTableData
    basicTable(caption,label,heading,data,TD)
    print('\n',heading)
    print(*data,sep='\n')
# ===============================================

# TABLE 2 - sensitivity MAE ===================== 
if 't_rsltSvty' in locals():
    caption='Sensitivity to simulation parameters'
    label='rsltSvty'
    heading = ['Model','Precond. calc effort, \%','Precond. MAE','Nmc MAE']
    data = rsltSvtyData
    basicTable(caption,label,heading,data,TD)
    print('\n',heading)
    print(*data,sep='\n')
# ===============================================


dx = 0.175; ddx=dx/1.5
X = np.arange(len(kCdfLin),0,-1)
i=0
# clrA,clrB,clrC,clrD,clrE = cm.tab10(np.arange(5))
clrA,clrB,clrC,clrD,clrE = cm.matlab(np.arange(5))

# # RESULTS 1 - opendss vs linear model, k =====================
if 'f_dssVlinWght' in locals():
    fig = plt.figure(figsize=figSze0)
    ax = fig.add_subplot(111)
    i=0
    for x in X:
        ax = plotBoxWhisk(ax,x+dx,ddx,kCdfLin[i][1:-1],clrA,bds=kCdfLin[i][[0,-1]],transpose=True)
        ax = plotBoxWhisk(ax,x-dx,ddx,kCdfDss[i][1:-1],clrB,bds=kCdfDss[i][[0,-1]],transpose=True)
        i+=1
    ax.plot(0,0,'-',color=clrA,label='Linear')
    ax.plot(0,0,'-',color=clrB,label='OpenDSS')
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
    feederPlot='8500node'
    rsltM1 = rsltsFrac[feederPlot]
    pdf = rsltM1['pdfData']
    linRsl = rsltM1['linHcRslNom']
    dssRsl = rsltM1['dssHcRslTgt']
    
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

    ax0.annotate('OpenDSS',xytext=(53,62),xy=(69,38),arrowprops={'arrowstyle':'->','linewidth':1.0})
    ax0.annotate('Linear',xytext=(75,12),xy=(75,40),arrowprops={'arrowstyle':'->','linewidth':1.0})
    
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
    
    
    ax1.annotate('Before',xytext=(8,80),xy=(19,54),arrowprops={'arrowstyle':'->','linewidth':1.0})
    ax1.annotate('After',xytext=(53,50),xy=(62,24),arrowprops={'arrowstyle':'->','linewidth':1.0})
    
    ax1.set_ylabel('Constraint violations, %');
    ax1.set_xlabel('Fraction of loads with PV, %');
    ax1.set_xlim((0,100))
    ax1.set_ylim((-3,103))
    ax1.grid(True)
    
    ax0.plot(x_vals,rsltBef['Vp_pct'],'-')
    ax0.plot(x_vals,rsltAft['Vp_pct'],'-.')
    
    ax0.legend(['Before','After'],loc='center left', bbox_to_anchor=(1.04, 0.5),fontsize='small',title='Model')
    
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
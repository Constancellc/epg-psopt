import pickle, sys, os, getpass
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

WD = os.path.dirname(sys.argv[0])
sys.path.append(os.path.dirname(WD))

from dss_python_funcs import basicTable
from linSvdCalcs import plotBoxWhisk, getKcdf, plotCns

# feeders = ['13bus','34bus','123bus','8500node','eulv','usLv','epriJ1','epriK1','epriM1','epri5','epri7','epri24']
# feeders = ['34bus','123bus','8500node','epriJ1','epriK1','epriM1','epri5','epri7','epri24']
feeders = ['34bus','123bus','epri5','epri7','epriK1','epriM1']

feedersTidy = {'34bus':'34 Bus','123bus':'123 Bus','8500node':'8500 Node','epriJ1':'Ckt. J1','epriK1':'Ckt. K1','epriM1':'Ckt. M1','epri5':'Ckt. 5','epri7':'Ckt. 7','epri24':'Ckt. 24'}

feeders_dcp = ['8500node','epriJ1','epriK1','epriM1','epri24']

# t_timeTable = 1 # timeTable
# t_rsltSvty = 1 # sensitivity table
# f_dssVlinWght = 1 # gammaFrac boxplot results
# f_linMcSns = 1
f_plotCns = 1 # <--- also useful for debugging.

# pltSave=True
pltShow=True

figSze0 = (5.2,3.4)
figSze1 = (5.2,2.5)
figSze2 = (5.2,3.0)
TD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\tables\\"
FD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\figures\\"

rsltsFrac = {}; rsltsSns = {}
for feeder in feeders:
    RD = os.path.join(WD,feeder,'linHcCalcsRslt_gammaFrac_final.pkl')
    with open(RD,'rb') as handle:
        rsltsFrac[feeder] = pickle.load(handle)

for feeder in feeders_dcp:
    RDsns = os.path.join(WD,feeder,'linHcCalcsSns_gammaFrac_new.pkl')
    with open(RDsns,'rb') as handle:
        rsltsSns[feeder] = pickle.load(handle)

kCdfLin = [];    kCdfDss = []; 
LmeanNorm = []; feederTidySet = []
timeTableData = []
rsltSvtyData = []
rslt34 = rsltsFrac['34bus'] # useful for debugging

for rslt in rsltsFrac.values():
    dataSet = []
    dataSet.append(feedersTidy[rslt['feeder']])
    dataSet.append('%.2f' % (rslt['dssHcRslNom']['runTime']))
    dataSet.append('%.2f' % (rslt['linHcRsl']['runTime']))
    dataSet.append('%.2f' %  (rslt['dssTgtMae']))
    dataSet.append('%.2f' %  (rslt['dssNomMae']))
    timeTableData.append(dataSet)
    
    dataSet = []
    dataSet.append(feedersTidy[rslt['feeder']])
    dataSet.append('%.1f' % rslt['preCndLeft'])
    dataSet.append('%.2f' % rslt['nomMae'])
    dataSet.append('%.2f' % (rslt['nmcMae'])) # <--- to do!
    rsltSvtyData.append(dataSet)
    
    kCdfLin.append(rslt['linHcRsl']['kCdf'][[0,1,5,10,15,19,20]])
    kCdfDss.append(rslt['dssHcRslTgt']['kCdf'][[0,1,5,10,15,19,20]])
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
KcdkLin = []; KcdkSeq = []; KcdkPar = []; # kCdfLin = []
timeSeq = [];timePar = [];

print('Sensitivity errors:')
print(*LsnsNorm,sep='\n')

i=0
linHcRsl = rslt['linHcRsl']

# TABLE 1 - timings + MAE ======================= 
if 't_timeTable' in locals():
    caption='OpenDSS and Linear models result comparison'
    label='timeTable'
    heading = ['Model','OpenDSS time','Linear time','MAE (tight), \%','MAE (nominal), \%']
    data = timeTableData
    basicTable(caption,label,heading,data,TD)
    print(heading)
    print(*data,sep='\n')
# ===============================================

# TABLE 2 - sensitivity MAE ===================== 
if 't_rsltSvty' in locals():
    caption='Sensitivity to simulation parameters'
    label='rsltSvty'
    heading = ['Model','Precond. calc effort, \%','Precond. MAE','Nmc MAE']
    data = rsltSvtyData
    basicTable(caption,label,heading,data,TD)
    print(heading)
    print(*data,sep='\n')
# ===============================================


dx = 0.175; ddx=dx/1.5
X = np.arange(len(kCdfLin),0,-1)
i=0
clrA,clrB,clrC,clrD,clrE = cm.tab10(np.arange(5))

# # RESULTS 1 - opendss vs linear model, k =====================
if 'f_dssVlinWght' in locals():
    fig = plt.figure(figsize=figSze0)
    ax = fig.add_subplot(111)
    i=0
    for x in X:
        ax = plotBoxWhisk(ax,x+dx,ddx,kCdfLin[i][1:-1],clrA,bds=kCdfLin[i][[0,-1]],transpose=True)
        ax = plotBoxWhisk(ax,x-dx,ddx,kCdfDss[i][1:-1],clrB,bds=kCdfDss[i][[0,-1]],transpose=True)
        i+=1
    ax.plot(0,0,'-',color=clrA,label='Lin.')
    ax.plot(0,0,'-',color=clrB,label='O\'DSS.')
    ax.tick_params(direction="in",bottom=1,top=1,left=1,right=1,grid_linewidth=0.4,width=0.4,length=2.5)
    legend = plt.legend()
    # legend = plt.legend(framealpha=1.0,fancybox=0,edgecolor='k',loc='lower right')
    legend = plt.legend(framealpha=1.0,fancybox=0,edgecolor='k')
    legend.get_frame().set_linewidth(0.4)
    [i.set_linewidth(0.4) for i in ax.spines.values()]
    
    plt.xlim((-3,103))
    plt.ylim((0.4,9.6))
    # plt.grid(True,axis='y')
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
    ax.tick_params(direction="in",bottom=1,top=1,left=1,right=1,grid_linewidth=0.4,width=0.4,length=2.5)
    legend = plt.legend()
    legend = plt.legend(framealpha=1.0,fancybox=0,edgecolor='k')
    legend.get_frame().set_linewidth(0.4)
    [i.set_linewidth(0.4) for i in ax.spines.values()]

    plt.grid(True)
    plt.yticks(Xsns,feederSnsSmart)
    plt.xlabel('Loads with PV installed, %')
    plt.tight_layout()
    if 'pltSave' in locals():
        plt.savefig(FD+'linMcSns.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'linMcSns.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()

        
# RESULTS 7 - OpenDSS vs Linear pltCons
if 'f_plotCns' in locals():
    # feeders = ['34bus','123bus','8500node','epriJ1','epriK1','epriM1','epri5','epri7','epri24']
    feederPlot='8500node'
    feederPlot='34bus'
    feederPlot='epriM1'
    rsltM1 = rsltsFrac[feederPlot]
    pdf = rsltM1['pdfData']
    linRsl = rsltM1['linHcRsl']
    dssRsl = rsltM1['dssHcRslTgt']
    # dssRsl = rsltM1['dssHcRslNom']
    fig, ax = plt.subplots(figsize=figSze2)
    
    # ax = plotCns(pdf['mu_k'],pdf['prms'],dssRsl['Cns_pct'],feeder=rsltM1['feeder'],lineStyle='-',ax=ax,pltShow=False)
    # ax = plotCns(pdf['mu_k'],pdf['prms'],linRsl['Cns_pct'],feeder=rsltM1['feeder'],lineStyle='--',ax=ax,pltShow=False)
    # plt.legend(('$\Delta V$','$V^{+}_{\mathrm{MV,LS}}$','$V^{-}_{\mathrm{MV,LS}}$','$V^{+}_{\mathrm{LV,LS}}$','$V^{-}_{\mathrm{LV,LS}}$','$V^{+}_{\mathrm{MV,HS}}$','$V^{-}_{\mathrm{MV,HS}}$','$V^{+}_{\mathrm{LV,HS}}$','$V^{-}_{\mathrm{LV,HS}}$'))
    
    clrs = cm.nipy_spectral(np.linspace(0,1,9))
    clrs = cm.viridis(np.linspace(0,1,4))
    ax.set_prop_cycle(color=clrs)
    Cns_dss = dssRsl['Cns_pct']
    Cns_lin = linRsl['Cns_pct']
    x_vals = 100*pdf['prms']
    y_dss = Cns_dss[:,0,:][:,[7,1,3,0]]
    y_lin = Cns_lin[:,0,:][:,[7,1,3,0]]
    
    # plt.legend(('$\Delta V$','$V^{+}_{\mathrm{MV,LS}}$','$V^{-}_{\mathrm{MV,LS}}$','$V^{+}_{\mathrm{LV,LS}}$','$V^{-}_{\mathrm{LV,LS}}$','$V^{+}_{\mathrm{MV,HS}}$','$V^{+}_{\mathrm{LV,HS}}$','$V^{-}_{\mathrm{LV,HS}}$'))
    
    ax.plot(x_vals,y_dss,'-',linewidth=1)
    ax.plot(x_vals,y_lin,'--',linewidth=1)
    
    # legend = ax.legend(['$V^{+}_{\mathrm{LV,Hi\,P}}$','$V^{+}_{\mathrm{MV,Lo\,P}}$','$V^{+}_{\mathrm{LV,Lo\,P}}$','$\Delta V$'],loc='lower right',framealpha=1.0,fancybox=0,edgecolor='k')
    legend = ax.legend(['$V^{+}_{\mathrm{LV,Hi\,P}}$','$V^{+}_{\mathrm{MV,Lo\,P}}$','$V^{+}_{\mathrm{LV,Lo\,P}}$','$\Delta V$'],framealpha=1.0,fancybox=0,edgecolor='k')
    legend.get_frame().set_linewidth(0.4)
    
    
    [i.set_linewidth(0.4) for i in ax.spines.values()]
    ax.tick_params(direction="in",bottom=1,top=1,left=1,right=1,grid_linewidth=0.4,width=0.4,length=2.5)
    # ax.plot(####,linewidth=1,markersize=4)

    
    ax.annotate('OpenDSS',xytext=(60,80),xy=(90,72),arrowprops={'arrowstyle':'->'})
    ax.annotate('Linear',xytext=(65,55),xy=(89,52),arrowprops={'arrowstyle':'->'})
    
    plt.ylabel('Fraction of runs w/ violations, %');
    plt.xlabel('Fraction of loads with PV, %');
    plt.xlim((0,100))
    plt.ylim((-3,103))
    plt.grid(True)
    plt.tight_layout()
    if 'pltSave' in locals():
        plt.savefig(FD+'plotCns.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'plotCns.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()
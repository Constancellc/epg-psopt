import pickle, sys, os, getpass
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

WD = os.path.dirname(sys.argv[0])
sys.path.append(os.path.dirname(WD))

from dss_python_funcs import basicTable
from linSvdCalcs import plotBoxWhisk, getKcdf, plotCns

# feeders = ['13bus','34bus','123bus','8500node','eulv','usLv','epriJ1','epriK1','epriM1','epri5','epri7','epri24']
feeders = ['34bus','123bus','8500node','epriJ1','epriK1','epriM1','epri5','epri7','epri24']
# feeders = ['13bus','34bus','123bus','8500node','eulv','usLv','epriK1','epriM1','epri5','epri7','epri24']
pLoad = {'epriJ1':11.6,'epriK1':12.74,'epriM1':15.67,'epri5':16.3,'epri7':19.3,'epri24':28.8,'8500node':12.05,'eulv':0.055,'usLv':42.8,'13bus':3.6,'34bus':2.0,'123bus':3.6}

feedersTidy = {'34bus':'34 Bus','123bus':'123 Bus','8500node':'8500 Node','epriJ1':'Ckt. J1','epriK1':'Ckt. K1','epriM1':'Ckt. M1','epri5':'Ckt. 5','epri7':'Ckt. 7','epri24':'Ckt. 24'}

feeders_dcp = ['8500node','epriJ1','epriK1','epriM1','epri24']

# t_timeTable = 1 # timeTable
# f_dssVlinWght = 1 # gammaFrac boxplot results
# f_linMcVal = 1 # monte carlo no. validation
# f_logTimes = 1 # 
# f_linMcSns = 1
# f_dssVlinWghtErr = 1
# f_dssSeqPar = 1
# f_plotCns = 1

pltSave=True
pltShow=True

figSze0 = (5.2,3.4)
figSze1 = (5.2,2.5)
TD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\tables\\"
FD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\figures\\"

rsltsFrac = {}; rsltsVal = {}; rsltsPar = {}; rsltsSns = {}
for feeder in feeders:
    # RD = os.path.join(WD,feeder,'linHcCalcs'+rsltType+pdfName+num+regID+'.pkl')
    RDval = os.path.join(WD,feeder,'linHcCalcsVal_gammaFrac50_new.pkl')
    # RDfrac = os.path.join(WD,feeder,'linHcCalcsRslt_gammaFrac_reg0_new.pkl')
    RDfrac = os.path.join(WD,feeder,'linHcCalcsRslt_gammaFrac_reg0_bw.pkl') # bw as in reduced bw
    RDpar = os.path.join(WD,feeder,'linHcCalcsRslt_gammaFrac_reg0_par.pkl') # par as in parallel (faster for J1)
    with open(RDfrac,'rb') as handle:
        rsltsFrac[feeder] = pickle.load(handle)
    with open(RDval,'rb') as handle:
        rsltsVal[feeder] = pickle.load(handle)
    with open(RDpar,'rb') as handle:
        rsltsPar[feeder] = pickle.load(handle)
for feeder in feeders_dcp:
    # RDsns = os.path.join(WD,feeder,'linHcCalcsSns_gammaFrac.pkl')
    RDsns = os.path.join(WD,feeder,'linHcCalcsSns_gammaFrac_new.pkl')
    with open(RDsns,'rb') as handle:
        rsltsSns[feeder] = pickle.load(handle)
timeStrLin = [];    timeStrDss = []; timeLin = []; timeDss = []
kCdfLin = [];    kCdfDss = []; 
LrelNorm = []; LmeanNorm = []; feederTidySet = []
feederData = []
for rslt in rsltsFrac.values():
    dataSet = []
    dataSet.append(feedersTidy[rslt['feeder']])
    dataSet.append('%.2f' % (rsltsPar[rslt['feeder']]['linHcRsl']['runTime']))
    dataSet.append('%.2f' % (rsltsPar[rslt['feeder']]['dssHcRslPar']['runTime']))
    # dataSet.append('%.2f' % (rslt['dssHcRsl']['runTime']))
    # dataSet.append('%.2f' % (rslt['linHcRsl']['runTime']))
    # dataSet.append('%.2f' % (rslt['dssHcRsl']['runTime']))
    dataSet.append('%.2f' %  (np.mean(np.abs(rslt['dssHcRsl']['Vp_pct']-rslt['linHcRsl']['Vp_pct']))) )
    timeStrLin.append('%.3f' % (rslt['linHcRsl']['runTime']/60.))
    timeStrDss.append('%.3f' % (rslt['dssHcRsl']['runTime']/60.))
    timeLin.append(rslt['linHcRsl']['runTime']/60.)
    timeDss.append(rslt['dssHcRsl']['runTime']/60.)
    kCdfLin.append(rslt['linHcRsl']['kCdf'][[0,1,5,10,15,19,20]])
    kCdfDss.append(rslt['dssHcRsl']['kCdf'][[0,1,5,10,15,19,20]])
    LmeanNorm.append( np.mean(np.abs(rslt['dssHcRsl']['Vp_pct']-rslt['linHcRsl']['Vp_pct']))*0.01 )
    LrelNorm.append(rslt['regError'])
    feederData.append(dataSet)
    feederTidySet.append(feedersTidy[rslt['feeder']])
        
kCdfVal = []; LvalNorm = []; # kCdfLin = []
for rslt in rsltsVal.values():
    kCdfVal.append(rslt['linHcVal']['kCdf'][[0,1,5,10,15,19,20]])
    LvalNorm.append(rslt['regError'])
kCdfSns0 = [];    kCdfSns1 = []; kCdfLinSns = []; LsnsNorm = [] # new lin needed coz this is only some models
feederSnsSmart = []
for rslt in rsltsSns.values():
    kCdfLinSns.append(rslt['linHcRsl']['kCdf'][[0,1,5,10,15,19,20]])
    kCdfSns0.append(rslt['linHcSns0']['kCdf'][[0,1,5,10,15,19,20]])
    kCdfSns1.append(rslt['linHcSns1']['kCdf'][[0,1,5,10,15,19,20]])
    LsnsNorm.append(rslt['regErrors'])
    feederSnsSmart.append(feedersTidy[rslt['feeder']])
KcdkLin = []; KcdkSeq = []; KcdkPar = []; # kCdfLin = []
timeSeq = [];timePar = [];
i=0
for rslt in rsltsPar.values():
    KcdkLin.append(rslt['linHcRsl']['kCdf'][0::5])
    KcdkSeq.append(rslt['dssHcRslSeq']['kCdf'][0::5])
    KcdkPar.append(rslt['dssHcRslPar']['kCdf'][0::5])
    timeSeq.append(rslt['dssHcRslSeq']['runTime'])
    timePar.append(rslt['dssHcRslPar']['runTime'])
    feederData[i].append('%.2f' %  (np.mean(np.abs(rslt['dssHcRslPar']['Vp_pct']-rslt['linHcRsl']['Vp_pct']))) )
    i+=1

linHcRsl = rslt['linHcRsl']

# TABLE 1 ======================= 
if 't_timeTable' in locals():
    caption='OpenDSS and Linear models result comparison'
    label='timeTable'
    # heading = ['']+feeders
    # data = [['Full Model']+timeStrDss,['Linear Model']+timeStrLin]
    heading = ['Model','OpenDSS time','Linear time','MAE (tight), \%','MAE (nominal), \%']
    # data = feederT + timeStrLin + timeStrDss + timeStrLin + timeStrDss
    data = feederData
    basicTable(caption,label,heading,data,TD)
# ===============================

dx = 0.175; ddx=dx/1.5
X = np.arange(len(kCdfLin),0,-1)
i=0
clrA,clrB,clrC,clrD,clrE = cm.tab10(np.arange(5))

# # RESULTS 1 - opendss vs linear model, k =====================
# if 'f_dssVlinWght' in locals():
    # fig = plt.figure(figsize=figSze0)
    # ax = fig.add_subplot(111)
    # i=0
    # for x in X:
        # ax = plotBoxWhisk(ax,x+dx,ddx,kCdfDss[i][1:-1],clrB,bds=kCdfDss[i][[0,-1]])
        # ax = plotBoxWhisk(ax,x-dx,ddx,kCdfLin[i][1:-1],clrA,bds=kCdfLin[i][[0,-1]])
        # i+=1
    # ax.plot(0,0,'-',color=clrA,label='Linear')
    # ax.plot(0,0,'-',color=clrB,label='OpenDSS')
    # plt.legend()
    # plt.ylim((-2,102))
    # plt.grid(True)
    # plt.xticks(X,feeders,rotation=90)
    # plt.ylabel('Loads with PV installed, \%')
    # plt.tight_layout()
    # if 'pltSave' in locals():
        # plt.savefig(FD+'dssVlinWght.png',pad_inches=0.02,bbox_inches='tight')
        # plt.savefig(FD+'dssVlinWght.pdf',pad_inches=0.02,bbox_inches='tight')
    # if 'pltShow' in locals():
        # plt.show()

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
    legend = plt.legend(framealpha=1.0,fancybox=0,edgecolor='k',loc='lower right')
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

# RESULTS 3 - linear model vs linear rerun, k =====================
if 'f_linMcVal' in locals():
    fig = plt.figure(figsize=figSze0)
    ax = fig.add_subplot(111)
    i=0
    for x in X:
        ax = plotBoxWhisk(ax,x+dx,ddx,kCdfVal[i][1:-1],clrB,bds=kCdfVal[i][[0,-1]])
        ax = plotBoxWhisk(ax,x-dx,ddx,kCdfLin[i][1:-1],clrA,bds=kCdfLin[i][[0,-1]])
        i+=1
    ax.plot(0,0,'-',color=clrA,label='Run A')
    ax.plot(0,0,'-',color=clrB,label='Run B')
    plt.legend()
    plt.xticks(X,feeders,rotation=90)
    plt.ylim((-2,102))
    plt.grid(True)
    plt.ylabel('Loads with PV installed, %')
    plt.tight_layout()
    if 'pltSave' in locals():
        plt.savefig(FD+'linMcVal.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'linMcVal.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()

        
# RESULTS 4 - linear model vs linear rerun, seconds =====================
if 'f_logTimes' in locals():
    fig = plt.figure(figsize=figSze0)
    ax = fig.add_subplot(111)
    ax.bar(X+dx,timeLin,width=ddx*2,color=clrA,zorder=10)
    ax.bar(X-dx,timeDss,width=ddx*2,color=clrB,zorder=10)
    ax.plot(0,0,'-',color=clrA,label='Linear')
    ax.plot(0,0,'-',color=clrB,label='OpenDSS')
    ax.set_yscale('log')
    ax.legend()
    ax.set_xticks(X)
    ax.set_xticklabels(feeders,rotation=90)
    ax.grid(True)
    ax.set_ylabel('Run time, min')
    ax.set_ylim((10**-3,10**3))
    plt.tight_layout()
    if 'pltSave' in locals():
        plt.savefig(FD+'logTimes.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'logTimes.pdf',pad_inches=0.02,bbox_inches='tight')
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

if 'f_dssVlinWghtErr' in locals():
    # plt.bar(X,LrelNorm)
    plt.bar(X-0.2,LrelNorm,width=0.3)
    plt.bar(X+0.2,LmeanNorm,width=0.3)
    plt.xticks(X,feeders,rotation=90)
    plt.title('Error')
    plt.ylabel('Relative error, $\epsilon = \dfrac{||f_{\mathrm{lin}}(x) - f_{\mathrm{dss}}(x)||_{1}}{1 + ||f_{\mathrm{dss}}||_{1}}$')
    plt.tight_layout()
    if 'pltSave' in locals():
        plt.savefig(FD+'dssVlinWghtErr.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'dssVlinWghtErr.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()

# RESULTS 6 - opendss vs linear model, k =====================
if 'f_dssSeqPar' in locals():
    fig = plt.figure(figsize=figSze0)
    ax = fig.add_subplot(111)
    i=0
    for x in X:
        ax = plotBoxWhisk(ax,x-dx,0.66*ddx,KcdkLin[i],clrC)
        ax = plotBoxWhisk(ax,x   ,0.66*ddx,KcdkSeq[i],clrD)
        ax = plotBoxWhisk(ax,x+dx,0.66*ddx,KcdkPar[i],clrE)
        i+=1
    ax.plot(0,0,'-',color=clrC,label='Lin')
    ax.plot(0,0,'-',color=clrD,label='Seq')
    ax.plot(0,0,'-',color=clrE,label='Par')
    plt.legend()
    plt.ylim((-2,102))
    plt.grid(True)
    plt.xticks(X,feeders,rotation=90)
    plt.ylabel('Loads with PV installed, %')
    plt.tight_layout()
    # if 'pltSave' in locals():
        # plt.savefig(FD+'dssVlinWght.png',pad_inches=0.02,bbox_inches='tight')
        # plt.savefig(FD+'dssVlinWght.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()

# RESULTS 7 - OpenDSS vs Linear pltCons
if 'f_plotCns' in locals():
    # feeders = ['34bus','123bus','8500node','epriJ1','epriK1','epriM1','epri5','epri7','epri24']
    feederPlot='8500node'
    rsltM1 = rsltsFrac[feederPlot]
    pdf = rsltM1['pdfData']
    linRsl = rsltM1['linHcRsl']
    dssRsl = rsltM1['dssHcRsl']
    fig, ax = plt.subplots(figsize=(5.2,3.4))
    
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
    
    ax.plot(x_vals,y_dss,'-')
    ax.plot(x_vals,y_lin,'--')
    
    ax.legend(['$V^{+}_{\mathrm{LV,Hi\,P}}$','$V^{+}_{\mathrm{MV,Lo\,P}}$','$V^{+}_{\mathrm{LV,Lo\,P}}$','$\Delta V$'],loc='lower right')
    
    ax.annotate('OpenDSS',xytext=(60,80),xy=(90,72),arrowprops={'arrowstyle':'->'})
    ax.annotate('Linear',xytext=(65,55),xy=(89,52),arrowprops={'arrowstyle':'->'})
    
    plt.ylabel('Fraction of runs w/ violations, %');
    plt.xlabel('Fraction of loads with PV, %');
    plt.xlim((0,100))
    plt.grid(True)
    plt.tight_layout()
    if 'pltShow' in locals():
        plt.show()
    if 'pltSave' in locals():
        plt.savefig(FD+'plotCns.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'plotCns.pdf',pad_inches=0.02,bbox_inches='tight')
import pickle, sys, os, getpass
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

WD = os.path.dirname(sys.argv[0])
sys.path.append(os.path.dirname(WD))

from dss_python_funcs import basicTable
from linSvdCalcs import plotBoxWhisk, getKcdf, plotCns

feeders = ['13bus','34bus','123bus','8500node','eulv','usLv','epriJ1','epriK1','epriM1','epri5','epri7','epri24']
# feeders = ['13bus','34bus','123bus','8500node','epriJ1','epriK1','epriM1','epri24']
pLoad = {'epriJ1':11.6,'epriK1':12.74,'epriM1':15.67,'epri5':16.3,'epri7':19.3,'epri24':28.8,'8500node':12.05,'eulv':0.055,'usLv':42.8,'13bus':3.6,'34bus':2.0,'123bus':3.6}

feeders_dcp = ['8500node','epriJ1','epriK1','epriM1','epri24']

# t_timeTable = True # timeTable
# f_dssVlinWght = True # gammaFrac boxplot results
# f_linMcVal = True # monte carlo no. validation
# f_logTimes = True # 
# f_linMcSns = True
f_dssVlinWghtErr = True

# pltSave=True
pltShow=True

figSze0 = (5,4)
TD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\tables\\"
FD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\figures\\"

rsltsFrac = {}; rsltsVal = {}; rsltsSns = {}
for feeder in feeders:
    # RD = os.path.join(WD,feeder,'linHcCalcs'+rsltType+pdfName+num+regID+'.pkl')
    RDval = os.path.join(WD,feeder,'linHcCalcsVal_gammaFrac50_new.pkl')
    RDfrac = os.path.join(WD,feeder,'linHcCalcsRslt_gammaFrac_reg0_new.pkl')
    RDfrac = os.path.join(WD,feeder,'linHcCalcsRslt_gammaFrac_reg0_bw.pkl')
    with open(RDfrac,'rb') as handle:
        rsltsFrac[feeder] = pickle.load(handle)
    with open(RDval,'rb') as handle:
        rsltsVal[feeder] = pickle.load(handle)
for feeder in feeders_dcp:
    # RDsns = os.path.join(WD,feeder,'linHcCalcsSns_gammaFrac.pkl')
    RDsns = os.path.join(WD,feeder,'linHcCalcsSns_gammaFrac_new.pkl')
    with open(RDsns,'rb') as handle:
        rsltsSns[feeder] = pickle.load(handle)

timeStrLin = [];    timeStrDss = []; timeLin = []; timeDss = []
kCdfLin = [];    kCdfDss = []; 
LrelNorm = []; LmeanNorm = []
for rslt in rsltsFrac.values():
    timeStrLin.append('%.3f' % (rslt['linHcRsl']['runTime']/60.))
    timeStrDss.append('%.3f' % (rslt['dssHcRsl']['runTime']/60.))
    timeLin.append(rslt['linHcRsl']['runTime']/60.)
    timeDss.append(rslt['dssHcRsl']['runTime']/60.)
    KcdkLin = rslt['linHcRsl']['kCdf']
    KcdkDss = rslt['dssHcRsl']['kCdf']
    kCdfLin.append(KcdkLin[[0,1,5,10,15,19,20]])
    kCdfDss.append(KcdkDss[[0,1,5,10,15,19,20]])
    LmeanNorm.append( np.mean(np.abs(rslt['dssHcRsl']['Vp_pct']-rslt['linHcRsl']['Vp_pct'])) )
    LrelNorm.append(rslt['regError'])
        
kCdfVal = []; LvalNorm = []; # kCdfLin = []
for rslt in rsltsVal.values():
    KcdkLin = rslt['linHcRsl']['kCdf']
    KcdkVal = rslt['linHcVal']['kCdf']
    kCdfVal.append(KcdkVal[[0,1,5,10,15,19,20]])
    LvalNorm.append(rslt['regError'])
kCdfSns0 = [];    kCdfSns1 = []; kCdfLinSns = []; LsnsNorm = [] # new lin needed coz this is only some models
for rslt in rsltsSns.values():
    KcdkLinSns = rslt['linHcRsl']['kCdf']
    KcdkSns0 = rslt['linHcSns0']['kCdf']
    KcdkSns1 = rslt['linHcSns1']['kCdf']
    kCdfLinSns.append(KcdkLinSns[[0,1,5,10,15,19,20]])
    kCdfSns0.append(KcdkSns0[[0,1,5,10,15,19,20]])
    kCdfSns1.append(KcdkSns1[[0,1,5,10,15,19,20]])
    LsnsNorm.append(rslt['regErrors'])

linHcRsl = rslt['linHcRsl']

# TABLE 1 ======================= 
if 't_timeTable' in locals():
    caption='Linear and non-linear models HC run times (min).'
    label='timeTable'
    heading = ['']+feeders
    data = [['Full Model']+timeStrDss,['Linear Model']+timeStrLin]
    basicTable(caption,label,heading,data,TD)
# ===============================

dx = 0.175; ddx=dx/1.5
X = np.arange(len(kCdfLin))
i=0
clrA,clrB,clrC,clrD,clrE = cm.tab10(np.arange(5))
# RESULTS 1 - opendss vs linear model, k =====================
if 'f_dssVlinWght' in locals():
    fig = plt.figure(figsize=figSze0)
    ax = fig.add_subplot(111)
    i=0
    for x in X:
        ax = plotBoxWhisk(ax,x+dx,ddx,kCdfDss[i][1:-1],clrB,bds=kCdfDss[i][[0,-1]])
        ax = plotBoxWhisk(ax,x-dx,ddx,kCdfLin[i][1:-1],clrA,bds=kCdfLin[i][[0,-1]])
        i+=1
    ax.plot(0,0,'-',color=clrA,label='Linear')
    ax.plot(0,0,'-',color=clrB,label='OpenDSS')
    plt.legend()
    plt.ylim((-2,102))
    plt.grid(True)
    plt.xticks(X,feeders,rotation=90)
    plt.ylabel('Loads with PV installed, \%')
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
    plt.ylabel('Loads with PV installed, \%')
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
Xsns = np.arange(len(kCdfLinSns))
if 'f_linMcSns' in locals():
    fig = plt.figure(figsize=figSze0)
    ax = fig.add_subplot(111)
    i=0
    for x in Xsns:
        ax = plotBoxWhisk(ax,x-dx,0.66*ddx,kCdfLinSns[i][1:-1],clrC,bds=kCdfLinSns[i][[0,-1]])
        ax = plotBoxWhisk(ax,x   ,0.66*ddx,kCdfSns0[i][1:-1],clrD,bds=kCdfSns0[i][[0,-1]])
        ax = plotBoxWhisk(ax,x+dx,0.66*ddx,kCdfSns1[i][1:-1],clrE,bds=kCdfSns1[i][[0,-1]])
        i+=1
    ax.plot(0,0,'-',color=clrC,label='Nom')
    ax.plot(0,0,'-',color=clrD,label='T+1')
    ax.plot(0,0,'-',color=clrE,label='T-1')
    plt.legend()
    plt.ylim((-2,102))
    plt.grid(True)
    plt.xticks(Xsns,feeders_dcp,rotation=90)
    plt.ylabel('Loads with PV installed, \%')
    plt.tight_layout()
    if 'pltSave' in locals():
        plt.savefig(FD+'linMcSns.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'linMcSns.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()

if 'f_dssVlinWghtErr' in locals():
    plt.bar(X,LrelNorm)
    plt.xticks(X,feeders,rotation=90)
    plt.title('Error')
    plt.ylabel('Relative error, $\epsilon = \dfrac{||f_{\mathrm{lin}}(x) - f_{\mathrm{dss}}(x)||_{1}}{1 + ||f_{\mathrm{dss}}||_{1}}$')
    plt.tight_layout()
    if 'pltSave' in locals():
        plt.savefig(FD+'dssVlinWghtErr.png',pad_inches=0.02,bbox_inches='tight')
        plt.savefig(FD+'dssVlinWghtErr.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()
        
        
        
# rsltM1 = rsltsFrac['epriM1']
# pdf = rsltM1['pdfData']
# linRsl = rsltM1['linHcRsl']
# dssRsl = rsltM1['dssHcRsl']
# fig, ax = plt.subplots()
# ax = plotCns(pdf['mu_k'],pdf['prms'],dssRsl['Cns_pct'],feeder=rsltM1['feeder'],lineStyle='-',ax=ax,pltShow=False)
# ax = plotCns(pdf['mu_k'],pdf['prms'],linRsl['Cns_pct'],feeder=rsltM1['feeder'],lineStyle='--',ax=ax,pltShow=False)
# plt.show()
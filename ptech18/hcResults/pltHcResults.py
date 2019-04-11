import pickle, sys, os, getpass
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

WD = os.path.dirname(sys.argv[0])
sys.path.append(os.path.dirname(WD))

from dss_python_funcs import basicTable
from linSvdCalcs import plotBoxWhisk, getKcdf, plotCns

feeders = ['13bus','34bus','123bus','8500node','eulv','usLv','epriJ1','epriK1','epriM1','epri5','epri7','epri24']
# feeders = ['13bus','34bus','123bus','8500node','eulv','usLv','epriK1','epriM1','epri5','epri7','epri24']
pLoad = {'epriJ1':11.6,'epriK1':12.74,'epriM1':15.67,'epri5':16.3,'epri7':19.3,'epri24':28.8,'8500node':12.05,'eulv':0.055,'usLv':42.8,'13bus':3.6,'34bus':2.0,'123bus':3.6}

feeders_dcp = ['8500node','epriJ1','epriK1','epriM1','epri24']

# t_timeTable = 1 # timeTable
# f_dssVlinWght = 1 # gammaFrac boxplot results
# f_linMcVal = 1 # monte carlo no. validation
# f_logTimes = 1 # 
# f_linMcSns = 1
f_dssVlinWghtErr = 1
# f_dssSeqPar = 1

# pltSave=True
pltShow=True

figSze0 = (5,4)
TD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\tables\\"
FD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\figures\\"

rsltsFrac = {}; rsltsVal = {}; rsltsPar = {}; rsltsSns = {}
for feeder in feeders:
    # RD = os.path.join(WD,feeder,'linHcCalcs'+rsltType+pdfName+num+regID+'.pkl')
    RDval = os.path.join(WD,feeder,'linHcCalcsVal_gammaFrac50_new.pkl')
    RDfrac = os.path.join(WD,feeder,'linHcCalcsRslt_gammaFrac_reg0_new.pkl')
    RDfrac = os.path.join(WD,feeder,'linHcCalcsRslt_gammaFrac_reg0_bw.pkl')
    RDpar = os.path.join(WD,feeder,'linHcCalcsRslt_gammaFrac_reg0_par.pkl')
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
LrelNorm = []; LmeanNorm = []
for rslt in rsltsFrac.values():
    timeStrLin.append('%.3f' % (rslt['linHcRsl']['runTime']/60.))
    timeStrDss.append('%.3f' % (rslt['dssHcRsl']['runTime']/60.))
    timeLin.append(rslt['linHcRsl']['runTime']/60.)
    timeDss.append(rslt['dssHcRsl']['runTime']/60.)
    kCdfLin.append(rslt['linHcRsl']['kCdf'][[0,1,5,10,15,19,20]])
    kCdfDss.append(rslt['dssHcRsl']['kCdf'][[0,1,5,10,15,19,20]])
    LmeanNorm.append( np.mean(np.abs(rslt['dssHcRsl']['Vp_pct']-rslt['linHcRsl']['Vp_pct']))*0.01 )
    LrelNorm.append(rslt['regError'])
        
kCdfVal = []; LvalNorm = []; # kCdfLin = []
for rslt in rsltsVal.values():
    kCdfVal.append(rslt['linHcVal']['kCdf'][[0,1,5,10,15,19,20]])
    LvalNorm.append(rslt['regError'])
kCdfSns0 = [];    kCdfSns1 = []; kCdfLinSns = []; LsnsNorm = [] # new lin needed coz this is only some models
for rslt in rsltsSns.values():
    kCdfLinSns.append(rslt['linHcRsl']['kCdf'][[0,1,5,10,15,19,20]])
    kCdfSns0.append(rslt['linHcSns0']['kCdf'][[0,1,5,10,15,19,20]])
    kCdfSns1.append(rslt['linHcSns1']['kCdf'][[0,1,5,10,15,19,20]])
    LsnsNorm.append(rslt['regErrors'])
KcdkLin = []; KcdkSeq = []; KcdkPar = []; # kCdfLin = []
timeSeq = [];timePar = [];
for rslt in rsltsPar.values():
    KcdkLin.append(rslt['linHcRsl']['kCdf'][0::5])
    KcdkSeq.append(rslt['dssHcRslSeq']['kCdf'][0::5])
    KcdkPar.append(rslt['dssHcRslPar']['kCdf'][0::5])
    timeSeq.append(rslt['dssHcRslSeq']['runTime'])
    timePar.append(rslt['dssHcRslPar']['runTime'])

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

# RESULTS 1 - opendss vs linear model, k =====================
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
    plt.ylabel('Loads with PV installed, \%')
    plt.tight_layout()
    # if 'pltSave' in locals():
        # plt.savefig(FD+'dssVlinWght.png',pad_inches=0.02,bbox_inches='tight')
        # plt.savefig(FD+'dssVlinWght.pdf',pad_inches=0.02,bbox_inches='tight')
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
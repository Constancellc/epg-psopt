import pickle, sys, os, getpass
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

WD = os.path.dirname(sys.argv[0])
sys.path.append(os.path.dirname(WD))

from dss_python_funcs import basicTable
from linSvdCalcs import plotBoxWhisk, getKcdf

feeders = ['13bus','34bus','123bus','8500node','eulv','usLv','epriJ1','epriK1','epriM1','epri5','epri7','epri24']
# feeders = ['eulv','13bus','34bus','123bus','usLv','epri5','epri7','epri24','epriK1','epriM1']
pLoad = {'epriJ1':11.6,'epriK1':12.74,'epriM1':15.67,'epri5':16.3,'epri7':19.3,'epri24':28.8,'8500node':12.05,'eulv':0.055,'usLv':42.8,'13bus':3.6,'34bus':2.0,'123bus':3.6}

# t_timeTable = True # timeTable
f_dssVlinWght = True # gammaFrac boxplot results
f_linMcVal = True # monte carlo no. validation
f_logTimes = True # 

# pltShow=True

figSze0 = (5,4)
TD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\tables\\"
FD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\figures\\"

rsltsFrac = {}; rsltsVal = {}
for feeder in feeders:
    # RD = os.path.join(WD,feeder,'linHcCalcs'+rsltType+pdfName+num+regID+'.pkl')
    RDval = os.path.join(WD,feeder,'linHcCalcsVal_gammaFrac50.pkl')
    RDfrac = os.path.join(WD,feeder,'linHcCalcsRslt_gammaFrac_reg0.pkl')
    with open(RDfrac,'rb') as handle:
        rsltsFrac[feeder] = pickle.load(handle)
    with open(RDval,'rb') as handle:
        rsltsVal[feeder] = pickle.load(handle)


timeStrLin = [];    timeStrDss = []; timeLin = []; timeDss = []
kCdfLin = [];    kCdfDss = []
for rslt in rsltsFrac.values():
    timeStrLin.append('%.3f' % (rslt['linHcRsl']['runTime']/60.))
    timeStrDss.append('%.3f' % (rslt['dssHcRsl']['runTime']/60.))
    timeLin.append(rslt['linHcRsl']['runTime']/60.)
    timeDss.append(rslt['dssHcRsl']['runTime']/60.)
    KcdkLin = 100*np.array(getKcdf(rslt['pdfData']['prms'],rslt['linHcRsl']['Vp_pct'])[0])
    KcdkDss = 100*np.array(getKcdf(rslt['pdfData']['prms'],rslt['dssHcRsl']['Vp_pct'])[0])
    KcdkLin[np.isnan(KcdkLin)] = 100.
    KcdkDss[np.isnan(KcdkDss)] = 100.
    kCdfLin.append(KcdkLin[0::5])
    kCdfDss.append(KcdkDss[0::5])
kCdfLin = [];    kCdfVal = []
for rslt in rsltsVal.values():
    KcdkLin = 100*np.array(rslt['linHcRsl']['kCdf'])
    KcdkVal = 100*np.array(rslt['linHcVal']['kCdf'])
    KcdkLin[np.isnan(KcdkLin)] = 100.
    KcdkVal[np.isnan(KcdkVal)] = 100.
    kCdfLin.append(KcdkLin[0::5])
    kCdfVal.append(KcdkVal[0::5])
    
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
clrA,clrB = cm.tab10([0,1])
# RESULTS 1 - opendss vs linear model, k =====================
if 'f_dssVlinWght' in locals():
    fig = plt.figure(figsize=figSze0)
    ax = fig.add_subplot(111)
    i=0
    for x in X:
        ax = plotBoxWhisk(ax,x+dx,ddx,kCdfDss[i],clrB)
        ax = plotBoxWhisk(ax,x-dx,ddx,kCdfLin[i],clrA)
        i+=1
    ax.plot(0,0,'-',color=clrA,label='Linear')
    ax.plot(0,0,'-',color=clrB,label='OpenDSS')
    plt.legend()
    plt.ylim((-2,102))
    plt.grid(True)
    plt.xticks(X,feeders,rotation=90)
    plt.ylabel('Loads with PV installed, \%')
    plt.tight_layout()
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
        ax = plotBoxWhisk(ax,x+dx,ddx,kCdfVal[i],clrB)
        ax = plotBoxWhisk(ax,x-dx,ddx,kCdfLin[i],clrA)
        i+=1
    ax.plot(0,0,'-',color=clrA,label='Run A')
    ax.plot(0,0,'-',color=clrB,label='Run B')
    plt.legend()
    plt.xticks(X,feeders,rotation=90)
    plt.ylim((-2,102))
    plt.grid(True)
    plt.ylabel('Loads with PV installed, \%')
    plt.tight_layout()
    plt.savefig(FD+'linMcVal.png',pad_inches=0.02,bbox_inches='tight')
    plt.savefig(FD+'linMcVal.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()

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
    plt.savefig(FD+'logTimes.png',pad_inches=0.02,bbox_inches='tight')
    plt.savefig(FD+'logTimes.pdf',pad_inches=0.02,bbox_inches='tight')
    if 'pltShow' in locals():
        plt.show()
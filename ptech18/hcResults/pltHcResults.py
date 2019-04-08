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

pdfName = 'gammaWght'
pdfName = 'gammaFrac'

rsltType = 'Rslt_'

table1 = True
table1 = False
res1 = True
# res1 = False
res2 = True
res2 = False

num = ''
# num = '300'

# rsltType = 'Val_'
res3 = True
res3 = False


figSze0 = (5,4)

TD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\tables\\"
rslts = {}

for feeder in feeders:
    # RD = os.path.join(WD,feeder,'linHcCalcs'+rsltType+pdfName+num+'.pkl')
    RD = os.path.join(WD,feeder,'linHcCalcs'+rsltType+pdfName+num+'_reg0.pkl')
    with open(RD,'rb') as handle:
        rslts[feeder] = pickle.load(handle)


if rsltType=='Rslt_':
    timeStrLin = [];    timeStrDss = []
    kCdfLin = [];    kCdfDss = []
    ppCdfLin = [];    ppCdfDss = []
    for rslt in rslts.values():
        timeStrLin.append('%.3f' % (rslt['linHcRsl']['runTime']/60.))
        timeStrDss.append('%.3f' % (rslt['dssHcRsl']['runTime']/60.))
        KcdkLin = 100*np.array(getKcdf(rslt['pdfData']['prms'],rslt['linHcRsl']['Vp_pct'])[0])
        KcdkDss = 100*np.array(getKcdf(rslt['pdfData']['prms'],rslt['dssHcRsl']['Vp_pct'])[0])
        KcdkLin[np.isnan(KcdkLin)] = 100.
        KcdkDss[np.isnan(KcdkDss)] = 100.
        kCdfLin.append(KcdkLin[0::5])
        kCdfDss.append(KcdkDss[0::5])
        ppCdfLin.append(1e-3*np.array(rslt['linHcRsl']['ppCdf'][0::5])/pLoad[rslt['feeder']]) # range plus quartiles
        ppCdfDss.append(1e-3*np.array(rslt['dssHcRsl']['ppCdf'][0::5])/pLoad[rslt['feeder']])
elif rsltType=='Val_':
    kCdfLin = [];    kCdfVal = []
    for rslt in rslts.values():
        KcdkLin = 100*np.array(rslt['linHcRsl']['kCdf'])
        KcdkVal = 100*np.array(rslt['linHcVal']['kCdf'])
        KcdkLin[np.isnan(KcdkLin)] = 100.
        KcdkVal[np.isnan(KcdkVal)] = 100.
        kCdfLin.append(KcdkLin[0::5])
        kCdfVal.append(KcdkVal[0::5])
    
linHcRsl = rslt['linHcRsl']

# TABLE 1 ======================= 
if table1:
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
if res1:
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
    plt.show()

# RESULTS 2 - opendss vs linear model, p =====================
if res2:
    fig = plt.figure(figsize=figSze0)
    ax = fig.add_subplot(111)
    i=0
    for x in X:
        ax = plotBoxWhisk(ax,x+dx,ddx,ppCdfDss[i],clrB)
        ax = plotBoxWhisk(ax,x-dx,ddx,ppCdfLin[i],clrA)
        i+=1
    ax.plot(0,0,'-',color=clrA,label='Linear Model')
    ax.plot(0,0,'-',color=clrB,label='OpenDSS Model')
    plt.legend()
    plt.xticks(X,feeders,rotation=90)
    plt.tight_layout()
    plt.show()
    
# RESULTS 3 - linear model vs linear rerun, k =====================
if res3:
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
    plt.show()


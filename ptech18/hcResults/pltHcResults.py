import pickle, sys, os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

WD = os.path.dirname(sys.argv[0])
sys.path.append(os.path.dirname(WD))

from dss_python_funcs import basicTable
from linSvdCalcs import plotBoxWhisk

feeders = ['epriJ1','epriK1','epriM1','epri5','epri7','epri24','8500node','eulv','usLv','13bus','34bus','123bus']
pLoad = 1000*np.array([11.6,12.74,15.67,16.3,19.3,28.8,12.05,0.055,42.8,3.6,2.0,3.6])

pLoad = {'epriJ1':11.6,'epriK1':12.74,'epriM1':15.67,'epri5':16.3,'epri7':19.3,'epri24':28.8,'8500node':12.05,'eulv':0.055,'usLv':42.8,'13bus':3.6,'34bus':2.0,'123bus':3.6}


# feeders = ['eulv','13bus','34bus','123bus','usLv','epri5','epri7','epri24','epriK1','epriM1']
figSze0 = (5,4)

TD = r"C:\Users\chri3793\Documents\DPhil\papers\psfeb19\tables\\"

rslts = {}

for feeder in feeders:
    RD = os.path.join(WD,feeder,'linHcCalcsRslt_gammaWght.pkl')
    with open(RD,'rb') as handle:
        rslts[feeder] = pickle.load(handle)


timeStrLin = []
timeStrDss = []
kCdfLin = []
kCdfDss = []
ppCdfLin = []
ppCdfDss = []

for rslt in rslts.values():    
    timeStrLin.append('%.3f' % (rslt['linHcRsl']['runTime']/60.))
    timeStrDss.append('%.3f' % (rslt['dssHcRsl']['runTime']/60.))
    
    kCdfLin.append(rslt['linHcRsl']['kCdf'][0::5]) # range plus quartiles
    kCdfDss.append(rslt['dssHcRsl']['kCdf'][0::5])
    ppCdfLin.append(1e-3*np.array(rslt['linHcRsl']['ppCdf'][0::5])/pLoad[rslt['feeder']]) # range plus quartiles
    ppCdfDss.append(1e-3*np.array(rslt['dssHcRsl']['ppCdf'][0::5])/pLoad[rslt['feeder']])
    
    

# # TABLE 1 ======================= 
# caption='Linear and non-linear models HC run times (min).'
# label='timeTable'
# heading = ['']+feeders
# data = [['Full Model']+timeStrDss,['Linear Model']+timeStrLin]
# basicTable(caption,label,heading,data,TD)
# # ===============================


# # RESULTS 1 =====================
# fig = plt.figure(figsize=figSze0)
# ax = fig.add_subplot(111)

# X = np.arange(len(kCdfLin))
# dx = 0.1; ddx=dx/4
# i=0
# clrA,clrB = cm.tab10([0,1])
# for x in X:
    # ax = plotBoxWhisk(ax,x+dx,ddx,kCdfDss[i],clrB)
    # ax = plotBoxWhisk(ax,x-dx,ddx,kCdfLin[i],clrA)
    # i+=1
# ax.plot(0,0,'-',color=clrA,label='Linear Model')
# ax.plot(0,0,'-',color=clrB,label='OpenDSS Model')
# plt.legend()
# plt.xticks(X,feeders,rotation=90)
# plt.tight_layout()
# plt.show()

# RESULTS 2 =====================
fig = plt.figure(figsize=figSze0)
ax = fig.add_subplot(111)

X = np.arange(len(kCdfLin))
dx = 0.1; ddx=dx/4
i=0
clrA,clrB = cm.tab10([0,1])
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







# plt.bar(x-(dx/2),p10/pLoad,width=dx,zorder=3)
# plt.bar(x+(dx/2),p10lin/pLoad,width=dx,zorder=3)
# plt.title('P10')
# plt.xticks(x,feeders,rotation=90)
# plt.ylabel('Fraction of peak load')
# plt.grid(True,zorder=0)
# plt.tight_layout()
# plt.show()
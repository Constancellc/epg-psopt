import pickle, sys, os
import matplotlib.pyplot as plt
import numpy as np

WD = os.path.dirname(sys.argv[0])
sys.path.append(os.path.dirname(WD))

from dss_python_funcs import basicTable

feeders = ['epriJ1','epriK1','epriM1','epri5','epri7','epri24','8500node','eulv','usLv','13bus','34bus','123bus']
pLoad = 1000*np.array([11.6,12.74,15.67,16.3,19.3,28.8,12.05,0.055,42.8,3.6,2.0,3.6])

feeders = ['eulv','13bus','34bus','123bus']


TD = r"C:\Users\chri3793\Documents\DPhil\papers\psfeb19\tables\\"

rslts = {}

for feeder in feeders:
    RD = os.path.join(WD,feeder,'linHcCalcsRslt_gammaWght.pkl')
    with open(RD,'rb') as handle:
        rslts[feeder] = pickle.load(handle)

p0 = []; p0lin = []; p10 = []; p10lin = []
k0 = []; k0lin = []; k10 = []; k10lin = []

timeStrLin = []
timeStrDss = []
for rslt in rslts.values():
    p0 = p0+[rslt[0]['p0']]
    p0lin = p0lin + [rslt[0]['p0lin']]
    p10 = p10+[rslt[0]['p10']]
    p10lin = p10lin + [rslt[0]['p10lin']]
    k0 = k0+[rslt[0]['k0']]
    k0lin = k0lin + [rslt[0]['k0lin']]
    k10 = k10+[rslt[0]['k10']]
    k10lin = k10lin + [rslt[0]['k10lin']]
    
    timeStrLin = timeStrLin + ['%.3f' % (rslt[0]['time2runLin']/60.)]
    timeStrDss = timeStrDss + ['%.3f' % (rslt[0]['time2runDss']/60.)]

# TABLE 1 ======================= 
caption='Linear and non-linear models HC run times (min).'
label='timeTable'
heading = ['']+feeders
data = [['Full Model']+timeStrDss,['Linear Model']+timeStrLin]
basicTable(caption,label,heading,data,TD)
# ===============================


# x = np.arange(len(feeders))
# dx = 0.25

# p0 = np.array(p0)
# p0lin = np.array(p0lin)
# p10 = np.array(p10)
# p10lin = np.array(p10lin)

# # plt.subplot(221)
# # plt.bar(x-(dx/2),p0/pLoad,width=dx,zorder=3)
# # plt.bar(x+(dx/2),p0lin/pLoad,width=dx,zorder=3)
# # plt.title('P0')
# # plt.legend(('OpenDSS','Linear Model'))
# # plt.xticks(x,feeders,rotation=90)
# # plt.ylabel('Fraction of peak load')
# # plt.grid(True,zorder=0)

# # plt.subplot(222)
# # plt.bar(x-(dx/2),p10/pLoad,width=dx,zorder=3)
# # plt.bar(x+(dx/2),p10lin/pLoad,width=dx,zorder=3)
# # plt.title('P10')
# # plt.xticks(x,feeders,rotation=90)
# # plt.ylabel('Fraction of peak load')
# # plt.grid(True,zorder=0)

# # plt.subplot(223)
# # plt.bar(x-(dx/2),k0,width=dx,zorder=3)
# # plt.bar(x+(dx/2),k0lin,width=dx,zorder=3)
# # plt.title('K0')
# # plt.xticks(x,feeders,rotation=90)
# # plt.grid(True,zorder=0)

# # plt.subplot(224)
# # plt.bar(x-(dx/2),k10,width=dx,zorder=3)
# # plt.bar(x+(dx/2),k10lin,width=dx,zorder=3)
# # plt.title('K10')
# # plt.tight_layout()
# # plt.grid(True,zorder=0)
# # plt.xticks(x,feeders,rotation=90)

# # plt.show()


# plt.bar(x-(dx/2),p10/pLoad,width=dx,zorder=3)
# plt.bar(x+(dx/2),p10lin/pLoad,width=dx,zorder=3)
# plt.title('P10')
# plt.xticks(x,feeders,rotation=90)
# plt.ylabel('Fraction of peak load')
# plt.grid(True,zorder=0)
# plt.tight_layout()
# plt.show()
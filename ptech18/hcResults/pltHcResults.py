import pickle, sys, os
import matplotlib.pyplot as plt
import numpy as np

WD = os.path.dirname(sys.argv[0])
feeders = ['13bus','34bus','123bus','epri5','epri7']

rslts = {}

for feeder in feeders:
    RD = os.path.join(WD,feeder,'linHcCalcsRslt.pkl')
    with open(RD,'rb') as handle:
        rslts[feeder] = pickle.load(handle)

p0 = []; p0lin = []; p10 = []; p10lin = []
k0 = []; k0lin = []; k10 = []; k10lin = []

for rslt in rslts.values():
    p0 = p0+[rslt[0]['p0']]
    p0lin = p0lin + [rslt[0]['p0lin']]
    p10 = p10+[rslt[0]['p10']]
    p10lin = p10lin + [rslt[0]['p10lin']]
    k0 = k0+[rslt[0]['k0']]
    k0lin = k0lin + [rslt[0]['k0lin']]
    k10 = k10+[rslt[0]['k10']]
    k10lin = k10lin + [rslt[0]['k10lin']]
    
x = np.arange(len(feeders))
dx = 0.25


plt.subplot(221)
plt.bar(x-(dx/2),p0,width=dx,zorder=3)
plt.bar(x+(dx/2),p0lin,width=dx,zorder=3)
plt.title('P0')
plt.legend(('OpenDSS','Linear Model'))
plt.xticks(x,feeders)
plt.grid(True,zorder=0)

plt.subplot(222)
plt.bar(x-(dx/2),p10,width=dx,zorder=3)
plt.bar(x+(dx/2),p10lin,width=dx,zorder=3)
plt.title('P10')
plt.xticks(x,feeders)
plt.grid(True,zorder=0)

plt.subplot(223)
plt.bar(x-(dx/2),k0,width=dx,zorder=3)
plt.bar(x+(dx/2),k0lin,width=dx,zorder=3)
plt.title('K0')
plt.xticks(x,feeders)
plt.grid(True,zorder=0)

plt.subplot(224)
plt.bar(x-(dx/2),k10,width=dx,zorder=3)
plt.bar(x+(dx/2),k10lin,width=dx,zorder=3)
plt.title('K10')
plt.tight_layout()
plt.grid(True,zorder=0)
plt.xticks(x,feeders)

plt.show()
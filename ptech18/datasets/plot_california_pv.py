import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, getpass
from math import gamma
# from matplotlib import rc
# rc('text',usetex=True)

# pltShow=1
pltSave=1


WD = os.path.dirname(sys.argv[0])
fn = os.path.join(WD,'california_residential_pv.csv')

figSze0 = (5.2,2.5)
SD = r"C:\Users\\"+getpass.getuser()+r"\Documents\DPhil\papers\psfeb19\figures\\"

QWE = pd.read_csv(fn)

WER = QWE['Customer Sector']=='Residential'
QWE = QWE[WER]

pmax=20

RTY = QWE['System Size AC']
RTY = RTY[RTY<pmax]
RTY = RTY[RTY>0]

RTYnp = np.array(RTY)

s = np.log(np.mean(RTYnp)) - sum(np.log(RTYnp))/len(RTYnp) # see gamma distribution wiki

k = ((3 - s) + np.sqrt( (s-3)**2 + 24*s ))/(12*s)
th = np.mean(RTYnp)/k

hist = plt.hist(RTY, bins=200,density=True)
plt.close()

histx = hist[1][:-1]
histy = hist[0]

X = np.linspace(0,20,len(histx))
dX = X[1] - X[0]
gX = (X**(k-1))*np.exp(-X/th)/(gamma(k)*(th**k))


fig = plt.figure(figsize=figSze0)
ax = plt.subplot()

ax.step(histx,histy)
ax.plot(X,gX)
ax.legend(('California residential solar','Fitted gamma distribution'))
ax.set_xlabel('AC System Size (kW)')
ax.set_ylabel('Probability density')
# ax.title('California Residential PV Sizes (Oct 18)')
ax.set_xlim((0,pmax))
ax.set_ylim((0,ax.get_ylim()[1]))
ax.set_xticks(np.arange(0,pmax+2,2))
# ax.grid(True)
plt.tight_layout()

if 'pltSave' in locals():
    plt.savefig(SD+"plot_california_pv.pdf",bbox_inches='tight',pad_inches=0)
    plt.savefig(SD+"plot_california_pv.png",bbox_inches='tight',pad_inches=0)
if 'pltShow' in locals():
    plt.show()

print("k (approx):",k)
print("Theta (approx):",th)
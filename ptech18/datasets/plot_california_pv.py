import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, getpass,time
from math import gamma
# from matplotlib import rc
# rc('text',usetex=True)
plt.style.use('tidySettings')


# pltShow=1
# pltSave=1
# pltSaveTss=1




WD = os.path.dirname(sys.argv[0])
fn = os.path.join(WD,'california_residential_pv.csv')

SDT = os.path.join(os.path.join(os.path.expanduser('~')), 'Documents','DPhil','thesis','c4tech2','c4figures')

figSze0 = (5.2,2.5)
figSzeTss = (5.5,2.2)
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
legend = ax.legend(('California residential solar','Fitted gamma distribution'))
ax.tick_params(direction="in",bottom=1,top=1,left=1,right=1,grid_linewidth=0.4,width=0.4,length=2.5)
ax.set_xlabel('AC System Size (kW)')
ax.set_ylabel('Probability density')
# ax.title('California Residential PV Sizes (Oct 18)')
ax.set_xlim((0,pmax))
ax.set_ylim((0,ax.get_ylim()[1]))
ax.set_xticks(np.arange(0,pmax+2,2))

plt.tight_layout()

if 'pltSave' in locals():
    plt.savefig(SD+"plot_california_pv.pdf")
    plt.savefig(SD+"plot_california_pv.png")
if 'pltShow' in locals():
    plt.show()

print("k (approx):",k)
print("Theta (approx):",th)

if 'pltSaveTss' in locals():
    plt.close()
    fig = plt.figure(figsize=figSzeTss)
    ax = plt.subplot()
    ax.step(histx,histy)
    ax.plot(X,gX)
    legend = ax.legend(('California residential solar','Fitted gamma distribution'))
    ax.tick_params(direction="in",bottom=1,top=1,left=1,right=1,grid_linewidth=0.4,width=0.4,length=2.5)
    # ax.set_xlabel('AC System Size (kW)')
    ax.set_xlabel('Capacity (kW)')
    ax.set_ylabel('Probability density')
    # ax.title('California Residential PV Sizes (Oct 18)')
    ax.set_xlim((0,pmax))
    ax.set_ylim((0,ax.get_ylim()[1]))
    ax.set_xticks(np.arange(0,pmax+2,2))
    plt.tight_layout()
    plt.savefig(SDT+"\\plot_california_pv_tss.pdf")
    plt.savefig(SDT+"\\plot_california_pv_tss.png")
    plt.close()

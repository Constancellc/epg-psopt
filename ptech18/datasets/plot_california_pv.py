import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import getpass
from math import gamma

if getpass.getuser()=='chri3793':
    fn = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18\datasets\california_residential_pv.csv"
elif getpass.getuser()=='Matt':
    sn = r"C:\Users\Matt\Desktop\wc190128\\"
    fn = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18\datasets\california_residential_pv.csv"
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
plt.step(histx,histy)
plt.plot(X,gX)
plt.legend(('CEC data','Gamma MLE approx.'))
plt.xlabel('x = AC System Size (kW)')
plt.ylabel('p(x)')
plt.title('California Residential PV Sizes (Oct 18)')
plt.xlim((0,pmax))
plt.ylim((0,plt.ylim()[1]))
plt.xticks(np.arange(0,pmax+2,2))
plt.grid(True)
plt.savefig(sn+"plot_california_pv.png")
# plt.show()

print("k (approx):",k)
print("Theta (approx):",th)
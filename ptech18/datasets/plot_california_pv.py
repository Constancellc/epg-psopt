import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import getpass
from math import gamma

if getpass.getuser()=='chri3793':
    fn = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18\datasets\california_residential_pv.csv"

QWE = pd.read_csv(fn)

WER = QWE['Customer Sector']=='Residential'
QWE = QWE[WER]

RTY = QWE['System Size AC']
RTY = RTY[RTY<20]
RTY = RTY[RTY>0]


RTYnp = np.array(RTY)

s = np.log(np.mean(RTYnp)) - sum(np.log(RTYnp))/len(RTYnp)

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
plt.plot(X,gX); plt.show()
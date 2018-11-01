import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

fn = "C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18\datasets\california_residential_pv.csv"

QWE = pd.read_csv(fn)

WER = QWE['Customer Sector']=='Residential'
QWE = QWE[WER]

RTY = QWE['System Size AC']
RTY = RTY[RTY<20]
RTY = RTY[RTY>0]

plt.hist(RTY, bins=200)
plt.show()
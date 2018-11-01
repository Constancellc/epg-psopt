import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

fn = "C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18\datasets\\fit_report_part1.csv"

data = pd.read_csv(fn)

pmax = 10

data = data[data['Technology']=='Photovoltaic']
data = data[data['Installation type']=='Domestic']
data = data[data['Declared net capacity']<pmax]
data = data[data['Declared net capacity']>0]

codes = data['Installation Postcode'].iloc[0:16]
i = 1
for code in codes:
	pv_set = data[data['Installation Postcode']==code]['Declared net capacity']
	ax = plt.subplot(4,4,i)
	plt.hist(pv_set,bins=40)
	i = i+1
	ax.set_title(code)
	ax.set_xlim([0,pmax])
plt.show()
# plt.show()

# NG2 = data[data['Installation Postcode']=='NG2 ']['Declared net capacity']
# TS10 = data[data['Installation Postcode']=='TS10 ']['Declared net capacity']
# PO16 = data[data['Installation Postcode']=='PO16 ']['Declared net capacity']
# SL6 = data[data['Installation Postcode']=='SL6 ']['Declared net capacity']
# TS17 = data[data['Installation Postcode']=='TS17 ']['Declared net capacity']

# plt.hist(TS17,bins=40)
# plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,getpass
plt.style.use('tidySettings')

# WD = os.path.dirname(sys.argv[0])
SDT = os.path.join(os.path.join(os.path.expanduser('~')), 'Documents','DPhil','thesis','c4tech2','c4figures')

if getpass.getuser()=='chri3793':
    fn = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18\datasets\\fit_report_part1.csv"
    fn1 = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18\datasets\\fit_report_part1.csv"
    fn2 = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18\datasets\fit_report_part2.csv"
    fn3 = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18\datasets\fit_report_part3.csv"
elif getpass.getuser()=='Matt':
    sn = r"C:\Users\Matt\Desktop\wc190128\\"
    fn = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18\datasets\fit_report_part1.csv"
    fn1 = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18\datasets\fit_report_part1.csv"
    fn2 = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18\datasets\fit_report_part2.csv"
    fn3 = r"C:\Users\Matt\Documents\MATLAB\epg-psopt\ptech18\datasets\fit_report_part3.csv"

data = pd.read_csv(fn,encoding="latin")
# data = pd.read_csv(fn)
data1 = pd.read_csv(fn1,encoding="latin")
data2 = pd.read_csv(fn2,encoding="latin")
data3 = pd.read_csv(fn3,encoding="latin")

dataTot = pd.concat((data1,data2,data3),ignore_index=True)
dataTot = pd.concat((data1,data2,data3))

pmax = 20

dataTot = dataTot[dataTot['Technology']=='Photovoltaic']
dataTot = dataTot[dataTot['Installation type']=='Domestic']
dataTot = dataTot[dataTot['Declared net capacity']<pmax]
dataTot = dataTot[dataTot['Declared net capacity']>0]

data = data[data['Technology']=='Photovoltaic']
data = data[data['Installation type']=='Domestic']
data = data[data['Declared net capacity']<pmax]
data = data[data['Declared net capacity']>0]

hist=plt.hist(data['Declared net capacity'],bins=200,range=(0,20),density=True)
histTot=plt.hist(dataTot['Declared net capacity'],bins=200,range=(0,20),density=True)
histTotIns=plt.hist(dataTot['Installed capacity'],bins=200,range=(0,20),density=True)

plt.close()

histx = hist[1][:-1]
histy = hist[0]
histxTot = histTot[1][:-1]
histyTot = histTot[0]
histxTotIns = histTotIns[1][:-1]
histyTotIns = histTotIns[0]
# plt.step(histx,histy)

fig,ax = plt.subplots(figsize=(4.7,2.2))
# plt.step(histxTotIns,histyTotIns)
# plt.step(histxTot,histyTot)
# plt.title('UK Domestic PV FIT Sizes (Oct 18)')
# plt.xlabel('x = Capacity (kW)')
plt.step(histxTot,histyTot,label='UK FIT Domestic Installations')
plt.xlabel('Capacity (kW)')
# plt.ylabel('p(x)')
plt.ylabel('Probability Density')
plt.grid(True)
plt.xlim((0,pmax))
plt.ylim((0,plt.ylim()[1]))
plt.xticks(np.arange(0,22,2))
# plt.legend(('Installed Capacity','Declared Net Capacity'))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SDT,"plot_uk_pv.png"),pad_inches=0.02,bbox_inches='tight')
plt.savefig(os.path.join(SDT,"plot_uk_pv.pdf"),pad_inches=0.02,bbox_inches='tight')
plt.show()


# codes = data['Installation Postcode'].iloc[0:16]
# i = 1
# for code in codes:
	# pv_set = data[data['Installation Postcode']==code]['Declared net capacity']
	# ax = plt.subplot(4,4,i)
	# plt.hist(pv_set,bins=40)
	# i = i+1
	# ax.set_title(code)
	# ax.set_xlim([0,pmax])
# plt.show()



# plt.show()

# NG2 = data[data['Installation Postcode']=='NG2 ']['Declared net capacity']
# TS10 = data[data['Installation Postcode']=='TS10 ']['Declared net capacity']
# PO16 = data[data['Installation Postcode']=='PO16 ']['Declared net capacity']
# SL6 = data[data['Installation Postcode']=='SL6 ']['Declared net capacity']
# TS17 = data[data['Installation Postcode']=='TS17 ']['Declared net capacity']

# plt.hist(TS17,bins=40)
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import csv

a = []
b = []
c = []
with open('data/eulv_k_error.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        a.append(100*float(row[1]))
        b.append(100*float(row[2]))
        c.append(100*float(row[3]))

p = np.arange(-0.7,1.8,0.1)
plt.figure(figsize=(6,2))
plt.rcParams["font.family"] = 'serif'
plt.rcParams["font.size"] = '10'
plt.plot(p,a,c='b',label='0.3 kW')
plt.plot(p,b,c='r',ls='--',label='0.6 kW')
plt.plot(p,c,c='g',ls=':',label='1.0 kW')
plt.xlim(-0.7,1.7)
plt.ylim(0,0.04)
plt.xlabel('Power (kW)')
plt.ylabel('Relative Voltage\nError (%)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/losses/img/v-error.eps', format='eps', dpi=1000)
plt.show()

import csv
import matplotlib.pyplot as plt
import numpy as np

outfile = 'riskDayLoadIn.csv'
infile = '../../../Documents/riskDaySpecifiedVsAppliedLoaad.csv'

appl = []
spec = []

inLoads = []

with open(outfile,'rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        inLoads.append(float(row[0]))

with open(infile,'rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        if row == []:
            continue
        spec.append(float(row[0]))
        appl.append(float(row[1]))

tot_a = []

i = 0
while i < len(appl):
    total = 0
    for j in range(55):
        total += appl[i]
        i += 1
    tot_a.append(total)

x = np.arange(0,80,0.2)
y_ls = [0]*len(x)
y_od = [0]*len(x)

for i in range(len(inLoads)):
    y_od[int(5*inLoads[i])] += 1
    y_ls[int(5*tot_a[i])] += 1

plt.figure(1)
plt.bar(x,y_od,alpha=0.5,width=0.2,label='openDSS')
plt.bar(x,y_ls,alpha=0.5,width=0.2,label='load sum')
plt.legend()
plt.xlim(15,50)
plt.show()

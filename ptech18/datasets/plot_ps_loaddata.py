import csv,sys,os
import matplotlib.pyplot as plt
import numpy as np

WD = os.path.dirname(sys.argv[0])
filename = os.path.join(WD,"ps-dataport-export.csv")

dataDict = {}

with open(filename,'rt') as csvfile:
    psReader = csv.reader(csvfile)
    next(psReader)
    for row in psReader:
        if row[1] in dataDict.keys():
            before = dataDict.pop(row[1])
            before[0].append(row[0])
            before[1].append(float(row[2]))
            before[2].append(float(row[3]))
            dataDict[row[1]]=before
        else:
            dataDict[row[1]]=[[row[0]],[float(row[2])],[float(row[3])]]

loadIds = list(dataDict.keys())
iSet = 5
data0 = dataDict[loadIds[iSet]]

t0 = np.arange(len(data0[2]))

R = 10
downSampled1 = (np.array(data0[1][:len(data0[1]) - (len(data0[1]) % R)])).reshape(-1,R).mean(axis=1)
downSampled2 = (np.array(data0[2][:len(data0[1]) - (len(data0[1]) % R)])).reshape(-1,R).mean(axis=1)

t1 = np.arange(len(downSampled2))*R

plt.plot(t0,data0[2])
plt.plot(t0,data0[1]); 
plt.plot(t1,downSampled1)
plt.plot(t1,downSampled2)
plt.show()

plt.hist(downSampled2,bins=60)
plt.show()
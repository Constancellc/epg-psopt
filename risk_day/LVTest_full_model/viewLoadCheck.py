import csv
import matplotlib.pyplot as plt
import numpy as np

file = '../../../Documents/riskDaySpecifiedVsAppliedLoaad.csv'

appl = []
spec = []
ratio = []

with open(file,'rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        if row == []:
            continue
        spec.append(float(row[0]))
        appl.append(float(row[1]))
        ratio.append(float(row[0])/float(row[1]))

xx = np.arange(0,1,0.001)
yr = [0]*len(xx)

for i in range(len(ratio)):
    yr[int(1000*ratio[i])] += 1


x = np.arange(0,8,0.01)
y1 = [0]*len(x)
y2 = [0]*len(x)

bigger = 0
smaller = 0
same = 0

for i in range(len(appl)):
    y1[int(100*appl[i])] += 1/len(appl)
    y2[int(100*spec[i])] += 1/len(appl)

    if spec[i] == appl[i]:
        same += 1
    elif appl[i] > spec[i]:
        bigger += 1
    else:
        smaller += 1

print(same)
print(bigger)
print(smaller)

plt.figure(1)
plt.plot(x,y1,label='applied')
plt.plot(x,y2,label='specified')
plt.legend()
plt.xlim(0,3)
plt.ylim(0,0.016)

plt.figure(2)
plt.bar(xx,yr,width=0.001)
plt.xlim(0.9,1)
plt.show()

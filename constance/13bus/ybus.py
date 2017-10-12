import csv
import numpy as np
from cvxopt import matrix

nodeNames = []
with open('nodeNames.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        cell = row[0][1:]
        nodeNames.append(cell[:len(cell)-1])


R = []
X = []
with open('yBus.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        rRow = []
        xRow = []
        for cell in row:
            reNum = ''
            i = 0
            while cell[i] != ' ':
                reNum += cell[i]
                i += 1

            imNum=cell[i+1]
            i += 3

            imNum += cell[i:len(cell)-1]

            reNum = float(reNum)
            imNum = float(imNum)

            try:
                z = -1/(complex(reNum,imNum))
            except:
                z = complex(0,0)

            rRow.append(z.real)
            xRow.append(z.imag)
                   

        R.append(rRow)
        X.append(xRow)

for i in range(0,len(R)):
    R[i][i] = 0
    X[i][i] = 0





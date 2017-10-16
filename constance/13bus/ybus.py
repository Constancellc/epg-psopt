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
with open('../YBus.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row == []:
            continue
        rRow = []
        xRow = []
        for cell in row:
            if cell == '0j':
                z = complex(0,0)
            else:
                
                reNum = cell[1]
                i = 2
                while cell[i] not in ['+', '-']:
                    reNum += cell[i]
                    i += 1

                imNum = cell[i:len(cell)-2]

                reNum = float(reNum)
                imNum = float(imNum)

                z = -1/(complex(reNum,imNum))


            rRow.append(z.real)
            xRow.append(z.imag)
                   

        R.append(rRow)
        X.append(xRow)

for i in range(0,len(R)):
    R[i][i] = 0
    X[i][i] = 0
    

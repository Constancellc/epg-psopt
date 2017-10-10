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
Y = []
with open('yBus.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        rRow =[]
        xRow = []
        yRow = []
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

            rRow.append(float(reNum)*-1*mag)
            xRow.append(float(imNum)*mag)

            yRow.append(complex(reNum,imNum))
        
        R.append(rRow)
        X.append(xRow)
        Y.append(yRow)


print(matrix(R))

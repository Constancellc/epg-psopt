import win32com.client
import csv

filename = 'IEEE13Nodeckt'

engine=win32com.client.Dispatch("OpenDSSEngine.DSS")
engine.Start("0")

engine.Text.Command='clear'
circuit = engine.ActiveCircuit

engine.text.Command='compile C:\Users/Constance/Documents/13bus/IEEE13Nodeckt.dss'

# comment out loads
engine.text.Command='Export Y'

Y = []
nodeNames = []

n = 0
with open(filename+'_EXP_Y.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row == []:
            continue
        n += 1

        if n <= 10:
            continue

        i = 0

        newRow = []
        nodeNames.append(row[0])

        i += 1

        while i < len(row):
            if row[i] == '':
                break
            newRow.append(complex(float(row[i]),float(row[i+1][4:])))
            i += 2
        
        Y.append(newRow)

print(len(Y[0][44-38:]))
print(len(nodeNames))

print(nodeNames)

with open('Ybus.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for row in Y:
        writer.writerow(row[len(Y[0])-len(nodeNames):])

with open('nodeNames.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for row in nodeNames:
        writer.writerow([row])
# put back loads
'''
engine.Text.Command='compile '+filename+'.dss'
'''

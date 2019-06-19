import os,sys

WD = os.path.dirname(sys.argv[0])

fileName = os.path.join(WD,'ckt7','Loads_ckt7.dss')
alfName = os.path.join(WD,'ckt7','AllocationFactors.dss')
fileOut = os.path.join(WD,'ckt7','Loads_ckt7_1ph.dss')

# nAccuracy=

strIn = ''
alfIn = ''
alfDict = {}
with open(fileName) as file:
    for line in file:
        strIn = strIn+line
with open(alfName) as file:
    for line in file:
        alfIn = alfIn+line
        load = line.split('.')[1].lower()
        af = line.split('=')[-1][:-1]
        alfDict[load] = af
strIn = strIn.replace('\t','  ')
strList = strIn.split('\n')

strOut = ''
for line in strList:
    if len(line)>0:
        # need the loadname for all.
        items = line.split(' ')
        for item in items:
            if 'load.' in item.lower():
                loadName = item[5:].split('.')[0]
        
        if line[0]!='!' and 'Phases=3' in line:
            strOut = strOut + '\n!' + line
            newLine = ''
            for item in items:
                if 'load.' in item.lower():
                    loadName = item[5:].split('.')[0]
                    item = 'Load.' + loadName + '__insertphase_'
                if 'kw' in item.lower():
                    kwVal = item.split('=')[1]
                    kwVal1 = "%.7f" %  (float(kwVal)/3) #accuracy to 0.1 mW
                    item = 'kW='+kwVal1
                if 'kvar' in item.lower():
                    kvar = item.split('=')[1]
                    kvar1 = "%.7f" %  (float(kvar)/3) #accuracy to 0.1 mW
                    item = 'kvar='+kvar1
                if 'xfkva' in item.lower():
                    xfkvaVal = item.split('=')[1]
                    xfkvaVal1 = "%.7f" %  (float(xfkvaVal)/3) #accuracy to 0.1 mW
                    afact = alfDict[loadName.lower()]
                    item = 'xfkVA='+xfkvaVal1+' AllocationFactor='+afact
                if 'bus1' in item.lower():
                    bus = item.split('=')[1]
                    item = bus+'._insertphase_'
                if 'phases=3' in item.lower():
                    item = 'phases=1'
                newLine = newLine + item + ' '
                
            strOut = strOut + '\n' + newLine.replace('_insertphase_','1')
            strOut = strOut + '\n' + newLine.replace('_insertphase_','2')
            strOut = strOut + '\n' + newLine.replace('_insertphase_','3')
            
        else:
            strOut = strOut + '\n' + line + ' AllocationFactor='+alfDict[loadName.lower()]

                
print(strOut)

with open(fileOut,'w') as file:
    file.write(strOut)



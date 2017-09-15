import xlrd

folder = '../../Documents/feeder13/'

linePhases = {}
nodes = {}
lines = []
spotLoads = {}

baseVoltages = {}
transformers = {}


# get transformer info
workbook = xlrd.open_workbook(folder+'Transformer Data.xls')
sheet = workbook.sheet_by_index(0)

for rowx in range(sheet.nrows):
    cols = sheet.row_values(rowx)

    if cols[0] == 'Transformer Data' or cols[0] == '':
        continue

    baseVoltages[cols[0]] = [float(cols[2][0:4])*1000,float(cols[3][0:4])*1000]

print(baseVoltages)
    
# first read in config phase infomation
workbook = xlrd.open_workbook(folder+'config.xls')
sheet = workbook.sheet_by_index(0)

for rowx in range(sheet.nrows):
    cols = sheet.row_values(rowx)
    try:
        config = int(cols[0])
    except:
        continue

    linePhases[str(config)] = [0,0,0]

    for ph in cols[1]:
        if ph == 'A':
            linePhases[str(config)][0] = 1
        elif ph == 'B':
            linePhases[str(config)][1] = 1
        elif ph == 'C':
            linePhases[str(config)][2] = 1
            
workbook = xlrd.open_workbook(folder+'UG config.xls')
sheet = workbook.sheet_by_index(0)

for rowx in range(sheet.nrows):
    cols = sheet.row_values(rowx)
    try:
        config = int(cols[0])
    except:
        continue

    linePhases[str(config)] = [0,0,0]

    for ph in cols[1]:
        if ph == 'A':
            linePhases[str(config)][0] = 1
        elif ph == 'B':
            linePhases[str(config)][1] = 1
        elif ph == 'C':
            linePhases[str(config)][2] = 1

# next get line data
workbook = xlrd.open_workbook(folder+'line data.xls')
sheet = workbook.sheet_by_index(0)

for rowx in range(sheet.nrows):
    cols = sheet.row_values(rowx)

    try:
        nodeA = int(cols[0])
        nodeB = int(cols[1])
        length = float(cols[2])
        config = str(int(cols[3]))
    except:
        # transformer or switch
        continue

    if nodeA not in nodes:
        nodes[nodeA] = [0,0,0]
    if nodeB not in nodes:
        nodes[nodeB] = [0,0,0]

    lines.append([nodeA,nodeB,length,config,linePhases[config]])

    for i in range(0,3):
        if linePhases[config][i] == 1:
            nodes[nodeA][i] = 1
            nodes[nodeB][i] = 1

print(lines)

# assigning base voltages

# start with the knowns
nodes[650][1] = baseVoltages['Substation:'][1]
nodes[633][1] = baseVoltages['XFM-1'][0]
nodes[634][1] = baseVoltages['XFM-1'][1]



# next get spot loads
workbook = xlrd.open_workbook(folder+'spot load data.xls')
sheet = workbook.sheet_by_index(0)

for rowx in range(sheet.nrows):
    cols = sheet.row_values(rowx)

    try:
        node = int(cols[0])
    except:
        continue
    loads = []
    for i in range(2,8):
        loads.append(float(cols[i]))

    spotLoads[node] = loads

# and capacitors - assume negative reactive loads
workbook = xlrd.open_workbook(folder+'cap data.xls')
sheet = workbook.sheet_by_index(0)

for rowx in range(sheet.nrows):
    cols = sheet.row_values(rowx)

    try:
        node = int(cols[0])
    except:
        continue

    if node not in spotLoads:
        spotLoads[node] = [0.0]*6

    for i in range(0,2):
        spotLoads[node][2*i+1] = -1*float(cols[1+i])
        
# and distributed loads - assume seen as spot load half at either end
workbook = xlrd.open_workbook(folder+'distributed load data.xls')
sheet = workbook.sheet_by_index(0)

for rowx in range(sheet.nrows):
    cols = sheet.row_values(rowx)

    try:
        nodeA = int(cols[0])
        nodeB = int(cols[1])
    except:
        continue

    if nodeA not in spotLoads:
        spotLoads[nodeA] = [0.0]*6
        
    if nodeB not in spotLoads:
        spotLoads[nodeB] = [0.0]*6

    for i in range(3,9):
        spotLoads[nodeA][i-3] += float(cols[i])/2
        spotLoads[nodeB][i-3] += float(cols[i])/2

print(spotLoads)
        
        
# and regulator data, if I understood what to do with it
workbook = xlrd.open_workbook(folder+'Regulator Data.xls')
sheet = workbook.sheet_by_index(0)

for rowx in range(sheet.nrows):
    cols = sheet.row_values(rowx)
    print(cols)

# well I think i've proven concept...





# ok, this is assuming that I have the admittance matrix

# y(3i+ph1,3j+ph2) is the admittance from node i phase 1 to node j phase 2


nPhases = 3

lines = [[nodei, nodej, [0,0,0], [r,r,r], [x,x,x]],
         []]
nodes = [[node,[0,0,0]],
         []]

nNodes = len(nodes)

variableList = []

slackBus = [node]

nLineVariables = 0
for line in lines:
    nodei = line[0]
    nodej = line[1]
    for ph in range(0,nPhases):
        if line[2][ph] != 0:
            variableList.append(['power',nodei,nodej,ph,'p',r])
            variableList.append(['power',nodei,nodej,ph,'q',x])
    
    nLineVariables += sum(line[2])*2

nNodeVariables = 0
for node in nodes:
    for ph in range(0,nPhases):
        if node[1][ph] != 0:
            variableList.append(['voltage',node[0],ph])
    nNodeVariables += sum(node[1])

M = nNodeVariables + nLineVariables

A = matrix(0.0,(M,M))
b = matrix(0.0,(M,1))

# first continuity - power in equals power out
for i in range(0,nLineVariables):
    nodei = variableList[i][1]
    nodej = variableList[i][2]
    ph = variableList[i][3]
    kind = variableList[i][4]

    A[i,i] = 1.0 # line flow in question

    # looking for things leaving from j
    for i2 in range(0,nLineVariables):
        if i2 == i:
            continue
        if variableList[i2][1] != nodej:
            continue
        if variableList[i2][3] != ph:
            continue
        if variableList[i2][4] != kind:
            continue

        A[i,i2] = -1.0

    # now add the load
    b[i] = # load for nodej phase and kind

    # now adding resistances

    # looking for node voltages
    for i2 in range(nLineVariables,M):
        
        if variableList[i][1] == nodei:
            if variableList[i][2] != ph:
                continue
            Vi_index = i2
            
        elif variableList[i][1] == nodej:
            if variableList[i][2] != ph:
                continue
            Vj_index = i2

        else:
            continue

    A[Vj_index,Vi_index] = 1.0
    A[Vj_index,Vj_index] = -1.0

    A[Vj_index,i] = variableList[i][5] # r, x

# finally the slack bus
for i in range(nLineVariables,M):
    if variableList[i][1] == slackBus:
        A[i,i] = 1.0
        b[i] = 1.0
    

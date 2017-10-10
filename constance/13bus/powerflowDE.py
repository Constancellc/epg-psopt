from cvxopt import matrix
import numpy as np
# ok, this is assuming that I have the admittance matrix

from ybus import R, X, nodeNames


# first things first, I must convert from ohms to per unit. For this I need
# the vbases if node is 634 vbase is 480, else 4160

Sbase = 5000000 # kW

Zbase = Sbase/(np.power(4160,2))

loads = {'634.1':complex(160000,110000),
         '634.2':complex(120000,90000),
         '634.3':complex(120000,90000),
         '645.2':complex(170000,125000),
         '646.2':complex(230000,132000),
         '652.1':complex(128000,86000),
         '671.1':complex(385000+9000,220000+5000),
         '671.2':complex(385000+33000,220000+19000),
         '671.3':complex(385000+59000,220000+34000),
         '675.1':complex(485000,190000-220000),
         '675.2':complex(68000,60000-200000),
         '675.3':complex(290000,212000-200000),
         '692.3':complex(170000,151000),
         '611.3':complex(170000,80000-100000),
         '632.1':complex(9000,5000),
         '632.2':complex(33000,19000),
         '632.3':complex(59000,34000)}

nNodes = len(nodeNames)
nFlows = 0
flows = []

# converting resistances to per unit
for i in range(0,len(R)):
    nodei = nodeNames[i]
    for j in range(0,len(R[0])):
        nodej = nodeNames[j]


        if j > i:
            if R[i][j] != 0:
                '''
                if nodei[:3] == nodej[:3]:
                    continue
                if nodei[-1] != nodej[-1]:
                    continue
                '''
                flows.append([nodei,nodej])
                nFlows += 1
        
        R[i][j] = R[i][j]/Zbase
        X[i][j] = X[i][j]/Zbase

        print(R[i][j])


A = matrix(0.0,(nNodes+2*nFlows,nNodes+2*nFlows))
b = matrix(0.0,(nNodes+2*nFlows,1))

print(flows)
for i in range(0,3): # source bus
    A[i,i] = 1
    b[i] = 1


for ln in range(0,len(flows)):
    cn = ln+3 # constraint number
    
    nodei = flows[ln][0]
    nodej = flows[ln][1]

    # find vi and vj index
    for k in range(0,len(nodeNames)):
        if nodeNames[k] == nodei:
            i_in = k
        elif nodeNames[k] == nodej:
            j_in = k

    A[cn,i_in] = 1.0
    A[cn,j_in] = -1.0
    A[cn,int(nNodes+2*ln)] = -R[i_in][j_in]
    A[cn,int(nNodes+2*ln+1)] = -X[i_in][j_in]

# now continuity equations
for i in range(3,len(nodeNames)):
    cn += 1

    node = nodeNames[i]

    # looking for lines in
    for l in range(0,len(flows)):
        if flows[l][0] == node:
            # line leaves from node
            A[cn,int(nNodes+2*l)] = 1.0
            A[cn+1,int(nNodes+2*l+1)] = 1.0
            
        elif flows[l][1] == node:
            # line enters node
            A[cn,int(nNodes+2*l)] = -1.0
            A[cn+1,int(nNodes+2*l+1)] = -1.0

    try:
        load = loads[node]
    except:
        load = complex(0,0)
            
    b[cn] = load.real/Sbase
    b[cn+1] = load.imag/Sbase

    cn += 1

print(flows)

sol = np.linalg.solve(A,b)

'''
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

# let x contian the solution

# first 
    
'''

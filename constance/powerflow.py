from cvxopt import matrix, spdiag, solvers, sparse
import numpy as np

from test13 import nodes, lines, spotLoads, Z, transformers, slackbus, nodeVoltageBases

Pbase = 500 # kW
Qbase = 500 # kVa

N = len(nodes) # number of nodes
L = len(lines) # number of lines

lineVariables = []
n = 0
for i in range(0,len(lines)):
    line = lines[i]
    if i == 0:
        nPhases = len(line[4])
    for ph in range(0,nPhases):
        if line[4][ph] == 0:
            continue
        lineVariables.append([i,ph,'p',n])
        n += 1
        lineVariables.append([i,ph,'q',n])
        n += 1


nodeVariables = []
slackVariables = []
i = 0
for node in nodes:
    for ph in range(0,nPhases):
        if nodes[node][ph] == 0:
            continue
        nodeVariables.append([i,node,ph])
        if node == slackbus[0]:
            slackVariables.append(i)
        i += 1


R = []
X = []

for line in lines:
    length = float(line[2])*0.00018939 # ft -> miles
    config = line[3]

    Zbase = np.power(nodeVoltageBases[lines[0]],2)/Pbase

    r = []
    x = []

    if config in transformers:
        for ph in range(0,nPhases):
            if line[4][ph] == 0:
                r.append(0.0)
                x.append(0.0)
            else:
                r.append(transformers[config][0])
                x.append(transformers[config][1])

    else:
        for ph in range(0,nPhases):
            r.append((length*Z[config][ph][0].real)/Zbase)
            x.append((length*Z[config][ph][0].imag)/Zbase)

    R.append(r)
    X.append(x)
    
M = len(nodeVariables)+len(lineVariables)

A = matrix(0.0,(M,M))
b = matrix(0.0,(M,1))

for variable in lineVariables:
    pq_index = variable[3]
    line = lines[variable[0]]
    ph = variable[1]
    kind = variable[2]

    nodei = line[0]
    nodej = line[1]

    A[pq_index,pq_index] = 1.0 # line flow in question

    # now look for things connected to j
    for variable2 in lineVariables:
        if lines[variable2[0]][0] != nodej:
            continue

        if variable[1] != variable2[1]:
            continue

        if variable[2] != variable2[2]:
            continue

        if variable[0] == variable2[0]:
            continue

        # so variable 2 leaves from j and is the same phase and type as variable
        A[pq_index,variable2[3]] = -1.0

    # lastly look for real loads at j

    if kind == 'p':
        offset = 0
    elif kind == 'q':
        offset = 1
        
    try:
        load = float(spotLoads[nodej][ph*2+offset])
    except:
        load = 0.0

    b[pq_index] = load

    r = R[variable[0]][ph]
    x = R[variable[0]][ph]

    for node in nodeVariables:
        if node[2] != ph:
            continue

        if node[1] == nodei:
            Vi_index = node[0]
        elif node[1] == nodej:
            Vj_index = node[0]

    if kind == 'p':           
        A[len(lineVariables)+Vj_index,len(lineVariables)+Vi_index] = 1.0 # Vi   
        A[len(lineVariables)+Vj_index,len(lineVariables)+Vj_index] = 1.0 # Vj
        
        A[len(lineVariables)+Vj_index,pq_index] = r # Pij  

    else:
        A[len(lineVariables)+Vj_index,pq_index] = x # Qij
     
# and the slack bus
for V_index in slackVariables:
    A[len(lineVariables)+V_index,len(lineVariables)+V_index] = 1.0
    b[len(lineVariables)+V_index] = float(slackbus[1])
    

   
sol = np.linalg.solve(A,b)
print(sol)

'''
print 'PRINTING LINE POWERS'
print ''
print('Node A   Node B          Real Power               Reactive Power')
print('------   ------    ----------------------     ----------------------')
print('                   Ph 1  |  Ph 2  |  Ph 3     Ph 1  |  Ph 2  |  Ph 3')

for l in range(0,len(lines)):
    line = lines[l]
    print ' ',
    print(line[0],end='')
    print '   ',
    print(line[1],end='')
    print '    ',
    for ph in range(0,3):
        realPower = None
        for variable in lineVariables:
            if variable[0] != l:
                continue
            if variable[1] != ph:
                continue
            if variable[2] != 'p':
                continue
            realPower = float(int(sol[variable[3]][0]/100))/100
        if realPower == None:
            print ' -- ',
        else:
            print realPower,
        if realPower > 100:
            print ' ',
        else:
            print '  ',

    for ph in range(0,3):
        reactivePower = None
        for variable in lineVariables:
            if variable[0] != l:
                continue
            if variable[1] != ph:
                continue
            if variable[2] != 'p':
                continue
            reactivePower = float(int(sol[variable[3]][0]/100))/100
        if reactivePower == None:
            print ' --- ',
        else:
            print reactivePower,
        if ph == 2:
            print ''
        else:
            if reactivePower > 100:
                print ' ',
            else:
                print '  ',
print ''
print ''

print 'PRINTING VOLTAGES'
print ''
print '  Node            Phase 1            Phase 2            Phase 3'
print '  ----            -------            -------            -------'
for node in nodes:
    print '  ',
    print node,
    print '           ',
    for ph in range(0,3):
        voltage = None
        for variable in nodeVariables:
            if variable[1] != node:
                continue
            if variable[2] != ph:
                continue
            voltage = float(int(sol[len(lineVariables)+variable[0]][0]/100))/100
        if voltage == None:
            print ' -- ',
        else:
            print voltage,
        if voltage > 100:
            print '         ',
        else:
            print '          ',
    print ''
'''
'''
linesPower = []
nodeVoltages = []

c = 0
for i in range(0,L):
    line = []
    for ii in range(0,2*nPhases):
        line.append(sol[c][0])
        c += 1
    linesPower.append(line)
for i in range(L,L+N):
    node = []
    for ii in range(0,nPhases):
        node.append(sol[c][0])
        c += 1
    nodeVoltages.append(node)

linesPower = matrix(linesPower)
print linesPower.T
nodeVoltages = matrix(nodeVoltages)
print nodeVoltages.T
'''



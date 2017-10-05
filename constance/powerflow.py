from cvxopt import matrix, spdiag, solvers, sparse
import matplotlib.pyplot as plt
import numpy as np

from test13 import nodes, lines, spotLoads, Z, transformers, slackbus, nodeVoltageBases, voltageSolutions

# converting Z into symmetrical matrix, 3 phase case
for config in Z:
    Z[config][1] = Z[config][0][1] + Z[config][1]
    Z[config][2] = Z[config][0][2] + Z[config][1][2] + Z[config][2]
    
Pbase = 7000000 # W
Qbase = 7000000 # Va

N = len(nodes) # number of nodes
L = len(lines) # number of lines

plotNodes = {}

i = 1
for node in nodes:
    plotNodes[node] = i
    i += 1
    
lineVariables = []
n = 0
for i in range(0,len(lines)):
    line = lines[i] #[nodea, nodeb, length(ft), config, [phases]]
    if i == 0:
        nPhases = len(line[4])
    for ph in range(0,nPhases):
        if line[4][ph] == 0: 
            continue
        # n is the variable number, i the line number and ph the phase number
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
            # store the index of the slack bus variables
            slackVariables.append(i) 
        i += 1


R = []
X = []

R2 = {}

for i in range(0,len(lines)):
    line = lines[i]

    if i not in R2:
        R2[i] = {}
        X2[i] = {}
    
    length = float(line[2])*0.00018939 # ft -> miles
    config = line[3]

    Zbase = np.power(nodeVoltageBases[line[0]],2)/Pbase

    r = []
    x = []

    if config in transformers:
        for ph in range(0,nPhases):
            
            if line[4][ph] == 0:
                
                R2[i][ph] = [0.0]*nPhases
                X2[i][ph] = [0.0]*nPhases
                r.append(0.0)
                x.append(0.0)
            else:
                R2[i][ph] = [0.0]*nPhases
                X2[i][ph] = [0.0]*nPhases

                R2[i][ph][ph] = transformers[config][0]
                X2[i][ph][ph] = transformers[config][1]
                r.append(transformers[config][0])
                x.append(transformers[config][1])

    else:
        for ph in range(0,nPhases):
            R2[i][ph] = [0.0]*nPhases

            for j in range(0,nPhases):
                R2[i][ph][j] = (length*Z[config][ph][j].real)/Zbase
                X2[i][ph][j] = (length*Z[config][ph][j].real)/Zbase
                
            r.append((length*Z[config][ph][0].real)/Zbase)
            x.append((length*Z[config][ph][0].imag)/Zbase)

    R.append(r)
    X.append(x)

print(R)
print(X)
    
M = len(nodeVariables)+len(lineVariables)

A = matrix(0.0,(M,M))
b = matrix(0.0,(M,1))

for variable in lineVariables:
    #pq_index = variable[3]
    pq_index = [0]*nPhases
    # I need to also find the pq_index of the other phases on that line

    line = lines[variable[0]]
    ph = variable[1]
    kind = variable[2]

    nodei = line[0]
    nodej = line[1]

    #A[pq_index,pq_index] = 1.0 # line flow in question


    # now look for things leaving from node j
    for variable2 in lineVariables:

        # looking for pq_index of all phases on the line
        if lines[variable2[0]][0] == nodei and lines[variable2[0]][1] == nodej\
           and variable[2] == variable[2][2]:
            pq_index[variable2[1]] = variable2[3]
            
        if lines[variable2[0]][0] != nodej:
            continue

        if variable[2] != variable2[2]: # if not the same flow kind
            continue

        if variable[1] != variable2[1]: # if not the same phase
            continue


        # so variable 2 leaves from j and is the same phase and type as variable
        A[pq_index[ph],variable2[3]] = -1.0

    A[pq_index[ph],pq_index[ph]] = 1.0 # line flow in question

    # lastly look for real loads at j
    if kind == 'p':
        offset = 0
    elif kind == 'q':
        offset = 1
        
    try:
        load = float(spotLoads[nodej][ph*2+offset])/Pbase
    except:
        load = 0.0

    b[pq_index] = load

    r = R[variable[0]][ph]
    x = R[variable[0]][ph]

    ra = R[variable[0]][ph][0]
    rb = R[variable[0]][ph][1]
    rc = R[variable[0]][ph][2]
    

    xa = X[variable[0]][ph][0]
    xb = X[variable[0]][ph][1]
    xc = X[variable[0]][ph][2]
    
    for node in nodeVariables:
        if node[2] != ph:
            continue

        if node[1] == nodei:
            Vi_index = node[0]
        elif node[1] == nodej:
            Vj_index = node[0]

    # not convinced about use of vj index

    if kind == 'p':           
        A[len(lineVariables)+Vj_index,len(lineVariables)+Vi_index] = 1.0 # Vi   
        A[len(lineVariables)+Vj_index,len(lineVariables)+Vj_index] = -1.0 # Vj
        
        A[len(lineVariables)+Vj_index,pq_index] = -r # Pij  

    else:
        A[len(lineVariables)+Vj_index,pq_index] = -x # Qij
     
# and the slack bus
for V_index in slackVariables:
    A[len(lineVariables)+V_index,len(lineVariables)+V_index] = 1.0
    b[len(lineVariables)+V_index] = 1.0
   
sol = np.linalg.solve(A,b)
#print(sol)

phs = {0:'A',1:'B',2:'C'}

plotVoltages = {0:{'x':[],'y':[]},1:{'x':[],'y':[]},2:{'x':[],'y':[]}}
plotSolutions = {0:[],1:[],2:[]}

print('PRINTING VOLTAGES')
for i in range(0,len(nodeVariables)):
    print('NODE:'+str(nodeVariables[i][1])+' PHASE:'+phs[nodeVariables[i][2]],end='  ')
#    print(str(int(sol[len(lineVariables)+i]*nodeVoltageBases[nodeVariables[i][1]]))+'V')
    print(str(int(sol[len(lineVariables)+i]*1000)/1000))
    plotVoltages[nodeVariables[i][2]]['x'].append(plotNodes[nodeVariables[i][1]])
    plotVoltages[nodeVariables[i][2]]['y'].append(int(sol[len(lineVariables)+i]*1000)/1000)
    plotSolutions[nodeVariables[i][2]].append(voltageSolutions[nodeVariables[i][1]][nodeVariables[i][2]])
    

'''
for variable in lineVariables:
    pq_index = variable[3]
    line = lines[variable[0]]
    ph = variable[1]
    kind = variable[2]

    nodei = line[0]
    nodej = line[1]
    
    print(str(nodei)+'->'+str(nodej)+' PHASE:'+phs[ph],end='  ')
    print(str(int(sol[pq_index]*Pbase)/1000),end='')
    if kind == 'p':
        print(' kW')
    else:
        print(' kVa')
'''
plt.figure(1)
for i in range(0,3):
    plt.subplot(3,1,i+1)
    plt.scatter(plotVoltages[i]['x'],plotVoltages[i]['y'],label='approx')
    plt.scatter(plotVoltages[i]['x'],plotSolutions[i],label='actual')
    plt.grid()
    plt.ylabel('Voltage (p.u.)')
    plt.title('Phase '+phs[nodeVariables[i][2]],y=0.7)
    plt.ylim(0.9,1.1)
plt.legend()
plt.xlabel('Node #')
plt.show()
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



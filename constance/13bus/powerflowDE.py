from cvxopt import matrix
import numpy as np
import matplotlib.pyplot as plt
# ok, this is assuming that I have the admittance matrix

from ybus import R, X, nodeNames


# first things first, I must convert from ohms to per unit. For this I need
# the vbases if node is 634 vbase is 480, else 4160

Sbase = 5000000 # kW

Zbase = np.power(4160,2)/Sbase

loads = {'634.1':complex(160000,110000),
         '634.2':complex(120000,90000),
         '634.3':complex(120000,90000),
         '645.2':complex(170000,125000),
         '646.2':complex(230000,132000),
         '652.1':complex(128000,86000),
         '671.1':complex(385000,220000),
         '671.2':complex(385000,220000),
         '671.3':complex(385000,220000),
         '675.1':complex(485000,190000-220000),
         '675.2':complex(68000,60000-200000),
         '675.3':complex(290000,212000-200000),
         '692.3':complex(170000,151000),
         '611.3':complex(170000,80000-100000),
         '670.1':complex(18000,10000),
         '670.2':complex(66000,38000),
         '670.3':complex(118000,68000)}

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
                
                if nodei[:3] == nodej[:3]:
                    continue
                if nodei[-1] != nodej[-1]:
                    continue
                
                flows.append([nodei,nodej])
                nFlows += 1
        
        R[i][j] = R[i][j]/Zbase
        X[i][j] = X[i][j]/Zbase


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
            A[cn,int(nNodes+2*l)] = -1.0
            A[cn+1,int(nNodes+2*l+1)] = -1.0
            
        elif flows[l][1] == node:
            # line enters node
            A[cn,int(nNodes+2*l)] = 1.0
            A[cn+1,int(nNodes+2*l+1)] = 1.0

    try:
        load = loads[node]
    except:
        load = complex(0,0)
            
    b[cn] = load.real/Sbase
    b[cn+1] = load.imag/Sbase

    cn += 1

print(nodeNames)

sol = np.linalg.solve(A,b)

print(sol)

V = sol[:len(nodeNames)]

vSols = {'RG60.1':1.0,
         'RG60.2':1.0,
         'RG60.3':1.0,
         '632.1':1.021,
         '632.2':1.042,
         '632.3':1.0687,
         '671.1':0.99,
         '671.2':1.0529,
         '671.3':1.0174,
         '680.1':0.99,
         '680.2':1.0529,
         '680.3':0.9778,
         '633.1':1.018,
         '633.2':1.0401,
         '633.3':1.0174,
         '645.3':1.0155,
         '645.2':1.0329,
         '646.3':1.0134,
         '646.2':1.0311,
         '692.1':0.99,
         '692.2':1.0529,
         '692.3':0.9777,
         '675.1':0.9835,
         '675.2':1.0553,
         '675.3':0.9758,
         '684.1':0.9881,
         '684.3':0.9758,
         '611.3':0.9738,
         '652.1':0.9825,
         '634.1':0.994,
         '634.2':1.0218,
         '634.3':0.9960}

est = []
tru = []
x_ticks = []

for i in range(0,3):
    est.append([])
    tru.append([])
    x_ticks.append([])

for i in range(0,len(nodeNames)):
    try:
        sol = vSols[nodeNames[i]]
    except:
        continue

    ph = int(nodeNames[i][-1])-1
    node = nodeNames[i][:3]

    est[ph].append(V[i])
    tru[ph].append(sol)
    x_ticks[ph].append(node)

'''
for ph in range(0,3):
    scale = tru[ph][1]
    for i in range(0,len(tru[ph])):
        tru[ph][i] = tru[ph][i]/scale
'''

plt.figure(1)
for ph in range(0,3):
    plt.subplot(3,1,ph+1)
    plt.plot(range(0,len(est[ph])),est[ph])
    plt.plot(range(0,len(est[ph])),tru[ph])
    plt.xticks(range(0,len(est[ph])),x_ticks[ph])
plt.show()

# I think the problem with the non decoupled might be in the continuity equations with things flowing when not physically connected

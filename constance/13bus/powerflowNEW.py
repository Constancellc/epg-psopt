from cvxopt import matrix, spdiag, solvers, sparse
import numpy as np
import matplotlib.pyplot as plt
import csv
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

sourceBus = ['RG60.1','RG60.2','RG60.3']
nNodes = len(nodeNames)
nFlows = 0
flows = []

print(nodeNames)
# converting resistances to per unit
for i in range(0,len(R)):
    nodei = nodeNames[i]
    for j in range(0,len(R[0])):
        nodej = nodeNames[j]

        if j > i:
            if R[i][j] != 0:
                
                if nodei[:3] == nodej[:3]:
                    continue
                '''
                if nodei[-1] != nodej[-1]:
                    continue
                '''
                flows.append([nodei,nodej])
                nFlows += 1
        
        R[i][j] = R[i][j]/Zbase
        X[i][j] = X[i][j]/Zbase

print(flows)
A1 = matrix(0.0,(nFlows,int(nNodes+2*nFlows)))
A2 = matrix(0.0,(int(2*(nNodes)-len(sourceBus)),int(nNodes+2*nFlows)))
b1 = matrix(0.0,(nFlows,1))
b2 = matrix(0.0,(int(2*(nNodes)-len(sourceBus)),1))


# voltage drop
for ln in range(0,len(flows)):
            
    nodei = flows[ln][0]
    nodej = flows[ln][1]

    # find vi and vj index
    for k in range(0,len(nodeNames)):
        if nodeNames[k] == nodei:
            i_in = k
        elif nodeNames[k] == nodej:
            j_in = k

    A1[ln,i_in] = 1.0
    A1[ln,j_in] = -1.0
    A1[ln,int(nNodes+2*ln)] = -R[i_in][j_in]
    A1[ln,int(nNodes+2*ln+1)] = -X[i_in][j_in]


cn = 0
# continuity
for i in range(len(nodeNames)):
    node = nodeNames[i]
    if node in sourceBus:
        A2[cn,i] = 1.0
        b2[cn] = 1.0
        cn += 1
        continue

    try:
        ld = loads[node]
    except:
        ld = complex(0.0,0.0)
    
    for l in range(len(flows)):
        nodei = flows[l][0]
        nodej = flows[l][1]
        if nodei == node:
            A2[cn,int(nNodes+2*l)] = -1.0
            A2[cn+1,int(nNodes+2*l+1)] = -1.0
        elif nodej == node:
            A2[cn,int(nNodes+2*l)] = 1.0
            A2[cn+1,int(nNodes+2*l+1)] = 1.0

    b2[cn] = ld.real/Sbase
    b2[cn+1] = ld.imag/Sbase
            
    cn += 2


A_ = sparse([A1,A2])
b_ = sparse([b1,b2])

print(A_.size)
print(b_.size)

with open('A.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(A_.size[0]):
        row = []
        for j in range(A_.size[1]):
            row.append(A_[i,j])
        writer.writerow(row)
        
q = matrix(-1*A_.T*b_)
P = A_.T*A_

G1 = sparse([[matrix(0.0,(nNodes-3,3))],[spdiag([-1]*(nNodes-3))],[matrix(0.0,(nNodes-3,2*nFlows))]])
G2 = sparse([[matrix(0.0,(nNodes-3,3))],[spdiag([1]*(nNodes-3))],[matrix(0.0,(nNodes-3,2*nFlows))]])
h1 = matrix(-0.8,(nNodes-3,1))
h2 = matrix(1.2,(nNodes-3,1))

G = sparse([G1,G2])
h = matrix(sparse([h1,h2]))
print(G.size)
print(h.size)
sol = solvers.qp(P,q,G,h,A2,b2) # solve quadratic program
X = sol['x']

vEst = X[:nNodes]

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

tru = []
est = []

for i in range(0,len(nodeNames)):
    try:
        tru.append(vSols[nodeNames[i]])
    except:
        continue
    est.append(vEst[i])

plt.figure(1)
plt.scatter(range(len(est)),tru)
plt.scatter(range(len(est)),est)
plt.show()

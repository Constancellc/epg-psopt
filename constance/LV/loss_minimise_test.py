import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers
import random

Pmax = 6 # kW
pph = 60
T = 24*pph
t_int = int(60/pph)

# okay so this script is going to calculate the optimal losses for some false
# energy requirements

# first I need to set up the energy demands
energy = [] # kWh
for hh in range(55):
    energy.append(100*random.random())

# then I need to acquire the losses model
P0 = matrix(0.0,(55,55))
with open('P.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        for j in range(len(row)):
            P0[i,j] += float(row[j])
        i += 1

P = spdiag([P0]*T)

q0 = []
with open('q.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        q0.append(float(row[0]))

q = []
for t in range(T):
    q += q0
q = matrix(q)
    
# then I want to estimate the losses for load flattening
x_lf = [0.0]*55*T

for hh in range(55):
    for t in range(T):
        x_lf[t*55+hh] = -energy[hh]*1000/24
        
x_lf = matrix(x_lf)

losses_lf = x_lf.T*P*x_lf + q.T*x_lf
# then I want to find the optimal losses
A = matrix(0.0,(55,55*T))
b = matrix(0.0,(55,1))
        
for j in range(55):
    for t in range(T):
        A[j,55*t+j] = 1.0/pph # energy requirement

    b[j] = -energy[j]*1000

G = sparse([spdiag([-1.0]*(55*T)),spdiag([1.0]*(55*T))])
h = matrix([Pmax*1000]*(55*T)+[0.0]*(55*T))

sol=solvers.qp(P,q,G,h,A,b)
x = sol['x']
losses_lm = x.T*P*x + q.T*x
print(losses_lf[0])
print(losses_lm[0])

av_lm = [0.0]*T
av_lf = [sum(energy)/24]*T

for i in range(55):
    for t in range(T):
        av_lm[t] -= x[55*t+i]/1000

plt.figure(1)
plt.plot(av_lm)
plt.plot(av_lf)
plt.show()

# finally I should store and / or plot the results

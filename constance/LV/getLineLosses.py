import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from lv_optimization_new import LVTestFeeder
import pickle

#outfile = '../../../Documents/simulation_results/LV/voltages.csv'
stem = '../../../Documents/ccModels/eulv/'
alpha = 0.328684513701

g = open(stem+'lnsYprims.pkl','rb')
data = pickle.load(g)
g.close()


# data is a dictionary where the key is the line number and it points to
# [bus a, bus b, Yprim]

# so we need to build up a dictionary of the voltages 

a = np.load(stem+'eulvLptaCc060.npy')
My = np.load(stem+'eulvLptMyCc060.npy')
v0 = np.load(stem+'eulvLptV0Cc060.npy')
Y = np.load(stem+'eulvLptYbusCc060.npy')
Y = Y.flatten()[0]
Y = Y.conj()
YNodeOrder = np.load(stem+'eulvNmtYNodeOrderCc060.npy')
buses = []
for node in YNodeOrder:
     buses = buses+[node.split('.')[0]]

def get_losses(Vtot):
    losses = {}
    for line in data:
        data0 = data[line]

        bus1 = data0[0]
        bus2 = data0[1]
        Yprim = data0[2]
        
        idx1 = [i for i, x in enumerate(buses) if x == bus1]
        idx2 = [i for i, x in enumerate(buses) if x == bus2]

        Vidx = Vtot[idx1+idx2]
        Iphs = Yprim.dot(Vidx)
        Sinj = Vidx*(Iphs.conj())
        Sloss = sum(Sinj)

        losses[line] = [bus1,bus2,Sloss.real]
    return losses
     
fdr = LVTestFeeder('manc_models/1',1)
fdr.set_households_NR('../../../Documents/netrev/TC2a/03-Dec-2013.csv')
fdr.set_evs_MEA('../../../Documents/My_Electric_Avenue_Technical_Data/'+
                'constance/ST1charges/')

voltages = fdr.get_all_voltages(My,a,alpha,v0)
losses_no_evs = {}
print(fdr.predict_losses())
for t in voltages:
    ls = get_losses(voltages[t])
    for l in ls:
        if l not in losses_no_evs:
            losses_no_evs[l] = 0
        losses_no_evs[l] += ls[l][2]

fdr.uncontrolled()
voltages = fdr.get_all_voltages(My,a,alpha,v0)
losses_unc = {}
print(fdr.predict_losses())
for t in voltages:
    ls = get_losses(voltages[t])
    for l in ls:
        if l not in losses_unc:
            losses_unc[l] = 0
        losses_unc[l] += ls[l][2]

fdr.load_flatten()
voltages = fdr.get_all_voltages(My,a,alpha,v0)
losses_lf = {}
print(fdr.predict_losses())
for t in voltages:
    ls = get_losses(voltages[t])
    for l in ls:
        if l not in losses_lf:
            losses_lf[l] = 0
        losses_lf[l] += ls[l][2]

fdr.loss_minimise()
voltages = fdr.get_all_voltages(My,a,alpha,v0)
losses_lm = {}
print(fdr.predict_losses())
for t in voltages:
    ls = get_losses(voltages[t])
    for l in ls:
        if l not in losses_lm:
            losses_lm[l] = 0
        losses_lm[l] += ls[l][2]

with open('lv test/branch_losses.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['line','no evs','unc','lf','lm'])
    for l in losses_unc:
        writer.writerow([l,losses_no_evs[l],losses_unc[l],losses_lf[l],
                         losses_lm[l]])

'''
busV = {}
for i in range(907):
    busV[i+1] = [complex(0,0)]*3

for i in range(3):
    busV[1][i] = v0[i]

for i in range(len(voltages)):
    bn = int(i/3)+2
    pn = i%3
    busV[bn][pn] = voltages[i]

lineI = {}
for l in data:
    b1 = data[l][0]
    b2 = data[l][1]
    Yp = data[l][2] 
    v_ = np.hstack((busV[int(b1)],busV[int(b2)]))
    i = np.matmul(Yp,v_)[:3]

    iT = 0
    for ii in range(3):
        iT += abs(i[ii]/1000)
    lineI[l] = iT

with open('lv test/no_evs.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for l in lineI:
        writer.writerow([l,lineI[l]])

busV = {}
for i in range(907):
    busV[i+1] = [complex(0,0)]*3

for i in range(3):
    busV[1][i] = v0[i]

for i in range(len(voltages)):
    bn = int(i/3)+2
    pn = i%3
    busV[bn][pn] = voltages[i]

lineI = {}
for l in data:
    b1 = data[l][0]
    b2 = data[l][1]
    Yp = data[l][2] 
    v_ = np.hstack((busV[int(b1)],busV[int(b2)]))
    i = np.matmul(Yp,v_)[:3]

    iT = 0
    for ii in range(3):
        iT += abs(i[ii]/1000)
    lineI[l] = iT

with open('lv test/uncontrolled.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for l in lineI:
        writer.writerow([l,lineI[l]])
busV = {}
for i in range(907):
    busV[i+1] = [complex(0,0)]*3

for i in range(3):
    busV[1][i] = v0[i]

for i in range(len(voltages)):
    bn = int(i/3)+2
    pn = i%3
    busV[bn][pn] = voltages[i]

lineI = {}
for l in data:
    b1 = data[l][0]
    b2 = data[l][1]
    Yp = data[l][2] 
    v_ = np.hstack((busV[int(b1)],busV[int(b2)]))
    i = np.matmul(Yp,v_)[:3]

    iT = 0
    for ii in range(3):
        iT += abs(i[ii]/1000)
    lineI[l] = iT

with open('lv test/lf.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for l in lineI:
        writer.writerow([l,lineI[l]])


busV = {}
for i in range(907):
    busV[i+1] = [complex(0,0)]*3

for i in range(3):
    busV[1][i] = v0[i]

for i in range(len(voltages)):
    bn = int(i/3)+2
    pn = i%3
    busV[bn][pn] = voltages[i]

lineI = {}
for l in data:
    b1 = data[l][0]
    b2 = data[l][1]
    Yp = data[l][2] 
    v_ = np.hstack((busV[int(b1)],busV[int(b2)]))
    i = np.matmul(Yp,v_)[:3]

    iT = 0
    for ii in range(3):
        iT += abs(i[ii]/1000)
    lineI[l] = iT

with open('lv test/lm.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for l in lineI:
        writer.writerow([l,lineI[l]])
    
    

# now I need to work out the line flows from the current injections
'''

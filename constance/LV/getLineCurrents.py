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

fdr = LVTestFeeder('manc_models/1',1)
fdr.set_households_NR('../../../Documents/netrev/TC2a/03-Dec-2013.csv')
fdr.set_evs_MEA('../../../Documents/My_Electric_Avenue_Technical_Data/'+
                'constance/ST1charges/')

voltages = fdr.get_average_voltages(My,a,alpha)

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

fdr.uncontrolled()
voltages = fdr.get_average_voltages(My,a,alpha)
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

fdr.load_flatten()
voltages = fdr.get_average_voltages(My,a,alpha)
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

fdr.loss_minimise()
voltages = fdr.get_average_voltages(My,a,alpha)
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

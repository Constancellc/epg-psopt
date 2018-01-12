import csv
import numpy as np
from decimal import *

folder = '../../../Documents/LV_LPF/'

hh_nodes = []

n = -3
with open(folder+'sY.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[0] != '0':
            hh_nodes.append(n)
            hh_nodes.append(n+2718)
        n += 1

a_r = np.empty((2721,1))
a_i = np.empty((2721,1))

with open(folder+'a_r.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        a_r[i][0] = Decimal(row[0])
        i += 1
#        a_r.append(float(row[0]))
        
with open(folder+'a_i.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        a_i[i][0] = Decimal(row[0])
        i += 1

#a_r = np.array(a_r)
#a_i = np.array(a_i)
               
My_r = np.empty((2721,110))
My_i = np.empty((2721,110))

with open(folder+'My_r.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        j = 0
        for node in range(len(row)):
            if node not in hh_nodes:
                continue
            My_r[i][j] = Decimal(row[node])
            j += 1
        i += 1


with open(folder+'My_i.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        j = 0
        for node in range(len(row)):
            if node not in hh_nodes:
                continue
            My_i[i][j] = Decimal(row[node])
            j += 1
        i += 1

#My_r = np.array(My_r)
#My_i = np.array(My_i)

Y_r = np.empty((2721,2721))
Y_i = np.empty((2721,2721))

with open(folder+'Y_r.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        for j in range(len(row)):
            Y_r[i][j] = Decimal(row[j])
        i += 1

with open(folder+'Y_i.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        for j in range(len(row)):
            Y_i[i][j] = Decimal(row[j])
        i += 1
'''
Y_r = np.array(Y_r)
Y_i = np.array(Y_i)
'''
v0_r = []
v0_i = []

with open(folder+'v0_r.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        v0_r.append(Decimal(row[0]))

with open(folder+'v0_i.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        v0_i.append(Decimal(row[0]))



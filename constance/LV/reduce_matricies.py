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

a_r = []
a_i = []

with open(folder+'a_r.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a_r.append(Decimal(row[0]))
#        a_r.append(float(row[0]))
        
with open(folder+'a_i.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a_i.append(Decimal(row[0]))

a_r = np.array(a_r)
a_i = np.array(a_i)
               
My_r = []
My_i = []

with open(folder+'My_r.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        new = []
        for i in range(len(row)):
            if i not in hh_nodes:
                continue
            new.append(Decimal(row[i]))
        My_r.append(new)


with open(folder+'My_i.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        new = []
        for i in range(len(row)):
            if i not in hh_nodes:
                continue
            new.append(Decimal(row[i]))
        My_i.append(new)

My_r = np.array(My_r)
My_i = np.array(My_i)

Y_r = []
Y_i = []

with open(folder+'Y_r.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        new = []
        for cell in row:
            new.append(Decimal(cell))
        Y_r.append(new)

with open(folder+'Y_i.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        new = []
        for cell in row:
            new.append(Decimal(cell))
        Y_i.append(new)

Y_r = np.array(Y_r)
Y_i = np.array(Y_i)

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

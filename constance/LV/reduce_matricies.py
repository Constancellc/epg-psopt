import csv
import numpy as np

folder = '../../../Documents/LV_LPF/'

hh_nodes = [2,33,44]

My_r = []
My_i = []

r = 0
with open(folder+'My.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        
        real = []
        imag = []

        for i in range(len(row)):

            if i not in hh_nodes:
                continue
            '''
            elif i > 2718: # as no imag loads - not true if assuming pf != 1
                continue
            '''
            c = 0
            while row[i][c] != 'e':
                c += 1
            c += 4

            reNum = float(row[i][:c])
            imNum = float(row[i][c:-1])

            real.append(reNum)
            imag.append(imNum)

        My_r.append(real)
        My_i.append(imag)

My_r = np.array(My_r)
My_i = np.array(My_i)

Y_r = []
Y_i = []

with open(folder+'Ybus.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        
        real = []
        imag = []

        for i in range(len(row)):
            
            c = 0
            while row[i][c] != 'e':
                c += 1
            c += 4

            reNum = float(row[i][:c])
            imNum = float(row[i][c:-1])

            real.append(reNum)
            imag.append(imNum)

        Y_r.append(real)
        Y_i.append(imag)

Y_r = np.array(Y_r)
Y_i = np.array(Y_i)
#

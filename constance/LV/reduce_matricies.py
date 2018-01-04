import csv
import numpy as np

folder = '../../../Documents/LV_LPF/'

hh_nodes = []

n = -3
with open(folder+'Sy.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[0] != '0':
            hh_nodes.append(n)
        n += 1

a_r = []
a_i = []
with open(folder+'a.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        c = 1
        while row[0][c] not in ['+','-']:
            c += 1
        if row[0][c-1] == 'e':
            c += 1
            while row[0][c] not in ['+','-']:
                c += 1

        a_r.append(float(row[0][:c]))
        a_i.append(float(row[0][c:-1]))

#a = a_r+a_i
            
print(hh_nodes)
        
My_r = []
My_i = []
'''
vs_r = []
vs_i = []
with open(folder+'v0.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        c = 1
        while row[0][c] not in ['+','-']:
            c += 1
        if row[0][c-1] == 'e':
            c += 1
            while row[0][c] not in ['+','-']:
                c += 1

        vs_r.append(float(row[0][:c]))
        vs_i.append(float(row[0][c:-1]))
'''       
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
            c = 1
            while row[i][c] not in ['+','-']:
                c += 1
            if row[i][c-1] == 'e':
                c += 1
                while row[i][c] not in ['+','-']:
                    c += 1

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

n = 0
with open(folder+'Ybus.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        #'''
        if n < 3:
            n += 1
            continue
        #'''
        
        real = []
        imag = []

        for i in range(len(row)):
            #'''
            if i < 3:
                continue
            #'''
            if row[i] == '0':
                reNum = 0
                imNum = 0
                
            else:
                c = 1
                while row[i][c] not in ['+','-']:
                    c += 1
                if row[i][c-1] == 'e':
                    c += 1
                    while row[i][c] not in ['+','-']:
                        c += 1

                reNum = float(row[i][:c])
                imNum = float(row[i][c:-1])

            real.append(reNum)
            imag.append(imNum)

        Y_r.append(real)
        Y_i.append(imag)

Y_r = np.array(Y_r)
Y_i = np.array(Y_i)
#

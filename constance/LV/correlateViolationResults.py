import csv
import random
import copy
import matplotlib.pyplot as plt
import scipy.stats as sts

res_stem = '../../../Documents/simulation_results/LV/LA/'
stem = '../../../Documents/simulation_results/NTS/clustering/power/locationsLA/'
r_type_data = '../../../Documents/census/LA_rural_urban.csv'

e7_data = '../../../Documents/census/e7.csv'
car_data = '../../../Documents/census/cars.csv'
hh_data = '../../../Documents/census/hhSize.csv'
w_data = '../../../Documents/census/dist.csv'

_p = {}
_v = {}
with open('res2.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        _p[row[1]] = float(row[0])
with open('res1.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        _v[row[1]] = float(row[0])


v = []
p = []
la = []

for l in _p:
    la.append(l)
    p.append(_p[l])
    v.append(_v[l])
def rescale(x):
    mn = min(x)
    mx = max(x)
    s = 1/(mx-mn)
    x2 = []
    for i in range(len(x)):
        x2.append(s*(x[i]-mn)-0.5)
    return x2

v = rescale(v)
p = rescale(p)

rType = {}
with open(r_type_data,'rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        rType[row[0]] = float(row[1])

e7 = {}
with open(e7_data,'rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        e7[row[0]] = float(row[1])

cars = {}
with open(car_data,'rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        cars[row[0]] = float(row[1])

hh = {}
with open(hh_data,'rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        hh[row[0]] = float(row[1])

wd = {}
with open(w_data,'rU') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        wd[row[0]] = float(row[1])
r = []
e = []
c = []
h = []
d = []
for l in la:
    r.append(rType[l])
    e.append(e7[l])
    try:
        c.append(cars[l])
    except:
        c.append(1)
    try:
        h.append(hh[l])
    except:
        h.append(2.4)
    try:
        d.append(wd[l])
    except:
        d.append(11)

r = rescale(r)
e = rescale(e)
c = rescale(c)
h = rescale(h)
d = rescale(d)

print('Transformer')
print(sts.pearsonr(r,p))
print(sts.pearsonr(e,p))
print(sts.pearsonr(c,p))
print(sts.pearsonr(h,p))
print(sts.pearsonr(d,p))

print('Voltage')
print(sts.pearsonr(r,v))
print(sts.pearsonr(e,v))
print(sts.pearsonr(c,v))
print(sts.pearsonr(h,v))
print(sts.pearsonr(d,v))


print(sts.pearsonr(p,v))


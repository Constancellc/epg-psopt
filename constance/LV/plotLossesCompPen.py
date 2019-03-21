import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filt

pens = []
for p in range(5,60,5):
    pens.append(str(p))
    
stem = '../../../Documents/simulation_results/LV/varying-pen/'
# losses first
m = [0]
q1 = [0]
q3 = [0]
u = [0]
l = [0]

_m = [0]
_q1 = [0]
_q3 = [0]
_u = [0]
_l = [0]

for p in pens:
    diff = []
    diff2 = []
    with open(stem+p+'-losses.csv','rU') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            diff.append(float(row[2])-float(row[3]))
            diff2.append(float(row[2])-float(row[4]))
            if diff[-1] < 0:
                diff[-1] = 0
            if diff2[-1] < 0:
                diff2[-1] = 0
    diff = sorted(diff)
    diff2 = sorted(diff2)

    m.append(1000*sum(diff)/(len(diff)*int(p)))
    
    #m.append(1000*diff[int(len(diff)/2)]/int(p))
    q1.append(1000*diff[int(len(diff)*0.25)]/int(p))
    q3.append(1000*diff[int(len(diff)*0.75)]/int(p))
    l.append(1000*diff[0]/int(p))
    u.append(1000*diff[-1]/int(p))
    
    _m.append(1000*sum(diff2)/(len(diff)*int(p)))
    
    #_m.append(1000*diff2[int(len(diff)/2)]/int(p))
    _q1.append(1000*diff2[int(len(diff)*0.25)]/int(p))
    _q3.append(1000*diff2[int(len(diff)*0.75)]/int(p))
    _l.append(1000*diff2[0]/int(p))
    _u.append(1000*diff2[-1]/int(p))
'''
m = filt.gaussian_filter1d(m,1)
q1 = filt.gaussian_filter1d(q1,1)
q3 = filt.gaussian_filter1d(q3,1)

_m = filt.gaussian_filter1d(_m,1)
_q1 = filt.gaussian_filter1d(_q1,1)
_q3 = filt.gaussian_filter1d(_q3,1)
'''
x = []
x_ticks = []
for p in range(0,120,20):
    x_ticks.append(str(p)+'%')
    x.append(p*11/100)
plt.figure(figsize=(6,2))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 9
plt.xticks(x,x_ticks)
plt.xlabel('% Households with an EV')

plt.fill_between(range(len(m)),q1,q3,color='#CCCCFF')
plt.plot(m,c='b',label='Loss Minimising')

plt.fill_between(range(len(m)),_q1,_q3,color='#FFCCCC')
plt.plot(_m,c='r',label='LF+Phase Balancing')
plt.xlim(0,11)
plt.ylim(0,35)
plt.legend(loc=2)
plt.ylabel('Losses reduction\n(Wh per household)')
plt.grid(linestyle=':')
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/losses/img/pen_comp.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0)
plt.show()

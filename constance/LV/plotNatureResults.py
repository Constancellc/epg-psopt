import csv
import random
import copy
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cbook
import scipy.stats as st

res_stem = '../../../Documents/simulation_results/LV/LA/'
stem = '../../../Documents/simulation_results/NTS/clustering/power/locationsLA/'
# first i want to pick an example simulation (preferrably a bad one)

la = 'E08000005'

# power demand first 
time = np.arange(0,1440,10)
ttls = ['No Charging','Uncontrolled','Controlled']
xt = ['04:00','12:00','20:00']
xt_ = [240,720,1200]
plt.figure(figsize=(8,3))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 14

res = []
for i in range(9):
    res.append([])
with open(res_stem+la+'_load.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        for i in range(9):
            res[i].append(float(row[i+1]))

for i in range(3):    
    plt.subplot(1,3,i+1)
    plt.plot(time,res[3*i+1],c='g')
    plt.title(ttls[i],y=0.8)
    if i == 0:
        plt.ylabel('Power Demand (kW)')
    else:
        plt.yticks([0,200,400,600],['','','',''])
    plt.fill_between(time,res[3*i],res[3*i+2],color='#CCFFCC')
    plt.ylim(0,600)
    plt.xlim(0,1439)
    plt.grid(ls=':')
    plt.xticks(xt_,xt)
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/Nature/img/lv_power.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0.1)

# this is voltages
plt.figure(figsize=(8,3))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 11
# minimum
res = []
for i in range(9):
    res.append([])
with open(res_stem+la+'_voltages_m.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        for i in range(9):
            res[i].append(float(row[i+1]))
for i in range(3):
    for t in range(len(res[0])):
        res[3*i][t] = res[3*i+1][t]-2*(res[3*i+1][t]-res[3*i][t])/3 
        res[3*i+2][t] = res[3*i+1][t]+2*(res[3*i+2][t]-res[3*i+1][t])/3 
    plt.subplot(1,3,i+1)
    plt.title(ttls[i],y=0.85)
    if i == 0:
        plt.plot(time,res[3*i+1],c='b',label='Minimum')
        plt.ylabel('Power Demand (kW)')
    else:
        plt.plot(time,res[3*i+1],c='b')
        #plt.yticks([0,20,40,60,80,100,120],['','','','','','',''])
    plt.ylim(0.85,1.1)
    plt.fill_between(time,res[3*i],res[3*i+2],color='#CCCCFF')
    plt.xlim(0,1439)
    plt.grid(ls=':')
    plt.xticks(xt_,xt)

# maximum
res = []
for i in range(9):
    res.append([])
with open(res_stem+la+'_voltages_p.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        for i in range(9):
            res[i].append(float(row[i+1]))
for i in range(3):
    for t in range(len(res[0])):
        res[3*i][t] = res[3*i+1][t]-2*(res[3*i+1][t]-res[3*i][t])/3 
        res[3*i+2][t] = res[3*i+1][t]+2*(res[3*i+2][t]-res[3*i+1][t])/3    
    plt.subplot(1,3,i+1)
    if i == 0:
        plt.plot(time,res[3*i+1],c='r',label='Maximum')
        plt.ylabel('Voltage (p.u.)')
    else:
        plt.plot(time,res[3*i+1],c='r')
        #plt.yticks([0,20,40,60,80,100,120],['','','','','','',''])
    plt.ylim(0.85,1.1)
    plt.fill_between(time,res[3*i],res[3*i+2],color='#FFCCCC')
    plt.xlim(0,1439)
    plt.grid(ls=':')
    plt.xticks(xt_,xt)

# average
res = []
for i in range(9):
    res.append([])
with open(res_stem+la+'_voltages_a.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        for i in range(9):
            res[i].append(float(row[i+1]))
for i in range(3):
    for t in range(len(res[0])):
        res[3*i][t] = res[3*i+1][t]-2*(res[3*i+1][t]-res[3*i][t])/3 
        res[3*i+2][t] = res[3*i+1][t]+2*(res[3*i+2][t]-res[3*i+1][t])/3                          
    plt.subplot(1,3,i+1)
    if i == 0:
        plt.plot(time,res[3*i+1],c='k',label='Average')
        plt.legend(loc=3)
    else:
        plt.plot(time,res[3*i+1],c='k')
        #plt.yticks([0,20,40,60,80,100,120],['','','','','','',''])
    plt.ylim(0.85,1.1)
    plt.xlim(0,1439)
    plt.grid(ls=':')
    plt.xticks(xt_,xt)
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/Nature/img/lv_voltages.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0)

# this is losses

plt.rcParams['font.size'] = 14
plt.figure(figsize=(5,3))

# losses first
m = []
q1 = []
q3 = []
u = []
l = []

res = []
for i in range(3):
    res.append([])
with open(res_stem+la+'_losses.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        for i in range(3):
            res[i].append(float(row[i+1]))
            
for i in range(3):
    x = sorted(res[i])
    
    m.append(1*x[int(len(x)/2)])
    q1.append(1*x[int(len(x)*0.25)])
    q3.append(1*x[int(len(x)*0.75)])
    l.append(1*x[0])
    u.append(1*x[-1])


plt.scatter(range(1,len(m)+1),l,marker='_',c='gray')
plt.scatter(range(1,len(m)+1),u,marker='_',c='gray')
for i in range(len(m)):
    plt.plot([i+1,i+1],[l[i],q1[i]],c='gray')
    plt.plot([i+1,i+1],[q3[i],u[i]],c='gray')

# box
for i in range(len(m)):
    plt.plot([i+0.81,i+1.19],[m[i],m[i]],c='b',lw='2')
    plt.plot([i+0.8,i+1.2],[q1[i],q1[i]],c='k')
    plt.plot([i+0.8,i+1.2],[q3[i],q3[i]],c='k')
    plt.plot([i+1.2,i+1.2],[q1[i],q3[i]],c='k')
    plt.plot([i+0.8,i+0.8],[q1[i],q3[i]],c='k')    
plt.grid(ls=':')
plt.ylim(0,250)
plt.ylabel('Losses (kWh)')
plt.xticks([1,2,3],ttls)
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/Nature/img/lv_losses.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0.1)

# ok, but next I'm ging to want to do something about nationally

# get locations
locs = {}
with open(stem+'LA-lat-lon.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        locs[row[1]] = [float(row[3]),float(row[2])]

# What metrics do I want to plot - probability of undervoltage?
# z score would be good i think
# increase in losses?

v_zscore = {'b':{},'u':{},'f':{}}
ty = ['b','u','f']
for la in locs:
    try:
        with open(res_stem+la+'_voltages_m.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            res = []
            for i in range(9):
                res.append([])
            for row in reader:
                for i in range(9):
                    res[i].append(float(row[i+1]))
            for i in range(3):
                mu = min(res[3*i+1])
                sigma = (mu-min(res[3*i]))/3
                z = (0.9-mu)/sigma
                v_zscore[ty[i]][la] = z
    except:
        continue
    
p_zscore = {'b':{},'u':{},'f':{}}
for la in locs:
    try:
        with open(res_stem+la+'_load.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            res = []
            for i in range(9):
                res.append([])
            for row in reader:
                for i in range(9):
                    res[i].append(float(row[i+1]))
            for i in range(3):
                mu = max(res[3*i+1])
                sigma = (max(res[3*i+2]))/3

                z = (mu-760)/sigma
                p_zscore[ty[i]][la] = z
    except:
        continue

def find_nearest(p1):
    closest = 100000
    best = None

    for ii in range(len(pList)):
        p = pList[ii]
        d = np.power(p[0]-p1[1],2)+np.power(p[1]-p1[0],2)
        if d < closest:
            closest = d
            best = ii

    return best

# so one good option would be to plot the 
pList = []
z = [[],[],[]]
z2 = [[],[],[]]

for la in v_zscore['f']:
    pList.append(locs[la])
    z[0].append(v_zscore['b'][la])
    z[1].append(v_zscore['u'][la])
    z[2].append(v_zscore['f'][la])
    z2[0].append(p_zscore['b'][la])
    z2[1].append(p_zscore['u'][la])
    z2[2].append(p_zscore['f'][la])
# create new figure, axes instances.

fig=plt.figure(figsize=(8,4))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 9

carry = {}
titles = ['No Charging','Uncontrolled','Controlled']
for pn in range(3):
    plt.subplot(1,3,pn+1)
    plt.title(titles[pn])
    ax = plt.gca()
    #ax=fig.add_axes([0.1,0.1,0.8,0.8])
    # setup mercator map projection.
    m = Basemap(llcrnrlon=-7,llcrnrlat=49.9,urcrnrlon=2.2,urcrnrlat=58.7,\
                resolution='l',projection='merc',\
                lat_0=40.,lon_0=-20.,lat_ts=20.)
    # make these smaller to increase the resolution
    x = np.arange(-7,3,0.05)
    y = np.arange(49,59,0.05)

    Z = np.zeros((len(x),len(y)))
    Z2 = np.zeros((len(x),len(y)))
    X = np.zeros((len(x),len(y)))
    Y = np.zeros((len(x),len(y)))
    m.drawcoastlines()
    for i in range(len(x)):
        for j in range(len(y)):
            p = [x[i],y[j]]
            best = find_nearest(p)
            xpt,ypt = m(x[i],y[j])
            X[i,j] = xpt
            Y[i,j] = ypt
            if m.is_land(xpt,ypt) == True:
                if xpt < 200000 and ypt < 970000 and ypt > 300000:
                    continue
                if xpt > 885000 and ypt < 175000:
                    continue
                if xpt > 766000 and ypt < 104000:
                    continue
                Z[i,j] = 100*st.norm.cdf(z[pn][best])
                Z2[i,j] = 100*st.norm.cdf(z2[pn][best])
            else:
                continue

    im = m.pcolor(X,Y,Z,vmin=0,vmax=100,cmap='Blues')
    carry[pn] = [X,Y,Z2]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/Nature/img/v_zscore.eps', format='eps', dpi=1000,
            bbox_inches='tight', pad_inches=0)



fig=plt.figure(figsize=(8,4))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 9

titles = ['No Charging','Uncontrolled','Controlled']
for pn in range(3):
    plt.subplot(1,3,pn+1)
    plt.title(titles[pn])
    ax = plt.gca()
    m = Basemap(llcrnrlon=-7,llcrnrlat=49.9,urcrnrlon=2.2,urcrnrlat=58.7,\
                resolution='l',projection='merc',\
                lat_0=40.,lon_0=-20.,lat_ts=20.)
    m.drawcoastlines()

    [X,Y,Z] = carry[pn]

    im = m.pcolor(X,Y,Z,vmin=0,vmax=50,cmap='Greens')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/Nature/img/p_zscore.eps', format='eps', dpi=1000,
            bbox_inches='tight', pad_inches=0)


plt.show()

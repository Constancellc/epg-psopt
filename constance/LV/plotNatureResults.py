import csv
import random
import copy
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cbook
import scipy.stats as st
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

census_stem = '../../../EPA-Code/sales-stats/'
name = {}
data = {}
total = {}
# get s curve stuff
def scurve(data,x):
    _x = np.arange(2011.75,2018.75,0.25)
    y = []
    for i in range(len(data)):
        y.append(np.log(data[i]/(1-data[i])))
    [m,c] = np.polyfit(_x,y,1)

    f = []
    for i in range(len(x)):
        f.append(1/(1+np.exp(-1*(m*x[i]+c))))
    return f

def national_flat(eV,n=48):
    p = [31.83083333333333, 32.601, 32.89416666666666, 32.49383333333333,
         32.263666666666666, 32.05883333333333, 31.681333333333335,
         31.143833333333333, 30.832666666666665, 30.95, 31.4, 32.43416666666667,
         35.81033333333333, 40.05233333333334, 42.334, 44.2905, 45.543,
         45.87716666666666, 45.596999999999994, 45.485166666666665,
         45.17733333333333, 44.85, 44.293499999999995, 43.85566666666667,
         43.73, 43.622166666666665, 43.32533333333333, 43.23733333333334,
         43.3825, 43.69533333333334, 44.13633333333333, 44.99333333333333,
         46.2515, 46.83566666666666, 48.37216666666667, 48.721333333333334,
         48.56100000000001, 48.42366666666667, 47.844833333333334, 47.701,
         47.09883333333333, 45.95016666666667, 43.94833333333333,
         41.56033333333333, 39.278666666666666, 37.0585, 35.374,
         34.292833333333334]
    if n != 48:
        sr = int(n/48)
        _p = [0]*n
        for t in range(n):
            f1 = p[int(t/sr)]
            try:
                f2 = p[int(t/sr)+1]
            except:
                f2 = p[0]
            alpha = float(n%sr)/sr
            _p[t] = alpha*f2+(1-alpha)*f1
    p = _p
    p2 = []
    for i in range(len(p)):
        p2.append(max(p)+0.01-p[i])

    sf = eV/sum(p2)
    for i in range(len(p2)):
        p2[i] = p2[i]*sf

    return p2


with open(census_stem+'veh0105.csv','rU') as csvfile:
    reader = csv.reader(csvfile)
    for i in range(8):
        next(reader)
    for row in reader:
        la = row[0]
        if la == '':
            continue
        n = row[1]
        while n[0] == ' ':
            n = n[1:]
        name[la] = n

        try:
            total[la] = float(row[2].replace(',',''))*1000
        except:
            continue

with open(census_stem+'EVregionalSales.csv','r',encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        if row[0] == '':
            continue

        la = row[0]

        d = []
        try:
            for i in range(2,len(row)):
                d.append(float(row[len(row)+1-i])/total[la])
        except:
            continue

        data[la] = d


blues = cm.get_cmap('Blues', 1000)
new = blues(np.linspace(0, 1, 1000))
new[:1,:] =  np.array([1,1,1,1])
blue2 = ListedColormap(new)


greens = cm.get_cmap('Oranges', 256)
new = greens(np.linspace(0, 1, 256))
new[:1,:] =  np.array([1,1,1,1])
green2 = ListedColormap(new)

res_stem = '../../../Documents/simulation_results/LV/LA/'
stem = '../../../Documents/simulation_results/NTS/clustering/power/locationsLA/'
r_type_data = '../../../Documents/census/LA_rural_urban.csv'

admd = '../../../Documents/census/admd.csv'
# first i want to pick an example simulation (preferrably a bad one)

la = 'E08000005'

ruType = {}
with open(r_type_data,'rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        ruType[row[0]] = int(row[1])

        
nHH = {}
with open('../../../Documents/simulation_results/NTS/clustering/power/locationsLA_/censusParams.csv','r',encoding='ISO-8859-1') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        nHH[row[0]] = [int(row[3])]

# power demand first 
time = np.arange(0,1440,10)
ttls = ['No Charging','Uncontrolled','Controlled']
xt = ['04:00','12:00','20:00']
xt_ = [240,720,1200]
plt.figure(figsize=(8,3))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 11

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
    plt.plot([time[0],time[-1]],[300,300],c='r',ls='--')
    plt.plot(time,res[3*i+1],c='g')
    plt.title(ttls[i],y=0.85)
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
plt.savefig('../../../Dropbox/thesis/chapter6/img/lv_power.eps', format='eps',
            dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.figure(figsize=(8,3))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 11

for i in range(3):
    if i == 2:
        for j in range(3):
            ev = sum(res[3*i+j])-sum(res[j])
            ev_p = national_flat(ev,144)
            for t in range(144):
                res[3*i+j][t] = res[j][t] + ev_p[t]
    plt.subplot(1,3,i+1)
    plt.plot([time[0],time[-1]],[300,300],c='r',ls='--')
    plt.plot(time,res[3*i+1],c='g')
    plt.title(ttls[i],y=0.85)
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
plt.savefig('../../../Dropbox/thesis/chapter7/img/lv_power2.eps', format='eps',
            dpi=300, bbox_inches='tight', pad_inches=0.1)

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
    if i > 0:
        plt.plot([0,1439],[0.9,0.9],c='r',ls='--')
    plt.grid(ls=':')
    plt.xticks(xt_,xt)
plt.tight_layout()
plt.savefig('../../../Dropbox/thesis/chapter6/img/lv_voltages.eps', format='eps',
            dpi=300, bbox_inches='tight', pad_inches=0)

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
#plt.show()
plt.savefig('../../../Dropbox/thesis/chapter6/img/lv_losses.eps', format='eps',
            dpi=300, bbox_inches='tight', pad_inches=0.1)

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


maxP = {}
with open(admd,'rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        maxP[row[0]] = float(row[1])

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
                sigma = (mu-min(res[3*i]))/2
                z = (0.9-mu)/sigma
                v_zscore[ty[i]][la] = z
    except:
        continue

bs = [0]*40
p_zscore = {'b':{},'u':{},'f':{}}
p_zscore2 = {'f':{}}

addCap = []
addCap2 = []
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
                sigma = (max(res[3*i+2])-mu)/2

                z = (mu-maxP[la])/sigma
                p_zscore[ty[i]][la] = z

                if i == 2:
                    ev = sum(res[3*i+1])-sum(res[1])
                    ev_p = national_flat(ev,144)
                    for t in range(144):
                        ev_p[t] += res[1][t]
                    mu2 = max(ev_p)
                    ev = sum(res[3*i+2])-sum(res[2])
                    ev_p = national_flat(ev,144)
                    for t in range(144):
                        ev_p[t] += res[2][t]
                    sigma2 = (max(ev_p)-mu2)/2
                    z2 = (mu2-maxP[la])/sigma2
                    p_zscore2[ty[i]][la] = z2

                ex = maxP[la]-3*sigma-mu

                if i == 2:
                    h = float(nHH[la][0])

                    if ruType[la] == '1':
                        nN = h/64
                    elif ruType[la] == '2':
                        nN = h/200
                    else:
                        nN = h/471

                    addCap.append([ex/1e6,nN])
                    addCap2.append([100*ex/mu,la,nN])

                    if ex < 0:
                        bs[0] += nN
                    else:
                        bs[int(ex/10)] += nN

    except:
        continue

sf = sum(bs)/100
for i in range(len(bs)):
    bs[i] = bs[i]/sf
plt.figure()
plt.bar(range(40),bs)
plt.xticks([0,10,20,30,40],['Fully\nConstrained','100 kW','200 kW','300kW','400kW'],
           rotation=90)
plt.xlabel('Additional Capacity')
plt.grid()
plt.tight_layout()

addCap = sorted(addCap,reverse=True)
addCap2 = sorted(addCap2,reverse=True)

with open('headroom.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['LA','Headroom (%)','No. Networks'])
    for i in range(len(addCap2)):
        writer.writerow([addCap2[i][1],addCap2[i][0],addCap2[i][2]])

y = [0]
x = [0]

for i in range(len(addCap)):
    x.append(x[-1]+addCap[i][1]/sf)
    if addCap[i][0] > 0:
        y.append(y[-1]+addCap[i][0]*addCap[i][1])
    else:
        y.append(y[-1])
plt.figure()
plt.plot(x,y)
plt.xlim(0,100)
plt.ylim(0,10)
plt.grid()
plt.ylabel('Cumulative Excess Capacity (GW)')
plt.xlabel('Percentage of Networks')
plt.tight_layout()
plt.savefig('../../../Dropbox/thesis/chapter7/img/extra_cap.eps', format='eps', dpi=300,
            bbox_inches='tight', pad_inches=0)
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
z = [[],[],[],[],[]]
z2 = [[],[],[],[],[]]
z3 = []

res = []
res2 = []
res3 = []
res4 = []

n = 0
years = {}

for y in range(2020,2055):
    years[y] = {0:0,1:0,2:0,3:0,4:0}
for la in v_zscore['f']:
    pList.append(locs[la])
    z[0].append(100*st.norm.cdf(v_zscore['b'][la]))
    z[1].append(100*st.norm.cdf(v_zscore['u'][la])-z[0][-1])
    z[2].append(100*st.norm.cdf(v_zscore['f'][la])-z[0][-1])
    z2[0].append(100*st.norm.cdf(p_zscore['b'][la]))
    z2[1].append(100*st.norm.cdf(p_zscore['u'][la])-z2[0][-1])
    z2[2].append(100*st.norm.cdf(p_zscore['f'][la])-z2[0][-1])
    z3.append(100*st.norm.cdf(p_zscore2['f'][la])-z2[0][-1])
    # 2030
    try:
        per = scurve(data[la],[2030])[0]
    except:
        per = 0.5
    z[3].append(z[1][-1]*per)
    z[4].append(z[2][-1]*per)
    z2[3].append(z2[1][-1]*per)
    z2[4].append(z2[2][-1]*per)

    if per != 0.5:
        n += 1
        for y in years:
            per = scurve(data[la],[y])[0]
            years[y][0] += z[1][-1]*per
            years[y][1] += z[2][-1]*per
            years[y][2] += z2[1][-1]*per
            years[y][3] += z2[2][-1]*per
            years[y][4] += z3[-1]*per

    res.append([z[1][-1],la])
    res2.append([z2[1][-1],la])
    res3.append([z[3][-1],la])
    res4.append([z2[3][-1],la])

yp = []
for i in range(5):
    yp.append([])
    for y in years:
        yp[-1].append(years[y][i]/n)

plt.figure(figsize=(7.5,4))
plt.subplot(1,2,1)
plt.title('Transformer Violations')
plt.plot(range(2020,2055),yp[2],label='Uncontrolled',c='b')
plt.plot(range(2020,2055),yp[3],label='Controlled (D)',c='r',ls='--')
plt.plot(range(2020,2055),yp[4],label='Controlled (T)',c='g',ls=':')
plt.xlabel('Year')
plt.ylabel('Networks (%)')
plt.ylim(0,23)
plt.xlim(2018,2054)
plt.grid()
plt.subplot(1,2,2)
plt.title('Voltage Violations')
plt.plot(range(2020,2055),yp[0],label='Uncontrolled',c='b')
plt.plot(range(2020,2055),yp[1],label='Controlled (D)',c='r',ls='--')
yy = []
for i in range(len(yp[1])):
    yy.append(yp[1][i]*1.1)
plt.plot(range(2020,2055),yy,label='Controlled (T)',c='g',ls=':')
plt.xlabel('Year')
plt.ylim(0,23)
plt.xlim(2018,2054)
plt.legend()
plt.grid()
plt.tight_layout()
#plt.savefig('../../../Dropbox/papers/Nature/img/av_violations.eps', format='eps', dpi=300,
#            bbox_inches='tight', pad_inches=0)
plt.savefig('../../../Dropbox/thesis/chapter7/img/av_violations2.eps', format='eps', dpi=300,
            bbox_inches='tight', pad_inches=0)

plt.show()
rl = [res,res2,res3,res4]
for r in range(4):
    with open('res'+str(r+1)+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        rs = sorted(rl[r],reverse=True)
        for row in rs:
            writer.writerow(row)
print('DONE')

rType = {}
with open(r_type_data,'rU') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        rType[row[0]] = row[1]

'''
res = []
for la in v_zscore['u']:
    res.append([st.norm.cdf(p_zscore['u'][la]),st.norm.cdf(v_zscore['u'][la]),
                rType[la],la])
res = sorted(res,reverse=True)
with open('res.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for row in res:
        writer.writerow(row)
res = []
for la in v_zscore['u']:
    res.append([st.norm.cdf(p_zscore['f'][la]),st.norm.cdf(v_zscore['f'][la]),
                rType[la],la])
res = sorted(res,reverse=True)
with open('res2.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    for row in res:
        writer.writerow(row)
'''
# create new figure, axes instances.

fig=plt.figure(figsize=(10,3.5))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 9

carry = {}
titles = ['No\nCharging','Uncontrolled\n(100%)','Controlled\n(100%)',
          'Uncontrolled\n(2030)','Controlled\n(2030)']

for pn in range(1,5):
    plt.subplot(1,4,pn)
    plt.title(titles[pn])
    ax = plt.gca()
    #ax=fig.add_axes([0.1,0.1,0.8,0.8])
    # setup mercator map projection.
    m = Basemap(llcrnrlon=-7,llcrnrlat=49.9,urcrnrlon=2.2,urcrnrlat=58.7,\
                resolution='l',projection='merc',\
                lat_0=40.,lon_0=-20.,lat_ts=20.)
    # make these smaller to increase the resolution
    x = np.arange(-7,3,0.02)
    y = np.arange(49,59,0.02)

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
                Z[i,j] = z[pn][best]
                Z2[i,j] = z2[pn][best]
            else:
                continue

    im = m.pcolor(X,Y,Z,vmin=0,vmax=50,cmap=blue2)
    carry[pn] = [X,Y,Z2]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/Nature/img/v_zscore.eps', format='eps', dpi=300,
            bbox_inches='tight', pad_inches=0)


fig=plt.figure(figsize=(10,3.5))
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 9


for pn in range(1,5):
    plt.subplot(1,4,pn)
    plt.title(titles[pn])
    ax = plt.gca()
    m = Basemap(llcrnrlon=-7,llcrnrlat=49.9,urcrnrlon=2.2,urcrnrlat=58.7,\
                resolution='l',projection='merc',\
                lat_0=40.,lon_0=-20.,lat_ts=20.)
    m.drawcoastlines()

    [X,Y,Z] = carry[pn]

    im = m.pcolor(X,Y,Z,vmin=0,vmax=50,cmap=green2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
plt.tight_layout()
plt.savefig('../../../Dropbox/papers/Nature/img/p_zscore.eps', format='eps', dpi=300,
            bbox_inches='tight', pad_inches=0)


plt.show()

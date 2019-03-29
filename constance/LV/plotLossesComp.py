import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat

fds = ['041','213','162','1','3','2','193','074','024']

hhs = {'1':55,'2':175,'3':94,'4':24,'024':115,'041':24,'074':186,'162':73,
       '193':65,'213':67}

stem = '../../../Documents/simulation_results/LV/manc-models/'
# losses first
m = []
q1 = []
q3 = []
u = []
l = []
_m = []
_q1 = []
_q3 = []
_u = []
_l = []

for f in fds:
    diff = []
    diff2 = []
    with open(stem+f+'-losses.csv','rU') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            diff.append(float(row[2])-float(row[3]))
            if diff[-1] < 0:
                diff[-1] = 0
            diff2.append(float(row[2])-float(row[4]))
            if diff2[-1] < 0:
                diff2[-1] = 0
    diff = sorted(diff)
    diff2 = sorted(diff2)
    
    m.append(1000*diff[int(len(diff)/2)]/hhs[f])
    q1.append(1000*diff[int(len(diff)*0.25)]/hhs[f])
    q3.append(1000*diff[int(len(diff)*0.75)]/hhs[f])
    l.append(1000*diff[0]/hhs[f])
    u.append(1000*diff[-1]/hhs[f])
    
    _m.append(1000*diff2[int(len(diff2)/2)]/hhs[f])
    _q1.append(1000*diff2[int(len(diff2)*0.25)]/hhs[f])
    _q3.append(1000*diff2[int(len(diff2)*0.75)]/hhs[f])
    _l.append(1000*diff2[0]/hhs[f])
    _u.append(1000*diff2[-1]/hhs[f])

    if _l[-1] < l[-1]:
        _l[-1] = l[-1]

plt.figure()
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 9
fig,ax = plt.subplots(1,figsize=(6,3.3))

# whiskers
plt.scatter(range(1,len(m)+1),l,marker='_',c='gray')
plt.scatter(range(1,len(m)+1),u,marker='_',c='gray')
for i in range(len(m)):
    plt.plot([i+1,i+1],[l[i],q1[i]],c='gray')
    plt.plot([i+1,i+1],[q3[i],u[i]],c='gray')

x_ticks = []
for i in range(len(m)):
    x_ticks.append(str(i+1)+'\n('+str(hhs[fds[i]])+')')
# box
for i in range(len(m)):
    plt.plot([i+0.62,i+1.38],[m[i],m[i]],c='b',lw='2')
    plt.plot([i+0.6,i+1.4],[q1[i],q1[i]],c='k')
    plt.plot([i+0.6,i+1.4],[q3[i],q3[i]],c='k')
    plt.plot([i+1.4,i+1.4],[q1[i],q3[i]],c='k')
    plt.plot([i+0.6,i+0.6],[q1[i],q3[i]],c='k')

if True: # adds the phase imbalance stuff
    # phase imbalance
    '''
    plt.scatter(range(1,len(m)+1),_l,marker='_',c='r')
    plt.scatter(range(1,len(m)+1),_u,marker='_',c='r')
    for i in range(len(m)):
        plt.plot([i+1,i+1],[_l[i],_q1[i]],ls='--',c='r')
        plt.plot([i+1,i+1],[_q3[i],_u[i]],ls='--',c='r')'''

    # box
    for i in range(len(m)):
        plt.plot([i+0.72,i+1.28],[_m[i],_m[i]],c='r',lw='2')
        plt.plot([i+0.7,i+1.3],[_q1[i],_q1[i]],ls=':',c='k')
        plt.plot([i+0.7,i+1.3],[_q3[i],_q3[i]],ls=':',c='k')
        plt.plot([i+1.3,i+1.3],[_q1[i],_q3[i]],ls=':',c='k')
        plt.plot([i+0.7,i+0.7],[_q1[i],_q3[i]],ls=':',c='k')


def draw_box(x_l,y_l,x_u,y_u,c,ls):
    plt.plot([x_l,x_l],[y_l,y_u],c=c,ls=ls)
    plt.plot([x_u,x_u],[y_l,y_u],c=c,ls=ls)
    plt.plot([x_l,x_u],[y_l,y_l],c=c,ls=ls)
    plt.plot([x_l,x_u],[y_u,y_u],c=c,ls=ls)
    
        
plt.xticks(range(1,len(m)+1),x_ticks)
plt.ylabel('Losses reduction (Wh per household)')
plt.grid(linestyle=':')
plt.tight_layout()

rect = pat.Rectangle((0.5,91),3.4,35,facecolor='w',edgecolor='gray',zorder=2)
ax.add_patch(rect)
draw_box(0.6,110,1.2,120,'k','-')
draw_box(0.65,95,1.15,105,'k',':')
plt.plot([0.62,1.18],[115,115],c='b',lw='2')
plt.plot([0.67,1.13],[100,100],c='r',lw='2')
plt.annotate('Loss Minimizing',(1.3,113))
plt.annotate('LF+Phase Balancing',(1.3,98))

plt.savefig('../../../Dropbox/papers/losses/img/losses_comp.eps', format='eps',
            dpi=1000, bbox_inches='tight', pad_inches=0)
plt.show()

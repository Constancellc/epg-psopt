import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers

solvers.options['maxiters'] = 30

class LVTestFeeder:

    def __init__(self):
        self.hh = None
        
        profiles = []
        for i in range(55):
            profiles.append([0.0]*1440)
        self.ev = profiles

        # might as well get the loss model coefficients now
        self.P0 = matrix(0.0,(55,55))
        with open('P.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                for j in range(len(row)):
                    self.P0[i,j] += float(row[j])
                i += 1

        self.q0 = []
        with open('q.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.q0.append(float(row[0]))

        with open('c.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.c = float(row[0])

        self.M0 = matrix(0.0,(55,55))
        with open('M_.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                for j in range(len(row)):
                    self.M0[i,j] += float(row[j])
                i += 1

        self.a0 = []
        with open('a_.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.a0.append(float(row[0]))      

    def set_households(self,profiles):
        self.hh = profiles
        self.x_h = []
        self.base = [0.0]*1440
        
        for t in range(1440):
            for i in range(55):
                self.x_h.append(-profiles[i][t]*1000)
                self.base[t] += profiles[i][t]

    def set_evs(self,vehicles):
        # n foorm [kWh,departure,arrival]
        
        self.b = []
        self.map = {} # maps the hh number to the vehicle number
        self.times = []
        
        i = 0
        for j in range(55):
            if vehicles[j][0] == 0:
                continue
            self.map[i] = j
            i += 1
            
            self.b.append(vehicles[j][0])
            self.times.append(vehicles[j][1:])

            # hack to present singular optimisation
            if self.times[-1][0] >= self.times[-1][1]:
                self.times[-1][0] = self.times[-1][1]-60
            
        self.n = len(self.b)

    def uncontrolled(self,power):
        profiles = []
        for i in range(55):
            profiles.append([0.0]*1440)

        for i in range(self.n):
            e = self.b[i]

            if e < power/60:
                continue
            
            a = self.times[i][1]

            chargeTime = int(e*60/power)+1

            for t in range(a,a+chargeTime):
                if t < 1440:
                    profiles[self.map[i]][t] = power
                else:
                    profiles[self.map[i]][t-1440] = power
            
        self.ev = profiles

    def loss_minimise(self,Pmax,constrain=False):
        profiles = []
        for i in range(55):
            profiles.append([0.0]*1440)

        Pr = matrix(0.0,(self.n,self.n))
        qr = []
        for i in range(self.n):
            qr.append(self.q0[self.map[i]])
            for j in range(self.n):
                Pr[i,j] = self.P0[self.map[i],self.map[j]]

        x_h = []
        for t in range(1440):
            for v in range(self.n):
                x_h.append(self.x_h[t*55+self.map[v]])
        
        P = spdiag([Pr]*1440)
        x_h = matrix(x_h)
        q = matrix(qr*1440)
        q += (P+P.T)*x_h

        if constrain == False:
            A = matrix(0.0,(self.n,self.n*1440))
            b = matrix(0.0,(self.n,1))
        else:
            A = matrix(0.0,(2*self.n,self.n*1440))
            b = matrix(0.0,(2*self.n,1))

        for v in range(self.n):
            for t in range(1440):
                A[v,self.n*t+v] = 1.0/60
                
                if constrain == True:
                    if t > self.times[v][0] and t < self.times[v][1]:
                        A[v+self.n,self.n*t+v] = 1.0
                        
            b[v] = -self.b[v]*1000

        M1 = matrix(0.0,(self.n,self.n))
        a1 = []

        for i in range(self.n):
            a1.append(self.a0[self.map[i]]-240)
            for j in range(self.n):
                M1[i,j] = -1*self.M0[self.map[i],self.map[j]]

        G = spdiag([M1]*1440)

        G = sparse([G,spdiag([-1.0]*(self.n*1440)),spdiag([1.0]*(self.n*1440))])
        h = matrix(a1*1440+[Pmax*1000]*(self.n*1440)+[0.0]*(self.n*1440))

        sol=solvers.qp(P*2,q,G,h,A,b)
        x = sol['x']
        self.status = sol['status'] 

        del P
        del q
        del G
        del h
        del A
        del b

        for v in range(self.n):
            for t in range(1440):
                profiles[self.map[v]][t] -= x[self.n*t+v]/1000

        self.ev = profiles

    def load_flatten(self,Pmax,constrain=False):
        profiles = []
        for i in range(55):
            profiles.append([0.0]*1440)

        
        q = copy.copy(self.base)*self.n
        q = matrix(q)

        P = sparse([[spdiag([1]*1440)]*self.n]*self.n)

        if constrain == False:
            A = matrix(0.0,(self.n,self.n*1440))
            b = matrix(0.0,(self.n,1))
        else:
            A = matrix(0.0,(2*self.n,self.n*1440))
            b = matrix(0.0,(2*self.n,1))
        

        for v in range(self.n):
            for t in range(1440):
                A[v,1440*v+t] = 1.0/60
                
                if constrain == True:
                    if t > self.times[v][0] and t < self.times[v][1]:
                        A[v+self.n,1440*v+t] = 1.0

            b[v] = self.b[v]
        
        G = sparse([spdiag([-1.0]*(self.n*1440)),spdiag([1.0]*(self.n*1440))])
        h = matrix([0.0]*(self.n*1440)+[Pmax]*(self.n*1440))

        sol=solvers.qp(P,q,G,h,A,b)
        x = sol['x']
        self.status = sol['status'] 
        
        del P
        del q
        del G
        del h
        del A
        del b

        for v in range(self.n):
            for t in range(1440):
                profiles[self.map[v]][t] = x[1440*v+t]
        
        self.ev = profiles

    def regularised_loss_minimise(self,Pmax,alpha=0.00001,constrain=False):
        profiles = []
        for i in range(55):
            profiles.append([0.0]*1440)

        Pr = matrix(0.0,(self.n,self.n))
        for i in range(self.n):
            for j in range(self.n):
                Pr[i,j] = self.P0[self.map[i],self.map[j]]
        
        P = spdiag([Pr]*1440)
        P += spdiag([alpha]*(1440*self.n))
        x_h = matrix(self.x_h)
        q = matrix(self.q0*1440)
        q += (P+P.T)*x_h

        if constrain == False:
            A = matrix(0.0,(self.n,self.n*1440))
            b = matrix(0.0,(self.n,1))
        else:
            A = matrix(0.0,(2*self.n,self.n*1440))
            b = matrix(0.0,(2*self.n,1))

        for v in range(self.n):
            for t in range(1440):
                A[v,self.n*t+v] = 1.0/60
                
                if constrain == True:
                    if t > self.times[v][0] and t < self.times[v][1]:
                        A[v+self.n,self.n*t+v] = 1.0
                        
            b[v] = -self.b[v]*1000

        G = sparse([spdiag([-1.0]*(self.n*1440)),spdiag([1.0]*(self.n*1440))])
        h = matrix([Pmax*1000]*(self.n*1440)+[0.0]*(self.n*1440))

        sol=solvers.qp(P*2,q,G,h,A,b)
        x = sol['x']

        for v in range(self.n):
            for t in range(1440):
                profiles[self.map[v]][t] -= x[self.n*t+v]/1000

        self.ev = profiles

    def predict_losses(self):
        losses = []

        for t in range(1440):
            y = [0.0]*55
            for i in range(55):
                y[i] -= self.hh[i][t]*1000
            for v in range(self.n):
                i = self.map[v]
                y[i] -= self.ev[v][t]*1000

            y = matrix(y)

            losses.append((y.T*self.P0*y+matrix(self.q0).T*y)[0]+self.c)

        return losses
    
    def predict_voltage(self):
        v_ = []
        for i in range(55):
            v_.append([])

        for t in range(1440):
            y = [0.0]*55
            for i in range(55):
                y[i] -= self.hh[i][t]*1000
            for v in range(self.n):
                i = self.map[v]
                y[i] -= self.ev[v][t]*1000

            y = matrix(y)

            v_new = self.M0*y+matrix(self.a0)
            i = 0
            for vv in v_new:
                v_[i].append(vv)
                i += 1

        v_av = [0.0]*1440
        for t in range(1440):
            for i in range(55):
                v_av[t] += v_[i][t]/55

        return v_av
    
    def predict_lowest_voltage(self):
        v_ = []
        for i in range(55):
            v_.append([])

        for t in range(1440):
            y = [0.0]*55
            for i in range(55):
                y[i] -= self.hh[i][t]*1000
            for v in range(self.n):
                i = self.map[v]
                y[i] -= self.ev[v][t]*1000

            y = matrix(y)

            v_new = self.M0*y+matrix(self.a0)
            i = 0
            for vv in v_new:
                v_[i].append(vv)
                i += 1

        v_l = [1000.0]*1440
        for t in range(1440):
            for i in range(55):
                if v_[i][t] < v_l[t]:
                    v_l[t] = v_[i][t]

        return v_l
    

    def getLineCurrents(self):
        current110 = []
        current296 = []

        Ar = matrix(0.0,(6,55))
        Ai = matrix(0.0,(6,55))

        br = matrix(0.0,(6,1))
        bi = matrix(0.0,(6,1))

        with open('Ar.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                for j in range(len(row)):
                    Ar[i,j] = float(row[j])
                i += 1

        with open('Ai.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                for j in range(len(row)):
                    Ai[i,j] = float(row[j])
                i += 1

        with open('br.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                br[i] = float(row[0])
                i += 1
                
        with open('bi.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                bi[i] = float(row[0])
                i += 1
                
        for t in range(1440):
            y = [0.0]*55
            for i in range(55):
                y[i] -= self.hh[i][t]*1000
            for v in range(self.n):
                i = self.map[v]
                y[i] -= self.ev[v][t]*1000
            y = matrix(y)

            ir = (Ar*y+br)
            ii = (Ai*y+bi)

            # phase b and c
            '''
            current110.append(np.sqrt(np.power(ir[1,0]+ii[1,0],2))+\
                               np.sqrt(np.power(ir[2,0]+ii[2,0],2)))
            current296.append(np.sqrt(np.power(ir[4,0]+ii[4,0],2))+\
                              np.sqrt(np.power(ir[5,0]+ii[5,0],2)))
            '''

            # phase c only
            current110.append(10*np.sqrt(np.power(ir[2,0]+ii[2,0],2)))
            current296.append(10*np.sqrt(np.power(ir[3,0]+ii[3,0],2)))

        return [current110,current296]
                

    def get_feeder_load(self):
        total_load = [0.0]*1440

        for t in range(1440):
            for i in range(55):
                total_load[t] += self.hh[i][t] + self.ev[i][t]

        return total_load


    def get_inidividual_load(self,node):
        base = self.hh[node]
        combined = []
        for t in range(1440):
            combined.append(self.hh[node][t]+self.ev[node][t])

        return base, combined

    def start_opendss(self):
        return ''

    def opendss_losses(self):
        return ''

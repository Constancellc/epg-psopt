import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers

solvers.options['maxiters'] = 30

'''
OK, I'm going for a restructure.

Changes that I need to make:
- In the initiation point to location of the linearization parameters
- The EV data will now be MEA, however I would like to also have the possibility
to overwrite the constraints and the energy requirments. When I am using the MEA
vehicles though I need to use the same ones
- The household data will now be from Network revolution, although I would like
option to overright maybe
- It would be good if the sensitivity analysis could also be wrapped in the
framework.
- It might be good to split this model over several files
_ I also want to delete the code that I think is unnecessary.
- I would like to implicitly consider variability, or the MC into the code.
'''

class LVTestFeeder:

    def __init__(self,folderPath):

        with open(folderPath+'/c.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.c = float(row[0])

        self.q0 = []
        with open(folderPath+'/q.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.q0.append(float(row[0]))

        self.hh = len(self.q0)
        
        self.P0 = matrix(0.0,(self.hh,self.hh))
        with open(folderPath'/P.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                for j in range(len(row)):
                    self.P0[i,j] += float(row[j])
                i += 1    

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
        self.nV = len(vehicles)
        self.b = []
        self.map = {} # maps the hh number to the vehicle number
        self.times = []
        
        i = 0
        for j in range(self.nV):
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

    def set_MEA(self,folderPath,weekday=True,weekend=False):
        # day 4 is a Monday
        # day 7 and 0 are Thursdays
        # max day is 253

        # ok, if we are concerned with weekends
        days = []
        for day in range(253):
            if day%7 not in [2,3] and weekday == True:
                days.append(day)
            elif day%7 in [2,3] and weekend == True:
                days.append(day)

        vehicles = ['000','001','002','003','004','005','006','007','009',
                    '010','011','012','013','014','018','019','020','021',
                    '022','023','027','028','029','030','031','032','034',
                    '035','036','037','038','039','041','042','043','044',
                    '045','046','047','048','049','050','052','053','054',
                    '055','056','057','058','059','060','061','062','063',
                    '064','065','066','067','068','069','070','071','072',
                    '073','074','075','076','077','078','079','080','081',
                    '082','083','084','085','086','087','090','091','092',
                    '093'.'094','096','097','098','099','100','101','102',
                    '103','104','105','106','107','108','110','111','112',
                    '113','114']

        chosen = []
        while len(chosen) < self.hh:
            ran = int(random.random()*len(vehicles))
            if vehicles[ran] not in chosen:
                chosen.append(vehicles[ran])

        # when it comes to variation I need to think about both variation
        # between vehicles and variation over time. For now I don't care
        chosenDay = days[int(random.random()*len(days))]

        empty = True

        # I GOT TO HERE!!!

        # Hit a problem: what to do with multiple charges?

        v_req = []
        for v in chosenV:
            charges = []
            kWh = 0
            start = 0
            end = 1
            with open(folderPath+v+'.csv','rU') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    if int(row[0]) == day:
                        charges.append(row[1:4])
            for c in charges:
                kWh += 
            if len(charges) == 1:
                kWh = charges[0][3]
                start = charges[0][1]
                needed = charges[0][2]
            elif len(charges) > 1:
                for c in charges:
                    kWh += cahrg
                

        

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
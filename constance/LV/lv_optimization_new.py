import csv
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, sparse, solvers

solvers.options['maxiters'] = 40
solvers.options['reltol'] = 1e-10

'''

Currently working on the load flattening function

It's not currently converging, I need to check how I got the MEA kWh because
often it says that the vehicle is not there for very long, and the value of
kWh exceeds what is possible on full charge

Also, I should add the efficiency term.

'''

class LVTestFeeder:

    def __init__(self,folderPath,t_res):

        self.t_res = t_res
        self.T = int(1440/t_res)

        with open(folderPath+'/c.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.c = float(row[0])

        self.q0 = []
        with open(folderPath+'/q.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.q0.append(float(row[0]))

        self.nH = len(self.q0)
        
        self.P0 = matrix(0.0,(self.nH,self.nH))
        with open(folderPath+'/P.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            i = 0
            for row in reader:
                for j in range(len(row)):
                    self.P0[i,j] += float(row[j])
                i += 1    

    def set_households(self,profiles): # UPDATED
        self.x_h = []
        self.base = [0.0]*int(self.T)
        hh_profiles = []
        for i in range(self.nH):
            hh_profiles.append([0.0]*self.T)
        
        for t in range(1440):
            for i in range(self.nH):
                self.base[int(t/self.t_res)] += profiles[i][t]/self.t_res
                hh_profiles[i][int(t/self.t_res)] += profiles[i][t]/self.t_res

                
        self.hh_profiles = hh_profiles

    def set_households_NR(self,filePath): # UPDATED

        # first get all
        all_profiles = {}
        with open(filePath,'rU') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                for j in range(len(row)-1):
                    if j not in all_profiles:
                        all_profiles[j] = []
                    all_profiles[j].append(float(row[j+1]))

        if len(all_profiles) < self.nH:
            print('Not enough profiles')
            return None

        # now chose some
        chosen_ = [] # for indexes
        chosen = [] # for profiles

        while len(chosen_) < self.nH:
            ran = int(random.random()*len(all_profiles))
            if ran not in chosen_:
                chosen_.append(ran)
                chosen.append(all_profiles[ran])

        self.set_households(chosen)

    def set_households_synthetic(self,pMax):

        profiles = []
        for hh in range(self.nH):
            p = []
            for t in range(1440):
                p.append(random.random()*pMax)
            profiles.append(p)

        self.set_households(profiles)
            

    def set_evs(self,vehicles): # UPDATED

        # now input has form [[[kWh1,arrival1,needed1],[kWh2...]],..]
        self.nV = len(vehicles) #
        
        self.b = []
        self.v_map = {} # 
        self.rn_map = {} # 
        self.times = []
        
        v = 0
        r = 0
        for j in range(self.nV):
            kWh = []
            times = []
            
            for j2 in range(len(vehicles[j])):
                if vehicles[j][j2][0] == 0:
                    continue
                kWh.append(vehicles[j][j2][0])

                self.rn_map[r] = v
                self.b.append(kWh[-1])
                self.times.append(vehicles[j][j2][1:])
                r += 1

            if len(kWh) == 0:
                continue

            self.v_map[v] = j
            v += 1
        
        self.rN = len(self.b)
        self.n = v
        
        '''      
        
                
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
        '''

    def set_evs_MEA(self,folderPath,weekday=True,weekend=False): #UPDATED
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
                    '093','094','096','097','098','099','100','101','102',
                    '103','104','105','106','107','108','110','111','112',
                    '113','114']

        chosen = []
        while len(chosen) < self.nH:
            ran = int(random.random()*len(vehicles))
            if vehicles[ran] not in chosen:
                chosen.append(vehicles[ran])

        # when it comes to variation I need to think about both variation
        # between vehicles and variation over time. For now I don't care
        chosenDay = days[int(random.random()*len(days))]

        #self.loc_map = {}
        # this wil map the position that each vehicle i assigned to

        vehicles = []
        evs = []
        rn = 0 # requirement number 
        for hh in range(self.nH):
            vehicles.append([])
            evs.append([0.0]*self.T)
            v = chosen[hh]
            with open(folderPath+v+'.csv','rU') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    if int(row[0]) != chosenDay:
                        continue
                    kWh = float(row[3])
                    start = int(row[1])
                    needed = int(row[2])

                    if needed>start and needed-start<60:
                        needed += 30

                    vehicles[hh].append([kWh,int(start/self.t_res),
                                         int(needed/self.t_res)])
                    

            # check for no overlapping constraints
            if len(vehicles[hh]) > 1:
                for j in range(len(vehicles[hh])-1):
                    if vehicles[hh][j][2] >= vehicles[hh][j+1][1]:
                        vehicles[hh][j][2] = vehicles[hh][j+1][1]-1
                               
                if vehicles[hh][-1][2] < vehicles[hh][-1][1]:
                    # check no overlap with first journey:
                    if vehicles[hh][-1][2] > vehicles[hh][0][1]:
                        vehicles[hh][-1][2] = vehicles[hh][0][1]

            # adjust any unfeasible constraints here
            for j in range(len(vehicles[hh])):
                if vehicles[hh][j][1] < vehicles[hh][j][2]:
                    maxTime = vehicles[hh][j][2]-vehicles[hh][j][1]
                else:
                    maxTime = 1440+vehicles[hh][j][2]-vehicles[hh][j][1]
                maxEnergy = maxTime*self.t_res*6.3/60 # 7kw at 90% eff
                
                if vehicles[hh][j][0] > maxEnergy:
                    vehicles[hh][j][0] = maxEnergy
                    

        self.set_evs(vehicles)
        self.evs = evs

    def set_evs_synthetic(self,kWh,start=None,needed=None,nTrips=1,pUnused=0):
        
        vehicles = []
        evs = []
        
        for v in range(self.nH):
            vehicles.append([])
            evs.append([0.0]*self.T)
            if random.random() < pUnused:
                continue
            if start != None and needed != None and nTrips==1:
                vehicles[v].append([kWh,start,needed])
                
            if nTrips == 1:
                start_ = int(1440*random.random())
                l = 100+int(1200*random.random())
                needed_ = start_+l
                if needed_ >= 1440:
                    needed_ -= 1440
                vehicles[v].append([kWh,int(start_/self.t_res),
                                    int(needed_/self.t_res)])

            elif nTrips == 2:
                start_ = int(600*random.random())
                needed_ = start_+200+int((600-start_)*random.random())

                start_2 = int(400*random.random())+needed_
                needed_2 = start_2+200+int(random.random()*(start_))
                if needed_2 >= 1440:
                    needed_2 -= 1440

                vehicles[v].append([kWh,int(start_/self.t_res),
                                    int(needed_/self.t_res)])
                vehicles[v].append([kWh,int(start_2/self.t_res),
                                    int(needed_2/self.t_res)])              
                
        self.set_evs(vehicles)
        self.evs = evs

    def uncontrolled(self,power=3.5,c_eff=0.9): # UPDATED
        profiles = []
        for i in range(self.nH):
            profiles.append([0.0]*self.T)
        for j in range(self.rN):
            e = copy.deepcopy(self.b[j])/c_eff
            
            a = self.times[j][0]
            
            if e < power*self.t_res/60:
                profiles[self.v_map[self.rn_map[j]]][a] = e*60/self.t_res
                continue
            
            chargeTime = int(e*60/(self.t_res*power))+1

            for t in range(a,a+chargeTime):
                if t < self.T:
                    profiles[self.v_map[self.rn_map[j]]][t] = power
                else:
                    profiles[self.v_map[self.rn_map[j]]][t-self.T] = power
            
        self.evs = profiles

    def loss_minimise(self,Pmax=7,c_eff=0.9,constrain=True):
        profiles = []
        for i in range(self.nH):
            profiles.append([0.0]*self.T)

        Pr = matrix(0.0,(self.n,self.n))
        qr = []
        for i in range(self.n):
            qr.append(self.q0[self.v_map[i]])
            for j in range(self.n):
                Pr[i,j] = self.P0[self.v_map[i],self.v_map[j]]

        x_h = []
        for t in range(self.T):
            for v in range(self.n):
                x_h.append(-1000*self.hh_profiles[self.v_map[v]][t])
        
        P = spdiag([Pr]*self.T)
        x_h = matrix(x_h)
        q = matrix(qr*self.T)
        q += (P+P.T)*x_h

        A = matrix(0.0,(self.rN,self.n*self.T)) # won't work for unconstrained
        b = matrix(0.0,(self.rN,1))

        for rn in range(self.rN):
            b[rn] = -self.b[rn]*1000
            if constrain == False:
                for t in range(self.T):
                    A[rn,self.n*t+self.rn_map[rn]] = c_eff*self.t_res/60

            else:
                if self.times[rn][1] < self.times[rn][0]:
                        
                    for t in range(0,self.times[rn][1]):
                        A[rn,self.n*t+self.rn_map[rn]] = c_eff*self.t_res/60

                    for t in range(self.times[rn][0],self.T):
                        A[rn,self.n*t+self.rn_map[rn]] = c_eff*self.t_res/60
                                                
                else:
                        
                    for t in range(self.times[rn][0],self.times[rn][1]):
                        A[rn,self.n*t+self.rn_map[rn]] = c_eff*self.t_res/60
                    
        G = sparse([spdiag([-1.0]*(self.n*self.T)),spdiag([1.0]*(self.n*self.T))])
        h = matrix([Pmax*1000]*(self.n*self.T)+[0.0]*(self.n*self.T))

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
            for t in range(self.T):
                profiles[self.v_map[v]][t] -= x[self.n*t+v]/1000
                
        self.evs = profiles

    def load_flatten(self,Pmax=7,c_eff=0.9,constrain=True):
        profiles = []
        for i in range(self.nH):
            profiles.append([0.0]*self.T)

        q = copy.copy(self.base)*self.n
        q = matrix(q)

        P = sparse([[spdiag([1]*self.T)]*self.n]*self.n)

        A = matrix(0.0,(self.rN,self.n*self.T)) # won't work for unconstrained
        b = matrix(0.0,(self.rN,1))

        for rn in range(self.rN):
            b[rn] = self.b[rn]
            if constrain == False:
                for t in range(self.T):
                    A[rn,self.T*self.rn_map[rn]+t] = c_eff*self.t_res/60

            else:
                if self.times[rn][1] < self.times[rn][0]:
                    maxTime = self.times[rn][0]+self.T-self.times[rn][1]
                    if maxTime*Pmax*self.t_res/60 < b[rn]:
                        print(self.times[rn])
                        print(':(')
                        b[rn] = 0.99*maxTime*Pmax/(60*c_eff)
                        
                    for t in range(0,self.times[rn][1]):
                        A[rn,self.T*self.rn_map[rn]+t] = c_eff*self.t_res/60

                    for t in range(self.times[rn][0],self.T):
                        A[rn,self.T*self.rn_map[rn]+t] = c_eff*self.t_res/60
                                                
                else:
                    maxTime = self.times[rn][1]-self.times[rn][0]
                    if maxTime*Pmax*self.t_res/60 < b[rn]:
                        print(':(')
                        print(self.times[rn])
                        b[rn] = 0.99*maxTime*Pmax/(60*c_eff)
                        
                    for t in range(self.times[rn][0],self.times[rn][1]):
                        A[rn,self.T*self.rn_map[rn]+t] = c_eff*self.t_res/60

            
        G = sparse([spdiag([-1.0]*(self.n*self.T)),spdiag([1.0]*(self.n*self.T))])
        h = matrix([0.0]*(self.n*self.T)+[Pmax]*(self.n*self.T))

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
            for t in range(self.T):
                profiles[self.v_map[v]][t] = x[self.T*v+t]
        
        self.evs = profiles



    def predict_losses(self):
        losses = []

        for t in range(self.T):
            y = [0.0]*self.nH
            for i in range(self.nH):
                y[i] -= self.hh_profiles[i][t]*1000
                y[i] -= self.evs[i][t]*1000

            y = matrix(y)

            losses.append((y.T*self.P0*y+matrix(self.q0).T*y)[0]+self.c)

        return sum(losses)*self.t_res/60
    
    def predict_voltage(self):
        # THIS FUNCTION DOES NOT WORK IN THIS VERSION
        v_ = []
        for i in range(55):
            v_.append([])

        for t in range(1440):
            y = [0.0]*55
            for i in range(55):
                y[i] -= self.hh[i][t]*1000
            for v in range(self.n):
                i = self.map[v]
                y[i] -= self.evs[v][t]*1000

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
        # THIS FUNCTION DOES NOT WORK IN THIS VERSION
        v_ = []
        for i in range(55):
            v_.append([])

        for t in range(1440):
            y = [0.0]*55
            for i in range(55):
                y[i] -= self.hh[i][t]*1000
            for v in range(self.n):
                i = self.map[v]
                y[i] -= self.evs[v][t]*1000

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
        # THIS FUNCTION DOES NOT WORK IN THIS VERSION
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
                y[i] -= self.evs[v][t]*1000
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
        total_load = [0.0]*self.T

        for t in range(self.T):
            for i in range(self.nH):
                total_load[t] += self.hh_profiles[i][t] + self.evs[i][t]

        return total_load


    def get_inidividual_load(self,node):
        base = self.hh[node]
        combined = []
        for t in range(1440):
            combined.append(self.hh[node][t]+self.evs[node][t])

        return base, combined

    def montecarlo_simulation_losses(self,nRuns,hh,evs,outfile):
        losses = {'b':[],'u':[],'lf':[],'lm':[]}

        for mc in range(nRuns):
            self.set_households_NR('')
            network.set_evs_MEA('')

            l = self.predict_losses()
            losses['b'].append(l)

            self.uncontrolled()
            l = self.predict_losses()
            losses['u'].append(l)

            self.load_flatten()
            l = self.predict_losses()
            losses['lf'].append(l)

            self.loss_minimise()
            l = self.predict_losses()
            losses['lm'].append(l)

        with open(outfile,'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['sim','base','unc','load flat','min loss'])
            for mc in range(nRuns):
                writer.writerow([mc,losses['b'][mc],losses['u'][mc],
                                 losses['lf'][mc],losses['lm'][mc]])

    def compare_loading(self,power=3.5):
        # This function will return the loading under the different regimes
        # potentially a stupid idea - may scrap!

        base = self.base
        un = [0]*1440
        lf = [0]*1440
        lm = [0]*1440

        self.uncontrolled(power=power)
        for t in range(1440):
            for j in range(self.nH):
                un[t] += self.evs[j][t]

        self.load_flatten(Pmax=power)
        for t in range(1440):
            for j in range(self.nH):
                lf[t] += self.evs[j][t]

        self.loss_minimise(Pmax=power)
        for t in range(1440):
            for j in range(self.nH):
                lm[t] += self.evs[j][t]

        return [base,un,lf,lm]

import pickle, os, sys, win32com.client, time, scipy.stats
import numpy as np
from dss_python_funcs import *
import numpy.random as rnd
import matplotlib.pyplot as plt
from math import gamma
import dss_stats_funcs as dsf
from matplotlib import cm




class exampleClass:
    """A simple example class"""
    i = 12345
    def f(self):
        return 'hello world'

class linModel:
    """Linear model class with a whole bunch of useful things that we can do with it."""
    
    def __init__(self,feeder,WD):
        self.feeder = feeder
        self.WD = WD # for debugging
        
        with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','chooseLinPoint.pkl'),'rb') as handle:
            lp0data = pickle.load(handle)
        
        self.linPoint = lp0data['k']
        self.loadPointLo = lp0data['kLo']
        self.loadPointHi = lp0data['kHi']
        
        with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','busCoords.pkl'),'rb') as handle:
            self.busCoords = pickle.load(handle)
        with open(os.path.join(WD,'lin_models',feeder,'chooseLinPoint','branches.pkl'),'rb') as handle:
            self.branches = pickle.load(handle)
        
        # load the fixed model, as this always exists
        LMfix = loadLinMagModel(self.feeder,self.linPoint,WD,'Lpt')
        Kyfix=LMfix['Ky'];Kdfix=LMfix['Kd']
        dvBase = LMfix['vKvbase'] # NB: this is different to vBase for ltc/regulator models!

        KyPfix = Kyfix[:,:Kyfix.shape[1]//2]
        KdPfix = Kdfix[:,:Kdfix.shape[1]//2]
        Kfix = np.concatenate((KyPfix,KdPfix),axis=1)
        KfixCheck = np.sum(Kfix==0,axis=1)!=Kfix.shape[1] # [can't remember what this is for...]
        Kfix = Kfix[KfixCheck]
        dvBase = dvBase[KfixCheck]
        
        self.KfixPu = dsf.vmM(1/dvBase,Kfix)
        self.vFixYNodeOrder = LMfix['vYNodeOrder']
    
    def loadNetModel(self,netModel):
        if not netModel:
            # IF using the FIXED model:
            LM = loadLinMagModel(self.feeder,self.linPoint,self.WD,'Lpt')
            Ky=LM['Ky'];Kd=LM['Kd'];bV=LM['bV'];xhy0=LM['xhy0'];xhd0=LM['xhd0']
            vBase = LM['vKvbase']

            xhyN = xhy0/self.linPoint # needed seperately for lower down
            xhdN = xhd0/self.linPoint
            xNom = np.concatenate((xhyN,xhdN))
            b0lo = (Ky.dot(xhyN*self.loadPointLo) + Kd.dot(xhdN*self.loadPointLo) + bV)/vBase # in pu
            b0hi = (Ky.dot(xhyN*self.loadPointHi) + Kd.dot(xhdN*self.loadPointHi) + bV)/vBase # in pu

            KyP = Ky[:,:Ky.shape[1]//2]
            KdP = Kd[:,:Kd.shape[1]//2]
            Ktot = np.concatenate((KyP,KdP),axis=1)
            vYZ = LM['vYNodeOrder']
        elif netModel>0:
            # IF using the LTC model:
            LM = loadNetModel(self.feeder,self.linPoint,self.WD,'Lpt',netModel)
            A=LM['A'];bV=LM['B'];xhy0=LM['xhy0'];xhd0=LM['xhd0']
            vBase = LM['Vbase']
            
            xhyN = xhy0/self.linPoint # needed seperately for lower down
            xhdN = xhd0/self.linPoint
            xNom = np.concatenate((xhyN,xhdN))
            b0lo = (A.dot(xNom*self.loadPointLo) + bV)/vBase # in pu
            b0hi = (A.dot(xNom*self.loadPointHi) + bV)/vBase # in pu
            
            KyP = A[:,0:len(xhy0)//2] # these might be zero if there is no injection (e.g. only Q)
            KdP = A[:,len(xhy0):len(xhy0) + (len(xhd0)//2)]
            
            Ktot = np.concatenate((KyP,KdP),axis=1)
            vYZ = LM['YZ'][LM['v_idx']]
        KtotCheck = np.sum(Ktot==0,axis=1)!=Ktot.shape[1] # [this seems to get rid of fixed regulated buses]
        Ktot = Ktot[KtotCheck]
        
        self.xNom = xNom
        self.b0lo = b0lo[KtotCheck]
        self.b0hi = b0hi[KtotCheck]
        vBase = vBase[KtotCheck]
        
        self.KtotPu = dsf.vmM(1/vBase,Ktot) # scale to be in pu
        self.vTotYNodeOrder = vYZ[KtotCheck]
        
    # def plotNetworkLines(self):
        
    
    def plotFlatVoltage(self):
        Voltages = self.b0hi
        busCoords = self.busCoords
        vYZ = self.vTotYNodeOrder
        branches = self.branches

        bus0 = []
        for yz in vYZ:
            bus0 = bus0+[yz.split('.')[0].lower()]
        bus0 = np.array(bus0)

        for branch in branches:
            bus1 = branches[branch][0].split('.')[0]
            bus2 = branches[branch][1].split('.')[0]
            if branch.split('.')[0]=='Transformer':
                plt.plot((busCoords[bus1][0],busCoords[bus2][0]),(busCoords[bus1][1],busCoords[bus2][1]),Color='#777777')
            else:
                plt.plot((busCoords[bus1][0],busCoords[bus2][0]),(busCoords[bus1][1],busCoords[bus2][1]),Color='#cccccc')

        busVoltages = {}
        vmax = 0.95
        vmin = 1.05
        Vmax = [vmax]
        Vmin = [vmin]
        for bus in busCoords:
            voltage = Voltages[bus0==bus.lower()]
            if not len(voltage):
                busVoltages[bus] = np.nan
            else:
                busVoltages[bus] = np.mean(voltage)
                vmax = max(vmax,np.mean(voltage))
                vmin = min(vmin,np.mean(voltage))
                
                Vmax = Vmax + [vmax]
                Vmin = Vmin + [vmin]
        
        for bus in busCoords:
            if np.isnan(busVoltages[bus]):
                plt.plot(busCoords[bus][0],busCoords[bus][1],'.',Color='#cccccc')
            else:
                score = (busVoltages[bus]-vmin)/(vmax-vmin)
                plt.plot(busCoords[bus][0],busCoords[bus][1],'.',Color=cm.viridis(score))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        
        # plt.plot(Vmax)
        # plt.plot(Vmin)
        # plt.show()
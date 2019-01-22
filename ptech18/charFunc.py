# based on the script from 21/1, 'fft_calcs' in matlab.
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import getpass
from dss_python_funcs import loadLinMagModel
from math import gamma
import time

if getpass.getuser()=='chri3793':
    WD = r"C:\Users\chri3793\Documents\MATLAB\DPhil\epg-psopt\ptech18"

def cf(k,th,t,a,dgn):
    charFuncNeg = ((1 + th*1j*t*a)**(-k))*dgn + (1-dgn) # negation of t: see 'definition' in CF wiki
    return charFuncNeg

intgt = 9
intmax = 10
dgn = 1 - (intgt/intmax) # only this percentage of loads are installed.
Nmc = int(3e5)
iKtot = 25
dVpu = 1e-4;
DVpu = 0.3;

# Nmc = int(1e6)
# mults = [1,-1,2.5,-0.5]
# # mults = [-1.5,-1,-0.5,-0.25]
# # mults = [1.5,1,0.5,0.25]
# k = [2,2,0.9,1.5] # shape
# th = [0.15,0.65,1,0.8]
# dx = 1e-1;
# Dx = 50.0;

# # # to verify with WFA:
# # cf0 = cf(1.,1.,t,1.,1.)
# # plt.subplot(121)
# # plt.plot(t,cf0.real)
# # plt.subplot(122)
# # plt.plot(t,cf0.imag)
# # plt.show()

# g0 = rnd.gamma(k[0],th[0],Nmc)*(rnd.randint(1,intmax+1,Nmc)>intgt)
# g1 = rnd.gamma(k[1],th[1],Nmc)*(rnd.randint(1,intmax+1,Nmc)>intgt)
# g2 = rnd.gamma(k[2],th[2],Nmc)*(rnd.randint(1,intmax+1,Nmc)>intgt)
# g3 = rnd.gamma(k[3],th[3],Nmc)*(rnd.randint(1,intmax+1,Nmc)>intgt)

# gD = mults[0]*g0 + mults[1]*g1 + mults[2]*g2 + mults[3]*g3

# tmax = np.pi/dx # see WB 22-1-19
# Nt = int(Dx/dx)

# t = np.linspace(-tmax,tmax,int(Nt + 1))

# cf0 = cf(k[0],th[0],t,mults[0],dgn);
# cf1 = cf(k[1],th[1],t,mults[1],dgn);
# cf2 = cf(k[2],th[2],t,mults[2],dgn);
# cf3 = cf(k[3],th[3],t,mults[3],dgn);
# cf_tot = cf0*cf1*cf2*cf3

# gDnew = abs(np.fft.fftshift(np.fft.ifft(cf_tot)))/dx

# x = dx*np.arange(-Nt/2,Nt/2 + 1)

# hist = plt.hist(gD,bins=1000,density=True)
# histx = hist[1][:-1]
# histy = hist[0]

# plt.close()
# plt.figure
# plt.plot(histx,histy)
# plt.plot(x,gDnew)
# plt.xlim([-10,10])
# plt.show()

# STEPS:
# 1. Load linear model.
# 2. Choose bus; distribution
# 3. Calculate distribution
# 4. Run MC analysis

# 1. 
fdr_i = 5
fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod','13busRegModRx','usLv']
feeder = fdrs[fdr_i]
lin_point=1.0

LM = loadLinMagModel(feeder,lin_point,WD,'Lpt')
Ky=LM['Ky'];Kd=LM['Kd'];Kt=LM['Kt'];bV=LM['bV'];xhy0=LM['xhy0'];xhd0=LM['xhd0']
vBase = LM['vKvbase']

b0 = Ky.dot(xhy0) + Kd.dot(xhd0) + bV

KyP = Ky[:,:Ky.shape[1]//2]
KdP = Kd[:,:Kd.shape[1]//2]

Ktot = np.concatenate((KyP,KdP),axis=1)


# NB: mean of gamma distribution is k*th.
rndI = 1e4
ld2mean = 0.5 # ie the mean of those generators which install is 1/2 of their load.
xhy0rnd = ld2mean*rndI*np.round(xhy0[:xhy0.shape[0]//2]/rndI)
xhd0rnd = ld2mean*rndI*np.round(xhd0[:xhd0.shape[0]//2]/rndI)
k = 2.0;  # choose the same for all

Th = -np.concatenate((xhy0rnd,xhd0rnd))/k # negative, so that the pds are positive (generation)

x = np.linspace(0,max(Th)*5,int(1e4))
# t = np.linspace(-1e5,1e5,int(1e4 + 1))
# t = np.linspace(-1e4,1e4,int(3e4 + 1))

tmax = np.pi/(vBase[iKtot]*dVpu) # see WB 22-1-19
Nt = DVpu/dVpu
t = np.linspace(-tmax,tmax,int(Nt + 1))

i = 0
cfI = np.zeros((len(Th),len(t)),dtype='complex')

cfTot = np.ones((len(t)),dtype='complex')
gD = np.ones((Nmc))*b0[iKtot]

t0=time.process_time()
print('Start MC:',t0)
for th in Th:
    gI = rnd.gamma(k,th,Nmc)*(rnd.randint(1,intmax+1,Nmc)>intgt)
    gD = gD + Ktot[iKtot,i]*gI
    i+=1

print('Complete; run DFT:',time.process_time())
i=0
for th in Th:
    cfI[i,:] = cf(k,th,t,Ktot[iKtot,i],dgn);
    # ax0.plot(t,abs(cfI[i,:]))
    # ax1.plot(t,np.angle(cfI[i,:])*180/np.pi)
    cfTot = cfTot*cfI[i,:]
    # pdf = (x**(k-1))*np.exp(-x/th)/((th**k)*gamma(k))
    # plt.plot(x,pdf)
    # gI = rnd.gamma(k,th,Nmc)*(rnd.randint(1,intmax+1,Nmc)>intgt)
    # gD = gD - Ktot[iKtot,i]*gI
    i+=1
# plt.show()

# plt.subplot(121)
# plt.plot(t,abs(cfTot))
# plt.subplot(122)
# plt.plot(t,np.angle(cfTot)*180/np.pi)
# plt.show()

dV = np.pi/tmax
vDnew = abs(np.fft.fftshift(np.fft.ifft(cfTot)))*vBase[iKtot]/dV
N = len(t)
V = dV*np.arange(-N/2,N/2)
Vpu = (b0[iKtot]+V)/vBase[iKtot]

v0 = b0[iKtot]/vBase[iKtot]
print('Complete.',time.process_time())

hist = plt.hist(gD/vBase[iKtot],bins=1000,density=True)
histx = hist[1][:-1]
histy = hist[0]
plt.close()
plt.plot(histx,histy)

plt.plot(Vpu,vDnew)

plt.xlim((0.90,1.1))
ylm = plt.ylim()
plt.plot(v0*np.ones(2),ylm,'--')
plt.plot(0.95*np.ones(2),ylm,'r:')
plt.plot(1.05*np.ones(2),ylm,'r:')
plt.ylim(ylm)
plt.grid(True)
plt.show()


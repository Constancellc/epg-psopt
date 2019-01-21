# based on the script from 21/1, 'fft_calcs' in matlab.
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt


def cf(k,th,t,a,dgn):
    charFunc = ((1 - th*1j*t*a)**(-k))*dgn + (1-dgn);
    return charFunc

mults = [1,-1,2.5,-0.5]
k = [2,2,0.9,1.5] # shape
th = [0.15,0.65,1,0.8]

intgt = 2
intmax = 5
dgnW = 1 - (intgt/intmax)

Nmc = int(1e5)

g0 = rnd.gamma(k[0],th[0],Nmc)*(rnd.randint(1,intmax+1,Nmc)>intgt)
g1 = rnd.gamma(k[1],th[1],Nmc)*(rnd.randint(1,intmax+1,Nmc)>intgt)
g2 = rnd.gamma(k[2],th[2],Nmc)*(rnd.randint(1,intmax+1,Nmc)>intgt)
g3 = rnd.gamma(k[3],th[3],Nmc)*(rnd.randint(1,intmax+1,Nmc)>intgt)

gD = mults[0]*g0 + mults[1]*g1 + mults[2]*g2 + mults[3]*g3

t = np.linspace(-100,100,int(1e3 + 1),dtype=complex)

cf0 = cf(k[0],th[0],t,mults[0],dgnW);
cf1 = cf(k[1],th[1],t,mults[1],dgnW);
cf2 = cf(k[2],th[2],t,mults[2],dgnW);
cf3 = cf(k[3],th[3],t,mults[3],dgnW);
cf_tot = cf0*cf1*cf2*cf3

dt = np.diff(t)[0]
dx = np.real(1/(max(t) - min(t)))
N = len(t)

gDnew = abs(np.fft.fftshift(np.fft.ifft(cf_tot)))/(2*np.pi*dx)

x = -dx*np.arange(-N/2,N/2)*(2*np.pi)

hist = plt.hist(gD,bins=1000,density=True)
histx = hist[1][:-1]
histy = hist[0]

dnx = x[0] - x[1]
dhx = histx[1] - histx[0]

plt.close()
plt.figure
plt.plot(histx,histy)
plt.plot(x,gDnew)
plt.xlim([-10,10])
plt.show()


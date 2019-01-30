import numpy as np
from math import gamma
import mpmath
import matplotlib.pyplot as plt

# NORMAL
def pdf_nml(mu,sg2,x): # sg2 as sigma**2
    pdf = np.exp(-( ((x-mu)**2)/(2*sg2) ) )/np.sqrt(2*np.pi*sg2)
    return pdf

def cf_nml(mu,sg2,t):
    charFunc = np.exp(-1j*mu*t - (sg2*(t**2)/2)) # negation of t: see 'definition' in CF wiki
    return charFunc


# GAMMA
def cf_gm(k,th,t):
    charFunc = ((1 + th*1j*t)**(-k)) # negation of t: see 'definition' in CF wiki
    return charFunc

def pdf_gm(k,th,x):
    pdf = (x**(k-1))*np.exp(-x/th)/(gamma(k)*(th**k))
    return pdf

def cdf_gm(k,th,x):
    cdf = np.zeros(x.shape)
    for i in range(len(x)):
        cdf[i] = mpmath.gammainc(k,a=0,b=x[i]/th,regularized=True)
    return cdf

# GAMMA CUTOFF
def pdf_gm_Xc(k,th,x,xC):
    dx = x[1]-x[0] 
    pdf0 = pdf_gm(k,th,x)
    cdfXc = cdf_gm(k,th,np.array([xC]))
    pdf = pdf0*(x<xC) + ((x==xC)*(1 - cdfXc))/dx
    return pdf

# GAMMA DEGENERATE
def cf_gm_dgn(k,th,t,a,dgn):
    charFuncNeg = ((1 + th*1j*t*a)**(-k))*dgn + (1-dgn); # negation of t: see 'definition' in CF wiki
    return charFuncNeg

# UNIFORM
def cf_uni(a,b,t):
    # following convention from wiki, with negated t:
    charFuncNeg = (np.exp(-1j*b*t) - np.exp(-1j*a*t))/(-1j*t*(b - a))
    return charFuncNeg

# BINOMIAL-1
def pdf_bn1(x0,p,x):
    dx = x[1]-x[0]
    pdf = np.zeros(x.shape)
    pdf[x==0.] = (1-p)/dx
    pdf[x==x0] = p/dx
    return pdf
def cf_bn1(x0,p,t):
    charFunc = (1-p) + p*np.exp(-1j*x0*t) # as in a 1x binomial trial
    return charFunc

# CREATING x,t FROM dx, Dx/dt, Dt:

def dy2Yz(dy,Dy): # see WB 24-1-19
    Nt = int(Dy/dy)
    dz = 2*np.pi/Dy
    Y = dy*np.arange(-Nt/2,Nt/2 + 1)
    Z = dz*np.arange(-Nt/2,Nt/2 + 1)
    return Y,Z

def dy2YzR(dy,Dy): # Real version (positive t)
    Nt = int(Dy/dy)
    dz = 2*np.pi/Dy
    Y = dy*np.arange(-Nt/2,Nt/2 + 1)
    Z = dz*np.arange(0,Nt/2 + 1)
    return Y,Z




# VVVVVVVVVVVVVVVVV TESTING VVVVVVVVVVVVVVVVV
k = 4.214235442805903
th = 1.260966500373797
dx = 1e-1
Dx = 500

x,t = dy2YzR(dx,Dx)
x = x+(Dx/2)
xC = 4
pdf = pdf_gm(k,th,x)
pdfN = pdf_gm_Xc(k,th,x,xC)

ft_gm = np.fft.rfft(pdf)*dx # calculated version    
ft_cf = cf_gm(k,th,t)


# plt.plot(t,ft_gm.real)
# plt.plot(t,ft_gm.imag)
# plt.plot(-t,ft_gm.real)
# plt.plot(-t,-ft_gm.imag)
# plt.show()

x0 = 10.
p = 1/2
pdfB = pdf_bn1(x0,p,x)
# pdfB[x==0] = p*1/dx

# plt.plot(x,pdfB)
# plt.show()

ftB = np.fft.rfft(pdfB)*dx
cfB = cf_bn1(x0,p,t)

cfN = cfB*cfB

ftN = np.fft.rfft(pdfN)

# plt.plot(t,ftN.real)
# plt.plot(t,ftN.imag)
# plt.show()

# pdfBn = np.fft.irfft(cfN)
# plt.plot(x[:-1],pdfBn)
# plt.show()

# plt.plot(t,ftB.imag)
# plt.plot(t,cfB.real)
# plt.plot(t,cfB.imag)
# plt.show()

# cfU = cf_uni(0,10.,t)
# plt.plot(t,cfU.real)
# plt.plot(t,cfU.imag)
# plt.show()




# # Looking at cf as gamma k grows, constant k*th =====
# K = np.linspace(0.1,20)
# mu = 1

# dt = 1e-2
# Dt = 100
# t = np.arange(-Dt,Dt,dt)

# dx = 1e-3
# Dx = 3
# x = np.arange(0,Dx,dx)

# k = 50

# # for k in K:
# th = mu/k
# sg2 = k*(th**2)
# cfGm = cf_gm(k,th,t)
# cfNml = cf_nml(mu,sg2,t)
# pdfGm = pdf_gm(k,th,x)
# pdfNml = pdf_nml(mu,sg2,x)

# plt.plot(t,cfGm.real,'r')
# plt.plot(t,cfGm.imag,'b')
# plt.plot(t,cfNml.real,'r--')
# plt.plot(t,cfNml.imag,'b--')
# plt.show()

# plt.plot(x,pdfGm,'r')
# plt.plot(x,pdfNml,'r--')
# # plt.semilogy(x,pdfGm,'r')
# # plt.semilogy(x,pdfNml,'r--')
# plt.show()
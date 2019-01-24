import numpy as np
from math import gamma
import mpmath
import matplotlib.pyplot as plt

def cf_gm(k,th,t):
    charFunc = ((1 + th*1j*t)**(-k)) # negation of t: see 'definition' in CF wiki
    return charFunc

def cf_gm_dgn(k,th,t,a,dgn):
    charFuncNeg = ((1 + th*1j*t*a)**(-k))*dgn + (1-dgn); # negation of t: see 'definition' in CF wiki
    return charFuncNeg

def pdf_gm(k,th,x):
    pdf = (x**(k-1))*np.exp(-x/th)/(gamma(k)*(th**k))
    return pdf

def pdf_gm_Xc(k,th,x,xC):
    dx = x[1]-x[0] 
    pdf0 = pdf_gm(k,th,x)
    cdfXc = cdf_gm(k,th,np.array([xC]))
    pdf = pdf0*(x<xC) + ((x==xC)*(1 - cdfXc))/dx
    return pdf
def cdf_gm(k,th,x):
    cdf = np.zeros(x.shape)
    for i in range(len(x)):
        cdf[i] = mpmath.gammainc(k,a=0,b=x[i]/th,regularized=True)
    return cdf

def pdf_bn1(x0,p,x):
    dx = x[1]-x[0]
    pdf = np.zeros(x.shape)
    pdf[x==0.] = (1-p)/dx
    pdf[x==x0] = p/dx
    return pdf
    
def cf_bn1(x0,p,t):
    charFunc = (1-p) + p*np.exp(-1j*x0*t) # as in a 1x binomial trial
    return charFunc

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

k = 4.214235442805903
th = 1.260966500373797
dx = 1e-1
Dx = 500

x,t = dy2YzR(dx,Dx)
x = x+(Dx/2)
xC = 4
pdf = pdf_gm(k,th,x*(x>=0))
# pdfN = pdf_gm_Xc(k,th,x*(x>=0),xC)

ft_gm = np.fft.rfft(pdf)*dx # calculated version    
ft_cf = cf_gm(k,th,t)


plt.plot(t,ft_gm.real)
plt.plot(t,ft_gm.imag)
plt.plot(-t,ft_gm.real)
plt.plot(-t,-ft_gm.imag)
plt.show()

x0 = 10.
p = 1/2
pdfB = pdf_bn1(x0,p,x)
# pdfB[x==0] = p*1/dx

# plt.plot(x,pdfB)
# plt.show()

ftB = np.fft.rfft(pdfB)*dx
cfB = cf_bn1(x0,p,t)

cfN = cfB*cfB

# pdfBn = np.fft.irfft(cfN)
# plt.plot(x[:-1],pdfBn)
# plt.show()

# plt.plot(t,ftB.imag)
# plt.plot(t,cfB.real)
# plt.plot(t,cfB.imag)
# plt.show()
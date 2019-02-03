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
    Nt = int(Dy/dy) + 1
    dz = 2*np.pi/Dy
    Y = dy*np.arange(-(Nt//2),(Nt//2) + 1)
    Z = dz*(Nt - 1)*np.fft.fftfreq(Nt)
    return Y,Z

def dy2YzR(dy,Dy): # Real version (positive t)
    Nt = int(Dy/dy) + 1 # assume symmetric around zero.
    dz = 2*np.pi/Dy
    Y = dy*np.arange(-(Nt//2),Nt//2 + 1)
    Z = dz*(Nt-1)*np.fft.rfftfreq(Nt) # note different freqs if odd/even.
    return Y,Z

# VVVVVVVVVVVVVVVVV TESTING VVVVVVVVVVVVVVVVV
# # GETTING the RFFT working for faster processing: =============
# dx = 1e-1
# # dx = 5.0 # for easier visualization of correct results.
# Dx = 50.0

# x,t = dy2YzR(dx,Dx)
# x = x+(Dx/2)

# # testing binomial pdf/cdf and real ffts
# x0 = 10.
# p = 1/2
# pdfB = pdf_bn1(x0,p,x)

# ftB = np.fft.rfft(pdfB)*dx
# ftBN = ftB*ftB
# cfB = cf_bn1(x0,p,t)
# cfN = cfB*cfB

# pdfBn = np.fft.irfft(cfN,n=len(x)) # len needed as odd 'time domain' signal (see below)
# pdfBn_ift = np.fft.irfft(ftBN,n=len(x))

# # check the cf and ft are the same --- 
# plt.plot(t,ftB.real)
# plt.plot(t,ftB.imag)
# plt.plot(t,cfB.real)
# plt.plot(t,cfB.imag)
# plt.show()

# plt.subplot(121)
# plt.semilogy(t,abs(ftB+ np.finfo(np.float64).eps)/abs(cfB+ np.finfo(np.float64).eps))
# plt.subplot(122)
# plt.plot(t,np.angle(ftB+ np.finfo(np.float64).eps) - np.angle(cfB+ np.finfo(np.float64).eps))
# plt.show()

# # Finally, demonstrate that the final result is as expected.
# plt.subplot(221)
# plt.plot(x,pdfBn_ift + np.finfo(np.float64).eps)
# plt.subplot(222)
# plt.semilogy(x,pdfBn_ift + np.finfo(np.float64).eps)
# plt.subplot(223)
# plt.plot(x,pdfBn)
# plt.subplot(224)
# plt.semilogy(x,pdfBn  + np.finfo(np.float64).eps)
# plt.show()


# # REAL FFT with ODD POINTS =============
# # here we need to use n=N option to return the correct output.
# x = np.arange(-5,6,1)
# fx = np.zeros(len(x))
# fx[0]=.333; fx[5]=.333; fx[-1]=.333

# ft = np.fft.rfft(fx)
# fft = np.fft.fftshift(np.fft.fft(fx))

# fx_t0 = np.fft.irfft(ft)
# fx_t = np.fft.irfft(ft,n=11) # this returns the correct ifft.

# # Looking at cf as gamma k grows, constant k*th =============
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
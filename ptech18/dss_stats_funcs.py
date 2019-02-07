import numpy as np
from math import gamma
import mpmath
import matplotlib.pyplot as plt
import scipy.special # for the beta function
import scipy.stats
import time
def get_dx(x):
    return x[1] - x[0]
def get_Dx(x):
    return max(x) - min(x)

# NORMAL
def pdf_nml(mu,sg2,x): # sg2 as sigma**2
    pdf = np.exp(-( ((x-mu)**2)/(2*sg2) ) )/np.sqrt(2*np.pi*sg2)
    return pdf

def cf_nml(mu,sg2,t):
    charFunc = np.exp(-1j*mu*t - (sg2*(t**2)/2)) # negation of t: see 'definition' in CF wiki
    return charFunc

def cf_freqs(k,dx): # NB assumes x is positive only. 
    N = []
    w = []
    N = N + [int(np.floor(abs(k/dx)))]
    N = N + [int(np.ceil(abs(k/dx)))]
    w = w + [abs(k/dx) - np.floor(abs(k/dx))]
    w = w + [np.ceil(abs(k/dx)) - abs(k/dx)]
    return N,w
    
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

# DEGENERATE
def pdf_dgn(x):
    dx = x[1]-x[0]
    pdf = np.zeros(len(x))
    pdf[x==0.0]=1/dx
    return pdf

# GAMMA DEGENERATE
def cf_gm_dgn(k,th,t,a,dgn):
    cf = ((1 + th*1j*t*a)**(-k))*dgn + (1-dgn); # negation of t: see 'definition' in CF wiki
    return cf

# GAMMA CUTOFF DEGENERATE
def pdf_gm_XD(k,th,x,a,dgn,Xc):
    dx = get_dx(x)
    pdfD = pdf_dgn(x)
    pdfX = pdf_gm_Xc(k,th*a,x,Xc*a) # <=== HERE
    pdf = (pdfD*(1-dgn)) + (pdfX*dgn)
    return pdf

def cf_gm_XD(k,th,t,a,dgn,Xc):
    cfD = ((1 + th*1j*t*a)**(-k))*dgn + (1-dgn); # negation of t: see 'definition' in CF wiki
    cfX = cf_uni(0,Xc,t)
    cf = cfD

# UNIFORM
def cf_uni(a,b,t):
    # following convention from wiki, with negated t:
    cf = (np.exp(-1j*b*t) - np.exp(-1j*a*t))/(-1j*t*(b - a))
    return cf

# BINOMIAL-1
def pdf_bn1(x0,p,x):
    dx = get_dx(x)
    pdf = np.zeros(x.shape)
    pdf[x==0.] = (1-p)/dx
    pdf[x==x0] = p/dx
    return pdf

def cf_bn1(x0,p,t):
    cf = (1-p) + p*np.exp(-1j*x0*t) # as in a 1x binomial trial
    return cf

def cf_bn1_apx(x0,fr,p,t):
    # This version of the function checks for the applicability of the model.
    cf = (1-p) + p*(fr[0]*np.exp(-1j*x0[0]*t) + fr[1]*np.exp(-1j*x0[1]*t))
    return cf

# BETA (used for sampling over a range of values)    
def pdf_beta(a,b,x):
    pdf = (x**(a-1))*((1-x)**(b-1))/scipy.special.beta(a,b)
    return pdf

# SHIFTED MEAN charecteristic functions:
def cf_gm_sh(k,th,t):
    cf = cf_gm(k,th,t)*np.exp(1j*k*th*t)
    return cf
    
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



# # Looking at how to add a bunch of PDFs together with a wide range of constant values. ==========
# th = 4.5 # scale
# k = 2.0 # shape

# mu = k*th
# sgm2 = k*(th**2)

# K = np.random.triangular(-1.0,0.0,1.0,size=int(1e2)) # Modelling the Ktot matrix.

# muAll = sum(K)*mu
# sgm2all = (sum(K**2))*sgm2

# nMc = int(1e5)

# xOut0 = np.zeros((nMc))
# xOut1 = np.zeros((nMc))
# for km in K:
    # xOut0 = xOut0 + km*np.random.normal(mu,sgm2**0.5,size=nMc)
    # xOut1 = xOut1 + np.random.normal(km*mu,abs(km)*(sgm2**0.5),size=nMc)

# plt.hist(xOut0,bins=int(1e2))
# plt.hist(xOut1,bins=int(1e2)); plt.show()

# print('Sampled mean:')
# print(np.mean(xOut0))
# print(np.mean(xOut1))
# print('Sampled var:')
# print(np.var(xOut0))
# print(np.var(xOut1))
# print('Calc mean/var:')
# print(muAll)
# print(sgm2all)


# # ===== Finally, experiment with the sum of messy distributions
# K = np.random.triangular(-1.0,0.0,1.0,size=int(1e2)) # Modelling the Ktot matrix.
# k = 2.5
# th = 4.0
# dgn = 0.1
# dgn = 1.0
# Xc = 10.0
# nMc = int(1e4)

# Dx = 20.0
# x = np.linspace(0.0,Dx,int(3e3)+1)

# dx = get_dx(x)

# t = dy2YzR(dx,Dx)[1]
# pdfXD = pdf_gm_XD(k,th,x,1.0,dgn,Xc)
# pdfsmErr = 100.*(1-sum(pdfXD)*dx)
# print(pdfsmErr)

# pdfXDa = pdf_gm_XD(k,th,x,0.5,dgn,Xc)
# pdfsmErrA = 100.*(1-sum(pdfXDa)*dx)
# print(pdfsmErrA)

# # plt.plot(x,pdfXD)
# # plt.plot(x,pdfXDa)
# # plt.show()

# cfXD = np.fft.rfft(pdfXD)*dx
# cfXDa = np.fft.rfft(pdfXDa)*dx

# # cfXDs = np.interp(t,t[::2],cfXD[:len(t)//2+1])
# cfXDs = np.interp(t,t[::2],cfXD[:len(t)//2+1])

# cfXDs = cfXD[::2]
# cfXDs = np.interp(t,t[::2],cfXDs)

# plt.plot(t,cfXD.real,'r.')
# plt.plot(t,cfXD.imag,'b--')
# plt.plot(t,cfXDa.real,'r:')
# plt.plot(t,cfXDa.imag,'b:')
# plt.plot(t[0:len(t)//2+1],cfXDa[0::2].real,'g:')
# plt.plot(t[0:len(t)//2+1],cfXDa[0::2].imag,'g:')

# plt.plot(t,cfXDs.real,'b.')
# plt.plot(t,cfXDa.real,'r.')
# plt.show()

# plt.plot(x,np.fft.irfft(cfXD,n=len(x)))
# plt.plot(x,np.fft.irfft(cfXDa,n=len(x)))
# plt.plot(x,np.fft.irfft(cfXDs,n=len(x)))
# plt.show()

# xOut = np.zeros((nMc))
# for km in K:
    # randD = np.random.randint(1,11,size=nMc)
    # randG = np.random.gamma(k,th,size=nMc)
    # randX = np.min(np.array([randG,Xc*np.ones(len(randG))]),axis=0)
    # rand = (randD>9)*randX
    # xOut = xOut + km*rand

# plt.hist(rand,bins=int(1e2),density=True)
# plt.plot(x,pdfXD)
# plt.show()

# plt.hist(xOut,bins=int(1e2),density=True)
# plt.show()





# # ===== STUDYING the impact of x0 on reproducibility of signals.
# dx = 2.0
# Dx = 20.0
# x,t = dy2YzR(dx,Dx)
# x= x+(Dx/2)
# x0 = 2.0
# x0 = 2.01
# p = 0.40

# absxp = abs(x - x0)
# dxpArg = np.argsort(absxp)[0:2]
# frxp = 1.0 - absxp[dxpArg]/dx

# cf = cf_bn1(x0,p,t)
# cfa = cf_bn1_apx(x[dxpArg],frxp,p,t)

# pdf = np.fft.irfft(cf,n=len(x))
# pdfApx = np.fft.irfft(cfa,n=len(x))
# plt.subplot(121)
# plt.plot(x,pdf,'o-')
# plt.plot(x,pdfApx,'.-')
# plt.grid(True)
# plt.subplot(122)
# plt.semilogy(x,abs(pdf)+np.finfo(float).eps,'o-')
# plt.semilogy(x,abs(pdfApx)+np.finfo(float).eps,'.-')
# plt.grid(True)
# plt.show()


# FROM HERE DOWNWARDS is work from 06/02 for multiplying lots of CFs together to create new pdfs

# # ===== USE the beta-function for nice sampling for practise multiplying together.
# def pdf_beta(a,b,x):
    # pdf = (x**(a-1))*((1-x)**(b-1))/scipy.special.beta(a,b)
    # return pdf

# x = np.linspace(0.0,1.0)
# x = np.linspace(-1.0,1.0)
# a = 8.0
# b = 8.0
# pdfB = 0.5*pdf_beta(a,b,0.5*(x + 1))
# plt.plot(x,pdfB); plt.show()

# # ====== SAMPLE and adda bunch of pdfs together.
# Nk = int(1e4)
# Nk = int(1e1)
# Nk = int(1e4)

# aBeta = 8.0
# bBeta = 8.0

# fLo = -1.0
# fLo = 0.0
# fLo = 0.5
# xMl = 1-fLo

# Ktot = xMl*(np.random.beta(aBeta,bBeta,Nk)) + fLo

# # # show that this has worked:
# # xk = np.linspace(fLo,1.0)
# # plt.hist(Ktot,bins=100,density=True)
# # plt.plot(xk,pdf_beta(aBeta,bBeta,np.linspace(0.0,1.0))/xMl)
# # plt.show()

# nMc = int(3e4)
# k = 2
# th = 0.1

# # Demonstrate that variance goes up but mean stays at zero:
# std = []
# mns = []
# nkSet = np.logspace(1,4,10)
# nkSet = np.logspace(1,3,10)
# for nk in nkSet:
    # print('Nk:',int(nk))
    # Ktot = xMl*(np.random.beta(aBeta,bBeta,int(nk))) + fLo
    # mcTot = np.zeros((nMc))
    # for km in Ktot:
        # mcTot = mcTot + ((np.random.gamma(k,th,size=nMc) - k*th)*km) # zero mean
    # std = std + [np.std(mcTot)]
    # mns = mns + [abs(np.mean(mcTot))]

# plt.subplot(121)
# plt.loglog(nkSet,std,'x')
# plt.loglog(nkSet,(nkSet**0.5)*0.07)
# plt.ylabel('Standard deviation')
# plt.xlabel('Number of samples')
# plt.subplot(122)
# plt.loglog(nkSet,mns,'x')
# plt.loglog(nkSet,nkSet*0.1)
# plt.ylabel('Mean')
# plt.xlabel('Number of samples')
# plt.show()


# # Estimate the size of dx, Dx required to model K even when it is reasonably small (dEps)
# dEps = 1/1e2;
# dx0 = 2e-4;
# dx0 = 2e-3;
# Dx0 = 1.0;
# x0 = dy2YzR(dx0,Dx0)[0]
# x0 = x0 + Dx0/2
# pdfG = pdf_gm(k,th*dEps,x0)
# print('PDF integral at dEps:',sum(pdfG*dx0))


# # # demonstrate this working for one function:
# NkScale = 1.0 + np.sqrt(Nk)*0.5
# Dx = np.round(NkScale*2*Dx0) # more variance for more PDFs.
# dx = dx0
# x,t = dy2YzR(dx,Dx)

# K2a = -1.0

# cfG = cf_gm(k,th,t)
# cfGa = cf_gm(k,th*K2a,t)

# cfGsh = cf_gm_sh(k,th,t)
# cfGsha = cf_gm_sh(k,th*K2a,t)

# cfG2 = cfG*cfGa
# pdf2 = np.fft.fftshift(np.fft.irfft(cfG2,n=len(x)))
# cfG2sh = cfGsh*cfGsha
# pdf2sh = np.fft.fftshift(np.fft.irfft(cfG2sh,n=len(x)))
# xsh = x + k*th*(1 + K2a)

# plt.plot(x,pdf2); 
# plt.plot(xsh,pdf2sh); 
# plt.grid(True); plt.show()

# # ==== DEMONSTRATE the summation working nicely.
# cfTot = np.ones((len(t)))
# mcTot = np.zeros((nMc))

# print('--- Begin CF computations analysis ---\n',time.process_time())
# Ktot = xMl*(np.random.beta(aBeta,bBeta,Nk)) + fLo

# for km in Ktot:
    # cfTot = cfTot*cf_gm_sh(k,th*km,t)
# print('--- Begin MC analysis ---\n',time.process_time())

# for km in Ktot:
    # mcTot = mcTot + (np.random.gamma(k,th,size=nMc)*km)
# print('End\n',time.process_time())

# xsh = (k*th*sum(Ktot)) + x
# pdfSum = np.fft.fftshift(np.fft.irfft(cfTot,n=len(x)))
# hst = plt.hist(mcTot,bins=100,density=True)
# plt.plot(xsh,pdfSum/dx ); plt.show()



# # ===== PDF of a normal distribution using scipy.stats
# import scipy.stats
# x = np.linspace(-4,4)
# pdfNorm = scipy.stats.norm.pdf(x,scale=2)
# plt.plot(x,pdfNorm); plt.show()













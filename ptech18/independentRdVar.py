import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

A = np.array([[1.0,0,0,0,0],[1.0,0,0,0,0],[1.0,0,0,0,0],[1.0,0,0,0,0],[0,1.0,0,0,0]])
A = A + 0.3*np.random.normal(size=A.shape)
nZ = 2

[U,S,Vh] = np.linalg.svd(A)
Smat = np.zeros(A.shape)
i=0
for s in S:
    Smat[i,i] = s
    i+=1
Uz = U[:,:nZ]

nMc = int(3e3)
k = 1.0
X = np.random.normal(size=(len(A),nMc))
# X = (np.random.gamma(k,size=(len(A),nMc)) - k)/(np.sqrt(k))
# X = np.random.laplace(loc=0.0,scale=1/np.sqrt(2),size=(len(A),nMc))

Y = A.dot(X)
Az = Uz.T.dot(A)
Z = Az.dot(X)
Yz = Uz.dot(Z)

Ylim = np.ones((len(A)))

Yout = np.zeros(nMc)
for i in range(nMc):
    Yout[i] = np.all(Y[:,i]<Ylim)

yFrac = sum(Yout)/nMc

Zout = np.zeros(nMc)
for i in range(nMc):
    Zout[i] = np.all(Yz[:,i]<Ylim)

zFrac = sum(Zout)/nMc

print('y frac:',yFrac)
print('z frac:',zFrac)
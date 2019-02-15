import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# A = 0.7*np.array([[1.,0.,0.],[1.,0.1,0.],[0.,0.,1.]])
# A = 2.0*np.eye(2)
# A = np.array([[1.0,0.0],[0.0,2.0]])
# A = np.array([[1.0,0.0],[0.0,-1.0]]) # reflection in x=0
A = np.eye(2)
# ROT
th = -0.9*np.pi
A = np.array([[np.cos(th),np.sin(th)],[-np.sin(th),np.cos(th)]]).dot(A)

# A = 2.0*np.array([[1.0,0.0],[1.0,1.0]])
# A = 2.0*np.array([[1.0,0.0],[1.0,0.0]])

[U,S,Vh] = np.linalg.svd(A)
S[S==0] = np.inf
Sinv = np.diag(1/S)
# say: Y = AX, with all Y < 1.0, X~N(0,1). 

nMc = int(3e4)
k = 0.1
X = np.random.normal(size=(len(A),nMc))
X = (np.random.gamma(k,size=(len(A),nMc)) - k)/(np.sqrt(k))
# X = np.random.laplace(loc=0.0,scale=1/np.sqrt(2),size=(len(A),nMc))
Y = A.dot(X)

Ylim = np.ones((len(A)))
YlimX = Sinv.dot(np.ones((len(A))))
# YlimX = Sinv.dot(U.T.dot(np.ones((len(A)))))

Yout = np.zeros(nMc)
for i in range(nMc):
    Yout[i] = np.all(Y[:,i]<Ylim)

yFrac = sum(Yout)/nMc

Ainv = np.linalg.inv(A)
# Ainv = ((Vh.T).dot(Sinv)).dot(U.T)
Adet = np.linalg.det(A)

Xlim0 = Ainv.dot(Ylim)

Xout3 = np.zeros(nMc)
Xout4 = np.zeros(nMc)
xoutJ = np.zeros(len(A))
for i in range(nMc):
    j=0
    for xlim in Xlim0:
        if xlim>0:
            xoutJ[j] = X[j,i]<Xlim0[j]
        else:
            xoutJ[j] = X[j,i]>Xlim0[j]
        j+=1
    Xout3[i] = np.all(xoutJ)
    Xout4[i] = np.all(X[:,i]<YlimX)

xFrac3 = sum(Xout3)/nMc
xFrac4 = sum(Xout4)/nMc

print('x frac3:',xFrac3)
print('x frac4:',xFrac4)
print('y frac:',yFrac)


# plt.subplot(121)
# plt.hist(Y.T,bins=30);
# plt.subplot(122)
# plt.hist(X.T,bins=30)
# plt.show()
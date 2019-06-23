import lineariseDssModels, sys, os, pickle, random, time
from importlib import reload
import numpy as np
from dss_python_funcs import vecSlc, getBusCoords, getBusCoordsAug, tp_2_ar
import matplotlib.pyplot as plt
from lineariseDssModels import dirPrint
import dss_stats_funcs as dsf



from scipy import sparse

FD = sys.argv[0]

fdrs = ['eulv','n1f1','n1f2','n1f3','n1f4','13bus','34bus','37bus','123bus','8500node','37busMod','13busRegMod3rg','13busRegModRx','13busModSng','usLv','123busMod','13busMod','epri5','epri7','epriJ1','epriK1','epriM1','epri24','4busYy','epriK1cvr','epri24cvr']


def main(fdr_i=5,modelType=None,linPoint=1.0,pCvr=0.8,method='fpl',saveModel=False,pltSave=False):
    reload(lineariseDssModels)
    
    SD = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=fdr_i,linPoints=[linPoint],pCvr=pCvr,
                                                modelType=modelType,method=method,saveModel=saveModel,SD=SD,pltSave=pltSave)
    return blm


def getTrilIdxs(n):
    idxI = []
    idxJ = []
    for i in range(n):
        # idxI = idxI + [i]*(i+1)
        # idxJ = idxJ + list(range(i+1))
        for j in range(i+1):
            idxI.append(i)
            idxJ.append(j)
    
    return idxI, idxJ

sdf = getTrilIdxs(len(H))
sdf = getTrilIdxs(4)
Hsdf = H[sdf]

# self = main(5,'buildSave',linPoint=1.0)
# self = main(5); self.printQpSln()
# plt.plot( np.log10(np.abs(dsf.vmM(1/self.iXfmrLims,self.Mc2i).T)),'x-' ); plt.show()

# self = main(5,'loadAndRun'); self.showQpSln()

# self = main(8,'loadAndRun'); self.showQpSln()

self = main(25,'loadOnly')
self.runQpSet()

t = time.time(); 

# self = main(25,'loadOnly')

def QtoH(Q):
    L,D = ldl(Q)[0:2] #NB not implemented here, but could also use spectral decomposition.
    if min(np.diag(D))<0:
        print('Warning: not PSD, removing negative D elements')
        D[D<0]=0
        H = L.dot(np.sqrt(D)) # get rid of the smallest eigenvalue,
    else:
        H = L.dot(np.sqrt(D)) # get rid of the smallest eigenvalue,
    print('Q error norm:',np.linalg.norm( H.dot(H.T) - Q ))
    return H


# self.setupConstraints()
# self.oneHat,self.x0 = self.remControlVrlbs('full','opCst')
# Mc2iCtrl = self.Mc2i.dot(self.oneHat)

# a2iCtrl = self.aIxfmr + self.Mc2i.dot(self.X0ctrl)
# score = []
# ii=0
# for lim in self.iXfmrLims:
    # mc2iCpx = Mc2iCtrl[ii]
    # a2iCpx = a2iCtrl[ii]
    # aCpx = np.r_[ a2iCpx.real, a2iCpx.imag].reshape((2,1))
    # aCpx = np.r_[ a2iCpx.real, a2iCpx.imag].reshape((2,1))
    # mCpx = np.r_[ [mc2iCpx.real],[mc2iCpx.imag] ]
    # # score.append( np.linalg.norm( mc2iCpx/(lim*self.iScale*np.ones((2,1)) - aCpx) ) )
    # score.append( np.linalg.norm( mc2iCpx/(lim*self.iScale) ) )
    # # score.append( np.linalg.norm( lim*self.iScale*np.ones((2,1)) - aCpx)/np.linalg.norm(mc2iCpx) )
    # ii+=1

# score = np.array(score)
# plt.plot(np.log10(score[score>1e-3])); plt.show()
# plt.plot(np.log10(score[score>1e-3])); plt.show()

# Gv,hv = self.getVghConstraints()
# Gx,hx = self.getXghConstraints('hcGen')

# G = np.r_[Gv,Gx]
# h = np.r_[hv,hx]

# oneHatSp = sparse.csc_matrix(self.oneHat)

# Gp = G.dot(self.oneHat)

# GpSp = aDotBsp(G,oneHatSp)

# hp = h

# scores = np.sum(abs(GpSp),axis=1)
# scoresNorm = scores/hp.flatten()

# GpNz = (G.dot(oneHat)!=0)
# self.oneHat,self.x0 = self.remControlVrlbs('full','opCst')

# xScore = ( self.iScale*self.iXfmrLims - np.abs(self.aIxfmr) )/np.sqrt(np.sum(np.abs(Mc2iCtrl),axis=1))
# plt.plot( np.log10(xScore) ); plt.show()


# plt.plot(np.log10(np.abs(self.Mc2i.T))); plt.show()

# self = main(25,'loadOnly');
# self.setupConstraints()
# self.oneHat,self.x0 = self.remControlVrlbs('full','opCst')
# Mc2iCtrl = self.Mc2i.dot(self.oneHat)
# a2iCtrl = self.aIxfmr + self.Mc2i.dot(self.X0ctrl)
# score = []
# ii=0
# for lim in self.iXfmrLims:
    # mc2iCpx = Mc2iCtrl[ii]
    # a2iCpx = a2iCtrl[ii]
    # aCpx = np.r_[ a2iCpx.real, a2iCpx.imag].reshape((2,1))
    # aCpx = np.r_[ a2iCpx.real, a2iCpx.imag].reshape((2,1))
    # mCpx = np.r_[ [mc2iCpx.real],[mc2iCpx.imag] ]
    # # score.append( np.linalg.norm( mc2iCpx/(lim*self.iScale*np.ones((2,1)) - aCpx) ) )
    # score.append( np.linalg.norm( mc2iCpx/(lim*self.iScale) ) )
    # # score.append( np.linalg.norm( lim*self.iScale*np.ones((2,1)) - aCpx)/np.linalg.norm(mc2iCpx) )
    # ii+=1

# score = np.array(score)
# plt.plot(np.log10(score[score>1e-3])); plt.show()
# plt.plot(np.log10(score[score>1e-9])); plt.show()

# xScoreThing = np.sqrt(np.sum(np.abs(self.Mc2i),axis=1))/self.iScale*self.iXfmrLims
# plt.plot( np.log10(xScoreThing) ); plt.show()

# Mc2iX = dsf.vmM(1/self.iXfmrLims,self.Mc2i);
# Mc2iX = dsf.vmM(1/np.max(abs(Mc2iX),axis=1),abs(Mc2iX))
# plt.spy( Mc2iX>1e-4 ); plt.show()

# (0.1*np.ones(21))**np.linspace(0,20,21)

# plt.spy( np.abs(Mc2iX)<1e-10 ); plt.show()
# plt.spy( np.abs(Mc2iX)<1e-10 ); plt.show()
# plt.spy( np.abs(Mc2iX)<1e-7 ); plt.show()
# plt.spy( np.abs(Mc2iX)<1e-4 ); plt.show()

# print( 'num 0 bef:',np.sum(np.abs(Mc2iX)==0) )
# print( 'num 0 aft:',np.sum(np.abs(Mc2iX)<1e-14) )




# feeder = 0
# obj = 'opCst'
# strategy = 'full'
# pCvr = 0.8
# linPoint = 0.1
# self = main(feeder,pCvr=pCvr,modelType='loadOnly',linPoint=linPoint); # self.initialiseOpenDss();
# self.loadQpSet(); self.loadQpSln(strategy,obj); self.showQpSln()

# self.testGenSetting(k=np.arange(-10,11,2),dPlim=0.10,dQlim=0.10); plt.show()
# self.slnD = self.qpDssValidation(method='relaxT')

# # for feeder epri24
# cns['mvHi'] = 1.10
# cns['mvLo'] = 0.92
# cns['lvHi'] = 1.10
# cns['lvLo'] = 0.92
# cns['iScale'] = 4.0
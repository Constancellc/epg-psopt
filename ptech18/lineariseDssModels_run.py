import lineariseDssModels, sys, os
from importlib import reload
import numpy as np
from dss_python_funcs import vecSlc, getBusCoords, getBusCoordsAug
import matplotlib.pyplot as plt

FD = sys.argv[0]

def main(fdr_i=5,nrelTest=False,linPoint=1.0,pCvr=0.8):
    reload(lineariseDssModels)
    
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=fdr_i,linPoints=[linPoint],nrelTest=nrelTest,pCvr=pCvr)
    
    return blm

# self = main(23,nrelTest=False,linPoint=0.6)
# self = main(5,nrelTest=False,linPoint=0.6)
# self = main(5,True,0.6)
# self = main(8,pCvr=0.3,nrelTest=True)
self = main(5,pCvr=0.3,nrelTest=True)
# self = main('n1',pCvr=0.3,nrelTest=True)
# self = main('n1',pCvr=0.3,nrelTest=True)


# TODAY: start creating and validating a model that takes 
# load CVR stuff into account.



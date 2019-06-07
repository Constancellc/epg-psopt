import lineariseDssModels, sys, os
from importlib import reload
import numpy as np
from dss_python_funcs import vecSlc, getBusCoords, getBusCoordsAug
import matplotlib.pyplot as plt

from scipy.linalg import toeplitz

FD = sys.argv[0]

def main(fdr_i=5,linPoint=1.0,pCvr=0.8,method='fpl'):
    reload(lineariseDssModels)
    
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=fdr_i,linPoints=[linPoint],pCvr=pCvr,method=method)
    
    return blm

# self = main(5,method='fot',pCvr=0.0)
self = main(5)
# TO DO: 
# Results of 4x control schemes.

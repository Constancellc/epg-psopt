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

self = main(20)
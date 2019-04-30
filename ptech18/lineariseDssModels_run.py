import lineariseDssModels, sys, os
from importlib import reload
import numpy as np

FD = sys.argv[0]

def main(fdr_i=5,nrelTest=False):
    reload(lineariseDssModels)
    
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=fdr_i,linPoints=np.array([1.0]),nrelTest=nrelTest)
    return blm

blm = main(5,True)
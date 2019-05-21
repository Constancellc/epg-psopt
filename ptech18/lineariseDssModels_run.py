import lineariseDssModels, sys, os
from importlib import reload
import numpy as np
from dss_python_funcs import vecSlc
import matplotlib.pyplot as plt

FD = sys.argv[0]

def main(fdr_i=5,nrelTest=False,linPoint=1.0):
    reload(lineariseDssModels)
    
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=fdr_i,linPoints=[linPoint],nrelTest=nrelTest)
    return blm


# self = main(23,nrelTest=False,linPoint=0.6)
# self = main(5,nrelTest=False,linPoint=0.6)
# self = main(5,True,0.6)
self = main(17)
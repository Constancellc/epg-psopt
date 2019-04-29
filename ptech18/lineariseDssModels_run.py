import lineariseDssModels, sys, os
from importlib import reload
import numpy as np

FD = sys.argv[0]

def main():
    reload(lineariseDssModels)
    blm = lineariseDssModels.buildLinModel(FD=FD,fdr_i=6,linPoints=np.array([1.0]))
    
main()